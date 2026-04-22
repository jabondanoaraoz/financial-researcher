"""
Valuation Agent
"What is this business worth?" - three independent methods, one consensus.

Calculates intrinsic value using three methodologies and triangulates a
weighted fair value estimate. Uses LLM only to generate the reasoning
narrative - all numbers are computed deterministically.

Methods:
    1. DCF (50% weight)       - 5-year revenue projection → FCF → discount at WACC
    2. Peer Multiples (30%)   - median EV/EBITDA of peers × target EBITDA
    3. Graham Number (20%)    - sqrt(22.5 × EPS × BVPS)

If peer data is unavailable, weights shift to 65% DCF / 35% Graham.

Output: estimated fair value, implied upside/downside, signal.

Key differentiators vs Damodaran agent:
    • Uses revenue-based (not FCFF) DCF - simpler, faster, complementary
    • Adds peer-relative valuation (market-implied benchmark)
    • Graham Number as hard floor anchor
    • Combines all three into a single consensus price target

"""

import logging
import math
import numpy as np
from typing import Optional

from agents.base_agent import BaseAgent, AgentSignal
from agents.base_agent import SIGNAL_BULLISH, SIGNAL_NEUTRAL, SIGNAL_BEARISH
from agents.base_agent import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

EQUITY_RISK_PREMIUM = 0.055
TAX_RATE            = 0.21
TERMINAL_GROWTH     = 0.03
PROJECTION_YEARS    = 5


SYSTEM_PROMPT = """You are a senior equity analyst performing a multi-method stock valuation.
You have three independent intrinsic value estimates - DCF, peer multiples, and Graham Number —
and a weighted consensus fair value. Your job is to synthesize these into a coherent narrative
that explains what the numbers mean and whether the current price is justified.

You distrust any single valuation method in isolation. You look for convergence across methods
as a signal of reliability, and divergence as a warning about model sensitivity.

You will receive the valuation outputs and current price. Produce your opinion as JSON only:

{
  "signal": "bullish" | "neutral" | "bearish",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<2–3 paragraphs citing each method's fair value, convergence/divergence, and implied upside>",
  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "target_action": "buy" | "hold" | "sell"
}"""


# Helpers

def _annual_series(df, row: str, n: int = 5) -> list[float]:
    if df is None or df.empty or row not in df.index:
        return []
    out = []
    for col in df.columns[:n]:
        try:
            v = float(df.loc[row, col])
            if not math.isnan(v):
                out.append(v)
        except (TypeError, ValueError):
            pass
    return out


def _first(df, *rows):
    if df is None or df.empty:
        return None
    col = df.columns[0]
    for row in rows:
        if row in df.index:
            try:
                v = float(df.loc[row, col])
                if not math.isnan(v):
                    return v
            except (TypeError, ValueError):
                pass
    return None


def _wacc(key_metrics: dict, financials: dict, risk_free: float) -> float:
    """Compute WACC from beta, risk-free rate, and capital structure."""
    beta = key_metrics.get("beta") or 1.0
    ke   = risk_free + beta * EQUITY_RISK_PREMIUM   # cost of equity

    bs  = financials.get("balance_sheet")
    inc = financials.get("income_statement")

    if bs is not None and not bs.empty and inc is not None and not inc.empty:
        equity    = _first(bs, "Stockholders Equity", "Total Stockholders Equity") or 0
        debt      = _first(bs, "Total Debt") or 0
        int_exp   = abs(_first(inc, "Interest Expense") or 0)
        kd_pretax = int_exp / debt if debt > 0 else 0
        kd        = kd_pretax * (1 - TAX_RATE)
        V         = equity + debt
        if V > 0:
            return (equity / V) * ke + (debt / V) * kd

    return ke   # all-equity fallback


# Valuation methods

def _dcf_value(financials: dict, key_metrics: dict, risk_free: float) -> Optional[dict]:
    """
    5-year revenue-based DCF.
    Projects revenue at historical CAGR, applies FCF margin, discounts at WACC.
    Returns per-share intrinsic value.
    """
    inc = financials.get("income_statement")
    cf  = financials.get("cash_flow")
    bs  = financials.get("balance_sheet")

    revenues = _annual_series(inc, "Total Revenue", n=4)
    if len(revenues) < 2:
        return None

    # Revenue CAGR
    cagr = ((revenues[0] / revenues[-1]) ** (1 / (len(revenues) - 1)) - 1)
    # Clamp to reasonable range: 0% – 40%
    cagr = min(max(cagr, 0.0), 0.40)

    # FCF margin from most recent year
    ocf   = _first(cf, "Operating Cash Flow")
    capex = _first(cf, "Capital Expenditure", "Capital Expenditures")
    rev0  = revenues[0]
    if ocf and capex and rev0 > 0:
        fcf_margin = (ocf + capex) / rev0   # capex is negative
    else:
        fcf_margin = 0.05   # 5% fallback

    # WACC
    wacc = _wacc(key_metrics, financials, risk_free)
    wacc = max(wacc, 0.06)   # floor at 6%

    # Project FCFs for 5 years
    pv_fcfs = 0.0
    rev_proj = rev0
    for yr in range(1, PROJECTION_YEARS + 1):
        rev_proj *= (1 + cagr)
        fcf_proj  = rev_proj * fcf_margin
        pv_fcfs  += fcf_proj / ((1 + wacc) ** yr)

    # Terminal value (Gordon Growth)
    fcf_terminal = rev_proj * fcf_margin * (1 + TERMINAL_GROWTH)
    tv = fcf_terminal / (wacc - TERMINAL_GROWTH)
    pv_tv = tv / ((1 + wacc) ** PROJECTION_YEARS)

    # Enterprise value → equity value per share
    ev_implied = pv_fcfs + pv_tv
    cash  = _first(bs, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents") or 0
    debt  = _first(bs, "Total Debt") or 0
    equity_val = ev_implied + cash - debt

    shares = key_metrics.get("shares_outstanding")
    if not shares or shares <= 0:
        return None

    per_share = equity_val / shares

    return {
        "fair_value":   round(per_share, 2),
        "wacc":         round(wacc * 100, 2),
        "cagr_used":    round(cagr * 100, 2),
        "fcf_margin":   round(fcf_margin * 100, 2),
        "ev_implied":   ev_implied,
    }


def _peer_multiple_value(financials: dict, key_metrics: dict, peers_data: dict) -> Optional[dict]:
    """
    EV/EBITDA peer median × target EBITDA → equity value per share.
    peers_data: {ticker: {"ev_ebitda": float, ...}}
    """
    if not peers_data:
        return None

    peer_multiples = []
    for pt, pm in peers_data.items():
        ev_eb = (pm or {}).get("ev_ebitda")
        if ev_eb and ev_eb > 0:
            peer_multiples.append(ev_eb)

    if not peer_multiples:
        return None

    median_multiple = float(np.median(peer_multiples))

    # Target EBITDA (Operating Income + D&A proxy)
    inc = financials.get("income_statement")
    cf  = financials.get("cash_flow")
    bs  = financials.get("balance_sheet")

    ebit = _first(inc, "Operating Income", "EBIT")
    da   = abs(_first(cf, "Depreciation And Amortization", "Depreciation Amortization Depletion") or 0)
    if not ebit:
        return None

    ebitda = ebit + da
    if ebitda <= 0:
        return None

    ev_implied = median_multiple * ebitda
    cash  = _first(bs, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents") or 0
    debt  = _first(bs, "Total Debt") or 0
    equity_val = ev_implied + cash - debt

    shares = key_metrics.get("shares_outstanding")
    if not shares or shares <= 0:
        return None

    per_share = equity_val / shares

    return {
        "fair_value":      round(per_share, 2),
        "median_multiple": round(median_multiple, 2),
        "ebitda":          ebitda,
        "n_peers":         len(peer_multiples),
    }


def _graham_number_value(financials: dict, key_metrics: dict, av: dict) -> Optional[dict]:
    """
    Graham Number = sqrt(22.5 × EPS × BVPS)
    EPS from income statement; BVPS from balance sheet.
    """
    # EPS
    eps = None
    if av:
        eps = av.get("eps") or av.get("trailing_eps")
    if eps is None:
        inc    = financials.get("income_statement")
        net_ni = _first(inc, "Net Income")
        shares = key_metrics.get("shares_outstanding")
        if net_ni and shares and shares > 0:
            eps = net_ni / shares

    # BVPS
    bvps = None
    bs     = financials.get("balance_sheet")
    equity = _first(bs, "Stockholders Equity", "Total Stockholders Equity")
    shares = key_metrics.get("shares_outstanding")
    if equity and shares and shares > 0:
        bvps = equity / shares

    if eps is None or bvps is None or eps <= 0 or bvps <= 0:
        return None

    gn = math.sqrt(22.5 * eps * bvps)
    return {"fair_value": round(gn, 2), "eps": round(eps, 2), "bvps": round(bvps, 2)}


# Agent class

class ValuationAgent(BaseAgent):
    """Intrinsic value by three methods → weighted consensus fair value."""

    def __init__(self, llm: Optional[LLMClient] = None):
        super().__init__(agent_id="valuation", agent_name="Valuation Analyst")
        self.llm = llm or LLMClient()

    def analyze(self, data: dict, ticker: str) -> AgentSignal:
        metrics    = data.get("key_metrics")   or {}
        financials = data.get("financials")    or {}
        av         = data.get("av_overview")   or {}
        peers_data = data.get("peers_data")    or {}
        risk_free  = data.get("risk_free_rate") or 0.043

        price = metrics.get("current_price")
        shares = metrics.get("shares_outstanding")

        # Run three methods
        dcf    = _dcf_value(financials, metrics, risk_free)
        peer   = _peer_multiple_value(financials, metrics, peers_data)
        graham = _graham_number_value(financials, metrics, av)

        # Weighted consensus
        values, weights = [], []
        detail = {}

        if dcf:
            w = 0.50 if peer else 0.65
            values.append(dcf["fair_value"])
            weights.append(w)
            detail["dcf"] = {
                "value": f"${dcf['fair_value']} (WACC={dcf['wacc']}%, CAGR={dcf['cagr_used']}%, FCF margin={dcf['fcf_margin']}%)",
                "pts": w, "max": 0.65,
            }

        if peer:
            w = 0.30
            values.append(peer["fair_value"])
            weights.append(w)
            detail["peer_multiples"] = {
                "value": f"${peer['fair_value']} (median EV/EBITDA {peer['median_multiple']}x, {peer['n_peers']} peers)",
                "pts": w, "max": 0.30,
            }

        if graham:
            w = 0.20 if dcf else 0.35
            values.append(graham["fair_value"])
            weights.append(w)
            detail["graham_number"] = {
                "value": f"${graham['fair_value']} (EPS=${graham['eps']}, BVPS=${graham['bvps']})",
                "pts": w, "max": 0.35,
            }

        if not values:
            fair_value = None
            upside     = None
        else:
            # Normalize weights to 1.0
            total_w    = sum(weights)
            norm_w     = [w / total_w for w in weights]
            fair_value = round(sum(v * w for v, w in zip(values, norm_w)), 2)
            upside     = round((fair_value / price - 1) * 100, 1) if (price and price > 0) else None

        detail["consensus_fair_value"] = {
            "value": f"${fair_value} (upside {upside:+.1f}%)" if (fair_value and upside is not None) else "N/A",
            "pts": 0, "max": 0,
        }

        # Score: map upside to 0-20
        if upside is not None:
            if upside >= 40:    total = 18.0
            elif upside >= 25:  total = 15.0
            elif upside >= 10:  total = 12.0
            elif upside >= 0:   total = 10.0
            elif upside >= -15: total = 7.0
            elif upside >= -30: total = 4.0
            else:               total = 1.0
        else:
            total = 10.0   # neutral if no valuation possible

        total_max  = 20.0
        norm_score = total / total_max

        # LLM reasoning
        methods_block = ""
        if dcf:
            methods_block += f"\nDCF fair value: ${dcf['fair_value']} (WACC={dcf['wacc']}%, Revenue CAGR={dcf['cagr_used']}%, FCF margin={dcf['fcf_margin']}%)"
        if peer:
            methods_block += f"\nPeer Multiples: ${peer['fair_value']} (median EV/EBITDA={peer['median_multiple']}x across {peer['n_peers']} peers)"
        if graham:
            methods_block += f"\nGraham Number: ${graham['fair_value']} (EPS=${graham['eps']}, BVPS=${graham['bvps']})"
        if fair_value:
            methods_block += f"\n\nWeighted Consensus Fair Value: ${fair_value}"
            methods_block += f"\nCurrent Price: ${price}"
            methods_block += f"\nImplied Upside/Downside: {upside:+.1f}%" if upside is not None else ""

        company_name = (data.get("company_info", {}).get("name", ticker) if data.get("company_info") else ticker)

        user_prompt = f"""
Ticker: {ticker} ({company_name})

VALUATION METHODS:
{methods_block}

Provide your valuation opinion as JSON.
"""
        llm_result = self.llm.generate_json(SYSTEM_PROMPT, user_prompt)

        signal        = llm_result.get("signal",        _norm_to_signal(norm_score))
        confidence    = float(llm_result.get("confidence", norm_score))
        reasoning     = llm_result.get("reasoning",    "LLM response unavailable.")
        key_risks     = llm_result.get("key_risks",    [])
        target_action = llm_result.get("target_action", _signal_to_action(signal))

        if signal not in ("bullish", "neutral", "bearish"):
            signal = _norm_to_signal(norm_score)
        if target_action not in ("buy", "hold", "sell", "short", "cover"):
            target_action = _signal_to_action(signal)

        self.logger.info(
            f"{ticker} → {signal.upper()} (conf={confidence:.0%}) | "
            f"FV=${fair_value} upside={upside}% | "
            f"DCF=${dcf['fair_value'] if dcf else 'N/A'} "
            f"Peers=${peer['fair_value'] if peer else 'N/A'} "
            f"GN=${graham['fair_value'] if graham else 'N/A'}"
        )

        return AgentSignal(
            agent_id      = self.agent_id,
            agent_name    = self.agent_name,
            ticker        = ticker,
            signal        = signal,
            confidence    = round(min(max(confidence, 0), 1), 3),
            price_target  = fair_value,
            scores        = {
                "total":      round(total, 2),
                "total_max":  total_max,
                "methods":    {"score": round(total, 2), "max": total_max, "detail": detail},
                "fair_value": fair_value,
                "upside_pct": upside,
            },
            reasoning     = reasoning,
            key_risks     = key_risks,
            target_action = target_action,
        )


# Utilities

def _norm_to_signal(score: float) -> str:
    if score >= 0.60: return SIGNAL_BULLISH
    if score >= 0.35: return SIGNAL_NEUTRAL
    return SIGNAL_BEARISH


def _signal_to_action(signal: str) -> str:
    return {SIGNAL_BULLISH: ACTION_BUY, SIGNAL_NEUTRAL: ACTION_HOLD, SIGNAL_BEARISH: ACTION_SELL}.get(signal, ACTION_HOLD)
