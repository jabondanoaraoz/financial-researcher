"""
Warren Buffett Agent
====================
"Wonderful companies at fair prices" — hybrid quant + LLM agent.

Scoring weights (total /20):
    Moat                10 pts  (50%) — ROE consistency, ROIC, gross margin durability
    Management Quality   4 pts  (20%) — Owner Earnings, FCF, ROIC vs WACC
    Financial Discipline 3 pts  (15%) — D/E, interest coverage, current ratio
    Valuation            3 pts  (15%) — Owner Earnings Yield, P/B, P/E vs growth

Key differentiators vs other agents:
    • Measures ROE *consistency* across years (std_dev), not just a snapshot
    • Uses Owner Earnings (OCF - Capex) as Buffett's preferred cash metric
    • Penalises ROIC < WACC (capital destruction even with positive earnings)
    • Gross margin stability as a proxy for pricing power / durable moat

Author: Financial Researcher Team
"""

import logging
import numpy as np
from typing import Optional

from agents.base_agent import BaseAgent, AgentSignal
from agents.base_agent import SIGNAL_BULLISH, SIGNAL_NEUTRAL, SIGNAL_BEARISH
from agents.base_agent import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

EQUITY_RISK_PREMIUM = 0.055   # 5.5% — standard Damodaran ERP for US market
TAX_RATE_DEFAULT    = 0.21    # 21% US corporate tax rate


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Warren Buffett, Chairman of Berkshire Hathaway, performing a stock analysis.
Your philosophy: buy wonderful companies at fair prices and hold forever. You look for durable competitive moats
(high, consistent ROE and gross margins), excellent capital allocation (ROIC > WACC), and management
that treats shareholders as partners. You are patient, deeply sceptical of leverage, and you never
pay more than intrinsic value. You think in decades, not quarters.

You will receive quantitative scores (0–20) across four pillars plus the underlying metrics.
Based on this data, produce a disciplined investment opinion in the following JSON format — no prose outside the JSON:

{
  "signal": "bullish" | "neutral" | "bearish",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<2–3 paragraphs embodying your philosophy and referencing the specific numbers>",
  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "target_action": "buy" | "hold" | "sell"
}"""


# ──────────────────────────────────────────────────────────────────────────────
# Helper: multi-year series extractor
# ──────────────────────────────────────────────────────────────────────────────

def _annual_series(df, row: str, n_years: int = 4) -> list[float]:
    """Return up to n_years of annual values for a given row in a financial DataFrame."""
    if df is None or df.empty or row not in df.index:
        return []
    values = []
    for col in df.columns[:n_years]:
        val = df.loc[row, col]
        try:
            v = float(val)
            if not np.isnan(v):
                values.append(v)
        except (TypeError, ValueError):
            pass
    return values


# ──────────────────────────────────────────────────────────────────────────────
# Scoring pillars
# ──────────────────────────────────────────────────────────────────────────────

def _score_moat(financials: dict, av: dict) -> dict:
    """
    Moat Score /10

    Sub-metrics:
        ROE mean (3 pts)   — sustained returns signal an economic moat
        ROE stability (2 pts) — low std_dev = Buffett's "predictable earnings"
        ROIC (2.5 pts)     — true return on all invested capital
        Gross margin level (1.5 pts) — pricing power
        Gross margin stability (1 pt) — margin durability over time
    """
    score = 0.0
    detail = {}

    inc = financials.get("income_statement")
    bs  = financials.get("balance_sheet")

    # ── ROE consistency ───────────────────────────────────────────────
    net_incomes = _annual_series(inc, "Net Income")
    equities    = _annual_series(bs,  "Stockholders Equity") or \
                  _annual_series(bs,  "Total Stockholders Equity")

    roe_series = []
    for ni, eq in zip(net_incomes, equities):
        if eq and eq > 0:
            roe_series.append((ni / eq) * 100)

    if roe_series:
        roe_mean = np.mean(roe_series)
        roe_std  = np.std(roe_series) if len(roe_series) > 1 else 0.0

        # ROE mean: 0-3 pts
        if roe_mean >= 20:   roe_pts = 3.0
        elif roe_mean >= 15: roe_pts = 2.0
        elif roe_mean >= 10: roe_pts = 1.0
        else:                roe_pts = 0.0

        # ROE stability: 0-2 pts (lower std_dev = better)
        if roe_std <= 3:     stab_pts = 2.0
        elif roe_std <= 7:   stab_pts = 1.5
        elif roe_std <= 12:  stab_pts = 1.0
        elif roe_std <= 18:  stab_pts = 0.5
        else:                stab_pts = 0.0

        score += roe_pts + stab_pts
        detail["roe_mean"]    = {"value": round(roe_mean, 1), "pts": round(roe_pts, 1),  "max": 3}
        detail["roe_std_dev"] = {"value": round(roe_std,  1), "pts": round(stab_pts, 1), "max": 2}
    else:
        # Fall back to AV snapshot
        roe_snap = (av.get("return_on_equity_ttm") or 0) * 100
        roe_pts  = 1.5 if roe_snap >= 15 else (1.0 if roe_snap >= 10 else 0.0)
        score += roe_pts
        detail["roe_snapshot"] = {"value": round(roe_snap, 1), "pts": round(roe_pts, 1), "max": 3}

    # ── ROIC ──────────────────────────────────────────────────────────
    roic = None
    if inc is not None and bs is not None and not inc.empty and not bs.empty:
        col = inc.columns[0]
        op_inc = inc.loc["Operating Income", col] if "Operating Income" in inc.index else None
        tax_rate = TAX_RATE_DEFAULT
        # Try to estimate tax rate from data
        if "Tax Provision" in inc.index and "Pretax Income" in inc.index:
            pretax = inc.loc["Pretax Income", col]
            tax    = inc.loc["Tax Provision",  col]
            if pretax and pretax > 0:
                tax_rate = min(max(tax / pretax, 0), 0.40)

        bs_col    = bs.columns[0]
        total_eq  = bs.loc["Stockholders Equity",   bs_col] if "Stockholders Equity"   in bs.index else None
        total_dbt = bs.loc["Total Debt",             bs_col] if "Total Debt"             in bs.index else None
        cash      = bs.loc["Cash And Cash Equivalents", bs_col] if "Cash And Cash Equivalents" in bs.index else (
                    bs.loc["Cash Cash Equivalents And Short Term Investments", bs_col]
                    if "Cash Cash Equivalents And Short Term Investments" in bs.index else None)

        if op_inc and total_eq:
            nopat    = op_inc * (1 - tax_rate)
            inv_cap  = (total_eq or 0) + (total_dbt or 0) - (cash or 0)
            if inv_cap > 0:
                roic = (nopat / inv_cap) * 100

    if roic is not None:
        if roic >= 18:   roic_pts = 2.5
        elif roic >= 15: roic_pts = 2.0
        elif roic >= 12: roic_pts = 1.5
        elif roic >= 8:  roic_pts = 0.75
        else:            roic_pts = 0.0
        score += roic_pts
        detail["roic"] = {"value": round(roic, 1), "pts": round(roic_pts, 2), "max": 2.5}

    # ── Gross Margin level + stability ───────────────────────────────
    revenues = _annual_series(inc, "Total Revenue")
    grossps  = _annual_series(inc, "Gross Profit")

    gm_series = []
    for gp, rev in zip(grossps, revenues):
        if rev and rev > 0:
            gm_series.append((gp / rev) * 100)

    if gm_series:
        gm_mean = np.mean(gm_series)
        gm_std  = np.std(gm_series) if len(gm_series) > 1 else 0.0

        # Gross margin level: 0-1.5 pts
        if gm_mean >= 55:   gm_pts = 1.5
        elif gm_mean >= 45: gm_pts = 1.0
        elif gm_mean >= 35: gm_pts = 0.5
        else:               gm_pts = 0.0

        # Gross margin stability: 0-1 pts
        if gm_std <= 2:     gm_stab_pts = 1.0
        elif gm_std <= 5:   gm_stab_pts = 0.5
        else:               gm_stab_pts = 0.0

        score += gm_pts + gm_stab_pts
        detail["gross_margin_mean"]   = {"value": round(gm_mean, 1), "pts": round(gm_pts, 1),      "max": 1.5}
        detail["gross_margin_std_dev"]= {"value": round(gm_std,  1), "pts": round(gm_stab_pts, 1), "max": 1.0}

    return {"score": round(min(score, 10), 2), "max": 10, "detail": detail}


def _score_management(financials: dict, key_metrics: dict, risk_free_rate: float) -> dict:
    """
    Management Quality Score /4

    Sub-metrics:
        Owner Earnings Yield (2 pts) — (OCF - Capex) / Market Cap
        FCF consistently positive (1 pt) — Buffett wants reliable cash generation
        ROIC vs WACC spread (1 pt) — are they creating or destroying value?
    """
    score = 0.0
    detail = {}

    cf  = financials.get("cash_flow")
    inc = financials.get("income_statement")
    bs  = financials.get("balance_sheet")

    mkt_cap = key_metrics.get("enterprise_value") or \
              ((key_metrics.get("shares_outstanding") or 0) * (key_metrics.get("current_price") or 0))

    # ── Owner Earnings Yield ──────────────────────────────────────────
    if cf is not None and not cf.empty and mkt_cap and mkt_cap > 0:
        col = cf.columns[0]
        ocf   = cf.loc["Operating Cash Flow",    col] if "Operating Cash Flow"    in cf.index else None
        capex = cf.loc["Capital Expenditure",    col] if "Capital Expenditure"    in cf.index else (
                cf.loc["Capital Expenditures",   col] if "Capital Expenditures"   in cf.index else None)

        if ocf and capex is not None:
            owner_earnings = ocf + capex   # capex is usually negative in financial statements
            oey = (owner_earnings / mkt_cap) * 100

            if oey >= 6:     oey_pts = 2.0
            elif oey >= 4:   oey_pts = 1.5
            elif oey >= 2.5: oey_pts = 1.0
            elif oey >= 1:   oey_pts = 0.5
            else:            oey_pts = 0.0

            score += oey_pts
            detail["owner_earnings_yield"] = {"value": f"{round(oey, 2)}%", "pts": round(oey_pts, 1), "max": 2}

    # ── FCF consistency ───────────────────────────────────────────────
    if cf is not None and not cf.empty:
        fcf_values = []
        fcf_row = "Free Cash Flow" if "Free Cash Flow" in cf.index else None
        if not fcf_row:
            ocf_vals = _annual_series(cf, "Operating Cash Flow", n_years=4)
            cap_vals = _annual_series(cf, "Capital Expenditure",  n_years=4) or \
                       _annual_series(cf, "Capital Expenditures", n_years=4)
            if ocf_vals and cap_vals:
                fcf_values = [o + c for o, c in zip(ocf_vals, cap_vals)]
        else:
            fcf_values = _annual_series(cf, fcf_row, n_years=4)

        if fcf_values:
            n_positive = sum(1 for v in fcf_values if v > 0)
            pct = n_positive / len(fcf_values)
            fcf_pts = 1.0 if pct == 1.0 else (0.5 if pct >= 0.75 else 0.0)
            score += fcf_pts
            detail["fcf_consistency"] = {
                "value": f"{n_positive}/{len(fcf_values)} years positive",
                "pts": round(fcf_pts, 1), "max": 1
            }

    # ── ROIC vs WACC spread ───────────────────────────────────────────
    beta = key_metrics.get("beta") or 1.0
    cost_of_equity = risk_free_rate + beta * EQUITY_RISK_PREMIUM

    wacc = cost_of_equity   # simplified: treat as all-equity if we can't split

    if bs is not None and not bs.empty and inc is not None and not inc.empty:
        bs_col = bs.columns[0]
        ic_col = inc.columns[0]
        total_dbt = bs.loc["Total Debt", bs_col] if "Total Debt" in bs.index else 0
        total_eq  = bs.loc["Stockholders Equity", bs_col] if "Stockholders Equity" in bs.index else None
        int_exp   = inc.loc["Interest Expense", ic_col] if "Interest Expense" in inc.index else None

        if total_eq and (total_eq + (total_dbt or 0)) > 0:
            E = total_eq
            D = total_dbt or 0
            V = E + D
            Ke = cost_of_equity
            Kd = (abs(int_exp) / D * (1 - TAX_RATE_DEFAULT)) if (D > 0 and int_exp) else 0
            wacc = (E / V) * Ke + (D / V) * Kd

    # Get ROIC from detail if we computed it in moat (re-compute simplified)
    roic_approx = None
    if inc is not None and not inc.empty:
        ic = inc.columns[0]
        op_inc = inc.loc["Operating Income", ic] if "Operating Income" in inc.index else None
        if op_inc and bs is not None and not bs.empty:
            bsc = bs.columns[0]
            eq  = bs.loc["Stockholders Equity", bsc] if "Stockholders Equity" in bs.index else None
            dbt = bs.loc["Total Debt", bsc] if "Total Debt" in bs.index else 0
            csh = bs.loc["Cash And Cash Equivalents", bsc] if "Cash And Cash Equivalents" in bs.index else 0
            inv_cap = (eq or 0) + (dbt or 0) - (csh or 0)
            if inv_cap > 0:
                roic_approx = (op_inc * (1 - TAX_RATE_DEFAULT)) / inv_cap * 100

    if roic_approx is not None:
        spread = roic_approx - wacc * 100
        if spread >= 8:    spread_pts = 1.0
        elif spread >= 4:  spread_pts = 0.75
        elif spread >= 0:  spread_pts = 0.5
        else:              spread_pts = 0.0
        score += spread_pts
        detail["roic_vs_wacc"] = {
            "value": f"ROIC {round(roic_approx, 1)}% vs WACC {round(wacc*100, 1)}% (spread {round(spread, 1)}%)",
            "pts": round(spread_pts, 2), "max": 1
        }

    return {"score": round(min(score, 4), 2), "max": 4, "detail": detail}


def _score_financial_discipline(financials: dict, key_metrics: dict) -> dict:
    """
    Financial Discipline Score /3

    Sub-metrics:
        D/E ratio (1 pt)         — Buffett prefers companies that self-finance
        Interest coverage (1 pt) — can the company survive a downturn?
        Current ratio (1 pt)     — short-term liquidity buffer
    """
    score = 0.0
    detail = {}

    inc = financials.get("income_statement")
    bs  = financials.get("balance_sheet")

    if bs is not None and not bs.empty:
        col = bs.columns[0]

        total_dbt = bs.loc["Total Debt",            col] if "Total Debt"            in bs.index else None
        equity    = bs.loc["Stockholders Equity",   col] if "Stockholders Equity"   in bs.index else None
        cur_assets= bs.loc["Current Assets",        col] if "Current Assets"        in bs.index else None
        cur_liab  = bs.loc["Current Liabilities",   col] if "Current Liabilities"   in bs.index else None

        # D/E
        if total_dbt is not None and equity and equity > 0:
            de = total_dbt / equity
            if de <= 0.3:   de_pts = 1.0
            elif de <= 0.7: de_pts = 0.75
            elif de <= 1.2: de_pts = 0.5
            elif de <= 2.0: de_pts = 0.25
            else:           de_pts = 0.0
            score += de_pts
            detail["debt_to_equity"] = {"value": round(de, 2), "pts": round(de_pts, 2), "max": 1}

        # Current Ratio
        if cur_assets and cur_liab and cur_liab > 0:
            cr = cur_assets / cur_liab
            if cr >= 2.5:   cr_pts = 1.0
            elif cr >= 2.0: cr_pts = 0.75
            elif cr >= 1.5: cr_pts = 0.5
            elif cr >= 1.0: cr_pts = 0.25
            else:           cr_pts = 0.0
            score += cr_pts
            detail["current_ratio"] = {"value": round(cr, 2), "pts": round(cr_pts, 2), "max": 1}

    # Interest coverage
    if inc is not None and not inc.empty:
        col = inc.columns[0]
        ebit    = inc.loc["Operating Income", col] if "Operating Income" in inc.index else None
        int_exp = inc.loc["Interest Expense", col] if "Interest Expense" in inc.index else None
        if ebit and int_exp and int_exp != 0:
            coverage = ebit / abs(int_exp)
            if coverage >= 15:  ic_pts = 1.0
            elif coverage >= 8: ic_pts = 0.75
            elif coverage >= 5: ic_pts = 0.5
            elif coverage >= 2: ic_pts = 0.25
            else:               ic_pts = 0.0
            score += ic_pts
            detail["interest_coverage"] = {"value": round(coverage, 1), "pts": round(ic_pts, 2), "max": 1}

    return {"score": round(min(score, 3), 2), "max": 3, "detail": detail}


def _score_valuation(financials: dict, key_metrics: dict, av: dict) -> dict:
    """
    Valuation Score /3

    Sub-metrics:
        Owner Earnings Yield (1.5 pts) — primary Buffett valuation metric
        P/B ratio (0.75 pts)           — intrinsic value anchor
        P/E vs earnings growth (0.75 pts) — simplified PEG-style check
    """
    score = 0.0
    detail = {}

    cf      = financials.get("cash_flow")
    mkt_cap = (key_metrics.get("shares_outstanding") or 0) * (key_metrics.get("current_price") or 0)

    # ── Owner Earnings Yield ──────────────────────────────────────────
    if cf is not None and not cf.empty and mkt_cap > 0:
        col = cf.columns[0]
        ocf   = cf.loc["Operating Cash Flow",  col] if "Operating Cash Flow"  in cf.index else None
        capex = cf.loc["Capital Expenditure",  col] if "Capital Expenditure"  in cf.index else (
                cf.loc["Capital Expenditures", col] if "Capital Expenditures" in cf.index else None)
        if ocf and capex is not None:
            oe   = ocf + capex
            oey  = (oe / mkt_cap) * 100
            if oey >= 6:     oey_pts = 1.5
            elif oey >= 4:   oey_pts = 1.0
            elif oey >= 2.5: oey_pts = 0.5
            else:            oey_pts = 0.0
            score += oey_pts
            detail["owner_earnings_yield"] = {"value": f"{round(oey, 2)}%", "pts": round(oey_pts, 2), "max": 1.5}

    # ── P/B ratio ─────────────────────────────────────────────────────
    pb = key_metrics.get("pb_ratio") or (av.get("price_to_book") if av else None)
    if pb and pb > 0:
        if pb <= 2:     pb_pts = 0.75
        elif pb <= 3.5: pb_pts = 0.5
        elif pb <= 5:   pb_pts = 0.25
        else:           pb_pts = 0.0
        score += pb_pts
        detail["price_to_book"] = {"value": round(pb, 2), "pts": round(pb_pts, 2), "max": 0.75}

    # ── P/E vs growth ─────────────────────────────────────────────────
    pe = key_metrics.get("pe_ratio") or (av.get("pe_ratio") if av else None)
    growth = None
    if av and av.get("quarterly_earnings_growth_yoy"):
        growth = av["quarterly_earnings_growth_yoy"] * 100
    elif financials.get("income_statement") is not None:
        inc = financials["income_statement"]
        if not inc.empty and inc.shape[1] >= 2 and "Net Income" in inc.index:
            ni0 = inc.loc["Net Income", inc.columns[0]]
            ni1 = inc.loc["Net Income", inc.columns[1]]
            if ni1 and ni1 > 0:
                growth = ((ni0 - ni1) / abs(ni1)) * 100

    if pe and pe > 0 and growth is not None and growth > 0:
        peg = pe / growth
        if peg <= 1:    peg_pts = 0.75
        elif peg <= 1.5:peg_pts = 0.5
        elif peg <= 2:  peg_pts = 0.25
        else:           peg_pts = 0.0
        score += peg_pts
        detail["pe_vs_growth_peg"] = {"value": round(peg, 2), "pts": round(peg_pts, 2), "max": 0.75}
    elif pe and pe > 0:
        if pe <= 15:    pe_pts = 0.75
        elif pe <= 22:  pe_pts = 0.5
        elif pe <= 30:  pe_pts = 0.25
        else:           pe_pts = 0.0
        score += pe_pts
        detail["pe_ratio"] = {"value": round(pe, 2), "pts": round(pe_pts, 2), "max": 0.75}

    return {"score": round(min(score, 3), 2), "max": 3, "detail": detail}


# ──────────────────────────────────────────────────────────────────────────────
# Agent class
# ──────────────────────────────────────────────────────────────────────────────

class WarrenBuffettAgent(BaseAgent):
    """Warren Buffett — wonderful companies at fair prices."""

    def __init__(self, llm: Optional[LLMClient] = None):
        super().__init__(agent_id="warren_buffett", agent_name="Warren Buffett")
        self.llm = llm or LLMClient()

    def analyze(self, data: dict, ticker: str) -> AgentSignal:
        metrics    = data.get("key_metrics")   or {}
        financials = data.get("financials")    or {}
        av         = data.get("av_overview")   or {}
        risk_free  = data.get("risk_free_rate") or 0.043

        # ── Score all four pillars ─────────────────────────────────────
        moat     = _score_moat(financials, av)
        mgmt     = _score_management(financials, metrics, risk_free)
        disc     = _score_financial_discipline(financials, metrics)
        val      = _score_valuation(financials, metrics, av)

        total      = moat["score"] + mgmt["score"] + disc["score"] + val["score"]
        total_max  = 20.0
        norm_score = total / total_max          # 0.0 – 1.0

        company_name = data.get("company_info", {}).get("name", ticker) if data.get("company_info") else ticker

        # ── LLM reasoning ─────────────────────────────────────────────
        user_prompt = f"""
Ticker: {ticker} ({company_name})

QUANTITATIVE SCORES (total {round(total, 2)}/{total_max}):

1. MOAT SCORE: {moat['score']}/{moat['max']}
{_fmt_detail(moat['detail'])}

2. MANAGEMENT QUALITY: {mgmt['score']}/{mgmt['max']}
{_fmt_detail(mgmt['detail'])}

3. FINANCIAL DISCIPLINE: {disc['score']}/{disc['max']}
{_fmt_detail(disc['detail'])}

4. VALUATION: {val['score']}/{val['max']}
{_fmt_detail(val['detail'])}

Key market context:
  Current Price: ${metrics.get('current_price', 'N/A')}
  Market Cap: ${_fmt_big(metrics.get('enterprise_value'))}
  Beta: {metrics.get('beta', 'N/A')}
  Risk-Free Rate: {round(risk_free * 100, 2)}%

Provide your investment opinion as JSON.
"""

        llm_result = self.llm.generate_json(SYSTEM_PROMPT, user_prompt)

        # ── Parse LLM output ──────────────────────────────────────────
        signal        = llm_result.get("signal", _norm_score_to_signal(norm_score))
        confidence    = float(llm_result.get("confidence", norm_score))
        reasoning     = llm_result.get("reasoning", "LLM response unavailable.")
        key_risks     = llm_result.get("key_risks", [])
        target_action = llm_result.get("target_action", _signal_to_action(signal))

        # Validate
        if signal not in ("bullish", "neutral", "bearish"):
            signal = _norm_score_to_signal(norm_score)
        if target_action not in ("buy", "hold", "sell", "short", "cover"):
            target_action = _signal_to_action(signal)

        self.logger.info(
            f"{ticker} → {signal.upper()} (conf={confidence:.0%}) | "
            f"total={round(total,1)}/{total_max} | "
            f"moat={moat['score']} mgmt={mgmt['score']} disc={disc['score']} val={val['score']}"
        )

        return AgentSignal(
            agent_id      = self.agent_id,
            agent_name    = self.agent_name,
            ticker        = ticker,
            signal        = signal,
            confidence    = round(min(max(confidence, 0), 1), 3),
            scores        = {
                "total":      round(total, 2),
                "total_max":  total_max,
                "moat":       {"score": moat["score"],  "max": moat["max"],  "detail": moat["detail"]},
                "management": {"score": mgmt["score"],  "max": mgmt["max"],  "detail": mgmt["detail"]},
                "discipline": {"score": disc["score"],  "max": disc["max"],  "detail": disc["detail"]},
                "valuation":  {"score": val["score"],   "max": val["max"],   "detail": val["detail"]},
            },
            reasoning     = reasoning,
            key_risks     = key_risks,
            target_action = target_action,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_detail(detail: dict) -> str:
    lines = []
    for k, v in detail.items():
        lines.append(f"  {k}: {v['value']}  ({v['pts']}/{v['max']} pts)")
    return "\n".join(lines) if lines else "  (no data)"


def _fmt_big(v) -> str:
    if v is None: return "N/A"
    if v >= 1e12: return f"{v/1e12:.2f}T"
    if v >= 1e9:  return f"{v/1e9:.2f}B"
    return f"{v:,.0f}"


def _norm_score_to_signal(score: float) -> str:
    if score >= 0.60: return SIGNAL_BULLISH
    if score >= 0.35: return SIGNAL_NEUTRAL
    return SIGNAL_BEARISH


def _signal_to_action(signal: str) -> str:
    return {"bullish": ACTION_BUY, "neutral": ACTION_HOLD, "bearish": ACTION_SELL}.get(signal, ACTION_HOLD)
