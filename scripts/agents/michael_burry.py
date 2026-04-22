"""
Michael Burry Agent
"Find what everyone else is missing." - contrarian deep value.

Burry's philosophy: the market systematically misprices unloved, complex,
or misunderstood companies. He digs into financials to find businesses
trading below intrinsic value, with hidden balance-sheet strength and
real free cash flow - while the crowd ignores or actively bets against them.

Scoring (/20):
    Deep Value Metrics      5 pts  (25%) - EV/EBITDA absolute + FCF yield
    Hidden Value            5 pts  (25%) - Net cash position + P/Tangible Book
    Solvency & Survival     5 pts  (25%) - D/E + Interest coverage
    Contrarian Signals      5 pts  (25%) - Insider net buying + Short interest (high = bullish)

Key differentiators vs other agents:
    • EV/EBITDA with absolute thresholds - no story, just hard numbers
    • Tangible Book strips goodwill/intangibles: Burry wants real asset backing
    • Net cash as % of market cap - hidden balance sheet value the crowd ignores
    • High short interest scored as BULLISH: market hatred = potential opportunity
    • Insider net buying as primary contrarian confirmation signal

"""

import logging
import numpy as np
from typing import Optional

from agents.base_agent import BaseAgent, AgentSignal
from agents.base_agent import SIGNAL_BULLISH, SIGNAL_NEUTRAL, SIGNAL_BEARISH
from agents.base_agent import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

TAX_RATE_DEFAULT = 0.21


# SYSTEM PROMPT

SYSTEM_PROMPT = """You are Michael Burry, founder of Scion Asset Management and the investor who famously
shorted the 2007 housing bubble. Your philosophy: find deep value that the market is systematically
ignoring, mispricing, or actively betting against. You are contrarian, data-obsessed, and deeply
sceptical of consensus narratives. You seek companies with hard asset backing, real free cash flow,
and a catalyst that will force the market to recognize the gap between price and value.

You distrust intangible assets, goodwill, and "adjusted" earnings. You trust tangible book value,
operating cash flows, and balance sheet math. High short interest does not scare you - it tells you
the crowd has decided the stock is worthless, which is exactly where you start looking.

You will receive quantitative scores (0–20) across four pillars plus underlying metrics.
Produce your investment opinion as JSON only - no prose outside the JSON:

{
  "signal": "bullish" | "neutral" | "bearish",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<2–3 paragraphs citing EV/EBITDA, tangible book, net cash, insider activity, and short interest as contrarian setup>",
  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "target_action": "buy" | "hold" | "sell"
}"""


# Helpers

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


def _first(df, *rows):
    """Return the first available value from a set of row name candidates."""
    if df is None or df.empty:
        return None
    col = df.columns[0]
    for row in rows:
        if row in df.index:
            try:
                v = float(df.loc[row, col])
                if not np.isnan(v):
                    return v
            except (TypeError, ValueError):
                pass
    return None


# Scoring pillars

def _score_deep_value(financials: dict, key_metrics: dict) -> dict:
    """
    Deep Value Metrics Score /5

    Sub-metrics:
        EV/EBITDA (0-3 pts) - absolute thresholds; Burry wants hard-number cheapness
        FCF yield  (0-2 pts) - (OCF - Capex) / Market Cap; real cash vs hype
    """
    score = 0.0
    detail = {}

    # EV/EBITDA (0-3 pts)
    ev_ebitda = key_metrics.get("ev_ebitda")
    if ev_ebitda and ev_ebitda > 0:
        if ev_ebitda <= 5:    ev_pts = 3.0
        elif ev_ebitda <= 8:  ev_pts = 2.5
        elif ev_ebitda <= 12: ev_pts = 1.5
        elif ev_ebitda <= 18: ev_pts = 0.75
        else:                 ev_pts = 0.0
        score += ev_pts
        detail["ev_to_ebitda"] = {
            "value": f"{round(ev_ebitda, 2)}x", "pts": round(ev_pts, 2), "max": 3
        }

    # FCF yield (0-2 pts)
    mkt_cap = (
        (key_metrics.get("shares_outstanding") or 0)
        * (key_metrics.get("current_price") or 0)
    )

    cf  = financials.get("cash_flow")
    fcf = None

    if cf is not None and not cf.empty:
        ocf   = _first(cf, "Operating Cash Flow")
        capex = _first(cf, "Capital Expenditure", "Capital Expenditures")
        if ocf is not None and capex is not None:
            fcf = ocf + capex   # capex is negative in statements

    # Fallback: use price_to_fcf from key_metrics
    if fcf is None and key_metrics.get("price_to_fcf") and key_metrics["price_to_fcf"] > 0:
        fcf_yield = (1 / key_metrics["price_to_fcf"]) * 100
        if fcf_yield >= 10:   fcf_pts = 2.0
        elif fcf_yield >= 6:  fcf_pts = 1.5
        elif fcf_yield >= 3:  fcf_pts = 0.75
        else:                 fcf_pts = 0.0
        score += fcf_pts
        detail["fcf_yield"] = {
            "value": f"{round(fcf_yield, 2)}% (from P/FCF)", "pts": round(fcf_pts, 2), "max": 2
        }
    elif fcf is not None and mkt_cap > 0:
        fcf_yield = (fcf / mkt_cap) * 100
        if fcf_yield >= 10:   fcf_pts = 2.0
        elif fcf_yield >= 6:  fcf_pts = 1.5
        elif fcf_yield >= 3:  fcf_pts = 0.75
        else:                 fcf_pts = 0.0
        score += fcf_pts
        detail["fcf_yield"] = {
            "value": f"{round(fcf_yield, 2)}%", "pts": round(fcf_pts, 2), "max": 2
        }

    return {"score": round(min(score, 5), 2), "max": 5, "detail": detail}


def _score_hidden_value(financials: dict, key_metrics: dict) -> dict:
    """
    Hidden Value Score /5

    Sub-metrics:
        Net cash / Market cap (0-3 pts) - is the market paying negative enterprise value?
                                          Burry loves companies where cash + securities
                                          cover a large chunk of the market cap.
        P/Tangible Book       (0-2 pts) - strips goodwill & intangibles; hard asset floor
    """
    score = 0.0
    detail = {}

    bs = financials.get("balance_sheet")
    mkt_cap = (
        (key_metrics.get("shares_outstanding") or 0)
        * (key_metrics.get("current_price") or 0)
    )

    # Net cash / Market cap (0-3 pts)
    if bs is not None and not bs.empty and mkt_cap > 0:
        cash = _first(
            bs,
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
        )
        total_debt = _first(bs, "Total Debt", "Long Term Debt") or 0.0

        if cash is not None:
            net_cash = cash - total_debt
            net_cash_pct = (net_cash / mkt_cap) * 100

            if net_cash_pct >= 50:   nc_pts = 3.0   # market barely charges for the business
            elif net_cash_pct >= 30: nc_pts = 2.5
            elif net_cash_pct >= 15: nc_pts = 1.5
            elif net_cash_pct >= 5:  nc_pts = 0.75
            else:                    nc_pts = 0.0
            score += nc_pts
            detail["net_cash_to_mktcap"] = {
                "value": f"{round(net_cash_pct, 1)}% (net cash {_fmt_big(net_cash)})",
                "pts": round(nc_pts, 2), "max": 3,
            }

    # P/Tangible Book (0-2 pts)
    if bs is not None and not bs.empty and mkt_cap > 0:
        equity = _first(bs, "Stockholders Equity", "Total Stockholders Equity")
        goodwill    = _first(bs, "Goodwill") or 0.0
        intangibles = _first(bs, "Other Intangible Assets", "Intangible Assets") or 0.0

        if equity is not None:
            tangible_book = equity - goodwill - intangibles
            if tangible_book > 0:
                ptb = mkt_cap / tangible_book
                if ptb <= 1.0:   ptb_pts = 2.0   # trading below tangible book - Burry's dream
                elif ptb <= 1.5: ptb_pts = 1.5
                elif ptb <= 2.5: ptb_pts = 1.0
                elif ptb <= 4.0: ptb_pts = 0.5
                else:            ptb_pts = 0.0
                score += ptb_pts
                detail["p_tangible_book"] = {
                    "value": f"{round(ptb, 2)}x (tangible equity {_fmt_big(tangible_book)})",
                    "pts": round(ptb_pts, 2), "max": 2,
                }
            elif tangible_book <= 0:
                # Negative tangible book - automatic 0 and flag it
                detail["p_tangible_book"] = {
                    "value": f"negative ({_fmt_big(tangible_book)})", "pts": 0.0, "max": 2
                }

    return {"score": round(min(score, 5), 2), "max": 5, "detail": detail}


def _score_solvency(financials: dict) -> dict:
    """
    Solvency & Survival Score /5

    Burry buys beaten-down companies - but they must survive long enough
    for the market to recognise the value. Debt and coverage are non-negotiable.

    Sub-metrics:
        D/E ratio          (0-2.5 pts) - keeps the lights on during the holding period
        Interest coverage  (0-2.5 pts) - EBIT / interest expense; can they service debt?
    """
    score = 0.0
    detail = {}

    bs  = financials.get("balance_sheet")
    inc = financials.get("income_statement")

    # D/E ratio (0-2.5 pts)
    if bs is not None and not bs.empty:
        equity     = _first(bs, "Stockholders Equity", "Total Stockholders Equity")
        total_debt = _first(bs, "Total Debt") or 0.0

        if equity and equity > 0:
            de = total_debt / equity
            if de <= 0.3:   de_pts = 2.5
            elif de <= 0.7: de_pts = 2.0
            elif de <= 1.2: de_pts = 1.25
            elif de <= 2.0: de_pts = 0.5
            else:           de_pts = 0.0   # D/E > 2 = Burry red flag
            score += de_pts
            detail["debt_to_equity"] = {
                "value": round(de, 2), "pts": round(de_pts, 2), "max": 2.5
            }

    # Interest coverage (0-2.5 pts)
    if inc is not None and not inc.empty:
        ebit    = _first(inc, "Operating Income", "EBIT")
        int_exp = _first(inc, "Interest Expense")
        if ebit is not None and int_exp and int_exp != 0:
            coverage = ebit / abs(int_exp)
            if coverage >= 10:  ic_pts = 2.5
            elif coverage >= 5: ic_pts = 1.75
            elif coverage >= 3: ic_pts = 1.0
            elif coverage >= 1: ic_pts = 0.25
            else:               ic_pts = 0.0
            score += ic_pts
            detail["interest_coverage"] = {
                "value": f"{round(coverage, 1)}x", "pts": round(ic_pts, 2), "max": 2.5
            }
        elif ebit is not None:
            # No interest expense = no debt burden
            score += 2.5
            detail["interest_coverage"] = {
                "value": "no debt (full points)", "pts": 2.5, "max": 2.5
            }

    return {"score": round(min(score, 5), 2), "max": 5, "detail": detail}


def _score_contrarian_signals(key_metrics: dict, insider_data: dict) -> dict:
    """
    Contrarian Signals Score /5

    Sub-metrics:
        Insider net buying   (0-3 pts) - insiders have skin in the game and private information;
                                         net buying = conviction the stock is cheap
        Short interest       (0-2 pts) - high short interest = market hatred = Burry's hunting ground
                                         Scored bullish: the more the crowd hates it, the more
                                         interesting it becomes (assuming fundamentals hold)
    """
    score = 0.0
    detail = {}

    # Insider net buying (0-3 pts)
    if insider_data:
        summary      = insider_data.get("summary", "neutral")
        n_buyers     = insider_data.get("n_buyers", 0)
        n_sellers    = insider_data.get("n_sellers", 0)
        net_buy_val  = insider_data.get("net_buy_value", 0)

        if summary == "net_buying":
            # More buyers, bigger net value = stronger signal
            if n_buyers >= 3 and net_buy_val > 1_000_000:   ins_pts = 3.0
            elif n_buyers >= 2 or net_buy_val > 500_000:    ins_pts = 2.0
            else:                                            ins_pts = 1.0
        elif summary == "neutral":
            ins_pts = 0.5
        else:  # net_selling
            ins_pts = 0.0

        score += ins_pts
        buyers_label = f"{n_buyers} buyer{'s' if n_buyers != 1 else ''}"
        sellers_label = f"{n_sellers} seller{'s' if n_sellers != 1 else ''}"
        detail["insider_activity"] = {
            "value": f"{summary} ({buyers_label}, {sellers_label}, net {_fmt_big(net_buy_val)})",
            "pts": round(ins_pts, 1), "max": 3,
        }

    # Short interest (0-2 pts)
    # High short interest → market hates this stock → Burry's contrarian entry signal
    short_pct = key_metrics.get("short_percent_of_float")
    if short_pct is not None:
        if short_pct >= 20:   si_pts = 2.0   # heavily shorted = high contrarian interest
        elif short_pct >= 10: si_pts = 1.5
        elif short_pct >= 5:  si_pts = 1.0
        elif short_pct >= 2:  si_pts = 0.5
        else:                 si_pts = 0.0   # low short interest = market not pricing fear
        score += si_pts
        detail["short_interest"] = {
            "value": f"{round(short_pct, 1)}% of float",
            "pts": round(si_pts, 2), "max": 2,
        }

    return {"score": round(min(score, 5), 2), "max": 5, "detail": detail}


# Agent class

class MichaelBurryAgent(BaseAgent):
    """Michael Burry - contrarian deep value, find what everyone else is missing."""

    def __init__(self, llm: Optional[LLMClient] = None):
        super().__init__(agent_id="michael_burry", agent_name="Michael Burry")
        self.llm = llm or LLMClient()

    def analyze(self, data: dict, ticker: str) -> AgentSignal:
        metrics      = data.get("key_metrics")         or {}
        financials   = data.get("financials")           or {}
        av           = data.get("av_overview")          or {}
        insider_data = data.get("insider_transactions") or {}

        # Score all four pillars
        deep_val  = _score_deep_value(financials, metrics)
        hidden    = _score_hidden_value(financials, metrics)
        solvency  = _score_solvency(financials)
        contrarian = _score_contrarian_signals(metrics, insider_data)

        total      = deep_val["score"] + hidden["score"] + solvency["score"] + contrarian["score"]
        total_max  = 20.0
        norm_score = total / total_max

        company_name = (
            data.get("company_info", {}).get("name", ticker)
            if data.get("company_info") else ticker
        )

        # Build LLM prompt
        user_prompt = f"""
Ticker: {ticker} ({company_name})

QUANTITATIVE SCORES (total {round(total, 2)}/{total_max}):

1. DEEP VALUE METRICS: {deep_val['score']}/{deep_val['max']}
{_fmt_detail(deep_val['detail'])}

2. HIDDEN VALUE: {hidden['score']}/{hidden['max']}
{_fmt_detail(hidden['detail'])}

3. SOLVENCY & SURVIVAL: {solvency['score']}/{solvency['max']}
{_fmt_detail(solvency['detail'])}

4. CONTRARIAN SIGNALS: {contrarian['score']}/{contrarian['max']}
{_fmt_detail(contrarian['detail'])}

Key market context:
  Current Price: ${metrics.get('current_price', 'N/A')}
  Enterprise Value: {_fmt_big(metrics.get('enterprise_value'))}
  Beta: {metrics.get('beta', 'N/A')}
  Short % of Float: {f"{round(metrics['short_percent_of_float'], 1)}%" if metrics.get('short_percent_of_float') else 'N/A'}

Provide your investment opinion as JSON.
"""

        llm_result = self.llm.generate_json(SYSTEM_PROMPT, user_prompt)

        # Parse LLM output
        signal        = llm_result.get("signal",        _norm_score_to_signal(norm_score))
        confidence    = float(llm_result.get("confidence", norm_score))
        reasoning     = llm_result.get("reasoning",    "LLM response unavailable.")
        key_risks     = llm_result.get("key_risks",    [])
        target_action = llm_result.get("target_action", _signal_to_action(signal))

        if signal not in ("bullish", "neutral", "bearish"):
            signal = _norm_score_to_signal(norm_score)
        if target_action not in ("buy", "hold", "sell", "short", "cover"):
            target_action = _signal_to_action(signal)

        self.logger.info(
            f"{ticker} → {signal.upper()} (conf={confidence:.0%}) | "
            f"total={round(total, 1)}/{total_max} | "
            f"deep={deep_val['score']} hidden={hidden['score']} "
            f"solvency={solvency['score']} contrarian={contrarian['score']}"
        )

        return AgentSignal(
            agent_id      = self.agent_id,
            agent_name    = self.agent_name,
            ticker        = ticker,
            signal        = signal,
            confidence    = round(min(max(confidence, 0), 1), 3),
            scores        = {
                "total":       round(total, 2),
                "total_max":   total_max,
                "deep_value":  {"score": deep_val["score"],   "max": deep_val["max"],   "detail": deep_val["detail"]},
                "hidden_value":{"score": hidden["score"],     "max": hidden["max"],     "detail": hidden["detail"]},
                "solvency":    {"score": solvency["score"],   "max": solvency["max"],   "detail": solvency["detail"]},
                "contrarian":  {"score": contrarian["score"], "max": contrarian["max"], "detail": contrarian["detail"]},
            },
            reasoning     = reasoning,
            key_risks     = key_risks,
            target_action = target_action,
        )


# Internal utilities

def _fmt_detail(detail: dict) -> str:
    lines = []
    for k, v in detail.items():
        lines.append(f"  {k}: {v['value']}  ({v['pts']}/{v['max']} pts)")
    return "\n".join(lines) if lines else "  (no data)"


def _fmt_big(v) -> str:
    if v is None: return "N/A"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "N/A"
    if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:  return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"


def _norm_score_to_signal(score: float) -> str:
    if score >= 0.60: return SIGNAL_BULLISH
    if score >= 0.35: return SIGNAL_NEUTRAL
    return SIGNAL_BEARISH


def _signal_to_action(signal: str) -> str:
    return {
        "bullish": ACTION_BUY,
        "neutral": ACTION_HOLD,
        "bearish": ACTION_SELL,
    }.get(signal, ACTION_HOLD)
