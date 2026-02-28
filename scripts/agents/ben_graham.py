"""
Ben Graham Agent
================
"The Intelligent Investor" — the most quantitative agent in the suite.

Graham's philosophy: buy with a margin of safety. Numbers must justify the price.
He never paid for hope, growth, or brand — only for verified assets and earnings.

Scoring (/30):
    Graham Number & margin of safety  6 pts  (20%) — his most iconic metric
    P/E vs dynamic risk-free floor     5 pts  (17%) — earnings yield vs bonds
    P/B < 1.5                          5 pts  (17%) — asset-based floor
    Liquidity + NCAV (net-net)         7 pts  (23%) — his deepest value concept
    Earnings stability                 5 pts  (17%) — predictability over growth
    Dividend record (bonus)            2 pts  ( 6%) — income discipline

Key differentiators vs other agents:
    • Graham Number is a hard price ceiling — no moat, no narrative overrides it
    • NCAV (Current Assets - Total Liabilities) flags true net-net opportunities
    • P/E threshold is DYNAMIC: 1 / (1.5 × risk_free_rate) — tightens with rates
    • Dividend penalises lightly — Graham preferred them but didn't require them
    • Would score GOOGL harshly on valuation despite its quality (by design)

Author: Financial Researcher Team
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


SYSTEM_PROMPT = """You are Benjamin Graham, father of value investing and author of "The Intelligent Investor."
Your philosophy is ruthlessly quantitative: the market is a voting machine in the short run but a weighing
machine in the long run. You demand a margin of safety in every purchase — never pay for hope or narrative.
You distrust growth projections, prefer tangible assets, and always ask: "What do I own if the business fails?"

You will receive quantitative scores (/30) across six criteria. Produce your opinion as JSON only:

{
  "signal": "bullish" | "neutral" | "bearish",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<2–3 paragraphs citing specific numbers, Graham Number, NCAV, and margin of safety>",
  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "target_action": "buy" | "hold" | "sell"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _annual_series(df, row: str, n: int = 5) -> list:
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


def _bs_get(bs, row: str):
    if bs is None or bs.empty:
        return None
    col = bs.columns[0]
    return float(bs.loc[row, col]) if row in bs.index else None


def _inc_get(inc, row: str):
    if inc is None or inc.empty:
        return None
    col = inc.columns[0]
    return float(inc.loc[row, col]) if row in inc.index else None


def _fmt_detail(detail: dict) -> str:
    return "\n".join(
        f"  {k}: {v['value']}  ({v['pts']}/{v['max']} pts)"
        for k, v in detail.items()
    ) or "  (no data)"


def _fmt_big(v) -> str:
    if v is None: return "N/A"
    if abs(v) >= 1e12: return f"{v/1e12:.2f}T"
    if abs(v) >= 1e9:  return f"{v/1e9:.2f}B"
    return f"{v:,.0f}"


# ─────────────────────────────────────────────────────────────────────────────
# Scoring pillars
# ─────────────────────────────────────────────────────────────────────────────

def _score_graham_number(inc, bs, key_metrics: dict, av: dict) -> dict:
    """
    Graham Number /6  — margin of safety against intrinsic asset value.

    Formula: sqrt(22.5 × EPS × Book Value per Share)
    This is the maximum price Graham would pay; deeper discount = more pts.

        Price < 50% of GN  → 6 pts  (50% margin of safety — rare opportunity)
        Price < 66% of GN  → 5 pts  (33% MoS — Graham's recommended threshold)
        Price < 80% of GN  → 3 pts  (20% MoS — acceptable for defensive investor)
        Price < 100% of GN → 1 pt   (at or near fair value)
        Price > GN         → 0 pts  (no margin of safety)
    """
    detail = {}

    eps        = None
    book_value = None   # per share
    price      = key_metrics.get("current_price")

    # EPS — try yfinance metrics first, fall back to AV
    if inc is not None and not inc.empty:
        col = inc.columns[0]
        ni  = _inc_get(inc, "Net Income")
        shares = key_metrics.get("shares_outstanding")
        if ni and shares and shares > 0:
            eps = ni / shares

    if eps is None and av:
        raw = av.get("eps")
        if raw: eps = float(raw)

    # Book value per share
    if bs is not None and not bs.empty:
        eq     = _bs_get(bs, "Stockholders Equity") or _bs_get(bs, "Total Stockholders Equity")
        shares = key_metrics.get("shares_outstanding")
        if eq and shares and shares > 0:
            book_value = eq / shares

    if book_value is None and av:
        raw = av.get("book_value")
        if raw: book_value = float(raw)

    # Graham Number
    if eps and eps > 0 and book_value and book_value > 0 and price and price > 0:
        gn    = math.sqrt(22.5 * eps * book_value)
        ratio = price / gn       # < 1 = below Graham Number (good)

        if ratio <= 0.50:   pts = 6.0
        elif ratio <= 0.66: pts = 5.0
        elif ratio <= 0.80: pts = 3.0
        elif ratio <= 1.00: pts = 1.0
        else:               pts = 0.0

        mos = (1 - ratio) * 100   # margin of safety %

        detail["graham_number"] = {
            "value": f"GN=${gn:.2f}  Price=${price:.2f}  MoS={mos:.1f}%",
            "pts": pts, "max": 6,
            "graham_number_usd": round(gn, 2),
            "margin_of_safety_pct": round(mos, 1),
        }
        return {"score": pts, "max": 6, "detail": detail}

    # Can't compute — note why
    missing = []
    if not eps or eps <= 0:        missing.append("positive EPS")
    if not book_value:             missing.append("book value")
    if not price:                  missing.append("current price")
    detail["graham_number"] = {
        "value": f"Cannot compute — missing: {', '.join(missing)}",
        "pts": 0, "max": 6
    }
    return {"score": 0.0, "max": 6, "detail": detail}


def _score_pe_dynamic(key_metrics: dict, av: dict, risk_free_rate: float) -> dict:
    """
    P/E vs Dynamic Threshold /5

    Graham's rule: earnings yield (1/PE) must exceed 1.5× the AAA bond yield.
    Threshold PE = 1 / (1.5 × risk_free_rate).
    When rates rise, the acceptable P/E falls — making this rate-aware.
    """
    detail = {}

    pe_limit = 1.0 / (1.5 * risk_free_rate) if risk_free_rate > 0 else 15.0
    pe = key_metrics.get("pe_ratio") or (av.get("pe_ratio") if av else None)

    if pe and pe > 0:
        ratio = pe / pe_limit     # < 1 = below threshold (good)

        if ratio <= 0.60:   pts = 5.0   # PE at 60% of limit
        elif ratio <= 0.80: pts = 4.0
        elif ratio <= 1.00: pts = 3.0
        elif ratio <= 1.30: pts = 1.5
        elif ratio <= 1.60: pts = 0.5
        else:               pts = 0.0

        detail["pe_vs_threshold"] = {
            "value": f"P/E={pe:.1f}  Limit={pe_limit:.1f}x  (rfr={risk_free_rate*100:.2f}%)",
            "pts": pts, "max": 5,
        }
    elif pe and pe <= 0:
        detail["pe_negative"] = {"value": "Negative earnings — automatic 0", "pts": 0, "max": 5}
        pts = 0.0
    else:
        detail["pe_unavailable"] = {"value": "P/E not available", "pts": 0, "max": 5}
        pts = 0.0

    return {"score": pts, "max": 5, "detail": detail}


def _score_pb(key_metrics: dict, av: dict) -> dict:
    """
    Price-to-Book /5

    Graham's ceiling: P/B ≤ 1.5 (defensive) and P/E × P/B ≤ 22.5 (combined rule).
    He treated book value as the liquidation floor of intrinsic value.
    """
    detail = {}

    pb = key_metrics.get("pb_ratio") or (av.get("price_to_book") if av else None)
    pe = key_metrics.get("pe_ratio") or (av.get("pe_ratio")       if av else None)

    if pb and pb > 0:
        # Basic P/B score
        if pb <= 1.0:    pts = 5.0
        elif pb <= 1.5:  pts = 4.0
        elif pb <= 2.0:  pts = 2.5
        elif pb <= 3.0:  pts = 1.0
        else:            pts = 0.0

        detail["price_to_book"] = {"value": round(pb, 2), "pts": pts, "max": 5}

        # Combined Graham rule: PE × PB ≤ 22.5
        if pe and pe > 0:
            combined = pe * pb
            combined_ok = combined <= 22.5
            detail["pe_times_pb"] = {
                "value": f"PE×PB = {combined:.1f}  ({'≤22.5 ✓' if combined_ok else '>22.5 ✗'})",
                "pts": 0,  # informational — already captured in individual metrics
                "max": 0,
            }
    else:
        pts = 0.0
        detail["pb_unavailable"] = {"value": "P/B not available", "pts": 0, "max": 5}

    return {"score": pts, "max": 5, "detail": detail}


def _score_liquidity_ncav(bs, key_metrics: dict) -> dict:
    """
    Liquidity + NCAV /7

    Two complementary balance-sheet tests:

    Current Ratio (0-4 pts):
        Graham required CR > 2 for industrial companies.
        For utilities he accepted lower; we apply the general threshold.

    NCAV — Net Current Asset Value (0-3 pts):
        NCAV = Current Assets - Total Liabilities (not just current liabilities)
        NCAV/share vs current price:
            Price < NCAV/share    → 3 pts  (true "net-net" — Graham's holy grail)
            Price < 1.5× NCAV/sh  → 2 pts  (near net-net)
            Price < 2× NCAV/sh    → 1 pt   (reasonable asset backing)
            Price > 2× NCAV/sh    → 0 pts
    """
    detail = {}
    total_pts = 0.0

    if bs is None or bs.empty:
        detail["balance_sheet"] = {"value": "No balance sheet data", "pts": 0, "max": 7}
        return {"score": 0.0, "max": 7, "detail": detail}

    col = bs.columns[0]

    cur_assets = _bs_get(bs, "Current Assets")
    cur_liab   = _bs_get(bs, "Current Liabilities")
    total_liab = (_bs_get(bs, "Total Liabilities Net Minority Interest")
                  or _bs_get(bs, "Total Liabilities"))
    price      = key_metrics.get("current_price")
    shares     = key_metrics.get("shares_outstanding")

    # ── Current Ratio (/4) ────────────────────────────────────────────
    if cur_assets and cur_liab and cur_liab > 0:
        cr = cur_assets / cur_liab
        if cr >= 3.0:    cr_pts = 4.0
        elif cr >= 2.5:  cr_pts = 3.5
        elif cr >= 2.0:  cr_pts = 3.0
        elif cr >= 1.5:  cr_pts = 1.5
        elif cr >= 1.0:  cr_pts = 0.5
        else:            cr_pts = 0.0
        total_pts += cr_pts
        detail["current_ratio"] = {"value": round(cr, 2), "pts": round(cr_pts, 1), "max": 4}

    # ── NCAV (/3) ─────────────────────────────────────────────────────
    if cur_assets and total_liab and price and shares and shares > 0:
        ncav          = cur_assets - total_liab
        ncav_per_share = ncav / shares

        if ncav_per_share > 0:
            ratio = price / ncav_per_share    # < 1 = true net-net
            if ratio <= 1.0:    ncav_pts = 3.0
            elif ratio <= 1.5:  ncav_pts = 2.0
            elif ratio <= 2.0:  ncav_pts = 1.0
            else:               ncav_pts = 0.0
        else:
            ncav_pts = 0.0     # negative NCAV (more liabilities than current assets)

        total_pts += ncav_pts
        detail["ncav_per_share"] = {
            "value": (f"NCAV/sh=${ncav_per_share:.2f}  Price=${price:.2f}  "
                      f"Ratio={price/ncav_per_share:.2f}x" if ncav_per_share > 0
                      else f"NCAV/sh=${ncav_per_share:.2f} (negative)"),
            "pts": ncav_pts, "max": 3,
            "ncav_per_share": round(ncav_per_share, 2),
        }

    return {"score": round(min(total_pts, 7), 2), "max": 7, "detail": detail}


def _score_earnings_stability(inc, financials: dict) -> dict:
    """
    Earnings Stability /5

    Graham required 10 years of positive earnings (defensive investor).
    We use the years available from yfinance (typically 4-5 annual).

    Also checks for earnings growth trend (secondary criterion).
    """
    detail = {}

    ni_series = _annual_series(inc, "Net Income", n=5)

    if not ni_series:
        detail["earnings"] = {"value": "No earnings data", "pts": 0, "max": 5}
        return {"score": 0.0, "max": 5, "detail": detail}

    n_years    = len(ni_series)
    n_positive = sum(1 for v in ni_series if v > 0)
    pct        = n_positive / n_years

    # Base stability score
    if pct == 1.0:
        if n_years >= 5: base_pts = 4.0
        else:            base_pts = 3.0
    elif pct >= 0.80:    base_pts = 2.0
    elif pct >= 0.60:    base_pts = 1.0
    else:                base_pts = 0.0

    detail["earnings_positive_years"] = {
        "value": f"{n_positive}/{n_years} years positive earnings",
        "pts": base_pts, "max": 4,
        "note": "(Graham required 10 years; limited to available data)"
    }

    # Bonus: earnings growth trend (+1 pt)
    growth_pts = 0.0
    if len(ni_series) >= 3 and all(v > 0 for v in ni_series[:3]):
        # Most recent 3 years — check if trend is upward
        if ni_series[0] > ni_series[1] > ni_series[2]:
            growth_pts = 1.0
        elif ni_series[0] > ni_series[2]:
            growth_pts = 0.5

    if growth_pts > 0:
        detail["earnings_growth_trend"] = {
            "value": f"Trend: {', '.join(_fmt_big(v) for v in ni_series[:3])}",
            "pts": growth_pts, "max": 1,
        }

    total_pts = base_pts + growth_pts
    return {"score": round(min(total_pts, 5), 2), "max": 5, "detail": detail}


def _score_dividend(key_metrics: dict, av: dict) -> dict:
    """
    Dividend Record /2  (bonus — not penalised if absent)

    Graham preferred uninterrupted dividends for 20 years (defensive).
    We award bonus points for current dividend yield; no penalty for zero.
    """
    detail = {}

    # key_metrics stores dividend_yield already in % (adapter does ×100 on raw decimal)
    # AV stores as raw decimal (0.0027 = 0.27%) — must multiply ×100
    dy_km  = key_metrics.get("dividend_yield")
    dy_av  = float(av.get("dividend_yield")) if (av and av.get("dividend_yield")) else None
    dps    = av.get("dividend_per_share") if av else None

    if dy_km and dy_km > 0:
        dy_pct = dy_km                 # already in percent
    elif dy_av and dy_av > 0:
        dy_pct = dy_av * 100           # AV returns decimal → convert
    else:
        dy_pct = None

    if dy_pct and dy_pct > 0:

        if dy_pct >= 3.0:    pts = 2.0
        elif dy_pct >= 1.5:  pts = 1.5
        elif dy_pct >= 0.5:  pts = 1.0
        else:                pts = 0.5

        detail["dividend_yield"] = {
            "value": f"{dy_pct:.2f}%  (DPS=${dps:.2f})" if dps else f"{dy_pct:.2f}%",
            "pts": pts, "max": 2,
        }
        return {"score": pts, "max": 2, "detail": detail}

    detail["dividend"] = {
        "value": "No dividend paid (bonus not awarded — not penalised)",
        "pts": 0, "max": 2,
    }
    return {"score": 0.0, "max": 2, "detail": detail}


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────

class BenGrahamAgent(BaseAgent):
    """Ben Graham — margin of safety through quantitative rigor."""

    def __init__(self, llm: Optional[LLMClient] = None):
        super().__init__(agent_id="ben_graham", agent_name="Ben Graham")
        self.llm = llm or LLMClient()

    def analyze(self, data: dict, ticker: str) -> AgentSignal:
        metrics    = data.get("key_metrics")    or {}
        financials = data.get("financials")     or {}
        av         = data.get("av_overview")    or {}
        risk_free  = data.get("risk_free_rate") or 0.043

        inc = financials.get("income_statement")
        bs  = financials.get("balance_sheet")

        # ── Score all six criteria ─────────────────────────────────────
        gn_score   = _score_graham_number(inc, bs, metrics, av)
        pe_score   = _score_pe_dynamic(metrics, av, risk_free)
        pb_score   = _score_pb(metrics, av)
        liq_score  = _score_liquidity_ncav(bs, metrics)
        earn_score = _score_earnings_stability(inc, financials)
        div_score  = _score_dividend(metrics, av)

        total     = (gn_score["score"] + pe_score["score"] + pb_score["score"] +
                     liq_score["score"] + earn_score["score"] + div_score["score"])
        total_max = 30.0
        norm      = total / total_max

        # Pull Graham Number for prompt context
        gn_detail = gn_score["detail"].get("graham_number", {})
        gn_usd    = gn_detail.get("graham_number_usd")
        mos_pct   = gn_detail.get("margin_of_safety_pct")

        # NCAV per share for prompt context
        ncav_detail = liq_score["detail"].get("ncav_per_share", {})
        ncav_ps     = ncav_detail.get("ncav_per_share")

        company_name = (data.get("company_info") or {}).get("name", ticker)

        # ── LLM prompt ────────────────────────────────────────────────
        user_prompt = f"""
Ticker: {ticker} ({company_name})

GRAHAM SCORES — total {round(total, 2)}/{total_max}:

1. GRAHAM NUMBER (margin of safety): {gn_score['score']}/{gn_score['max']}
{_fmt_detail(gn_score['detail'])}

2. P/E vs DYNAMIC THRESHOLD: {pe_score['score']}/{pe_score['max']}
{_fmt_detail(pe_score['detail'])}

3. PRICE-TO-BOOK: {pb_score['score']}/{pb_score['max']}
{_fmt_detail(pb_score['detail'])}

4. LIQUIDITY + NCAV: {liq_score['score']}/{liq_score['max']}
{_fmt_detail(liq_score['detail'])}

5. EARNINGS STABILITY: {earn_score['score']}/{earn_score['max']}
{_fmt_detail(earn_score['detail'])}

6. DIVIDEND RECORD (bonus): {div_score['score']}/{div_score['max']}
{_fmt_detail(div_score['detail'])}

Key context:
  Current Price:      ${metrics.get('current_price', 'N/A')}
  Graham Number:      ${gn_usd if gn_usd else 'N/A'}
  Margin of Safety:   {f'{mos_pct:.1f}%' if mos_pct is not None else 'N/A'}
  NCAV/share:         ${ncav_ps if ncav_ps else 'N/A'}
  Risk-Free Rate:     {round(risk_free * 100, 2)}%  (dynamic P/E limit = {round(1/(1.5*risk_free), 1)}x)

As Graham, you are deeply sceptical of companies trading above their Graham Number.
Be honest — if the numbers don't justify the price, say so clearly.
"""

        llm_result    = self.llm.generate_json(SYSTEM_PROMPT, user_prompt)

        signal        = llm_result.get("signal",        _norm_to_signal(norm))
        confidence    = float(llm_result.get("confidence", norm))
        reasoning     = llm_result.get("reasoning",     "LLM response unavailable.")
        key_risks     = llm_result.get("key_risks",     [])
        target_action = llm_result.get("target_action", _signal_to_action(signal))

        if signal not in ("bullish", "neutral", "bearish"):
            signal = _norm_to_signal(norm)
        if target_action not in ("buy", "hold", "sell", "short", "cover"):
            target_action = _signal_to_action(signal)

        self.logger.info(
            f"{ticker} → {signal.upper()} (conf={confidence:.0%}) | "
            f"total={round(total,1)}/{total_max} | "
            f"GN={gn_score['score']} PE={pe_score['score']} "
            f"PB={pb_score['score']} LIQ={liq_score['score']} "
            f"EARN={earn_score['score']} DIV={div_score['score']}"
        )

        return AgentSignal(
            agent_id      = self.agent_id,
            agent_name    = self.agent_name,
            ticker        = ticker,
            signal        = signal,
            confidence    = round(min(max(confidence, 0), 1), 3),
            scores        = {
                "total":              round(total, 2),
                "total_max":          total_max,
                "graham_number":      {"score": gn_score["score"],   "max": gn_score["max"],   "detail": gn_score["detail"]},
                "pe_dynamic":         {"score": pe_score["score"],   "max": pe_score["max"],   "detail": pe_score["detail"]},
                "price_to_book":      {"score": pb_score["score"],   "max": pb_score["max"],   "detail": pb_score["detail"]},
                "liquidity_ncav":     {"score": liq_score["score"],  "max": liq_score["max"],  "detail": liq_score["detail"]},
                "earnings_stability": {"score": earn_score["score"], "max": earn_score["max"], "detail": earn_score["detail"]},
                "dividend":           {"score": div_score["score"],  "max": div_score["max"],  "detail": div_score["detail"]},
            },
            reasoning     = reasoning,
            key_risks     = key_risks,
            target_action = target_action,
        )


def _norm_to_signal(score: float) -> str:
    if score >= 0.60: return SIGNAL_BULLISH
    if score >= 0.35: return SIGNAL_NEUTRAL
    return SIGNAL_BEARISH

def _signal_to_action(signal: str) -> str:
    return {"bullish": ACTION_BUY, "neutral": ACTION_HOLD, "bearish": ACTION_SELL}.get(signal, ACTION_HOLD)
