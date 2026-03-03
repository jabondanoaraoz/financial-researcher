"""
Cathie Wood Agent
"We invest in disruptive innovation — the biggest growth opportunities in history."

Cathie Wood's philosophy: identify companies riding exponential S-curves in
disruptive technology. She accepts near-term losses and high valuations if
the total addressable market (TAM) is large and the revenue trajectory is
accelerating. Asset-light, high-gross-margin platforms with heavy R&D
reinvestment are her hallmark.

Scoring (/20):
    Revenue Growth & Acceleration   7 pts  (35%) — CAGR + acceleration signal
    Scalability                     4 pts  (20%) — Gross margin ≥ 50% level + R&D intensity
    Disruption Score                4 pts  (20%) — Operating leverage + Gross margin expansion
    Valuation Tolerance             5 pts  (25%) — P/S adjusted for growth rate

Key differentiators vs other agents:
    • Rewards revenue *acceleration*, not just level (momentum on the S-curve)
    • Gross margin > 50% as scalability threshold — measures LEVEL in pillar 2
    • Gross margin TREND measured separately in disruption pillar (level ≠ trajectory)
    • R&D/Revenue as proxy for TAM reinvestment (not a cost, a strategic asset)
    • P/S primary valuation metric: accepts high multiples when CAGR justifies it
    • Operating leverage scored via operating margin trajectory over 3-4 years

Author: Joaquin Abondano w/ Claude Code
"""

import logging
import numpy as np
from typing import Optional

from agents.base_agent import BaseAgent, AgentSignal
from agents.base_agent import SIGNAL_BULLISH, SIGNAL_NEUTRAL, SIGNAL_BEARISH
from agents.base_agent import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from agents.llm_client import LLMClient

logger = logging.getLogger(__name__)


# SYSTEM PROMPT

SYSTEM_PROMPT = """You are Cathie Wood, founder and CIO of ARK Invest, performing a stock analysis.
Your philosophy: invest early in disruptive innovation — genomics, AI, robotics, fintech, energy storage.
You believe the biggest risk is *not* owning the companies reshaping civilization. You tolerate near-term
losses and high P/S multiples if the revenue trajectory is accelerating and the total addressable market
is enormous. You think in 5-year horizons and position around exponential S-curves, not quarterly earnings.

You will receive quantitative scores (0–20) across four pillars plus the underlying metrics.
Produce your investment opinion as JSON only — no prose outside the JSON:

{
  "signal": "bullish" | "neutral" | "bearish",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<2–3 paragraphs citing revenue acceleration, scalability, disruption signals, and valuation relative to TAM>",
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


def _revenue_cagr(revenues: list[float]) -> Optional[float]:
    """
    Compute CAGR from oldest to most recent year.
    revenues[0] = most recent, revenues[-1] = oldest.
    Returns None if insufficient or invalid data.
    """
    if len(revenues) < 2:
        return None
    newest = revenues[0]
    oldest = revenues[-1]
    n = len(revenues) - 1
    if oldest <= 0 or newest <= 0:
        return None
    return ((newest / oldest) ** (1 / n) - 1) * 100


# Scoring pillars

def _score_revenue_growth(financials: dict, av: dict) -> tuple[dict, Optional[float]]:
    """
    Revenue Growth & Acceleration Score /7

    Sub-metrics:
        Revenue CAGR (0-4 pts)       — 3-4 year compound growth rate
        YoY acceleration (0-2 pts)   — is the most recent year faster than prior period?
        Revenue consistency (0-1 pt) — all periods show positive growth (no declines)

    Returns (result_dict, cagr_float) so cagr can be reused in the valuation pillar.
    """
    score = 0.0
    detail = {}

    inc = financials.get("income_statement")
    revenues = _annual_series(inc, "Total Revenue", n_years=4)

    # Fallback to AV snapshot
    if not revenues or len(revenues) < 2:
        av_growth = av.get("quarterly_revenue_growth_yoy") or av.get("revenue_growth_yoy")
        if av_growth:
            g = av_growth * 100
            if g >= 40:   cagr_pts = 3.0
            elif g >= 25: cagr_pts = 2.5
            elif g >= 15: cagr_pts = 1.5
            elif g >= 8:  cagr_pts = 0.75
            else:         cagr_pts = 0.0
            score += cagr_pts
            detail["revenue_growth_snapshot"] = {
                "value": f"{round(g, 1)}%", "pts": round(cagr_pts, 2), "max": 4
            }
        return {"score": round(min(score, 7), 2), "max": 7, "detail": detail}, None

    cagr = _revenue_cagr(revenues)

    # Revenue CAGR (0-4 pts)
    if cagr is not None:
        if cagr >= 40:   cagr_pts = 4.0
        elif cagr >= 25: cagr_pts = 3.0
        elif cagr >= 15: cagr_pts = 2.0
        elif cagr >= 8:  cagr_pts = 1.0
        else:            cagr_pts = 0.0
        score += cagr_pts
        detail["revenue_cagr"] = {
            "value": f"{round(cagr, 1)}%", "pts": round(cagr_pts, 1), "max": 4
        }

    # YoY acceleration (0-2 pts)
    # Compare most recent YoY vs the prior YoY to detect inflection
    if len(revenues) >= 3:
        latest_yoy = ((revenues[0] - revenues[1]) / abs(revenues[1])) * 100 if revenues[1] != 0 else None
        prev_yoy   = ((revenues[1] - revenues[2]) / abs(revenues[2])) * 100 if revenues[2] != 0 else None

        if latest_yoy is not None and prev_yoy is not None:
            accel = latest_yoy - prev_yoy
            if accel >= 10:   accel_pts = 2.0
            elif accel >= 3:  accel_pts = 1.5
            elif accel >= 0:  accel_pts = 1.0
            elif accel >= -5: accel_pts = 0.5
            else:             accel_pts = 0.0
            score += accel_pts
            detail["yoy_acceleration"] = {
                "value": f"latest YoY {round(latest_yoy, 1)}% vs prior {round(prev_yoy, 1)}% (Δ {round(accel, 1)}pp)",
                "pts": round(accel_pts, 1), "max": 2,
            }
        elif latest_yoy is not None and cagr is not None:
            # Only one comparison period: compare latest YoY to CAGR as proxy
            accel = latest_yoy - cagr
            if accel >= 5:   accel_pts = 1.5
            elif accel >= 0: accel_pts = 1.0
            else:            accel_pts = 0.5
            score += accel_pts
            detail["yoy_vs_cagr"] = {
                "value": f"latest YoY {round(latest_yoy, 1)}% vs CAGR {round(cagr, 1)}% (Δ {round(accel, 1)}pp)",
                "pts": round(accel_pts, 1), "max": 2,
            }
    elif len(revenues) == 2:
        yoy = ((revenues[0] - revenues[1]) / abs(revenues[1])) * 100 if revenues[1] != 0 else None
        if yoy is not None:
            accel_pts = 1.0 if yoy >= 20 else (0.5 if yoy >= 10 else 0.0)
            score += accel_pts
            detail["yoy_growth"] = {
                "value": f"{round(yoy, 1)}%", "pts": round(accel_pts, 1), "max": 2
            }

    # Revenue consistency (0-1 pt)
    # Every period should show positive growth (no regressions)
    growth_flags = []
    for i in range(len(revenues) - 1):
        if revenues[i + 1] != 0:
            growth_flags.append(revenues[i] > revenues[i + 1])  # newer > older

    if growth_flags:
        n_positive = sum(growth_flags)
        pct = n_positive / len(growth_flags)
        if pct == 1.0:    cons_pts = 1.0
        elif pct >= 0.75: cons_pts = 0.5
        else:             cons_pts = 0.0
        score += cons_pts
        detail["revenue_consistency"] = {
            "value": f"{n_positive}/{len(growth_flags)} years growing",
            "pts": round(cons_pts, 1), "max": 1,
        }

    return {"score": round(min(score, 7), 2), "max": 7, "detail": detail}, cagr


def _score_scalability(financials: dict) -> dict:
    """
    Scalability Score /4

    Sub-metrics:
        Gross margin LEVEL (0-2 pts)  — ≥ 50% = asset-light, scalable platform model
        R&D intensity      (0-2 pts)  — R&D/Revenue = reinvestment in future TAM

    Note: gross margin *trajectory* is captured in the Disruption pillar.
    """
    score = 0.0
    detail = {}

    inc = financials.get("income_statement")
    if inc is None or inc.empty:
        return {"score": 0.0, "max": 4, "detail": detail}

    revenues = _annual_series(inc, "Total Revenue", n_years=3)
    gross_ps = _annual_series(inc, "Gross Profit",  n_years=3)

    # Gross margin level (0-2 pts)
    gm_series = []
    for gp, rev in zip(gross_ps, revenues):
        if rev and rev > 0:
            gm_series.append((gp / rev) * 100)

    if gm_series:
        gm_mean = np.mean(gm_series)
        if gm_mean >= 70:   gm_pts = 2.0
        elif gm_mean >= 50: gm_pts = 1.5
        elif gm_mean >= 35: gm_pts = 0.75
        else:               gm_pts = 0.0
        score += gm_pts
        detail["gross_margin_level"] = {
            "value": f"{round(gm_mean, 1)}%", "pts": round(gm_pts, 2), "max": 2
        }

    # R&D intensity (0-2 pts)
    rnd_rows = [
        "Research And Development",
        "Research And Development Expenses",
        "R&D Expenses",
        "Research Development",
    ]
    rnd_series = []
    for rnd_row in rnd_rows:
        rnd_series = _annual_series(inc, rnd_row, n_years=3)
        if rnd_series:
            break

    if rnd_series and revenues:
        rnd_ratios = []
        for rnd, rev in zip(rnd_series, revenues):
            if rev and rev > 0 and rnd and rnd > 0:
                rnd_ratios.append((rnd / rev) * 100)

        if rnd_ratios:
            rnd_mean = np.mean(rnd_ratios)
            if rnd_mean >= 20:   rnd_pts = 2.0
            elif rnd_mean >= 12: rnd_pts = 1.5
            elif rnd_mean >= 6:  rnd_pts = 0.75
            else:                rnd_pts = 0.0
            score += rnd_pts
            detail["rnd_to_revenue"] = {
                "value": f"{round(rnd_mean, 1)}%", "pts": round(rnd_pts, 2), "max": 2
            }

    return {"score": round(min(score, 4), 2), "max": 4, "detail": detail}


def _score_disruption(financials: dict) -> dict:
    """
    Disruption Score /4

    Sub-metrics:
        Operating leverage   (0-2 pts) — operating margin TREND over 3-4 years
                                         Expanding margins signal that growth is being
                                         leveraged rather than consumed by costs.
        Gross margin expansion (0-2 pts) — GM TREND direction (level is in scalability)
                                           A platform improving its take-rate signals
                                           increasing pricing power in its disrupted market.
    """
    score = 0.0
    detail = {}

    inc = financials.get("income_statement")
    if inc is None or inc.empty:
        return {"score": 0.0, "max": 4, "detail": detail}

    revenues = _annual_series(inc, "Total Revenue",    n_years=4)
    op_incs  = _annual_series(inc, "Operating Income", n_years=4)
    gross_ps = _annual_series(inc, "Gross Profit",     n_years=4)

    # Operating leverage: margin trend (0-2 pts)
    op_margins = []
    for oi, rev in zip(op_incs, revenues):
        if rev and rev != 0:
            op_margins.append((oi / rev) * 100)

    if len(op_margins) >= 2:
        # Most recent (index 0) vs oldest available — positive delta = expanding
        margin_delta = op_margins[0] - op_margins[-1]
        if margin_delta >= 10:   oplev_pts = 2.0
        elif margin_delta >= 4:  oplev_pts = 1.5
        elif margin_delta >= 0:  oplev_pts = 1.0
        elif margin_delta >= -4: oplev_pts = 0.5
        else:                    oplev_pts = 0.0
        score += oplev_pts
        detail["operating_leverage"] = {
            "value": (
                f"op margin {round(op_margins[0], 1)}% "
                f"(was {round(op_margins[-1], 1)}%, Δ {round(margin_delta, 1)}pp)"
            ),
            "pts": round(oplev_pts, 1), "max": 2,
        }

    # Gross margin expansion: GM trend (0-2 pts)
    gm_series = []
    for gp, rev in zip(gross_ps, revenues):
        if rev and rev > 0:
            gm_series.append((gp / rev) * 100)

    if len(gm_series) >= 2:
        gm_delta = gm_series[0] - gm_series[-1]  # recent vs oldest
        if gm_delta >= 8:    gm_exp_pts = 2.0
        elif gm_delta >= 3:  gm_exp_pts = 1.5
        elif gm_delta >= 0:  gm_exp_pts = 1.0
        elif gm_delta >= -3: gm_exp_pts = 0.5
        else:                gm_exp_pts = 0.0
        score += gm_exp_pts
        detail["gross_margin_expansion"] = {
            "value": (
                f"GM {round(gm_series[0], 1)}% "
                f"(was {round(gm_series[-1], 1)}%, Δ {round(gm_delta, 1)}pp)"
            ),
            "pts": round(gm_exp_pts, 1), "max": 2,
        }

    return {"score": round(min(score, 4), 2), "max": 4, "detail": detail}


def _score_valuation_tolerance(
    financials: dict,
    key_metrics: dict,
    av: dict,
    revenue_cagr: Optional[float],
) -> dict:
    """
    Valuation Tolerance Score /5

    Cathie Wood's primary lens: Price/Sales adjusted for revenue growth rate.
    High P/S is acceptable — even desirable — when CAGR justifies the premium.
    Low-growth stocks with elevated P/S are disqualified.

    Sub-metrics:
        P/S ratio (growth-adjusted)  (0-3 pts) — core metric
        EV/Revenue                   (0-1 pt)  — enterprise-level corroboration
        Growth justification bonus   (0-1 pt)  — strong CAGR + reasonable P/S = bonus
    """
    score = 0.0
    detail = {}

    g = revenue_cagr or 0.0

    # P/S ratio (growth-adjusted) (0-3 pts)
    ps = None
    if av:
        ps = av.get("price_to_sales_ratio_ttm") or av.get("price_to_sales_ratio")

    if ps is None:
        # Compute from market cap / most recent annual revenue
        mkt_cap = (
            (key_metrics.get("shares_outstanding") or 0)
            * (key_metrics.get("current_price") or 0)
        )
        inc = financials.get("income_statement")
        revenues = _annual_series(inc, "Total Revenue", n_years=1)
        if mkt_cap > 0 and revenues:
            ps = mkt_cap / revenues[0]

    if ps and ps > 0:
        if g >= 30:
            # Hypergrowth: high P/S tolerance
            if ps <= 8:    ps_pts = 3.0
            elif ps <= 15: ps_pts = 2.5
            elif ps <= 25: ps_pts = 2.0
            elif ps <= 40: ps_pts = 1.0
            else:          ps_pts = 0.5   # extreme multiple, still partial credit
        elif g >= 15:
            # Solid growth: moderate tolerance
            if ps <= 5:    ps_pts = 3.0
            elif ps <= 10: ps_pts = 2.0
            elif ps <= 20: ps_pts = 1.0
            else:          ps_pts = 0.0
        else:
            # Low growth + high P/S = red flag for ARK
            if ps <= 3:    ps_pts = 2.0
            elif ps <= 6:  ps_pts = 1.0
            else:          ps_pts = 0.0
        score += ps_pts
        detail["price_to_sales"] = {
            "value": f"{round(ps, 2)}x (CAGR {round(g, 1)}%)",
            "pts": round(ps_pts, 2), "max": 3,
        }

    # EV/Revenue (0-1 pt)
    ev = key_metrics.get("enterprise_value")
    inc = financials.get("income_statement")
    revenues = _annual_series(inc, "Total Revenue", n_years=1)

    if ev and revenues and revenues[0] > 0:
        ev_rev = ev / revenues[0]
        if ev_rev <= 5:    evr_pts = 1.0
        elif ev_rev <= 10: evr_pts = 0.75
        elif ev_rev <= 20: evr_pts = 0.5
        elif ev_rev <= 35: evr_pts = 0.25
        else:              evr_pts = 0.0
        score += evr_pts
        detail["ev_to_revenue"] = {
            "value": f"{round(ev_rev, 2)}x", "pts": round(evr_pts, 2), "max": 1
        }

    # Growth justification bonus (0-1 pt)
    # Strong CAGR paired with a reasonable P/S = the ARK sweet spot
    if ps and g >= 30 and ps <= 20:
        score += 1.0
        detail["growth_justification_bonus"] = {
            "value": f"CAGR {round(g, 1)}% with P/S {round(ps, 2)}x",
            "pts": 1.0, "max": 1,
        }
    elif ps and g >= 25 and ps <= 15:
        score += 0.5
        detail["growth_justification_bonus"] = {
            "value": f"CAGR {round(g, 1)}% with P/S {round(ps, 2)}x",
            "pts": 0.5, "max": 1,
        }

    return {"score": round(min(score, 5), 2), "max": 5, "detail": detail}


# Agent class

class CathieWoodAgent(BaseAgent):
    """Cathie Wood — disruptive innovation and exponential growth."""

    def __init__(self, llm: Optional[LLMClient] = None):
        super().__init__(agent_id="cathie_wood", agent_name="Cathie Wood")
        self.llm = llm or LLMClient()

    def analyze(self, data: dict, ticker: str) -> AgentSignal:
        metrics    = data.get("key_metrics")  or {}
        financials = data.get("financials")   or {}
        av         = data.get("av_overview")  or {}

        # Score all four pillars
        # Revenue pillar also returns cagr so valuation can reuse it
        rev, cagr = _score_revenue_growth(financials, av)
        scale     = _score_scalability(financials)
        disrt     = _score_disruption(financials)
        val       = _score_valuation_tolerance(financials, metrics, av, cagr)

        total      = rev["score"] + scale["score"] + disrt["score"] + val["score"]
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

1. REVENUE GROWTH & ACCELERATION: {rev['score']}/{rev['max']}
{_fmt_detail(rev['detail'])}

2. SCALABILITY: {scale['score']}/{scale['max']}
{_fmt_detail(scale['detail'])}

3. DISRUPTION SCORE: {disrt['score']}/{disrt['max']}
{_fmt_detail(disrt['detail'])}

4. VALUATION TOLERANCE: {val['score']}/{val['max']}
{_fmt_detail(val['detail'])}

Key market context:
  Current Price: ${metrics.get('current_price', 'N/A')}
  Market Cap: {_fmt_big(metrics.get('enterprise_value'))}
  Beta: {metrics.get('beta', 'N/A')}
  Revenue CAGR (3-4yr): {f"{round(cagr, 1)}%" if cagr is not None else "N/A"}

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
            f"rev={rev['score']} scale={scale['score']} disrt={disrt['score']} val={val['score']}"
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
                "revenue":     {"score": rev["score"],   "max": rev["max"],   "detail": rev["detail"]},
                "scalability": {"score": scale["score"], "max": scale["max"], "detail": scale["detail"]},
                "disruption":  {"score": disrt["score"], "max": disrt["max"], "detail": disrt["detail"]},
                "valuation":   {"score": val["score"],   "max": val["max"],   "detail": val["detail"]},
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
    if v >= 1e12: return f"${v/1e12:.2f}T"
    if v >= 1e9:  return f"${v/1e9:.2f}B"
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
