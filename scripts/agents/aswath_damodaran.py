"""
Aswath Damodaran Agent
======================
"Every asset has an intrinsic value — the story must match the numbers."

Damodaran's framework: disciplined DCF valuation, narrative consistency,
and rigorous cost of capital. He believes that valuation is a bridge
between stories and numbers — both must be coherent and grounded in data.

Scoring (/20):
    Revenue growth story  (CAGR + sustainability)   5 pts  (25%)
    Margin trajectory     (level + trend)            5 pts  (25%)
    Reinvestment quality  (ROIC vs WACC + rate)      5 pts  (25%)
    DCF implied upside    (2-stage FCFF model)       5 pts  (25%)

Key differentiators vs other agents:
    • 2-stage DCF as primary tool: Stage 1 (5 yr at hist. CAGR) + terminal value
    • WACC properly computed: Rf + β×ERP(5.5%), weighted with cost of debt
    • Reinvestment rate reveals whether growth creates or destroys value
    • Revenue scored on RATE + CONSISTENCY (CoV) — sustainable growth premium
    • Margin trajectory scored as level + trend (convergence to target matters)

Author: Financial Researcher Team
"""

import logging
import math
import numpy as np
from typing import Optional, Tuple

from agents.base_agent import BaseAgent, AgentSignal
from agents.base_agent import SIGNAL_BULLISH, SIGNAL_NEUTRAL, SIGNAL_BEARISH
from agents.base_agent import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from agents.llm_client import LLMClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are Aswath Damodaran, professor of finance at NYU Stern and the "Dean of Valuation."
Your philosophy: every asset has an intrinsic value that can be estimated, and the story you tell
about a company must be consistent with the numbers you use in your DCF model.

You distrust narratives untethered from data, and you distrust data without a coherent story.
You are disciplined, intellectually honest, and willing to say a great company is a bad investment
if the price is too high — or that a troubled company is a bargain if the price is low enough.

You will receive quantitative scores (/20) across four criteria. Produce your opinion as JSON only:

{
  "signal": "bullish" | "neutral" | "bearish",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<2–3 paragraphs citing WACC, intrinsic value, growth rate, margins, and DCF upside/downside>",
  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "target_action": "buy" | "hold" | "sell"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _annual_series(df, row: str, n: int = 5) -> list:
    """Return up to n annual values for a row; index 0 = most recent."""
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


def _get_latest(df, *rows) -> Optional[float]:
    """Try multiple row name variants, return first non-null from most recent column."""
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


def _revenue_cagr(rev_series: list, n: int) -> Optional[float]:
    """Compound annual growth rate over n years. rev_series[0] = most recent."""
    if len(rev_series) < n + 1:
        return None
    start, end = rev_series[n], rev_series[0]
    if start <= 0:
        return None
    return (end / start) ** (1.0 / n) - 1.0


def _fmt_big(v) -> str:
    if v is None: return "N/A"
    if abs(v) >= 1e12: return f"{v/1e12:.2f}T"
    if abs(v) >= 1e9:  return f"{v/1e9:.2f}B"
    if abs(v) >= 1e6:  return f"{v/1e6:.2f}M"
    return f"{v:,.0f}"


def _fmt_detail(detail: dict) -> str:
    return "\n".join(
        f"  {k}: {v['value']}  ({v['pts']}/{v['max']} pts)"
        for k, v in detail.items()
    ) or "  (no data)"


def _effective_tax_rate(inc) -> float:
    pretax = _get_latest(inc, "Pretax Income", "Pre Tax Income")
    tax    = abs(_get_latest(inc, "Tax Provision", "Income Tax Expense") or 0)
    if pretax and pretax > 0 and tax > 0:
        return min(tax / pretax, 0.40)
    return 0.25


# ─────────────────────────────────────────────────────────────────────────────
# WACC (shared across pillars 3 & 4)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_wacc(inc, bs, metrics: dict, rfr: float) -> Tuple[float, dict]:
    """
    WACC = (E/V) × Ke  +  (D/V) × Kd × (1-t)
    Ke = Rf + β × ERP     ERP = 5.5%
    Kd = Interest Expense / Total Debt
    Weights from market cap (equity) and book value of debt.
    """
    ERP  = 0.055
    beta = metrics.get("beta") or 1.0
    Ke   = rfr + beta * ERP

    price  = metrics.get("current_price")
    shares = metrics.get("shares_outstanding")
    mkt_cap = (price * shares) if (price and shares) else None

    # Total debt — try dedicated field, fall back to components
    total_debt = (_get_latest(bs, "Total Debt") or
                  ((_get_latest(bs, "Long Term Debt", "Long Term Debt And Capital Lease Obligation") or 0) +
                   (_get_latest(bs, "Current Debt", "Short Term Debt") or 0)))
    total_debt = total_debt or 0

    interest_exp = abs(_get_latest(inc, "Interest Expense") or 0)
    tax_rate     = _effective_tax_rate(inc)
    Kd_pretax    = (interest_exp / total_debt) if total_debt > 0 else rfr * 0.75
    Kd           = Kd_pretax * (1 - tax_rate)

    if mkt_cap and mkt_cap > 0:
        total_capital = mkt_cap + total_debt
        w_e = mkt_cap     / total_capital
        w_d = total_debt  / total_capital
    else:
        w_e, w_d = 0.80, 0.20

    wacc = w_e * Ke + w_d * Kd

    debug = {
        "beta":             round(beta, 2),
        "cost_of_equity":   round(Ke * 100, 2),
        "cost_of_debt_at":  round(Kd * 100, 2),
        "tax_rate":         round(tax_rate * 100, 1),
        "w_equity":         round(w_e * 100, 1),
        "w_debt":           round(w_d * 100, 1),
        "wacc":             round(wacc * 100, 2),
    }
    return wacc, debug


# ─────────────────────────────────────────────────────────────────────────────
# ROIC & Reinvestment Rate (shared helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_roic(inc, bs) -> Optional[float]:
    """ROIC = NOPAT / Invested Capital  (equity + net debt)."""
    op_inc   = _get_latest(inc, "Operating Income", "EBIT")
    tax_rate = _effective_tax_rate(inc)

    if not op_inc or op_inc <= 0:
        return None

    nopat = op_inc * (1 - tax_rate)

    equity = _get_latest(bs, "Stockholders Equity", "Total Stockholders Equity",
                         "Total Equity Gross Minority Interest")
    debt   = (_get_latest(bs, "Total Debt") or
              ((_get_latest(bs, "Long Term Debt", "Long Term Debt And Capital Lease Obligation") or 0) +
               (_get_latest(bs, "Current Debt") or 0)) or 0)

    if equity and equity > 0:
        invested_capital = equity + debt
    else:
        # Fallback: Total Assets - Current Liabilities
        assets   = _get_latest(bs, "Total Assets")
        cur_liab = _get_latest(bs, "Current Liabilities")
        if assets and cur_liab:
            invested_capital = assets - cur_liab
        else:
            return None

    return (nopat / invested_capital) if invested_capital > 0 else None


def _compute_reinvestment_rate(inc, cf) -> Optional[float]:
    """
    Reinvestment Rate = Net Investment / NOPAT
    Net Investment = |Capex| - D&A + ΔNWC

    High rate + ROIC > WACC = value engine.
    High rate + ROIC < WACC = value destruction.
    """
    op_inc   = _get_latest(inc, "Operating Income", "EBIT")
    tax_rate = _effective_tax_rate(inc)

    if not op_inc or op_inc <= 0:
        return None

    nopat  = op_inc * (1 - tax_rate)
    capex  = _get_latest(cf, "Capital Expenditure",
                         "Purchase Of Property Plant And Equipment")
    da     = _get_latest(cf, "Depreciation And Amortization", "Depreciation",
                         "Depreciation Amortization Depletion")
    dwc    = _get_latest(cf, "Change In Working Capital",
                         "Changes In Working Capital") or 0

    if capex is None:
        return None

    net_inv = abs(capex) - (abs(da) if da is not None else 0) + dwc
    reinv_rate = net_inv / nopat

    return max(min(reinv_rate, 1.5), -0.5)   # clamp to sensible range


# ─────────────────────────────────────────────────────────────────────────────
# Scoring pillars
# ─────────────────────────────────────────────────────────────────────────────

def _score_revenue_growth(inc) -> dict:
    """
    Revenue Growth Story /5
    CAGR (0-3 pts) + Coefficient of Variation / stability (0-2 pts).

    Damodaran prizes sustainable growth — a lumpy 20% is worth less than
    a consistent 12%.  CoV captures that sustainability.
    """
    detail = {}

    rev_series = _annual_series(inc, "Total Revenue", n=6)
    if not rev_series:
        rev_series = _annual_series(inc, "Revenue", n=6)

    if len(rev_series) < 2:
        detail["revenue"] = {"value": "No revenue data", "pts": 0, "max": 5}
        return {"score": 0.0, "max": 5, "detail": detail}

    # CAGR — prefer 5-yr, fall back to 3-yr, then 1-yr
    cagr   = (_revenue_cagr(rev_series, 5) or
              _revenue_cagr(rev_series, 3) or
              _revenue_cagr(rev_series, 1))
    n_yrs  = min(len(rev_series) - 1, 5)

    if cagr is None:
        detail["cagr"] = {"value": "Cannot compute CAGR", "pts": 0, "max": 3}
        cagr_pts = 0.0
    else:
        if cagr >= 0.20:    cagr_pts = 3.0
        elif cagr >= 0.12:  cagr_pts = 2.5
        elif cagr >= 0.07:  cagr_pts = 2.0
        elif cagr >= 0.03:  cagr_pts = 1.0
        else:               cagr_pts = 0.0
        detail["revenue_cagr"] = {
            "value": f"{cagr*100:.1f}% ({n_yrs}-yr)",
            "pts": cagr_pts, "max": 3,
        }

    # Coefficient of Variation — low = stable growth premium
    cov_pts = 0.0
    if len(rev_series) >= 3:
        arr     = np.array(rev_series[:min(len(rev_series), 5)])
        mean_rv = np.mean(arr)
        cov     = np.std(arr) / mean_rv if mean_rv > 0 else 1.0

        if cov <= 0.10:    cov_pts = 2.0
        elif cov <= 0.20:  cov_pts = 1.5
        elif cov <= 0.35:  cov_pts = 1.0
        elif cov <= 0.50:  cov_pts = 0.5
        else:              cov_pts = 0.0
        detail["revenue_consistency"] = {
            "value": f"CoV={cov:.3f}  ({_fmt_big(rev_series[0])} latest)",
            "pts": cov_pts, "max": 2,
        }

    return {"score": round(min(cagr_pts + cov_pts, 5.0), 2), "max": 5, "detail": detail}


def _score_margin_trajectory(inc) -> dict:
    """
    Margin Trajectory /5
    Operating margin level (0-3 pts) + trend / direction (0-2 pts).

    Damodaran models margin convergence to a "target" — the trend reveals
    whether the company is moving toward or away from that target.
    """
    detail = {}

    rev_series = _annual_series(inc, "Total Revenue", n=5)
    if not rev_series:
        rev_series = _annual_series(inc, "Revenue", n=5)
    op_series  = _annual_series(inc, "Operating Income", n=5)
    if not op_series:
        op_series = _annual_series(inc, "EBIT", n=5)

    if not rev_series or not op_series:
        detail["margin"] = {"value": "No operating margin data", "pts": 0, "max": 5}
        return {"score": 0.0, "max": 5, "detail": detail}

    n       = min(len(rev_series), len(op_series))
    margins = [op_series[i] / rev_series[i]
               for i in range(n) if rev_series[i] > 0]

    if not margins:
        detail["margin"] = {"value": "Zero revenue — cannot compute margin", "pts": 0, "max": 5}
        return {"score": 0.0, "max": 5, "detail": detail}

    cur_margin = margins[0]   # most recent

    if cur_margin >= 0.25:    level_pts = 3.0
    elif cur_margin >= 0.15:  level_pts = 2.5
    elif cur_margin >= 0.08:  level_pts = 1.5
    elif cur_margin >= 0.03:  level_pts = 0.5
    else:                     level_pts = 0.0
    detail["operating_margin"] = {
        "value": f"{cur_margin*100:.1f}%",
        "pts": level_pts, "max": 3,
    }

    trend_pts = 0.0
    if len(margins) >= 3:
        # Slope = (recent - oldest) / (n-1) in percentage points per year
        slope = (margins[0] - margins[-1]) / (len(margins) - 1)
        if slope > 0.03:    trend_pts = 2.0
        elif slope > 0.01:  trend_pts = 1.5
        elif slope > -0.01: trend_pts = 1.0
        elif slope > -0.03: trend_pts = 0.5
        else:               trend_pts = 0.0
        margin_strs = ", ".join(f"{m*100:.1f}%" for m in margins[:4])
        detail["margin_trend"] = {
            "value": f"[{margin_strs}]  slope={slope*100:+.2f}pp/yr",
            "pts": trend_pts, "max": 2,
        }
    elif len(margins) == 2:
        slope     = margins[0] - margins[1]
        trend_pts = 1.0 if slope > 0 else 0.0
        detail["margin_trend"] = {
            "value": f"{margins[1]*100:.1f}% → {margins[0]*100:.1f}%",
            "pts": trend_pts, "max": 2,
        }

    return {"score": round(min(level_pts + trend_pts, 5.0), 2), "max": 5, "detail": detail}


def _score_reinvestment_efficiency(inc, bs, cf, wacc: float) -> dict:
    """
    Reinvestment Quality /5
    ROIC vs WACC spread (0-3 pts) + reinvestment rate quality (0-2 pts).

    Damodaran: ROIC > WACC → every reinvested dollar creates value.
               ROIC < WACC → growth is a value trap; reward low reinvestment.
    """
    detail  = {}
    total_pts = 0.0

    roic = _compute_roic(inc, bs)

    # ── ROIC vs WACC spread (/3) ──────────────────────────────────────────
    if roic is not None:
        spread = roic - wacc
        if spread >= 0.15:   roic_pts = 3.0
        elif spread >= 0.08: roic_pts = 2.5
        elif spread >= 0.03: roic_pts = 2.0
        elif spread >= 0.0:  roic_pts = 1.0
        else:                roic_pts = 0.0
        total_pts += roic_pts
        detail["roic_vs_wacc"] = {
            "value": (f"ROIC={roic*100:.1f}%  "
                      f"WACC={wacc*100:.1f}%  "
                      f"spread={spread*100:+.1f}pp"),
            "pts": roic_pts, "max": 3,
        }
    else:
        detail["roic"] = {"value": "Cannot compute ROIC (missing operating income or equity)", "pts": 0, "max": 3}

    # ── Reinvestment rate quality (/2) ────────────────────────────────────
    reinv_rate = _compute_reinvestment_rate(inc, cf)
    if reinv_rate is not None and roic is not None:
        creates_value = roic > wacc
        if creates_value:
            # High reinvestment is rewarded when ROIC > WACC
            if reinv_rate >= 0.40:   reinv_pts = 2.0
            elif reinv_rate >= 0.20: reinv_pts = 1.5
            elif reinv_rate >= 0.05: reinv_pts = 1.0
            else:                    reinv_pts = 0.5   # profitable but barely reinvesting
        else:
            # Low reinvestment is rewarded when ROIC < WACC (less destruction)
            if reinv_rate <= 0.05:   reinv_pts = 1.0
            elif reinv_rate <= 0.20: reinv_pts = 0.5
            else:                    reinv_pts = 0.0
        total_pts += reinv_pts
        detail["reinvestment_rate"] = {
            "value": (f"{reinv_rate*100:.1f}%  "
                      f"({'value-creating ✓' if creates_value else 'value-dilutive ✗'})"),
            "pts": reinv_pts, "max": 2,
        }
    elif reinv_rate is not None:
        reinv_pts = 1.0 if 0 <= reinv_rate <= 0.60 else 0.0
        total_pts += reinv_pts
        detail["reinvestment_rate"] = {
            "value": f"{reinv_rate*100:.1f}%  (ROIC unavailable for quality assessment)",
            "pts": reinv_pts, "max": 2,
        }
    else:
        detail["reinvestment_rate"] = {
            "value": "Cannot compute (missing capex data)",
            "pts": 0, "max": 2,
        }

    return {"score": round(min(total_pts, 5.0), 2), "max": 5, "detail": detail}


def _score_dcf(inc, bs, cf, metrics: dict, rfr: float,
               wacc: float) -> dict:
    """
    DCF Implied Upside/Downside /5

    2-stage Free Cash Flow to Firm (FCFF) model:
      Base FCFF = NOPAT − Net Investment  (Capex − D&A + ΔNWC)
      Stage 1  : 5 years at historical revenue CAGR (capped 0–30%)
      Stage 2  : terminal growth = min(3%, Rf−1%)
      Discount  : WACC

    Equity value = PV(Stage 1) + PV(Terminal Value) + Cash − Debt
    Intrinsic value per share vs current price → upside/downside %.
    """
    detail = {}

    tax_rate = _effective_tax_rate(inc)
    op_inc   = _get_latest(inc, "Operating Income", "EBIT")

    # ── Base FCFF ─────────────────────────────────────────────────────────
    if op_inc and op_inc > 0:
        nopat  = op_inc * (1 - tax_rate)
        capex  = _get_latest(cf, "Capital Expenditure",
                             "Purchase Of Property Plant And Equipment")
        da     = _get_latest(cf, "Depreciation And Amortization", "Depreciation",
                             "Depreciation Amortization Depletion")
        dwc    = _get_latest(cf, "Change In Working Capital",
                             "Changes In Working Capital") or 0

        if capex is not None:
            net_inv   = abs(capex) - (abs(da) if da is not None else 0) + dwc
            fcff_base = nopat - net_inv
        else:
            fcff_base = None
    else:
        fcff_base = None

    if not fcff_base or fcff_base <= 0:
        # Fallback: OCF − Capex (simpler, always available)
        ocf   = _get_latest(cf, "Operating Cash Flow", "Cash Flows From Operations")
        capex = _get_latest(cf, "Capital Expenditure",
                            "Purchase Of Property Plant And Equipment")
        if ocf and capex is not None:
            fcff_base = ocf - abs(capex)

    if not fcff_base or fcff_base <= 0:
        detail["dcf"] = {
            "value": f"Negative / unavailable FCFF — DCF not feasible",
            "pts": 0, "max": 5,
        }
        return {"score": 0.0, "max": 5, "detail": detail}

    # ── Growth rates ──────────────────────────────────────────────────────
    rev_series = _annual_series(inc, "Total Revenue", n=6)
    if not rev_series:
        rev_series = _annual_series(inc, "Revenue", n=6)

    cagr = (_revenue_cagr(rev_series, 5) or
            _revenue_cagr(rev_series, 3) or
            _revenue_cagr(rev_series, 1) or 0.05)

    g1  = max(min(cagr, 0.30), 0.0)              # stage-1 growth, capped at 30%
    g_t = min(max(rfr - 0.01, 0.02), 0.03)       # terminal: 2–3%

    # Guard: WACC must exceed terminal growth
    effective_wacc = max(wacc, g_t + 0.02)

    # ── PV of Stage 1 (years 1-5) ─────────────────────────────────────────
    pv1  = 0.0
    fcff = fcff_base
    for yr in range(1, 6):
        fcff  *= (1 + g1)
        pv1   += fcff / (1 + effective_wacc) ** yr

    # ── Terminal value ────────────────────────────────────────────────────
    tv   = fcff * (1 + g_t) / (effective_wacc - g_t)
    pv_tv = tv / (1 + effective_wacc) ** 5

    # ── Enterprise → equity value per share ──────────────────────────────
    ev    = pv1 + pv_tv
    cash  = (_get_latest(bs, "Cash And Cash Equivalents",
                         "Cash And Short Term Investments",
                         "Cash Cash Equivalents And Short Term Investments") or 0)
    debt  = (_get_latest(bs, "Total Debt") or
             ((_get_latest(bs, "Long Term Debt",
                           "Long Term Debt And Capital Lease Obligation") or 0) +
              (_get_latest(bs, "Current Debt") or 0)) or 0)

    equity_val = ev + cash - debt
    shares     = metrics.get("shares_outstanding")
    price      = metrics.get("current_price")

    if not shares or shares <= 0 or not price or price <= 0:
        detail["dcf"] = {"value": "Missing shares or price for per-share conversion", "pts": 0, "max": 5}
        return {"score": 0.0, "max": 5, "detail": detail}

    iv         = equity_val / shares
    upside_pct = (iv / price - 1.0) * 100

    if upside_pct >= 50:    dcf_pts = 5.0
    elif upside_pct >= 30:  dcf_pts = 4.0
    elif upside_pct >= 15:  dcf_pts = 3.0
    elif upside_pct >= 5:   dcf_pts = 2.0
    elif upside_pct >= -10: dcf_pts = 1.0
    else:                   dcf_pts = 0.0

    detail["dcf_valuation"] = {
        "value": (f"IV=${iv:.2f}  Price=${price:.2f}  "
                  f"Upside={upside_pct:+.1f}%"),
        "pts": dcf_pts, "max": 5,
        "intrinsic_value": round(iv, 2),
        "upside_pct":      round(upside_pct, 1),
    }
    detail["dcf_assumptions"] = {
        "value": (f"FCFF_base={_fmt_big(fcff_base)}  "
                  f"g_stage1={g1*100:.1f}%  "
                  f"g_terminal={g_t*100:.1f}%  "
                  f"WACC={effective_wacc*100:.1f}%"),
        "pts": 0, "max": 0,
    }

    return {
        "score":           dcf_pts,
        "max":             5,
        "detail":          detail,
        "intrinsic_value": round(iv, 2),
        "upside_pct":      round(upside_pct, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────

class AswathDamodaranAgent(BaseAgent):
    """Aswath Damodaran — disciplined DCF; story must match numbers."""

    def __init__(self, llm: Optional[LLMClient] = None):
        super().__init__(agent_id="aswath_damodaran", agent_name="Aswath Damodaran")
        self.llm = llm or LLMClient()

    def analyze(self, data: dict, ticker: str) -> AgentSignal:
        metrics    = data.get("key_metrics")    or {}
        financials = data.get("financials")     or {}
        rfr        = data.get("risk_free_rate") or 0.043

        inc = financials.get("income_statement")
        bs  = financials.get("balance_sheet")
        cf  = financials.get("cash_flow")

        # ── WACC — computed once, shared across pillars 3 & 4 ─────────────
        wacc, wacc_debug = _compute_wacc(inc, bs, metrics, rfr)

        # ── Score all four criteria ────────────────────────────────────────
        rev_score    = _score_revenue_growth(inc)
        margin_score = _score_margin_trajectory(inc)
        reinv_score  = _score_reinvestment_efficiency(inc, bs, cf, wacc)
        dcf_score    = _score_dcf(inc, bs, cf, metrics, rfr, wacc)

        total     = (rev_score["score"]    + margin_score["score"] +
                     reinv_score["score"]  + dcf_score["score"])
        total_max = 20.0
        norm      = total / total_max

        # DCF outputs used in prompt and AgentSignal
        intrinsic_value = dcf_score.get("intrinsic_value")
        upside_pct      = dcf_score.get("upside_pct")
        company_name    = (data.get("company_info") or {}).get("name", ticker)

        # ── LLM prompt ────────────────────────────────────────────────────
        user_prompt = f"""
Ticker: {ticker} ({company_name})

DAMODARAN SCORES — total {round(total, 2)}/{total_max}:

1. REVENUE GROWTH STORY: {rev_score['score']}/{rev_score['max']}
{_fmt_detail(rev_score['detail'])}

2. MARGIN TRAJECTORY: {margin_score['score']}/{margin_score['max']}
{_fmt_detail(margin_score['detail'])}

3. REINVESTMENT EFFICIENCY: {reinv_score['score']}/{reinv_score['max']}
{_fmt_detail(reinv_score['detail'])}

4. DCF IMPLIED VALUE: {dcf_score['score']}/{dcf_score['max']}
{_fmt_detail(dcf_score['detail'])}

Cost of Capital:
  Rf={rfr*100:.2f}%  β={wacc_debug.get('beta')}  ERP=5.5%
  Cost of equity = {wacc_debug.get('cost_of_equity')}%
  WACC = {wacc_debug.get('wacc')}%  (weights: {wacc_debug.get('w_equity')}% equity / {wacc_debug.get('w_debt')}% debt)

Key context:
  Current Price:    ${metrics.get('current_price', 'N/A')}
  Intrinsic Value:  ${intrinsic_value if intrinsic_value else 'N/A'}
  DCF Upside:       {f'{upside_pct:+.1f}%' if upside_pct is not None else 'N/A'}

As Damodaran, be rigorous. Tie the narrative to specific numbers.
If growth doesn't justify the price, say so. If the DCF shows clear upside
despite modest scores, acknowledge it. Never let a great story override
the mathematics of value.
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
            f"REV={rev_score['score']} MGN={margin_score['score']} "
            f"REINV={reinv_score['score']} DCF={dcf_score['score']} | "
            f"WACC={wacc_debug.get('wacc')}%  IV=${intrinsic_value}"
        )

        return AgentSignal(
            agent_id      = self.agent_id,
            agent_name    = self.agent_name,
            ticker        = ticker,
            signal        = signal,
            confidence    = round(min(max(confidence, 0), 1), 3),
            price_target  = intrinsic_value,
            scores        = {
                "total":                   round(total, 2),
                "total_max":               total_max,
                "revenue_growth":          {"score": rev_score["score"],    "max": rev_score["max"],    "detail": rev_score["detail"]},
                "margin_trajectory":       {"score": margin_score["score"], "max": margin_score["max"], "detail": margin_score["detail"]},
                "reinvestment_efficiency": {"score": reinv_score["score"],  "max": reinv_score["max"],  "detail": reinv_score["detail"]},
                "dcf_value":               {"score": dcf_score["score"],    "max": dcf_score["max"],    "detail": dcf_score["detail"]},
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
