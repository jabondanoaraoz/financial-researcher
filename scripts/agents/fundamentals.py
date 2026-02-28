"""
Fundamentals Agent
==================
100% quantitative agent — no LLM calls.

Scores a company across four pillars using hard thresholds:
    1. Valuation      — Is the stock cheap or expensive?
    2. Profitability  — Is the business high-quality?
    3. Growth         — Is it growing fast enough?
    4. Financial Health — Is the balance sheet safe?

Each pillar returns:
    • sub_signal  : "bullish" | "neutral" | "bearish"
    • score       : 0.0–1.0  (1 = best possible outcome)
    • details     : dict of individual metric results

The final signal is the plurality vote of the four pillars.
Confidence = fraction of pillars that agree with the final signal.

Author: Financial Researcher Team
"""

import logging
from typing import Optional

from agents.base_agent import BaseAgent, AgentSignal
from agents.base_agent import SIGNAL_BULLISH, SIGNAL_NEUTRAL, SIGNAL_BEARISH
from agents.base_agent import ACTION_BUY, ACTION_HOLD, ACTION_SELL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold tables
# ---------------------------------------------------------------------------

# Valuation — LOWER is better
VALUATION_THRESHOLDS = {
    # key            : (bullish_if_below, bearish_if_above)
    "pe_ratio"       : (10.0,  35.0),
    "pb_ratio"       : (2.0,   5.0),
    "ev_ebitda"      : (10.0,  22.0),
    "ps_ratio"       : (2.0,   8.0),
    "peg_ratio"      : (1.0,   2.5),
}

# Profitability — HIGHER is better (values as %)
PROFITABILITY_THRESHOLDS = {
    # key                  : (bullish_if_above, bearish_if_below)
    "gross_margin"         : (50.0, 25.0),
    "operating_margin"     : (20.0,  8.0),
    "net_margin"           : (15.0,  4.0),
    "roe"                  : (20.0,  8.0),
    "roa"                  : (10.0,  3.0),
}

# Growth — HIGHER is better (values as %)
GROWTH_THRESHOLDS = {
    # key                  : (bullish_if_above, bearish_if_below)
    "revenue_growth_yoy"   : (20.0,  3.0),
    "earnings_growth_yoy"  : (25.0,  0.0),
    "fcf_growth_yoy"       : (15.0, -5.0),
}

# Financial Health — mixed directions (annotated per metric)
HEALTH_THRESHOLDS = {
    # key                  : (bull_threshold, bear_threshold)
    "current_ratio"        : (2.0,  1.0),   # higher = better
    "debt_to_equity"       : (0.5,  2.0),   # lower  = better
    "interest_coverage"    : (8.0,  2.0),   # higher = better
    "net_debt_to_ebitda"   : (1.0,  4.0),   # lower  = better
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_low_is_good(value: float, bull_t: float, bear_t: float) -> tuple:
    """Signal + score for metrics where a lower value is bullish."""
    if value <= bull_t:
        score = 1.0 - (value / bull_t) * 0.5
        return SIGNAL_BULLISH, min(1.0, score)
    elif value >= bear_t:
        score = max(0.0, 1.0 - (value / bear_t) * 0.5)
        return SIGNAL_BEARISH, score
    else:
        span  = bear_t - bull_t
        pos   = (value - bull_t) / span
        return SIGNAL_NEUTRAL, round(0.7 - pos * 0.2, 3)


def _score_high_is_good(value: float, bull_t: float, bear_t: float) -> tuple:
    """Signal + score for metrics where a higher value is bullish."""
    if value >= bull_t:
        score = min(1.0, 0.7 + (value - bull_t) / (bull_t + 1e-9) * 0.3)
        return SIGNAL_BULLISH, score
    elif value <= bear_t:
        score = max(0.0, value / (bear_t + 1e-9) * 0.4)
        return SIGNAL_BEARISH, score
    else:
        span  = bull_t - bear_t
        pos   = (value - bear_t) / (span + 1e-9)
        return SIGNAL_NEUTRAL, round(0.4 + pos * 0.3, 3)


def _pillar_vote(signals: list) -> tuple:
    """Plurality vote → (winning_signal, confidence)."""
    if not signals:
        return SIGNAL_NEUTRAL, 0.5
    counts  = {s: signals.count(s) for s in (SIGNAL_BULLISH, SIGNAL_NEUTRAL, SIGNAL_BEARISH)}
    winner  = max(counts, key=counts.__getitem__)
    conf    = round(counts[winner] / len(signals), 3)
    return winner, conf


# ---------------------------------------------------------------------------
# Fundamentals Agent
# ---------------------------------------------------------------------------

class FundamentalsAgent(BaseAgent):
    """Quantitative fundamentals agent — no LLM required."""

    def __init__(self):
        super().__init__(agent_id="fundamentals", agent_name="Fundamentals Analyst")

    # ── Pillar 1: Valuation ───────────────────────────────────────────

    def _score_valuation(self, metrics: dict, av: dict) -> dict:
        results, signals, scores = {}, [], []

        def get(yf_key, av_key=None):
            v = metrics.get(yf_key)
            if v is None and av:
                v = av.get(av_key or yf_key)
            return v

        checks = [
            ("pe_ratio",  get("pe_ratio",  "pe_ratio")),
            ("pb_ratio",  get("pb_ratio",  "price_to_book")),
            ("ev_ebitda", get("ev_ebitda", "ev_to_ebitda")),
            ("ps_ratio",  get("ps_ratio",  "price_to_sales")),
            ("peg_ratio", get("peg_ratio", "peg_ratio")),
        ]

        for key, val in checks:
            if val is None or val <= 0:
                continue
            bull_t, bear_t = VALUATION_THRESHOLDS[key]
            sig, sc = _score_low_is_good(val, bull_t, bear_t)
            results[key] = {"value": round(val, 2), "signal": sig, "score": round(sc, 3)}
            signals.append(sig)
            scores.append(sc)

        p_sig, _ = _pillar_vote(signals)
        return {
            "signal": p_sig,
            "score":  round(sum(scores) / len(scores), 3) if scores else 0.5,
            "detail": results,
            "n_metrics": len(signals),
        }

    # ── Pillar 2: Profitability ───────────────────────────────────────

    def _score_profitability(self, metrics: dict, av: dict, financials: dict) -> dict:
        results, signals, scores = {}, [], []

        gross_margin = op_margin = net_margin = roe = roa = None

        # Extract margins from income statement
        if financials:
            inc = financials.get("income_statement")
            if inc is not None and not inc.empty:
                col = inc.columns[0]

                def _pct(numerator_row):
                    rev_key = "Total Revenue" if "Total Revenue" in inc.index else None
                    if numerator_row not in inc.index or rev_key is None:
                        return None
                    rev = inc.loc[rev_key, col]
                    val = inc.loc[numerator_row, col]
                    if rev and rev != 0:
                        return (val / rev) * 100
                    return None

                gross_margin = _pct("Gross Profit")
                op_margin    = _pct("Operating Income")
                net_margin   = _pct("Net Income")

        # ROE / ROA from Alpha Vantage (decimal → %)
        if av:
            if av.get("return_on_equity_ttm"):
                roe = av["return_on_equity_ttm"] * 100
            if av.get("return_on_assets_ttm"):
                roa = av["return_on_assets_ttm"] * 100

        for key, val in [
            ("gross_margin",     gross_margin),
            ("operating_margin", op_margin),
            ("net_margin",       net_margin),
            ("roe",              roe),
            ("roa",              roa),
        ]:
            if val is None:
                continue
            bull_t, bear_t = PROFITABILITY_THRESHOLDS[key]
            sig, sc = _score_high_is_good(val, bull_t, bear_t)
            results[key] = {"value": round(val, 2), "signal": sig, "score": round(sc, 3)}
            signals.append(sig)
            scores.append(sc)

        p_sig, _ = _pillar_vote(signals)
        return {
            "signal": p_sig,
            "score":  round(sum(scores) / len(scores), 3) if scores else 0.5,
            "detail": results,
            "n_metrics": len(signals),
        }

    # ── Pillar 3: Growth ──────────────────────────────────────────────

    def _score_growth(self, av: dict, financials: dict) -> dict:
        results, signals, scores = {}, [], []

        rev_growth = earn_growth = fcf_growth = None

        # From income statement (YoY)
        if financials:
            inc = financials.get("income_statement")
            if inc is not None and not inc.empty and inc.shape[1] >= 2:
                cols = inc.columns

                def _yoy(row):
                    if row not in inc.index:
                        return None
                    v0 = inc.loc[row, cols[0]]
                    v1 = inc.loc[row, cols[1]]
                    if v1 and v1 != 0:
                        return ((v0 - v1) / abs(v1)) * 100
                    return None

                rev_growth  = _yoy("Total Revenue")
                earn_growth = _yoy("Net Income")

            # FCF from cash flow
            cf = financials.get("cash_flow")
            if cf is not None and not cf.empty and cf.shape[1] >= 2:
                cols = cf.columns
                fcf_row = "Free Cash Flow" if "Free Cash Flow" in cf.index else (
                    "Operating Cash Flow" if "Operating Cash Flow" in cf.index else None
                )
                if fcf_row:
                    f0, f1 = cf.loc[fcf_row, cols[0]], cf.loc[fcf_row, cols[1]]
                    if f1 and f1 > 0:
                        fcf_growth = ((f0 - f1) / abs(f1)) * 100

        # AV quarterly overrides (more recent)
        if av:
            if av.get("quarterly_revenue_growth_yoy"):
                rev_growth  = av["quarterly_revenue_growth_yoy"] * 100
            if av.get("quarterly_earnings_growth_yoy"):
                earn_growth = av["quarterly_earnings_growth_yoy"] * 100

        for key, val in [
            ("revenue_growth_yoy",  rev_growth),
            ("earnings_growth_yoy", earn_growth),
            ("fcf_growth_yoy",      fcf_growth),
        ]:
            if val is None:
                continue
            bull_t, bear_t = GROWTH_THRESHOLDS[key]
            sig, sc = _score_high_is_good(val, bull_t, bear_t)
            results[key] = {"value": round(val, 2), "signal": sig, "score": round(sc, 3)}
            signals.append(sig)
            scores.append(sc)

        p_sig, _ = _pillar_vote(signals)
        return {
            "signal": p_sig,
            "score":  round(sum(scores) / len(scores), 3) if scores else 0.5,
            "detail": results,
            "n_metrics": len(signals),
        }

    # ── Pillar 4: Financial Health ────────────────────────────────────

    def _score_health(self, financials: dict) -> dict:
        results, signals, scores = {}, [], []

        if not financials:
            return {"signal": SIGNAL_NEUTRAL, "score": 0.5, "detail": {}, "n_metrics": 0}

        bs  = financials.get("balance_sheet")
        inc = financials.get("income_statement")

        current_ratio = debt_to_equity = interest_coverage = net_debt_ebitda = None

        if bs is not None and not bs.empty:
            col = bs.columns[0]

            def _bs(row):
                return bs.loc[row, col] if row in bs.index else None

            cur_assets = _bs("Current Assets")
            cur_liab   = _bs("Current Liabilities")
            total_debt = _bs("Total Debt") or _bs("Long Term Debt")
            equity     = _bs("Stockholders Equity") or _bs("Total Stockholders Equity")
            cash       = _bs("Cash And Cash Equivalents") or _bs("Cash Cash Equivalents And Short Term Investments")

            if cur_assets and cur_liab and cur_liab != 0:
                current_ratio = cur_assets / cur_liab
            if total_debt and equity and equity != 0:
                debt_to_equity = total_debt / abs(equity)

            # Net Debt / EBITDA
            if total_debt is not None and cash is not None and inc is not None and not inc.empty:
                ic = inc.columns[0]
                ebitda = inc.loc["EBITDA", ic] if "EBITDA" in inc.index else None
                if ebitda and ebitda > 0:
                    net_debt_ebitda = (total_debt - cash) / ebitda

        # Interest coverage: EBIT / Interest Expense
        if inc is not None and not inc.empty:
            ic = inc.columns[0]
            ebit     = inc.loc["Operating Income", ic]     if "Operating Income"  in inc.index else None
            interest = inc.loc["Interest Expense", ic]     if "Interest Expense"  in inc.index else None
            if ebit and interest and interest != 0:
                interest_coverage = ebit / abs(interest)

        # Score each metric
        health_checks = [
            ("current_ratio",      current_ratio,      True),   # high=good
            ("debt_to_equity",     debt_to_equity,     False),  # low=good
            ("interest_coverage",  interest_coverage,  True),   # high=good
            ("net_debt_to_ebitda", net_debt_ebitda,    False),  # low=good
        ]

        for key, val, high_is_good in health_checks:
            if val is None:
                continue
            bull_t, bear_t = HEALTH_THRESHOLDS[key]
            if high_is_good:
                sig, sc = _score_high_is_good(val, bull_t, bear_t)
            else:
                sig, sc = _score_low_is_good(val, bull_t, bear_t)
            results[key] = {"value": round(val, 3), "signal": sig, "score": round(sc, 3)}
            signals.append(sig)
            scores.append(sc)

        p_sig, _ = _pillar_vote(signals)
        return {
            "signal": p_sig,
            "score":  round(sum(scores) / len(scores), 3) if scores else 0.5,
            "detail": results,
            "n_metrics": len(signals),
        }

    # ── Reasoning & risks ─────────────────────────────────────────────

    def _build_reasoning(self, ticker: str, pillars: dict, final_signal: str, confidence: float) -> str:
        lines = [f"{ticker} — Fundamentals Analysis  |  {final_signal.upper()}  |  {confidence:.0%} confidence\n"]

        labels = {
            "valuation":     "Valuation",
            "profitability": "Profitability",
            "growth":        "Growth",
            "health":        "Financial Health",
        }

        for key, label in labels.items():
            p = pillars[key]
            sig_icon = {"bullish": "✅", "neutral": "⚖️", "bearish": "🔴"}.get(p["signal"], "")
            lines.append(f"  {sig_icon} {label} [{p['signal'].upper()} | score {p['score']:.2f}]")
            for m, info in p["detail"].items():
                arrow = "↑" if info["signal"] == SIGNAL_BULLISH else ("↓" if info["signal"] == SIGNAL_BEARISH else "→")
                lines.append(f"      {arrow} {m}: {info['value']}  ({info['signal']})")

        return "\n".join(lines)

    def _build_risks(self, pillars: dict) -> list:
        risks = []
        for p in pillars.values():
            for metric, info in p["detail"].items():
                if info["signal"] == SIGNAL_BEARISH:
                    risks.append(f"{metric.replace('_', ' ').title()} is stretched ({info['value']})")
        return risks[:5] if risks else ["No major fundamental red flags identified"]

    # ── BaseAgent.analyze() ───────────────────────────────────────────

    def analyze(self, data: dict, ticker: str) -> AgentSignal:
        metrics    = data.get("key_metrics")  or {}
        financials = data.get("financials")   or {}
        av         = data.get("av_overview")  or {}
        xbrl       = data.get("xbrl_facts")   or {}

        pillars = {
            "valuation":    self._score_valuation(metrics, av),
            "profitability": self._score_profitability(metrics, av, financials),
            "growth":       self._score_growth(av, financials),
            "health":       self._score_health(financials),
        }

        # Plurality vote across four pillars
        all_signals  = [p["signal"] for p in pillars.values()]
        counts       = {s: all_signals.count(s) for s in (SIGNAL_BULLISH, SIGNAL_NEUTRAL, SIGNAL_BEARISH)}
        final_signal = max(counts, key=counts.__getitem__)
        confidence   = round(counts[final_signal] / len(all_signals), 3)

        action_map  = {SIGNAL_BULLISH: ACTION_BUY, SIGNAL_NEUTRAL: ACTION_HOLD, SIGNAL_BEARISH: ACTION_SELL}

        self.logger.info(
            f"{ticker} → {final_signal.upper()} ({confidence:.0%}) | "
            f"val={pillars['valuation']['score']:.2f} "
            f"prof={pillars['profitability']['score']:.2f} "
            f"growth={pillars['growth']['score']:.2f} "
            f"health={pillars['health']['score']:.2f}"
        )

        return AgentSignal(
            agent_id      = self.agent_id,
            agent_name    = self.agent_name,
            ticker        = ticker,
            signal        = final_signal,
            confidence    = confidence,
            scores        = {k: {"signal": v["signal"], "score": v["score"]} for k, v in pillars.items()},
            reasoning     = self._build_reasoning(ticker, pillars, final_signal, confidence),
            key_risks     = self._build_risks(pillars),
            target_action = action_map[final_signal],
        )
