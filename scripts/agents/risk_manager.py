"""
Risk Manager Agent
100% quantitative - NO LLM.

Computes portfolio-level risk metrics from price history and maps them
into a position-sizing recommendation via a simplified Kelly Criterion.

Metrics (/20):
    Volatility      4 pts  - annualized std dev of daily returns (lower = better)
    Beta            4 pts  - systematic risk vs S&P 500
    Max Drawdown    4 pts  - worst peak-to-trough over 12 months (lower = better)
    Sharpe Proxy    4 pts  - risk-adjusted return estimate
    Kelly Position  4 pts  - recommended max position size (higher = safer to size up)

Signal interpretation:
    High score = LOW risk profile → bullish (safe to own)
    Low score  = HIGH risk profile → bearish (reduce or avoid)

The `scores` dict embeds a `risk_metrics` sub-dict for the orchestrator
to surface directly to the Portfolio Manager.

"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

from agents.base_agent import BaseAgent, AgentSignal
from agents.base_agent import SIGNAL_BULLISH, SIGNAL_NEUTRAL, SIGNAL_BEARISH
from agents.base_agent import ACTION_BUY, ACTION_HOLD, ACTION_SELL

logger = logging.getLogger(__name__)

TRADING_DAYS   = 252
MAX_KELLY_SIZE = 0.25   # hard cap at 25% of portfolio


# Metric calculators

def _daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def _annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def _beta(stock_returns: pd.Series, spy_returns: pd.Series) -> float:
    """Covariance(stock, SPY) / Variance(SPY). Aligns on common dates."""
    common = stock_returns.index.intersection(spy_returns.index)
    if len(common) < 30:
        return 1.0
    s = stock_returns.loc[common]
    m = spy_returns.loc[common]
    cov = float(np.cov(s, m)[0][1])
    var = float(m.var())
    return cov / var if var > 0 else 1.0


def _max_drawdown(prices: pd.Series) -> float:
    """Maximum peak-to-trough drawdown over the full price series."""
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    return float(drawdown.min())   # negative number


def _sharpe(returns: pd.Series, risk_free_annual: float) -> float:
    """Annualized Sharpe ratio proxy (using historical mean return)."""
    ann_return = float(returns.mean() * TRADING_DAYS)
    ann_vol    = _annualized_volatility(returns)
    if ann_vol == 0:
        return 0.0
    return (ann_return - risk_free_annual) / ann_vol


def _kelly_fraction(returns: pd.Series, risk_free_annual: float) -> float:
    """
    Simplified Kelly: f* = (mu - rf) / sigma^2
    Capped at MAX_KELLY_SIZE.
    """
    mu    = float(returns.mean() * TRADING_DAYS)
    sigma = _annualized_volatility(returns)
    if sigma == 0:
        return 0.0
    kelly = (mu - risk_free_annual) / (sigma ** 2)
    return round(min(max(kelly, 0.0), MAX_KELLY_SIZE), 4)


# Scoring functions

def _score_volatility(vol: float) -> tuple[float, str]:
    """Lower volatility = higher score (less risk)."""
    if vol < 0.15:    return 4.0, f"{vol:.1%} - low volatility"
    elif vol < 0.25:  return 3.0, f"{vol:.1%} - moderate"
    elif vol < 0.35:  return 2.0, f"{vol:.1%} - elevated"
    elif vol < 0.50:  return 1.0, f"{vol:.1%} - high"
    else:             return 0.0, f"{vol:.1%} - very high"


def _score_beta(beta: float) -> tuple[float, str]:
    """Beta close to 1 = systematic; higher = more volatile than market."""
    if beta < 0.5:    return 3.0, f"{round(beta,2)} - defensive"
    elif beta < 0.8:  return 4.0, f"{round(beta,2)} - low-beta"
    elif beta < 1.1:  return 3.0, f"{round(beta,2)} - market-like"
    elif beta < 1.4:  return 2.0, f"{round(beta,2)} - above-market"
    elif beta < 1.8:  return 1.0, f"{round(beta,2)} - high beta"
    else:             return 0.0, f"{round(beta,2)} - very high beta"


def _score_drawdown(mdd: float) -> tuple[float, str]:
    """Less negative drawdown = higher score."""
    pct = mdd * 100  # negative
    if pct > -10:     return 4.0, f"{pct:.1f}% max drawdown - minimal"
    elif pct > -20:   return 3.0, f"{pct:.1f}% - moderate"
    elif pct > -30:   return 2.0, f"{pct:.1f}% - significant"
    elif pct > -45:   return 1.0, f"{pct:.1f}% - severe"
    else:             return 0.0, f"{pct:.1f}% - extreme"


def _score_sharpe(sharpe: float) -> tuple[float, str]:
    if sharpe >= 2.0:   return 4.0, f"{round(sharpe,2)} - excellent"
    elif sharpe >= 1.0: return 3.0, f"{round(sharpe,2)} - good"
    elif sharpe >= 0.5: return 2.0, f"{round(sharpe,2)} - acceptable"
    elif sharpe >= 0.0: return 1.0, f"{round(sharpe,2)} - poor"
    else:               return 0.0, f"{round(sharpe,2)} - negative"


def _score_kelly(kelly: float) -> tuple[float, str]:
    """Higher Kelly fraction (up to cap) = better risk/reward → higher score."""
    pct = kelly * 100
    if pct >= 20:     return 4.0, f"{pct:.1f}% recommended max position"
    elif pct >= 12:   return 3.0, f"{pct:.1f}%"
    elif pct >= 6:    return 2.0, f"{pct:.1f}%"
    elif pct >= 2:    return 1.0, f"{pct:.1f}%"
    else:             return 0.0, f"{pct:.1f}% - Kelly negative (risk > reward)"


# Agent class

class RiskManagerAgent(BaseAgent):
    """Risk Manager - pure quantitative, no LLM."""

    def __init__(self):
        super().__init__(agent_id="risk_manager", agent_name="Risk Manager")

    def analyze(self, data: dict, ticker: str) -> AgentSignal:
        prices_df  = data.get("prices")
        spy_df     = data.get("spy_prices")
        risk_free  = data.get("risk_free_rate") or 0.043
        key_metrics = data.get("key_metrics") or {}

        # Fallback: use beta from key_metrics if no price data
        if prices_df is None or prices_df.empty or len(prices_df) < 30:
            fallback_beta = key_metrics.get("beta") or 1.0
            beta_pts, beta_lbl = _score_beta(fallback_beta)
            return AgentSignal(
                agent_id="risk_manager", agent_name="Risk Manager",
                ticker=ticker, signal=SIGNAL_NEUTRAL, confidence=0.3,
                scores={
                    "total": 10.0, "total_max": 20.0,
                    "risk_metrics": {"beta": fallback_beta, "note": "insufficient price data"},
                },
                reasoning=f"Insufficient price history. Beta from provider: {fallback_beta}.",
                key_risks=["Insufficient price data for full risk assessment"],
                target_action=ACTION_HOLD,
            )

        close   = prices_df["Close"].dropna()
        returns = _daily_returns(close)

        # Compute all metrics
        vol = _annualized_volatility(returns)
        mdd = _max_drawdown(close)

        # Beta - use SPY if available, else key_metrics fallback
        if spy_df is not None and not spy_df.empty:
            spy_close   = spy_df["Close"].dropna()
            spy_returns = _daily_returns(spy_close)
            beta = _beta(returns, spy_returns)
        else:
            beta = key_metrics.get("beta") or 1.0

        sharpe = _sharpe(returns, risk_free)
        kelly  = _kelly_fraction(returns, risk_free)
        ann_return = float(returns.mean() * TRADING_DAYS)

        # Score each metric
        vol_pts,    vol_lbl    = _score_volatility(vol)
        beta_pts,   beta_lbl   = _score_beta(beta)
        mdd_pts,    mdd_lbl    = _score_drawdown(mdd)
        sharpe_pts, sharpe_lbl = _score_sharpe(sharpe)
        kelly_pts,  kelly_lbl  = _score_kelly(kelly)

        total     = vol_pts + beta_pts + mdd_pts + sharpe_pts + kelly_pts
        total_max = 20.0
        norm      = total / total_max

        signal     = _norm_to_signal(norm)
        confidence = round(min(max(abs(norm - 0.5) * 2.0, 0.25), 0.95), 3)

        # Risk metrics dict (surfaced by orchestrator)
        risk_metrics = {
            "annualized_volatility":  round(vol, 4),
            "annualized_return":      round(ann_return, 4),
            "beta":                   round(beta, 3),
            "max_drawdown":           round(mdd, 4),
            "sharpe_proxy":           round(sharpe, 3),
            "kelly_fraction":         round(kelly, 4),
            "max_position_size_pct":  round(kelly * 100, 1),
        }

        # Build reasoning (no LLM)
        reasoning = (
            f"Risk profile for {ticker}: annualized vol={vol:.1%}, beta={round(beta,2)}, "
            f"max drawdown={mdd:.1%}, Sharpe={round(sharpe,2)}, "
            f"Kelly max position={round(kelly*100,1)}%. "
            f"{'Low-risk profile supports allocation.' if norm >= 0.6 else 'Elevated risk profile warrants smaller position or hedging.'}"
        )

        risks = []
        if vol > 0.40:  risks.append(f"High annualized volatility ({vol:.1%})")
        if beta > 1.5:  risks.append(f"High beta ({round(beta,2)}) amplifies market drawdowns")
        if mdd < -0.35: risks.append(f"Severe max drawdown ({mdd:.1%}) in last 12 months")
        if sharpe < 0:  risks.append("Negative Sharpe ratio - return below risk-free rate")
        if not risks:
            risks = ["Market regime change could alter risk profile", "Historical volatility may understate tail risk"]

        self.logger.info(
            f"{ticker} → {signal.upper()} (conf={confidence:.0%}) | "
            f"total={round(total,1)}/{total_max} | "
            f"vol={vol:.1%} beta={round(beta,2)} MDD={mdd:.1%} "
            f"Sharpe={round(sharpe,2)} Kelly={round(kelly*100,1)}%"
        )

        return AgentSignal(
            agent_id      = "risk_manager",
            agent_name    = "Risk Manager",
            ticker        = ticker,
            signal        = signal,
            confidence    = confidence,
            scores        = {
                "total":      round(total, 2),
                "total_max":  total_max,
                "volatility": {"score": vol_pts,    "max": 4, "detail": {"annualized_vol":  {"value": vol_lbl,    "pts": vol_pts,    "max": 4}}},
                "beta":       {"score": beta_pts,   "max": 4, "detail": {"beta":            {"value": beta_lbl,   "pts": beta_pts,   "max": 4}}},
                "drawdown":   {"score": mdd_pts,    "max": 4, "detail": {"max_drawdown":    {"value": mdd_lbl,    "pts": mdd_pts,    "max": 4}}},
                "sharpe":     {"score": sharpe_pts, "max": 4, "detail": {"sharpe_proxy":    {"value": sharpe_lbl, "pts": sharpe_pts, "max": 4}}},
                "kelly":      {"score": kelly_pts,  "max": 4, "detail": {"kelly_position":  {"value": kelly_lbl,  "pts": kelly_pts,  "max": 4}}},
                "risk_metrics": risk_metrics,   # surfaced by orchestrator
            },
            reasoning     = reasoning,
            key_risks     = risks[:3],
            target_action = _signal_to_action(signal),
        )


# Utilities

def _norm_to_signal(score: float) -> str:
    if score >= 0.60: return SIGNAL_BULLISH
    if score >= 0.40: return SIGNAL_NEUTRAL
    return SIGNAL_BEARISH


def _signal_to_action(signal: str) -> str:
    return {SIGNAL_BULLISH: ACTION_BUY, SIGNAL_NEUTRAL: ACTION_HOLD, SIGNAL_BEARISH: ACTION_SELL}.get(signal, ACTION_HOLD)
