"""
Technical Analysis Agent
100% quantitative — NO LLM.

Analyses price action and volume using five indicator groups.
Each group scores 0–4 pts (neutral = 2, bullish = 4, bearish = 0).
Total /20 → maps to bullish / neutral / bearish signal.

Scoring (/20):
    RSI              4 pts  — momentum oscillator (oversold/overbought)
    MACD             4 pts  — trend + crossover signal
    Bollinger Bands  4 pts  — price relative to volatility envelope
    Moving Averages  5 pts  — SMA50/200 cross + price position
    Volume           3 pts  — volume confirmation of price trend

All calculations use yfinance Close prices fetched by the data layer.
Falls back gracefully when insufficient history is available.

Author: Joaquin Abondano w/ Claude Code
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

from agents.base_agent import BaseAgent, AgentSignal
from agents.base_agent import SIGNAL_BULLISH, SIGNAL_NEUTRAL, SIGNAL_BEARISH
from agents.base_agent import ACTION_BUY, ACTION_HOLD, ACTION_SELL

logger = logging.getLogger(__name__)


# Indicator calculators

def _rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    val   = rsi.iloc[-1]
    return float(val) if pd.notna(val) else None


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line, histogram) — last values."""
    ema_f  = close.ewm(span=fast,   adjust=False).mean()
    ema_s  = close.ewm(span=slow,   adjust=False).mean()
    line   = ema_f - ema_s
    sig    = line.ewm(span=signal,  adjust=False).mean()
    hist   = line - sig
    return float(line.iloc[-1]), float(sig.iloc[-1]), float(hist.iloc[-1]), float(hist.iloc[-2]) if len(hist) > 1 else float(hist.iloc[-1])


def _bollinger(close: pd.Series, period=20, n_std=2):
    """Returns (upper, middle, lower) — last values."""
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return float(upper.iloc[-1]), float(mid.iloc[-1]), float(lower.iloc[-1])


# Scoring functions

def _score_rsi(close: pd.Series) -> dict:
    rsi = _rsi(close)
    if rsi is None:
        return {"score": 2.0, "max": 4, "detail": {}}

    if rsi < 20:      pts = 4.0; label = f"{round(rsi,1)} — extreme oversold"
    elif rsi < 30:    pts = 3.5; label = f"{round(rsi,1)} — oversold"
    elif rsi < 45:    pts = 3.0; label = f"{round(rsi,1)} — mildly bullish"
    elif rsi <= 55:   pts = 2.0; label = f"{round(rsi,1)} — neutral"
    elif rsi <= 70:   pts = 1.0; label = f"{round(rsi,1)} — mildly overbought"
    elif rsi <= 80:   pts = 0.5; label = f"{round(rsi,1)} — overbought"
    else:             pts = 0.0; label = f"{round(rsi,1)} — extreme overbought"

    return {
        "score": pts, "max": 4,
        "detail": {"rsi": {"value": label, "pts": pts, "max": 4}},
        "_rsi": rsi,
    }


def _score_macd(close: pd.Series) -> dict:
    if len(close) < 35:
        return {"score": 2.0, "max": 4, "detail": {}}

    line, sig, hist, prev_hist = _macd(close)
    momentum_building = hist > prev_hist   # histogram growing = acceleration

    if hist > 0:
        label = f"bullish {'& accelerating' if momentum_building else 'but fading'}"
        pts   = 4.0 if momentum_building else 3.0
    elif abs(hist) < abs(line) * 0.05:   # near crossover zone
        pts = 2.0; label = "near crossover"
    else:
        label = f"bearish {'& accelerating' if not momentum_building else 'but easing'}"
        pts   = 0.0 if not momentum_building else 1.0

    return {
        "score": pts, "max": 4,
        "detail": {"macd": {
            "value": f"MACD {round(line,2)} | Signal {round(sig,2)} | Hist {round(hist,3)} → {label}",
            "pts": pts, "max": 4,
        }},
        "_hist": hist,
    }


def _score_bollinger(close: pd.Series) -> dict:
    if len(close) < 20:
        return {"score": 2.0, "max": 4, "detail": {}}

    price = float(close.iloc[-1])
    upper, mid, lower = _bollinger(close)
    band_width = upper - lower
    if band_width <= 0:
        return {"score": 2.0, "max": 4, "detail": {}}

    pct = (price - lower) / band_width   # 0 = at lower, 1 = at upper

    if pct <= 0.05:    pts = 4.0; label = f"at/below lower band (${round(lower,2)})"
    elif pct <= 0.25:  pts = 3.0; label = f"near lower band"
    elif pct <= 0.65:  pts = 2.0; label = f"middle zone"
    elif pct <= 0.90:  pts = 1.0; label = f"near upper band"
    else:              pts = 0.0; label = f"at/above upper band (${round(upper,2)})"

    return {
        "score": pts, "max": 4,
        "detail": {"bollinger": {
            "value": f"price ${round(price,2)} | upper ${round(upper,2)} lower ${round(lower,2)} → {label}",
            "pts": pts, "max": 4,
        }},
    }


def _score_moving_averages(close: pd.Series) -> dict:
    score = 0.0
    detail = {}
    price = float(close.iloc[-1])

    sma50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    # Golden / Death Cross (SMA50 vs SMA200) — 2 pts
    if sma50 is not None and sma200 is not None:
        cross_pts = 2.0 if sma50 > sma200 else 0.0
        cross_lbl = "golden cross (SMA50 > SMA200)" if sma50 > sma200 else "death cross (SMA50 < SMA200)"
        score += cross_pts
        detail["sma_cross"] = {"value": f"SMA50={round(sma50,2)} SMA200={round(sma200,2)} → {cross_lbl}", "pts": cross_pts, "max": 2}

    # Price vs SMA50 — 1.5 pts
    if sma50 is not None:
        pct50 = (price - sma50) / sma50 * 100
        pts50 = 1.5 if pct50 >= 0 else (1.0 if pct50 >= -5 else 0.0)
        score += pts50
        detail["price_vs_sma50"] = {
            "value": f"{round(pct50,1)}% {'above' if pct50 >= 0 else 'below'} SMA50 (${round(sma50,2)})",
            "pts": pts50, "max": 1.5,
        }

    # Price vs SMA200 — 1.5 pts
    if sma200 is not None:
        pct200 = (price - sma200) / sma200 * 100
        pts200 = 1.5 if pct200 >= 0 else (0.75 if pct200 >= -10 else 0.0)
        score += pts200
        detail["price_vs_sma200"] = {
            "value": f"{round(pct200,1)}% {'above' if pct200 >= 0 else 'below'} SMA200 (${round(sma200,2)})",
            "pts": pts200, "max": 1.5,
        }

    max_pts = (2.0 if (sma50 and sma200) else 0) + (1.5 if sma50 else 0) + (1.5 if sma200 else 0)
    return {"score": round(min(score, 5), 2), "max": 5, "detail": detail, "_sma50": sma50, "_sma200": sma200}


def _score_volume(prices_df: pd.DataFrame) -> dict:
    if "Volume" not in prices_df.columns or len(prices_df) < 20:
        return {"score": 1.5, "max": 3, "detail": {"volume_trend": {"value": "insufficient data", "pts": 1.5, "max": 3}}}

    vol = prices_df["Volume"]
    recent_avg  = float(vol.iloc[-5:].mean())    # last 5 trading days
    baseline    = float(vol.iloc[-20:].mean())   # last 20 trading days

    ratio = recent_avg / baseline if baseline > 0 else 1.0

    if ratio >= 1.5:    pts = 3.0; label = f"surging ({round(ratio,2)}x 20-day avg)"
    elif ratio >= 1.1:  pts = 2.0; label = f"above avg ({round(ratio,2)}x)"
    elif ratio >= 0.9:  pts = 1.5; label = f"normal ({round(ratio,2)}x)"
    elif ratio >= 0.7:  pts = 1.0; label = f"below avg ({round(ratio,2)}x)"
    else:               pts = 0.0; label = f"drying up ({round(ratio,2)}x)"

    return {"score": pts, "max": 3, "detail": {"volume_trend": {"value": label, "pts": pts, "max": 3}}}


# Agent class

class TechnicalsAgent(BaseAgent):
    """Technical Analysis Agent — pure quantitative, no LLM."""

    def __init__(self):
        super().__init__(agent_id="technicals", agent_name="Technical Analyst")

    def analyze(self, data: dict, ticker: str) -> AgentSignal:
        prices_df = data.get("prices")

        if prices_df is None or prices_df.empty or len(prices_df) < 30:
            return AgentSignal(
                agent_id="technicals", agent_name="Technical Analyst",
                ticker=ticker, signal=SIGNAL_NEUTRAL, confidence=0.1,
                scores={"total": 10.0, "total_max": 20.0},
                reasoning="Insufficient price history for technical analysis.",
                key_risks=["No price data available"],
                target_action=ACTION_HOLD,
            )

        close = prices_df["Close"].dropna()
        price = float(close.iloc[-1])

        # Score all five groups
        rsi_r  = _score_rsi(close)
        macd_r = _score_macd(close)
        bb_r   = _score_bollinger(close)
        ma_r   = _score_moving_averages(close)
        vol_r  = _score_volume(prices_df)

        total     = rsi_r["score"] + macd_r["score"] + bb_r["score"] + ma_r["score"] + vol_r["score"]
        total_max = 20.0
        norm      = total / total_max

        signal = _norm_to_signal(norm)
        # Confidence: distance from 0.5 neutral midpoint, scaled to [0.2, 0.95]
        confidence = round(min(max(abs(norm - 0.5) * 2.0, 0.20), 0.95), 3)

        # Build reasoning string (no LLM)
        lines = [f"Technical analysis for {ticker} @ ${round(price, 2)}:"]
        for grp in [rsi_r, macd_r, bb_r, ma_r, vol_r]:
            for k, v in grp.get("detail", {}).items():
                lines.append(f"• {k}: {v['value']}")
        reasoning = "\n".join(lines)

        # Key risks
        risks = []
        rsi_val = rsi_r.get("_rsi")
        sma50   = ma_r.get("_sma50")
        sma200  = ma_r.get("_sma200")
        hist    = macd_r.get("_hist", 0)

        if rsi_val and rsi_val > 70:
            risks.append(f"RSI overbought at {round(rsi_val,1)}")
        if rsi_val and rsi_val < 30:
            risks.append(f"RSI oversold ({round(rsi_val,1)}) — potential further downside")
        if sma50 and sma200 and sma50 < sma200:
            risks.append("Death cross active: SMA50 < SMA200")
        if hist and hist < 0:
            risks.append("MACD bearish crossover")
        if not risks:
            risks = ["Technical signals may reverse quickly", "Low volume may reduce signal reliability"]

        self.logger.info(
            f"{ticker} → {signal.upper()} (conf={confidence:.0%}) | "
            f"total={round(total,1)}/{total_max} | "
            f"RSI={round(rsi_val,1) if rsi_val else 'N/A'} "
            f"MACD_hist={round(hist,3) if hist else 'N/A'} "
            f"SMA50={'✓' if (sma50 and price > sma50) else '✗'} "
            f"SMA200={'✓' if (sma200 and price > sma200) else '✗'}"
        )

        return AgentSignal(
            agent_id      = "technicals",
            agent_name    = "Technical Analyst",
            ticker        = ticker,
            signal        = signal,
            confidence    = confidence,
            scores        = {
                "total":           round(total, 2),
                "total_max":       total_max,
                "rsi":             {"score": rsi_r["score"],  "max": rsi_r["max"],  "detail": rsi_r["detail"]},
                "macd":            {"score": macd_r["score"], "max": macd_r["max"], "detail": macd_r["detail"]},
                "bollinger":       {"score": bb_r["score"],   "max": bb_r["max"],   "detail": bb_r["detail"]},
                "moving_averages": {"score": ma_r["score"],   "max": ma_r["max"],   "detail": ma_r["detail"]},
                "volume":          {"score": vol_r["score"],  "max": vol_r["max"],  "detail": vol_r["detail"]},
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
