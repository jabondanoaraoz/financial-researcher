"""
Base Agent
==========
Abstract base class and shared data structures for all investment agents.

Every agent in the Financial Researcher implements this interface,
ensuring a consistent output format that the orchestrator can aggregate.

Author: Financial Researcher Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Valid signal values
SIGNAL_BULLISH  = "bullish"
SIGNAL_NEUTRAL  = "neutral"
SIGNAL_BEARISH  = "bearish"

# Valid action values
ACTION_BUY   = "buy"
ACTION_HOLD  = "hold"
ACTION_SELL  = "sell"
ACTION_SHORT = "short"
ACTION_COVER = "cover"


@dataclass
class AgentSignal:
    """
    Standardized output produced by every agent after analyzing a ticker.

    Fields
    ------
    agent_id : str
        Machine-readable identifier (e.g. "fundamentals", "warren_buffett")
    agent_name : str
        Human-readable name (e.g. "Fundamentals Analyst", "Warren Buffett")
    signal : str
        Overall directional view — "bullish" | "neutral" | "bearish"
    confidence : float
        How strongly the agent holds its view — 0.0 (no view) to 1.0 (conviction)
    scores : dict
        Agent-specific sub-scores.  Keys and semantics vary by agent:
          - Fundamentals: {"valuation": float, "profitability": float, ...}
          - Buffett:       {"moat_score": float, "management_quality": float, ...}
          Each score is in [0.0, 1.0], where 1.0 = best possible.
    reasoning : str
        Human-readable explanation of the signal (1-3 paragraphs).
        Quant agents build this from rule logic;
        LLM agents generate it via Groq/Claude.
    key_risks : list[str]
        Top 3-5 risks the agent identified.
    target_action : str
        Concrete recommended action — "buy" | "hold" | "sell" | "short" | "cover"
    price_target : float | None
        Optional price target in USD.  Not all agents provide one.
    ticker : str
        Ticker symbol this signal is for.
    """
    agent_id:      str
    agent_name:    str
    signal:        str
    confidence:    float
    scores:        dict
    reasoning:     str
    key_risks:     list
    target_action: str
    ticker:        str = ""
    price_target:  Optional[float] = None

    def __post_init__(self):
        # Validate signal and action values
        valid_signals = {SIGNAL_BULLISH, SIGNAL_NEUTRAL, SIGNAL_BEARISH}
        valid_actions = {ACTION_BUY, ACTION_HOLD, ACTION_SELL, ACTION_SHORT, ACTION_COVER}

        if self.signal not in valid_signals:
            raise ValueError(f"Invalid signal '{self.signal}'. Must be one of {valid_signals}")
        if self.target_action not in valid_actions:
            raise ValueError(f"Invalid action '{self.target_action}'. Must be one of {valid_actions}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> dict:
        """Serialize to plain dict (useful for JSON export / Excel output)."""
        return {
            "agent_id":      self.agent_id,
            "agent_name":    self.agent_name,
            "ticker":        self.ticker,
            "signal":        self.signal,
            "confidence":    round(self.confidence, 4),
            "target_action": self.target_action,
            "price_target":  self.price_target,
            "scores":        self.scores,
            "reasoning":     self.reasoning,
            "key_risks":     self.key_risks,
        }

    def __str__(self) -> str:
        lines = [
            f"[{self.agent_name}] {self.ticker} → {self.signal.upper()} ({self.target_action.upper()})",
            f"  Confidence: {self.confidence:.0%}",
        ]
        if self.price_target:
            lines.append(f"  Price Target: ${self.price_target:.2f}")
        for key, val in self.scores.items():
            lines.append(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")
        lines.append(f"  Reasoning: {self.reasoning[:200]}...")
        if self.key_risks:
            lines.append(f"  Key Risks: {'; '.join(self.key_risks[:3])}")
        return "\n".join(lines)


class BaseAgent(ABC):
    """
    Abstract base class for all investment analysis agents.

    Subclasses must implement `analyze()`.  The orchestrator calls
    `safe_analyze()` which wraps the implementation with error handling,
    so an individual agent crash never kills the entire pipeline.

    Parameters
    ----------
    agent_id : str
        Machine-readable identifier.
    agent_name : str
        Human-readable display name.
    """

    def __init__(self, agent_id: str, agent_name: str):
        self.agent_id   = agent_id
        self.agent_name = agent_name
        self.logger     = logging.getLogger(f"agents.{agent_id}")

    @abstractmethod
    def analyze(self, data: dict, ticker: str) -> AgentSignal:
        """
        Core analysis logic.  Subclasses implement this.

        Parameters
        ----------
        data : dict
            Consolidated data bundle from the data layer.  Expected keys:
                company_info  → dict  (from yfinance_adapter)
                key_metrics   → dict  (from yfinance_adapter)
                financials    → dict  (from yfinance_adapter)
                av_overview   → dict | None  (from alpha_vantage)
                xbrl_facts    → dict | None  (from sec_edgar)
                risk_free_rate → float  (from fred)
                macro          → dict   (from fred)
        ticker : str
            Ticker symbol being analyzed.

        Returns
        -------
        AgentSignal
        """
        raise NotImplementedError

    def safe_analyze(self, data: dict, ticker: str) -> Optional[AgentSignal]:
        """
        Fault-tolerant wrapper around `analyze()`.

        Returns None instead of raising if the agent encounters an error,
        so one broken agent never stops the other nine.
        """
        try:
            signal = self.analyze(data, ticker)
            signal.ticker = ticker
            return signal
        except Exception as exc:
            self.logger.error(
                f"{self.agent_name} failed on {ticker}: {type(exc).__name__}: {exc}"
            )
            return None
