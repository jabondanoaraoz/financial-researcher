"""
Base agent class and AgentSignal dataclass shared by all investment agents.
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
    """Standardized output for all investment agents."""
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
    """Abstract base class for all investment agents. Subclasses implement analyze()."""

    def __init__(self, agent_id: str, agent_name: str):
        self.agent_id   = agent_id
        self.agent_name = agent_name
        self.logger     = logging.getLogger(f"agents.{agent_id}")

    @abstractmethod
    def analyze(self, data: dict, ticker: str) -> AgentSignal:
        raise NotImplementedError

    def safe_analyze(self, data: dict, ticker: str) -> Optional[AgentSignal]:
        """Fault-tolerant wrapper - returns None on error so one agent failure never kills the pipeline."""
        try:
            signal = self.analyze(data, ticker)
            signal.ticker = ticker
            return signal
        except Exception as exc:
            self.logger.error(
                f"{self.agent_name} failed on {ticker}: {type(exc).__name__}: {exc}"
            )
            return None
