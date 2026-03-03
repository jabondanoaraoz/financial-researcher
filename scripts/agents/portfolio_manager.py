"""
Portfolio Manager Agent
"Synthesize 9 analysts. Make the call." — senior portfolio manager.

Receives ALL AgentSignals from the other 9 agents plus risk metrics,
weighs the evidence, and produces a single final investment decision.

The LLM plays the role of a senior portfolio manager who reads the
analyst reports and decides: what to do, with how much conviction,
and at what price range.

Output (via LLM):
    signal          — bullish | neutral | bearish
    confidence      — 0.0–1.0
    target_action   — buy | hold | sell
    target_price_low  / target_price_high — price range estimate
    reasoning       — 3–4 sentence executive summary
    key_risks       — top 3 risks from all analyst reports

Author: Joaquin Abondano w/ Claude Code
"""

import json
import logging
from typing import Optional, List

from agents.base_agent import BaseAgent, AgentSignal
from agents.base_agent import SIGNAL_BULLISH, SIGNAL_NEUTRAL, SIGNAL_BEARISH
from agents.base_agent import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from agents.llm_client import LLMClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior portfolio manager at a multi-strategy hedge fund.
You have received research reports from 9 specialized analysts covering the same stock.
Your job is to synthesize their views into a single actionable investment decision,
weighing the quality of their reasoning, the consensus across different frameworks,
and the risk profile provided by the Risk Manager.

You do not simply vote-count. You weigh: which analysts have the most relevant lens
for this type of company? Where do they agree? Where do they disagree, and why?
What does the risk profile tell you about position sizing?

Produce your final decision as JSON only — no prose outside the JSON:

{
  "signal": "bullish" | "neutral" | "bearish",
  "confidence": <float 0.0–1.0>,
  "target_action": "buy" | "hold" | "sell",
  "target_price_low": <float — conservative fair value estimate>,
  "target_price_high": <float — optimistic fair value estimate>,
  "reasoning": "<3–4 sentences: consensus view, key agreement/disagreement, final rationale>",
  "key_risks": ["<top risk 1>", "<top risk 2>", "<top risk 3>"],
  "conviction": "high" | "medium" | "low"
}"""


class PortfolioManagerAgent(BaseAgent):
    """Portfolio Manager — synthesizes all 9 agent signals into final decision."""

    def __init__(self, llm: Optional[LLMClient] = None):
        super().__init__(agent_id="portfolio_manager", agent_name="Portfolio Manager")
        self.llm = llm or LLMClient()

    def analyze(self, data: dict, ticker: str) -> AgentSignal:
        all_signals: List[AgentSignal] = data.get("all_signals") or []
        risk_metrics: dict             = data.get("risk_metrics") or {}
        price = (data.get("key_metrics") or {}).get("current_price")

        if not all_signals:
            return AgentSignal(
                agent_id="portfolio_manager", agent_name="Portfolio Manager",
                ticker=ticker, signal=SIGNAL_NEUTRAL, confidence=0.1,
                scores={"total": 10.0, "total_max": 20.0},
                reasoning="No agent signals received.",
                key_risks=["Pipeline error — no agent data"],
                target_action=ACTION_HOLD,
            )

        # Consensus stats
        n_bullish = sum(1 for s in all_signals if s.signal == SIGNAL_BULLISH)
        n_neutral = sum(1 for s in all_signals if s.signal == SIGNAL_NEUTRAL)
        n_bearish = sum(1 for s in all_signals if s.signal == SIGNAL_BEARISH)
        n_total   = len(all_signals)

        # Confidence-weighted consensus
        bull_weight = sum(s.confidence for s in all_signals if s.signal == SIGNAL_BULLISH)
        bear_weight = sum(s.confidence for s in all_signals if s.signal == SIGNAL_BEARISH)
        neut_weight = sum(s.confidence for s in all_signals if s.signal == SIGNAL_NEUTRAL)
        total_weight = bull_weight + bear_weight + neut_weight

        # Collect all key risks for deduplication
        all_risks = []
        for s in all_signals:
            all_risks.extend(s.key_risks or [])

        # Price target from Valuation agent
        val_signal  = next((s for s in all_signals if s.agent_id == "valuation"), None)
        price_target = val_signal.price_target if val_signal else None

        # Build analyst summary for LLM
        analyst_rows = []
        for sig in all_signals:
            sc    = sig.scores or {}
            total = sc.get("total", "?")
            t_max = sc.get("total_max", 20)
            analyst_rows.append(
                f"  {sig.agent_name:<22} {sig.signal.upper():<8} "
                f"(conf={sig.confidence:.0%}, score={total}/{t_max})  "
                f"Action: {sig.target_action.upper()}"
            )

        risk_block = (
            f"  Annualized Volatility: {risk_metrics.get('annualized_volatility', 'N/A'):.1%}\n"
            f"  Beta vs S&P 500:       {risk_metrics.get('beta', 'N/A')}\n"
            f"  Max Drawdown (12m):    {risk_metrics.get('max_drawdown', 0):.1%}\n"
            f"  Sharpe Proxy:          {risk_metrics.get('sharpe_proxy', 'N/A')}\n"
            f"  Kelly Max Position:    {risk_metrics.get('max_position_size_pct', 'N/A')}% of portfolio"
        ) if risk_metrics else "  Risk metrics unavailable."

        company_name = (data.get("company_info", {}).get("name", ticker) if data.get("company_info") else ticker)

        user_prompt = f"""
Ticker: {ticker} ({company_name})
Current Price: ${price}
Valuation Agent Fair Value: ${price_target if price_target else "N/A"}

ANALYST SIGNALS ({n_total} analysts):
  Bullish: {n_bullish}  |  Neutral: {n_neutral}  |  Bearish: {n_bearish}
  Weighted bull score: {bull_weight:.2f}  |  Weighted bear score: {bear_weight:.2f}

{chr(10).join(analyst_rows)}

RISK METRICS (Risk Manager):
{risk_block}

TOP RISKS FLAGGED BY ANALYSTS:
{chr(10).join(f'  • {r}' for r in all_risks[:9])}

Synthesize all of the above and provide your final portfolio decision as JSON.
"""

        llm_result = self.llm.generate_json(SYSTEM_PROMPT, user_prompt)

        # Parse LLM output
        signal        = llm_result.get("signal",        _consensus_signal(n_bullish, n_neutral, n_bearish))
        confidence    = float(llm_result.get("confidence", total_weight / max(n_total, 1) / 1.0))
        reasoning     = llm_result.get("reasoning",    "LLM response unavailable.")
        key_risks     = llm_result.get("key_risks",    [])
        target_action = llm_result.get("target_action", _signal_to_action(signal))
        conviction    = llm_result.get("conviction",   "medium")
        price_low     = llm_result.get("target_price_low")
        price_high    = llm_result.get("target_price_high")

        if signal not in ("bullish", "neutral", "bearish"):
            signal = _consensus_signal(n_bullish, n_neutral, n_bearish)
        if target_action not in ("buy", "hold", "sell", "short", "cover"):
            target_action = _signal_to_action(signal)
        confidence = round(min(max(confidence, 0), 1), 3)

        price_target_mid = (
            round((float(price_low) + float(price_high)) / 2, 2)
            if (price_low and price_high) else price_target
        )

        self.logger.info(
            f"{ticker} → FINAL: {signal.upper()} {target_action.upper()} "
            f"(conf={confidence:.0%}, conviction={conviction}) | "
            f"▲{n_bullish} ●{n_neutral} ▼{n_bearish} | "
            f"target=${price_target_mid}"
        )

        return AgentSignal(
            agent_id      = self.agent_id,
            agent_name    = self.agent_name,
            ticker        = ticker,
            signal        = signal,
            confidence    = confidence,
            price_target  = price_target_mid,
            scores        = {
                "total":       round((n_bullish / n_total) * 20 if signal == SIGNAL_BULLISH else
                                     (n_neutral  / n_total) * 10 if signal == SIGNAL_NEUTRAL else
                                     (n_bearish  / n_total) * 20, 2),
                "total_max":   20.0,
                "consensus":   {
                    "bullish": n_bullish, "neutral": n_neutral, "bearish": n_bearish,
                    "conviction": conviction,
                    "target_price_low":  price_low,
                    "target_price_high": price_high,
                },
            },
            reasoning     = reasoning,
            key_risks     = key_risks,
            target_action = target_action,
        )


# Utilities

def _consensus_signal(n_bull: int, n_neut: int, n_bear: int) -> str:
    if n_bull > n_bear and n_bull > n_neut: return SIGNAL_BULLISH
    if n_bear > n_bull and n_bear > n_neut: return SIGNAL_BEARISH
    return SIGNAL_NEUTRAL


def _signal_to_action(signal: str) -> str:
    return {SIGNAL_BULLISH: ACTION_BUY, SIGNAL_NEUTRAL: ACTION_HOLD, SIGNAL_BEARISH: ACTION_SELL}.get(signal, ACTION_HOLD)
