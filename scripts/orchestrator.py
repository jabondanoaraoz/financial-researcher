"""
Orchestrator - runs the full 10-agent pipeline and returns a consolidated result dict.
Entry point: run_analysis(ticker, peers)
"""

import logging
from typing import Optional

# Data layer
from data.yfinance_adapter import (
    get_company_info, get_key_metrics, get_financials,
    get_prices, get_insider_transactions,
)
from data.alpha_vantage import get_company_overview
from data.sec_edgar      import get_xbrl_facts
from data.fred           import get_risk_free_rate, get_macro_context

# Agents
from agents.fundamentals        import FundamentalsAgent
from agents.ben_graham          import BenGrahamAgent
from agents.warren_buffett      import WarrenBuffettAgent
from agents.aswath_damodaran    import AswathDamodaranAgent
from agents.cathie_wood         import CathieWoodAgent
from agents.michael_burry       import MichaelBurryAgent
from agents.technicals          import TechnicalsAgent
from agents.valuation           import ValuationAgent
from agents.risk_manager        import RiskManagerAgent
from agents.portfolio_manager   import PortfolioManagerAgent

logger = logging.getLogger(__name__)

# Default peer universe if none provided (large-cap tech)
DEFAULT_PEERS = {
    "AAPL": None, "MSFT": None, "META": None, "AMZN": None,
}


def _fetch_peers_data(peers: dict) -> dict:
    """Fetch key_metrics for each peer ticker. Skips failures silently."""
    peers_data = {}
    for pt in peers:
        try:
            km = get_key_metrics(pt)
            if km:
                peers_data[pt] = km
        except Exception as e:
            logger.warning(f"Could not fetch peer data for {pt}: {e}")
    return peers_data


def run_analysis(ticker: str, peers: Optional[dict] = None) -> dict:
    """
    Run the full 10-agent pipeline on a single ticker.

    peers: {ticker: None} mapping. Defaults to DEFAULT_PEERS if None.
    Returns a dict with keys: ticker, company_data, financials, peers_data,
    agent_signals, portfolio_decision, risk_metrics, consensus.
    """
    ticker = ticker.upper().strip()
    logger.info(f"{'='*60}\n  Starting analysis for {ticker}\n{'='*60}")

    # 1. Fetch all data
    logger.info(f"[{ticker}] Fetching data layer...")

    company_info   = get_company_info(ticker)
    key_metrics    = get_key_metrics(ticker)
    financials     = get_financials(ticker, years=5)
    av_overview    = get_company_overview(ticker)
    xbrl_facts     = get_xbrl_facts(ticker)
    insider_txns   = get_insider_transactions(ticker)
    risk_free      = get_risk_free_rate()
    macro          = get_macro_context()

    # Price history - 2 years for SMA200 + beta calculation
    prices_df   = get_prices(ticker, period="2y", interval="1d")
    spy_prices  = get_prices("SPY",   period="2y", interval="1d")

    # Peer data for Valuation agent
    peer_universe = peers if peers is not None else DEFAULT_PEERS
    peers_data    = _fetch_peers_data(peer_universe)

    # Base data bundle (shared across all agents)
    base_data = {
        "company_info":         company_info,
        "key_metrics":          key_metrics,
        "financials":           financials,
        "av_overview":          av_overview,
        "xbrl_facts":           xbrl_facts,
        "insider_transactions": insider_txns,
        "risk_free_rate":       risk_free,
        "macro":                macro,
        "prices":               prices_df,
        "spy_prices":           spy_prices,
        "peers_data":           peers_data,
    }

    # 2. Instantiate agents
    agents_1_to_9 = [
        FundamentalsAgent(),
        BenGrahamAgent(),
        WarrenBuffettAgent(),
        AswathDamodaranAgent(),
        CathieWoodAgent(),
        MichaelBurryAgent(),
        TechnicalsAgent(),
        ValuationAgent(),
        RiskManagerAgent(),
    ]

    # 3. Run agents 1-9 sequentially
    agent_signals  = {}   # {agent_id: AgentSignal}
    risk_metrics   = {}

    for agent in agents_1_to_9:
        logger.info(f"[{ticker}] Running {agent.agent_name}...")
        signal = agent.safe_analyze(base_data, ticker)
        if signal:
            agent_signals[agent.agent_id] = signal
            # Extract risk_metrics from Risk Manager
            if agent.agent_id == "risk_manager":
                risk_metrics = signal.scores.get("risk_metrics", {})
        else:
            logger.warning(f"[{ticker}] {agent.agent_name} returned None - skipping.")

    # 4. Run Portfolio Manager (agent #10)
    logger.info(f"[{ticker}] Running Portfolio Manager (synthesizing {len(agent_signals)} signals)...")

    pm_data = {
        **base_data,
        "all_signals":  list(agent_signals.values()),
        "risk_metrics": risk_metrics,
    }
    pm_agent   = PortfolioManagerAgent()
    pm_signal  = pm_agent.safe_analyze(pm_data, ticker)

    # 5. Compute consensus stats
    signals_list = list(agent_signals.values())
    n_bull = sum(1 for s in signals_list if s.signal == "bullish")
    n_neut = sum(1 for s in signals_list if s.signal == "neutral")
    n_bear = sum(1 for s in signals_list if s.signal == "bearish")

    scores_20 = [
        s.scores.get("total", 0) / max(s.scores.get("total_max", 20), 1) * 20
        for s in signals_list
        if isinstance(s.scores.get("total"), (int, float))
    ]
    avg_score = round(sum(scores_20) / len(scores_20), 2) if scores_20 else 0.0

    consensus = {
        "bullish":   n_bull,
        "neutral":   n_neut,
        "bearish":   n_bear,
        "avg_score_20": avg_score,
    }

    logger.info(
        f"[{ticker}] Pipeline complete. "
        f"Consensus: ▲{n_bull} ●{n_neut} ▼{n_bear} | "
        f"Avg score={avg_score}/20 | "
        f"Final: {pm_signal.signal.upper() if pm_signal else 'N/A'} "
        f"({pm_signal.target_action.upper() if pm_signal else 'N/A'})"
    )

    # 6. Return consolidated result
    return {
        "ticker":             ticker,
        "company_data":       {
            "info":           company_info,
            "key_metrics":    key_metrics,
            "macro":          macro,
        },
        "financials":         financials,
        "peers_data":         peers_data,
        "risk_free_rate":     risk_free,
        "agent_signals":      agent_signals,
        "portfolio_decision": pm_signal,
        "risk_metrics":       risk_metrics,
        "consensus":          consensus,
    }
