"""
Test script — generates an Excel report from a mock result dict.
Run from the project root: python test_excel.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from agents.base_agent import AgentSignal
from excel.workbook import generate_report


def mock_result():
    def sig(aid, name, signal, conf, scores, reasoning, risks, action, pt=None):
        return AgentSignal(
            agent_id=aid, agent_name=name, ticker="AAPL",
            signal=signal, confidence=conf, scores=scores,
            reasoning=reasoning, key_risks=risks,
            target_action=action, price_target=pt,
        )

    agent_signals = {
        "fundamentals": sig(
            "fundamentals", "Fundamentals Analyst", "bullish", 0.72,
            {"total": 0.72, "total_max": 1.0},
            "Strong revenue growth, healthy gross margins above 40%, improving ROE trend. Balance sheet remains solid with manageable debt levels.",
            ["Margin compression risk from increased R&D", "Revenue concentration in iPhone"],
            "buy",
        ),
        "ben_graham": sig(
            "ben_graham", "Ben Graham", "bearish", 0.65,
            {"total": 11.5, "total_max": 30.0},
            "At current prices, AAPL trades well above Graham's conservative P/E and P/B thresholds. "
            "The stock fails the net-net test and the combined PE×PB rule. "
            "While earnings quality is high, the valuation premium is inconsistent with a margin-of-safety approach.",
            ["P/E far exceeds Graham's 1/(2×RFR) threshold", "P/B > 3x — premium to book not defensible on liquidation basis", "No margin of safety at current price"],
            "sell",
        ),
        "warren_buffett": sig(
            "warren_buffett", "Warren Buffett", "bullish", 0.81,
            {"total": 16.0, "total_max": 20.0},
            "Apple possesses an exceptional consumer brand moat and an ecosystem that creates powerful switching costs. "
            "Return on invested capital consistently exceeds cost of capital by a wide margin. "
            "Management capital allocation has been excellent — aggressive buybacks at rational prices.",
            ["Platform commoditization risk from Android ecosystem", "Regulatory pressure on App Store economics"],
            "buy",
        ),
        "aswath_damodaran": sig(
            "aswath_damodaran", "Aswath Damodaran", "neutral", 0.58,
            {"total": 11.0, "total_max": 20.0},
            "FCFF-based DCF using a WACC of 9.2% and terminal growth of 3.0% yields an intrinsic value "
            "roughly in line with current market price. The company creates value (ROIC > WACC) but the "
            "growth premium embedded in the stock leaves little room for error.",
            ["WACC sensitivity: +100bps reduces fair value by ~15%", "Services growth deceleration would impair terminal value"],
            "hold", 195.0,
        ),
        "cathie_wood": sig(
            "cathie_wood", "Cathie Wood", "bullish", 0.74,
            {"total": 14.0, "total_max": 20.0},
            "Apple's positioning in AI hardware (M-series chips) and its locked-in services ecosystem "
            "provide durable revenue streams. Revenue per user is expanding. "
            "The convergence of spatial computing and AI-native devices represents a multi-year growth vector.",
            ["Revenue growth may not justify high P/S vs disruptive peers", "Lagging public AI strategy vs competitors"],
            "buy",
        ),
        "michael_burry": sig(
            "michael_burry", "Michael Burry", "bearish", 0.69,
            {"total": 7.0, "total_max": 20.0},
            "Short interest in AAPL is low — no contrarian signal here. "
            "Insider selling has accelerated. Balance sheet leverage has increased despite buybacks. "
            "The stock is priced for perfection at a point where macro headwinds are building.",
            ["Elevated short-term debt load", "Insider net selling trend last 6 months", "Consumer spending slowdown risk"],
            "sell",
        ),
        "technicals": sig(
            "technicals", "Technical Analyst", "bullish", 0.66,
            {"total": 13.5, "total_max": 20.0},
            "Price above both SMA50 and SMA200 — trend is intact. RSI at 58 — not overbought. "
            "MACD positive and widening. Volume profile supports the current move.",
            ["RSI approaching overbought on weekly timeframe", "Key support at $185 — break would be bearish"],
            "buy",
        ),
        "valuation": sig(
            "valuation", "Valuation Analyst", "neutral", 0.55,
            {
                "total": 10.0, "total_max": 20.0,
                "fair_value": 198.50,
                "upside_pct": 1.4,
                "methods": {
                    "score": 10.0, "max": 20.0,
                    "detail": {
                        "dcf": {
                            "value": "$212.40 (WACC=9.2%, CAGR=8.5%, FCF margin=22.1%)",
                            "pts": 0.50, "max": 0.65,
                        },
                        "peer_multiples": {
                            "value": "$195.80 (median EV/EBITDA 22.5x, 4 peers)",
                            "pts": 0.30, "max": 0.30,
                        },
                        "graham_number": {
                            "value": "$87.30 (EPS=$6.57, BVPS=$4.14)",
                            "pts": 0.20, "max": 0.35,
                        },
                        "consensus_fair_value": {
                            "value": "$198.50 (upside +1.4%)",
                            "pts": 0, "max": 0,
                        },
                    },
                },
            },
            "Three independent methods produce a consensus fair value of $198.50, broadly in line with current prices. "
            "The Graham Number at $87 highlights the stock's premium to asset value, while the DCF at $212 "
            "suggests modest upside on a growth basis. Peer multiples cluster around $196. "
            "The convergence of DCF and peer values provides reasonable confidence in the $195–$215 range.",
            ["DCF is sensitive to FCF margin assumptions — 200bps compression reduces FV by ~12%", "Peer multiples may compress in a risk-off environment"],
            "hold", 198.50,
        ),
        "risk_manager": sig(
            "risk_manager", "Risk Manager", "bullish", 0.71,
            {
                "total": 14.0, "total_max": 20.0,
                "volatility": {"score": 3.0, "max": 4, "detail": {"annualized_vol": {"value": "22.3% — moderate", "pts": 3.0, "max": 4}}},
                "beta":       {"score": 3.0, "max": 4, "detail": {"beta":           {"value": "1.21 — above-market", "pts": 3.0, "max": 4}}},
                "drawdown":   {"score": 3.0, "max": 4, "detail": {"max_drawdown":   {"value": "-19.4% — moderate", "pts": 3.0, "max": 4}}},
                "sharpe":     {"score": 3.0, "max": 4, "detail": {"sharpe_proxy":   {"value": "1.42 — good", "pts": 3.0, "max": 4}}},
                "kelly":      {"score": 2.0, "max": 4, "detail": {"kelly_position": {"value": "10.2% recommended max position", "pts": 2.0, "max": 4}}},
                "risk_metrics": {
                    "annualized_volatility": 0.223,
                    "annualized_return":     0.187,
                    "beta":                  1.21,
                    "max_drawdown":          -0.194,
                    "sharpe_proxy":          1.42,
                    "kelly_fraction":        0.102,
                    "max_position_size_pct": 10.2,
                },
            },
            "AAPL presents a moderate risk profile. Annualized volatility of 22.3% is within acceptable bounds "
            "for a large-cap growth stock. Beta of 1.21 indicates above-market sensitivity. "
            "The Kelly criterion suggests a maximum position size of ~10% of portfolio.",
            ["Beta of 1.21 amplifies market drawdowns", "Historical volatility may understate tail risk"],
            "buy",
        ),
    }

    pm = sig(
        "portfolio_manager", "Portfolio Manager", "bullish", 0.73,
        {
            "total": 14.4, "total_max": 20.0,
            "consensus": {
                "bullish": 5, "neutral": 2, "bearish": 2,
                "conviction": "medium",
                "target_price_low":  190.0,
                "target_price_high": 225.0,
            },
        },
        "Five of nine analysts are bullish on AAPL, with strong conviction from Buffett and Wood on the moat and ecosystem thesis. "
        "The two bearish views (Graham and Burry) are primarily valuation-driven rather than fundamental concerns. "
        "Risk metrics are favorable and support a meaningful position. "
        "Given the neutral valuation signal and the 1.4% implied upside to consensus fair value, a BUY is warranted with a 6-12 month horizon and tight position sizing.",
        [
            "Valuation offers limited margin of safety — stock is priced for execution",
            "Macro slowdown could compress consumer spending on hardware upgrades",
            "Regulatory risk on App Store could impair Services segment growth",
        ],
        "buy", 207.50,
    )

    return {
        "ticker": "AAPL",
        "company_data": {
            "info": {
                "name":        "Apple Inc.",
                "sector":      "Technology",
                "industry":    "Consumer Electronics",
                "exchange":    "NASDAQ",
                "country":     "United States",
                "description": (
                    "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, "
                    "wearables, and accessories worldwide. The company offers iPhone, Mac, iPad, and Apple Watch "
                    "product lines, as well as the iOS, macOS, watchOS, and tvOS operating systems. Apple's "
                    "Services segment includes the App Store, Apple Music, Apple TV+, iCloud, Apple Pay, and "
                    "AppleCare, generating high-margin recurring revenue. The company is headquartered in "
                    "Cupertino, California."
                ),
                "employees":   161000,
                "market_cap":  3.02e12,
            },
            "key_metrics": {
                "current_price":  196.45,
                "pe_ratio":       29.9,
                "forward_pe":     27.1,
                "pb_ratio":       47.5,
                "ps_ratio":       7.9,
                "ev_ebitda":      22.5,
                "beta":           1.21,
                "52w_high":       237.23,
                "52w_low":        164.08,
                "dividend_yield": 0.53,
            },
            "macro": {},
        },
        "agent_signals":      agent_signals,
        "portfolio_decision": pm,
        "risk_metrics": {
            "annualized_volatility": 0.223,
            "annualized_return":     0.187,
            "beta":                  1.21,
            "max_drawdown":          -0.194,
            "sharpe_proxy":          1.42,
            "kelly_fraction":        0.102,
            "max_position_size_pct": 10.2,
        },
        "consensus": {
            "bullish":       5,
            "neutral":       2,
            "bearish":       2,
            "avg_score_20": 13.1,
        },
    }


if __name__ == "__main__":
    result = mock_result()
    path   = generate_report(result)
    print(f"Report generated: {path}")
