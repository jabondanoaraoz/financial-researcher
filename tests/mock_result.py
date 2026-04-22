"""
Shared mock result for tests — no API calls required.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pandas as pd
from agents.base_agent import AgentSignal


def _make_financials():
    dates = [
        pd.Timestamp("2024-09-28"),
        pd.Timestamp("2023-09-30"),
        pd.Timestamp("2022-09-24"),
        pd.Timestamp("2021-09-25"),
        pd.Timestamp("2020-09-26"),
    ]
    income = pd.DataFrame(
        {
            "Total Revenue":    [391.0e9, 383.3e9, 394.3e9, 365.8e9, 274.5e9],
            "Gross Profit":     [180.7e9, 169.1e9, 170.8e9, 152.8e9, 104.9e9],
            "EBITDA":           [134.7e9, 125.8e9, 130.5e9, 111.4e9,  77.3e9],
            "Operating Income": [123.2e9, 114.3e9, 119.4e9, 109.0e9,  66.3e9],
            "Net Income":       [ 93.7e9,  97.0e9,  99.8e9,  94.7e9,  57.4e9],
            "Interest Expense": [ -3.9e9,  -3.9e9,  -2.8e9,  -2.6e9,  -2.9e9],
        },
        index=dates,
    ).T
    cash_flow = pd.DataFrame(
        {
            "Operating Cash Flow":           [122.1e9, 113.0e9, 122.2e9, 104.0e9,  80.7e9],
            "Capital Expenditure":           [ -9.4e9, -10.9e9, -10.7e9, -11.1e9,  -7.3e9],
            "Free Cash Flow":                [112.7e9, 102.1e9, 111.4e9,  92.9e9,  73.4e9],
            "Depreciation And Amortization": [ 11.4e9,  11.5e9,  11.1e9,   9.6e9,  11.3e9],
        },
        index=dates,
    ).T
    balance_sheet = pd.DataFrame(
        {
            "Cash Cash Equivalents And Short Term Investments": [65.2e9,  61.6e9,  48.3e9,  62.6e9,  38.0e9],
            "Total Debt":                                       [101.3e9, 109.6e9, 120.1e9, 124.7e9, 112.4e9],
            "Stockholders Equity":                              [ 56.9e9,  62.1e9,  50.7e9,  63.1e9,  65.3e9],
            "Current Assets":                                   [152.9e9, 143.7e9, 135.4e9, 134.8e9, 143.7e9],
            "Current Liabilities":                              [176.4e9, 145.3e9, 153.9e9, 125.5e9, 105.4e9],
        },
        index=dates,
    ).T
    return {
        "income_statement": income,
        "cash_flow":        cash_flow,
        "balance_sheet":    balance_sheet,
    }


def _make_peers():
    return {
        "MSFT": {"current_price": 415.50, "market_cap": 3.09e12, "pe_ratio": 34.5,
                 "forward_pe": 29.2, "pb_ratio": 12.1, "ps_ratio": 12.3,
                 "ev_ebitda": 25.1, "price_to_fcf": 41.2, "beta": 0.90, "dividend_yield": 0.72},
        "GOOGL": {"current_price": 174.10, "market_cap": 2.15e12, "pe_ratio": 22.5,
                  "forward_pe": 18.8, "pb_ratio": 6.5, "ps_ratio": 6.1,
                  "ev_ebitda": 18.7, "price_to_fcf": 28.5, "beta": 1.05, "dividend_yield": 0.52},
        "META": {"current_price": 512.30, "market_cap": 1.31e12, "pe_ratio": 26.1,
                 "forward_pe": 22.4, "pb_ratio": 8.2, "ps_ratio": 10.5,
                 "ev_ebitda": 19.3, "price_to_fcf": 24.8, "beta": 1.22, "dividend_yield": 0.40},
        "AMZN": {"current_price": 201.40, "market_cap": 2.09e12, "pe_ratio": 43.2,
                 "forward_pe": 32.1, "pb_ratio": 9.3, "ps_ratio": 3.4,
                 "ev_ebitda": 20.8, "price_to_fcf": 52.1, "beta": 1.35, "dividend_yield": None},
    }


def _sig(aid, name, signal, conf, scores, reasoning, risks, action, pt=None):
    return AgentSignal(
        agent_id=aid, agent_name=name, ticker="AAPL",
        signal=signal, confidence=conf, scores=scores,
        reasoning=reasoning, key_risks=risks,
        target_action=action, price_target=pt,
    )


def make_mock_result():
    agent_signals = {
        "fundamentals": _sig(
            "fundamentals", "Fundamentals Analyst", "bullish", 0.72,
            {"total": 0.72, "total_max": 1.0,
             "valuation":     {"signal": "bearish",  "score": 0.42},
             "profitability": {"signal": "bullish",  "score": 0.88},
             "growth":        {"signal": "neutral",  "score": 0.61},
             "health":        {"signal": "bullish",  "score": 0.74}},
            "Strong revenue growth, healthy gross margins above 40%, improving ROE trend. "
            "Balance sheet remains solid with manageable debt levels.",
            ["Margin compression risk from increased R&D", "Revenue concentration in iPhone"],
            "buy",
        ),
        "ben_graham": _sig(
            "ben_graham", "Ben Graham", "bearish", 0.65,
            {"total": 11.5, "total_max": 30.0},
            "At current prices, AAPL trades well above Graham's conservative P/E and P/B thresholds. "
            "The stock fails the net-net test and the combined PE×PB rule.",
            ["P/E far exceeds Graham threshold", "P/B > 3x", "No margin of safety"],
            "sell",
        ),
        "warren_buffett": _sig(
            "warren_buffett", "Warren Buffett", "bullish", 0.81,
            {"total": 16.0, "total_max": 20.0},
            "Apple possesses an exceptional consumer brand moat and an ecosystem that creates powerful switching costs. "
            "Return on invested capital consistently exceeds cost of capital.",
            ["Platform commoditization risk", "Regulatory pressure on App Store"],
            "buy",
        ),
        "aswath_damodaran": _sig(
            "aswath_damodaran", "Aswath Damodaran", "neutral", 0.58,
            {"total": 11.0, "total_max": 20.0},
            "FCFF-based DCF yields an intrinsic value roughly in line with current market price. "
            "The growth premium embedded in the stock leaves little room for error.",
            ["WACC sensitivity: +100bps reduces FV by ~15%", "Services growth deceleration"],
            "hold", 195.0,
        ),
        "cathie_wood": _sig(
            "cathie_wood", "Cathie Wood", "bullish", 0.74,
            {"total": 14.0, "total_max": 20.0},
            "Apple's positioning in AI hardware and its locked-in services ecosystem provide durable revenue streams. "
            "Spatial computing and AI-native devices represent a multi-year growth vector.",
            ["Revenue growth may not justify high P/S", "Lagging public AI strategy"],
            "buy",
        ),
        "michael_burry": _sig(
            "michael_burry", "Michael Burry", "bearish", 0.69,
            {"total": 7.0, "total_max": 20.0},
            "Short interest in AAPL is low — no contrarian signal. Insider selling has accelerated. "
            "The stock is priced for perfection at a point where macro headwinds are building.",
            ["Elevated short-term debt", "Insider net selling", "Consumer spending slowdown"],
            "sell",
        ),
        "technicals": _sig(
            "technicals", "Technical Analyst", "bullish", 0.66,
            {"total": 13.5, "total_max": 20.0,
             "rsi":             {"score": 1.0,  "max": 4},
             "macd":            {"score": 4.0,  "max": 4},
             "bollinger":       {"score": 2.0,  "max": 4},
             "moving_averages": {"score": 5.0,  "max": 5},
             "volume":          {"score": 1.5,  "max": 3}},
            "Price above both SMA50 and SMA200 — trend is intact. RSI at 58. MACD positive and widening.",
            ["RSI approaching overbought on weekly", "Key support at $185"],
            "buy",
        ),
        "valuation": _sig(
            "valuation", "Valuation Analyst", "neutral", 0.55,
            {"total": 10.0, "total_max": 20.0, "fair_value": 198.50, "upside_pct": 1.4},
            "Three independent methods produce a consensus fair value of $198.50, broadly in line with current prices.",
            ["DCF sensitive to FCF margin assumptions", "Peer multiples may compress in risk-off"],
            "hold", 198.50,
        ),
        "risk_manager": _sig(
            "risk_manager", "Risk Manager", "bullish", 0.71,
            {"total": 14.0, "total_max": 20.0,
             "risk_metrics": {
                 "annualized_volatility": 0.223, "annualized_return": 0.187,
                 "beta": 1.21, "max_drawdown": -0.194,
                 "sharpe_proxy": 1.42, "kelly_fraction": 0.102,
                 "max_position_size_pct": 10.2,
             }},
            "AAPL presents a moderate risk profile. Annualized volatility of 22.3% is within acceptable bounds.",
            ["Beta of 1.21 amplifies drawdowns", "Historical vol may understate tail risk"],
            "buy",
        ),
    }

    pm = _sig(
        "portfolio_manager", "Portfolio Manager", "bullish", 0.73,
        {"total": 14.4, "total_max": 20.0,
         "consensus": {"bullish": 5, "neutral": 2, "bearish": 2, "conviction": "medium",
                       "target_price_low": 190.0, "target_price_high": 225.0}},
        "Five of nine analysts are bullish on AAPL. The two bearish views are valuation-driven. "
        "Risk metrics are favorable. A BUY is warranted with a 6-12 month horizon.",
        ["Valuation offers limited margin of safety", "Macro slowdown risk", "App Store regulatory risk"],
        "buy", 207.50,
    )

    return {
        "ticker": "AAPL",
        "company_data": {
            "info": {
                "name": "Apple Inc.", "longName": "Apple Inc.",
                "sector": "Technology", "industry": "Consumer Electronics",
                "exchange": "NASDAQ", "country": "United States",
                "website": "https://www.apple.com", "employees": 161000,
                "market_cap": 3.02e12,
                "description": (
                    "Apple Inc. designs, manufactures, and markets smartphones, personal computers, "
                    "tablets, wearables, and accessories worldwide."
                ),
            },
            "key_metrics": {
                "current_price": 196.45, "pe_ratio": 29.9, "forward_pe": 27.1,
                "pb_ratio": 47.5, "ps_ratio": 7.9, "peg_ratio": 2.8,
                "ev_ebitda": 22.5, "beta": 1.21,
                "52w_high": 237.23, "52w_low": 164.08,
                "dividend_yield": 0.53, "shares_outstanding": 15.1e9,
                "enterprise_value": 3.3e12, "short_percent_of_float": 0.71,
                "market_cap": 3.02e12,
            },
            "macro": {},
        },
        "financials":         _make_financials(),
        "peers_data":         _make_peers(),
        "risk_free_rate":     0.043,
        "agent_signals":      agent_signals,
        "portfolio_decision": pm,
        "risk_metrics": {
            "annualized_volatility": 0.223, "annualized_return": 0.187,
            "beta": 1.21, "max_drawdown": -0.194,
            "sharpe_proxy": 1.42, "kelly_fraction": 0.102,
            "max_position_size_pct": 10.2,
        },
        "consensus": {"bullish": 5, "neutral": 2, "bearish": 2, "avg_score_20": 13.1},
    }
