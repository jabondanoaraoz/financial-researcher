# Financial Researcher Skill

**Autonomous AI-powered financial analysis and investment research**

## Description

This skill performs comprehensive financial analysis on any public company by:
1. Extracting real-time financial data from multiple sources (yfinance, SEC EDGAR, Alpha Vantage, FRED)
2. Running 10 AI investment agents in parallel to analyze the company from different perspectives
3. Generating an IB-grade Excel workbook with valuation models, financial statements, and agent consensus

## Usage

### Basic Analysis
```
Analyze [TICKER] stock
```
Example: "Analyze AAPL stock"

### Custom Analysis
```
Analyze [TICKER] with focus on [ASPECT]
```
Example: "Analyze TSLA with focus on growth metrics"

### Compare Multiple Stocks
```
Compare [TICKER1] vs [TICKER2]
```
Example: "Compare MSFT vs GOOGL"

## What You Get

A multi-sheet Excel workbook containing:

1. **Executive Summary**: Key metrics, price performance, consensus rating
2. **Financial Statements**: Income statement, balance sheet, cash flow (5-year historical)
3. **Valuation Models**: DCF model with sensitivity analysis, peer comparables
4. **Financial Ratios**: Profitability, liquidity, leverage, efficiency metrics
5. **Agent Analysis**: Individual opinions from 10 AI investment agents:
   - Warren Buffett (value investing)
   - Ben Graham (margin of safety)
   - Aswath Damodaran (rigorous valuation)
   - Cathie Wood (disruptive innovation)
   - Michael Burry (deep value)
   - Fundamentals Analyst
   - Technical Analyst
   - Valuation Specialist
   - Risk Manager
   - Portfolio Manager
6. **Investment Thesis**: Synthesized recommendation with bull/bear cases

## Configuration

The skill uses these data sources:
- **yfinance**: Real-time quotes, historical prices, financial statements
- **SEC EDGAR**: Official filings (10-K, 10-Q, 8-K)
- **Alpha Vantage**: Advanced metrics, earnings data (requires free API key)
- **FRED**: Macroeconomic data (interest rates, GDP, inflation)

LLM provider (default: Groq with Llama 3.1 70B):
- Can be switched to Claude API, DeepSeek, or local Ollama
- Configure in `.env` file

## Requirements

- Python 3.11+
- API Keys:
  - Groq API (free tier available)
  - Alpha Vantage API (optional, free tier)

## Example Output

After running the skill on AAPL:
```
✓ Data extracted from 4 sources
✓ 10 agents completed analysis
✓ Excel workbook generated: output/AAPL_analysis_2026-02-18.xlsx

Agent Consensus:
- BUY: 6 agents (Buffett, Graham, Damodaran, Fundamentals, Valuation, Portfolio Mgr)
- HOLD: 3 agents (Technicals, Risk Mgr, Wood)
- SELL: 1 agent (Burry)

Price Target Range: $165 - $210
Current Price: $182.50
Upside Potential: +15%
```

## Notes

- Analysis typically takes 2-5 minutes depending on company size and data availability
- Results are cached for 24 hours to speed up repeat analyses
- Excel file is saved to `./output/` directory
- All data sources are free (except optional Alpha Vantage premium tier)

## Disclaimer

This tool generates AI-powered analysis for educational and research purposes only. It is NOT financial advice. Always conduct your own due diligence and consult licensed financial advisors before making investment decisions.
