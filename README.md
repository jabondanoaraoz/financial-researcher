# 📊 Financial Researcher

**Autonomous financial analysis engine powered by 10 AI investment agents.** Generates IB-grade investment research reports on any publicly traded company.

## 🎯 What It Does

Financial Researcher combines real financial data with multi-agent AI analysis to produce comprehensive investment research. Think of it as having 10 legendary investors and specialized analysts study a stock simultaneously and compile their findings into a professional-grade Excel workbook.

```bash
python run.py GOOGL
```

```
Analyzing GOOGL...

══════════════════════════════════════════════════════════════════════
  GOOGL  Alphabet Inc.
  Communication Services  |  2026-04-21
══════════════════════════════════════════════════════════════════════
  Price: $332.29   Mkt Cap: $4.02T   P/E: x30.8   Fwd P/E: x24.6

  CONSENSUS
  ▲ 4 Bullish   ● 2 Neutral   ▼ 3 Bearish
  Avg score: 10.6 / 20

  PORTFOLIO MANAGER
  BULLISH  BUY  |  Target: $350.00  |  Conf: 65%

Excel report saved → output/GOOGL_2026-04-21.xlsx
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                            │
│         yfinance · SEC EDGAR · Alpha Vantage · FRED         │
│                  ↓  cached in SQLite  ↓                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      AGENT ENGINE                           │
│   10 AI agents analyze from fundamentals to technicals      │
│         each scoring the stock on their own criteria        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                            │
│   Terminal summary  +  5-sheet IB-grade Excel workbook      │
└─────────────────────────────────────────────────────────────┘
```

## 🤖 The 10 Agents

Each agent embodies a distinct investment philosophy and scores the stock on its own criteria:

| Agent | Philosophy | Scoring |
|-------|-----------|---------|
| **Ben Graham** | Margin of safety, net-nets, conservative multiples | /30 (6 pillars) |
| **Warren Buffett** | Moats, quality businesses, capital allocation | /20 |
| **Aswath Damodaran** | Rigorous DCF, WACC, growth reinvestment | /20 |
| **Cathie Wood** | Disruptive innovation, revenue growth, P/S | /20 |
| **Michael Burry** | Deep value, contrarian signals, short interest | /20 |
| **Fundamentals Analyst** | Profitability, growth, balance sheet health | 0–1 |
| **Technical Analyst** | RSI, MACD, Bollinger Bands, SMA, volume | /20 |
| **Valuation Analyst** | DCF + peer multiples + Graham Number | /20 |
| **Risk Manager** | Beta, Sharpe, max drawdown, Kelly sizing | /20 |
| **Portfolio Manager** | Synthesizes all 9 signals → final recommendation | - |

## 📊 Excel Output

The generated workbook contains 5 IB-style sheets:

1. **Financials**: 5-year income statement, cash flow, and balance sheet with formula-driven ratios
2. **Multiples**: Peer comparables table with user-editable weighting
3. **DCF Model**: Full discounted cash flow model with Excel formulas, WACC build-up, and sensitivity table
4. **Analyst Panel**: All 10 agents with signal, confidence, pillar-level scores, reasoning, and key risks
5. **Summary**: Investment decision, consensus heatmap, Portfolio Manager thesis, and risk snapshot

## 🚀 Quick Start

```bash
git clone https://github.com/jabondanoaraoz/financial-researcher.git
cd financial-researcher
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
python run.py AAPL
```

→ Full setup instructions in **[INSTALL.md](INSTALL.md)**

Requires a free [Groq API key](https://console.groq.com). Alpha Vantage and FRED are optional.

## 🖥️ CLI Reference

```bash
# Single ticker
python run.py AAPL

# Batch mode: analyzes each ticker sequentially, one Excel per ticker
python run.py AAPL MSFT META GOOGL

# Custom peer universe
python run.py AAPL --peers MSFT GOOGL AMZN META

# Terminal summary only, skip Excel
python run.py AAPL --no-excel

# Custom Excel output path (single ticker only)
python run.py AAPL --output reports/AAPL_Q2.xlsx

# Suppress terminal output (silent Excel generation)
python run.py AAPL --quiet

# Verbose pipeline logging
python run.py AAPL --log-level INFO
```

## 🌐 Web UI

For a browser-based interface, start the Streamlit app:

```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501`. Enter a ticker and optional peers, click **Analyze**, and explore results interactively:

- Key valuation metrics (Price, Mkt Cap, P/E, Fwd P/E, P/S, EV/EBITDA)
- Consensus summary + Portfolio Manager decision with full reasoning
- Agent breakdown table (signal, confidence, score, action for all 10 agents)
- Full agent reasoning & key risks (expandable)
- Peer comparables table (P/S, P/E, Fwd P/E, EV/EBITDA, P/FCF, Beta)
- Risk snapshot (Beta, Sharpe, Max Drawdown, Volatility, Kelly sizing)
- One-click Excel report download

## 🧠 Claude Code Skill

This project is designed as a **Claude Code skill**. Once installed, Claude can run an analysis directly from a conversation:

> *"Analyze TSLA stock"* → `python run.py TSLA`

See **[SKILL.md](SKILL.md)** for the full skill definition and invocation patterns.

## 📂 Project Structure

```
financial-researcher/
├── run.py                   # CLI entry point
├── SKILL.md                 # Claude Code skill definition
├── INSTALL.md               # Step-by-step setup guide
├── requirements.txt
├── .env.example
├── scripts/
│   ├── orchestrator.py      # Main 10-agent pipeline
│   ├── display.py           # Terminal summary renderer
│   ├── agents/              # 10 investment agents + base class
│   ├── data/                # yfinance, SEC EDGAR, Alpha Vantage, FRED adapters
│   └── excel/               # 5-sheet workbook builder
├── tests/
│   ├── mock_result.py       # Shared mock (no API calls)
│   ├── test_excel.py        # Excel generation test
│   └── test_display.py      # Terminal display test
├── references/              # Agent philosophies, valuation methods, metrics glossary
├── templates/               # Peer universe mappings
├── output/                  # Generated reports (gitignored)
└── cache/                   # SQLite data cache (gitignored)
```

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Data** | `yfinance`, `sec-edgar-downloader`, `alpha_vantage`, `fredapi` |
| **LLM** | `groq` (llama-3.3-70b-versatile) swappable via `.env` |
| **Processing** | `pandas`, `numpy` |
| **Excel** | `openpyxl` |
| **Cache** | `sqlite3` via `sqlalchemy` |

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. The outputs (agent analyses, DCF models, price targets) do not constitute financial advice or investment recommendations. Agent reasoning is LLM-generated and may contain errors or outdated assumptions. Always conduct your own due diligence and consult a licensed financial advisor before making investment decisions.

---

## 👤 Author

**Joaquín Abondano Araoz** - Financial Modeling & Valuation · Value Investing · Data Analytics · AI & ML

[Website](https://www.joaquinabondano.com) · [LinkedIn](https://www.linkedin.com/in/joaquin-abondano) · [GitHub](https://github.com/jabondanoaraoz)

> Built with [Claude Code](https://claude.ai/code)
