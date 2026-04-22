# Financial Researcher

Autonomous financial analysis engine powered by 10 AI investment agents. Given a stock ticker, it fetches real data, runs multi-perspective analysis, and generates an IB-grade Excel report.

## What it does

```bash
python run.py GOOGL
```

1. Pulls data from yfinance, SEC EDGAR, Alpha Vantage, and FRED
2. Runs 10 AI agents — each embodying a different investment philosophy
3. Prints a terminal summary with consensus, scores, and risk snapshot
4. Generates a 5-sheet Excel workbook in `output/`

## The 10 agents

| Agent | Philosophy | Score |
|-------|-----------|-------|
| Ben Graham | Margin of safety, net-nets, conservative valuation | /30 |
| Warren Buffett | Moats, quality businesses, capital allocation | /20 |
| Aswath Damodaran | Rigorous DCF, WACC, growth accounting | /20 |
| Cathie Wood | Disruptive innovation, P/S, revenue growth | /20 |
| Michael Burry | Deep value, contrarian, short interest signals | /20 |
| Fundamentals Analyst | Profitability, growth, health ratios | 0–1 |
| Technical Analyst | RSI, MACD, Bollinger, SMA, volume | /20 |
| Valuation Analyst | DCF + peer multiples + Graham Number | /20 |
| Risk Manager | Beta, Sharpe, drawdown, Kelly sizing | /20 |
| Portfolio Manager | Synthesizes all 9 signals → final recommendation | — |

## Excel output

| Sheet | Contents |
|-------|----------|
| Financials | Income statement, cash flow, balance sheet (5-year) |
| Multiples | Peer comparables with editable weights |
| DCF Model | IB-style DCF with Excel formulas and sensitivity table |
| Analyst Panel | All 10 agents — signal, score, pillar breakdown, reasoning |
| Summary | Investment decision, consensus, PM thesis, risk snapshot |

## CLI reference

```bash
# Basic analysis (generates terminal summary + Excel)
python run.py AAPL

# Custom peer universe
python run.py AAPL --peers MSFT GOOGL AMZN META

# Terminal only, no Excel
python run.py AAPL --no-excel

# Custom output path
python run.py AAPL --output reports/AAPL_custom.xlsx

# Suppress terminal output (Excel only)
python run.py AAPL --quiet

# Verbose logging
python run.py AAPL --log-level INFO
```

## Project structure

```
financial-researcher/
├── run.py                   # CLI entry point
├── SKILL.md                 # Claude Code skill definition
├── INSTALL.md               # Setup guide
├── requirements.txt
├── .env.example             # API key template
├── scripts/
│   ├── orchestrator.py      # Main pipeline
│   ├── display.py           # Terminal summary
│   ├── agents/              # 10 investment agents + base class
│   ├── data/                # yfinance, SEC EDGAR, Alpha Vantage, FRED adapters
│   └── excel/               # 5-sheet workbook builder
├── tests/
│   ├── mock_result.py       # Shared mock (no API calls)
│   ├── test_excel.py        # Excel generation test
│   └── test_display.py      # Terminal display test
├── output/                  # Generated reports (gitignored)
├── cache/                   # SQLite data cache (gitignored)
├── references/              # Agent philosophies, valuation methods, metrics glossary
└── templates/               # Peer universe mappings
```

## Setup

See [INSTALL.md](INSTALL.md) for full installation instructions.

**Quick version:**
```bash
git clone https://github.com/jabondanoaraoz/financial-researcher.git
cd financial-researcher
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
python run.py AAPL
```

Requires a free [Groq API key](https://console.groq.com). Alpha Vantage and FRED keys are optional.

## Tech stack

- **Data**: yfinance, sec-edgar, alpha_vantage, fredapi
- **LLM**: Groq (llama-3.3-70b-versatile by default) — configurable via `.env`
- **Processing**: pandas, numpy
- **Excel**: openpyxl
- **Cache**: SQLite

## Disclaimer

For educational and research purposes only. Not financial advice. Always conduct your own due diligence before making investment decisions.

---

**Author:** Joaquín Abondano Araoz · [LinkedIn](https://www.linkedin.com/in/joaquin-abondano) · [GitHub](https://github.com/jabondanoaraoz)

> Built with [Claude Code](https://claude.ai/code)
