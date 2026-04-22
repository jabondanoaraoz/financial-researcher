# Financial Researcher — Claude Code Skill

Autonomous financial analysis engine. Given a stock ticker, runs 10 AI investment agents and generates an IB-grade Excel report.

## How to invoke

When the user asks to analyze a stock, run:

```bash
python run.py <TICKER>
```

### With custom peer universe
```bash
python run.py <TICKER> --peers MSFT GOOGL AMZN META
```

### Terminal summary only (no Excel)
```bash
python run.py <TICKER> --no-excel
```

### Custom output path
```bash
python run.py <TICKER> --output path/to/report.xlsx
```

### Suppress terminal output
```bash
python run.py <TICKER> --quiet
```

## What it does

1. Fetches data from yfinance, SEC EDGAR, Alpha Vantage, and FRED
2. Runs 10 AI investment agents sequentially:
   - Ben Graham, Warren Buffett, Aswath Damodaran, Cathie Wood, Michael Burry
   - Fundamentals Analyst, Technical Analyst, Valuation Analyst, Risk Manager
   - Portfolio Manager (synthesizes all 9 signals into a final recommendation)
3. Prints a terminal summary (consensus, agent breakdown, risk snapshot)
4. Generates an Excel workbook in `output/<TICKER>_<date>.xlsx`

## Excel output (5 sheets)

| Sheet | Contents |
|-------|----------|
| Financials | Income statement, cash flow, balance sheet (5-year historical) |
| Multiples | Peer comparables with user-editable weights |
| DCF Model | IB-style DCF with Excel formulas and sensitivity table |
| Analyst Panel | All 10 agents — signal, score, reasoning, key risks |
| Summary | Investment decision, consensus, PM thesis, risk snapshot |

## Notes

- Analysis takes 2–5 minutes (Groq rate limits cause automatic retries)
- Data is cached for 24 hours in `cache/financial_data.db`
- Excel is saved to `output/` (gitignored, directory tracked)
- Requires `.env` with GROQ_API_KEY (see INSTALL.md)
