# 📊 Financial Researcher

**Autonomous financial analysis engine powered by AI agents** — A Claude Code skill that generates IB-grade investment research reports.

## 🎯 What It Does

Financial Researcher combines real financial data with multi-agent AI analysis to produce comprehensive investment research reports in Excel format. Think of it as having 10 expert investors analyze a stock simultaneously and compile their findings into a professional-grade workbook.

### Key Features

- **📈 Multi-Source Data Extraction**: Pulls real-time data from yfinance, SEC EDGAR, Alpha Vantage, and FRED
- **🤖 10 AI Investment Agents**:
  - 5 legendary investors (Buffett, Graham, Damodaran, Wood, Burry)
  - 3 specialized analysts (Fundamentals, Technicals, Valuation)
  - 1 Risk Manager
  - 1 Portfolio Manager
- **📑 IB-Grade Excel Output**: Multi-sheet workbook with DCF model, peer comps, financial statements, and agent consensus
- **🔄 LLM Agnostic**: Works with Groq, Claude API, Ollama, or DeepSeek
- **💾 Smart Caching**: SQLite-based cache to minimize API calls

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                               │
│  yfinance · SEC EDGAR · Alpha Vantage · FRED                │
│                   ↓ (cached in SQLite)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   AGENT ENGINE                               │
│  10 AI agents analyze in parallel using LLM                 │
│  (Groq Llama 3.1 70B by default)                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   EXCEL OUTPUT                               │
│  9-sheet workbook: Overview · Financials · DCF · Comps ·    │
│  Ratios · Agent Consensus · Investment Thesis               │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

* **Data Engineering & Extraction:** `yfinance`, `sec-edgar-api`, `alpha_vantage`, `fredapi`
* **Agentic Framework & LLM Ops:** Custom Orchestrator, `groq`, Anthropic Claude API, Ollama (Local LLMs)
* **Data Processing:** `pandas`, `numpy`, `sqlite3` (Caching layer)
* **Reporting & Outputs:** `openpyxl`, `xlsxwriter`

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/[YOUR_USERNAME]/financial-researcher.git
cd financial-researcher

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configuration

Required API keys:
- **Groq API** (default LLM): Get free key at [console.groq.com](https://console.groq.com)
- **Alpha Vantage** (optional): Free tier at [alphavantage.co](https://www.alphavantage.co)

Optional providers:
- Anthropic Claude API
- DeepSeek API
- Ollama (local)

### 3. Usage

```python
# As Claude Code Skill
# Simply invoke from Claude Code:
"Analyze AAPL stock using financial-researcher"

# Or use directly as Python module
from scripts.orchestrator import FinancialResearcher

researcher = FinancialResearcher()
report = researcher.analyze(ticker="AAPL", output_path="./output/AAPL_analysis.xlsx")
```

## 📂 Project Structure

```
financial-researcher/
├── SKILL.md              # Claude Code skill definition
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
├── references/           # Knowledge base (valuation methods, metrics, etc.)
├── scripts/
│   ├── data/            # Data extraction adapters
│   ├── agents/          # AI investment agents
│   ├── excel/           # Excel workbook builder
│   └── orchestrator.py  # Main orchestration logic
└── templates/           # Configuration files (peer mappings, etc.)
```

## 🧠 Agent Philosophies

Each AI agent embodies the investment philosophy of legendary investors:

- **Warren Buffett**: Value investing, moats, quality businesses
- **Ben Graham**: Margin of safety, intrinsic value, contrarian plays
- **Aswath Damodaran**: Rigorous valuation, DCF modeling, market narratives
- **Cathie Wood**: Disruptive innovation, exponential growth, long-term tech trends
- **Michael Burry**: Deep value, asymmetric bets, credit analysis

Plus specialized agents for:
- Fundamental analysis (ratios, margins, growth)
- Technical analysis (charts, momentum, support/resistance)
- Valuation modeling (DCF, comps, multiples)
- Risk assessment (VaR, beta, scenario analysis)
- Portfolio construction (allocation, diversification)

## 📊 Excel Output

The generated workbook contains:

1. **Overview**: Key metrics, price performance, sector info
2. **Income Statement**: 5-year historical + projections
3. **Balance Sheet**: Assets, liabilities, equity breakdown
4. **Cash Flow**: Operating, investing, financing activities
5. **Financial Ratios**: Profitability, liquidity, efficiency, leverage
6. **DCF Model**: Discounted Cash Flow valuation featuring WACC calculation, terminal value assumptions, and margin of safety sensitivity analysis.
7. **Peer Comparables**: Valuation multiples vs. industry peers
8. **Agent Consensus**: 10 agents' ratings, price targets, and reasoning
9. **Investment Thesis**: Synthesized recommendation with bull/bear cases

## 🔧 Advanced Configuration

### Switch LLM Provider

```bash
# Use Claude API instead of Groq
LLM_PROVIDER=claude
LLM_MODEL=claude-sonnet-4-5-20250929

# Use local Ollama
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:70b
```

### Customize Agent Weightings

Edit `templates/agent_weights.json` to adjust how much each agent influences the consensus.

### Add Custom Peer Groups

Edit `templates/peers_mapping.json` to define industry-specific peer sets.

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss proposed changes or architectural shifts. 
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. The outputs generated by this repository, including agent analyses, DCF models, and technical screening, do not constitute financial advice, investment recommendations, or an offer to buy/sell any securities. The autonomous agents simulate the investment thesis of historical figures based on LLM processing, which may contain hallucinations, parsing errors, or outdated assumptions.

Always conduct your own due diligence (DD) and consult with a certified financial advisor before making capital allocation decisions. The author is not liable for any capital losses incurred resulting from the use of this software or its outputs.

---

## 👤 Author

**Joaquín Abondano Araoz** *Strategic & Financial Planning | Value Investing | Data Analytics | AI & ML Enthusiast*

Contact Me: [LinkedIn](https://www.linkedin.com/in/joaquin-abondano) • [GitHub](https://github.com/jabondanoaraoz)

> 🤖 **Note:** The architecture, debugging, and agent logic of this repository were accelerated utilizing [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview). Data sourced from public APIs.