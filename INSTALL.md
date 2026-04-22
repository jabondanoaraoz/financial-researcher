# Installation Guide

## Requirements

- Python 3.10 or higher
- pip
- A free [Groq API key](https://console.groq.com) (the LLM backbone)

## Step-by-step setup

### 1. Clone the repository

```bash
git clone https://github.com/jabondanoaraoz/financial-researcher.git
cd financial-researcher
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate it:
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
GROQ_API_KEY=your_key_here          # Required
ALPHA_VANTAGE_API_KEY=your_key_here # Optional
FRED_API_KEY=your_key_here          # Optional
```

#### Where to get them

| Key | Required | Where to get it | Cost |
|-----|----------|-----------------|------|
| `GROQ_API_KEY` | Yes | [console.groq.com](https://console.groq.com) | Free tier available |
| `ALPHA_VANTAGE_API_KEY` | No | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | Free tier available |
| `FRED_API_KEY` | No | [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) | Free |

> If Alpha Vantage or FRED keys are missing, those data sources are skipped gracefully — the analysis still runs.

### 5. Verify installation

Run the mock test (no API calls required):

```bash
python tests/test_excel.py
```

This generates `output/AAPL_test.xlsx`. If it opens correctly, the installation is complete.

### 6. Run your first analysis

```bash
python run.py AAPL
```

The terminal will print a summary and save the Excel report to `output/AAPL_<date>.xlsx`.

---

## Common issues

**`ModuleNotFoundError: No module named 'groq'`**
You're not in the virtual environment. Activate it with `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows).

**`AuthenticationError` or `401` from Groq**
Your `GROQ_API_KEY` is missing or incorrect in `.env`.

**Analysis hangs for a long time**
Groq's free tier has rate limits. The tool retries automatically — this is normal. It typically resolves within 1–2 minutes.

**`output/` directory missing**
It's created automatically on first run. If it doesn't appear, create it manually: `mkdir output`.
