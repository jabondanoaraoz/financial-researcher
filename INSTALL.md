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

> If Alpha Vantage or FRED keys are missing, those data sources are skipped gracefully - the analysis still runs.

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

## Optional: Install as a Claude Code skill

If you use [Claude Code](https://claude.ai/code), you can install Financial Researcher as a skill so Claude can run analyses directly from a conversation.

### What this enables

Once installed, you can type things like:

> *"Analyze AAPL"* or *"Research TSLA with peers RIVN LCID GM"*

and Claude will automatically run the pipeline and generate the Excel report - no need to remember the CLI commands.

### Install

Copy `SKILL.md` into Claude's skill directory:

```bash
# macOS / Linux
mkdir -p ~/.claude/skills/financial-researcher
cp SKILL.md ~/.claude/skills/financial-researcher/SKILL.md
```

```powershell
# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.claude\skills\financial-researcher"
Copy-Item SKILL.md "$env:USERPROFILE\.claude\skills\financial-researcher\SKILL.md"
```

### Verify

Open a new Claude Code session and ask:

> *"Analyze MSFT"*

Claude should recognize the skill and run `python run.py MSFT` from the project directory.

> **Note:** The skill file references an absolute path to this project directory. If you move the project folder after installing, update the path in `~/.claude/skills/financial-researcher/SKILL.md`.

---

## Common issues

**`ModuleNotFoundError: No module named 'groq'`**
You're not in the virtual environment. Activate it with `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows).

**`AuthenticationError` or `401` from Groq**
Your `GROQ_API_KEY` is missing or incorrect in `.env`.

**Analysis hangs for a long time**
Groq's free tier has rate limits. The tool retries automatically - this is normal. It typically resolves within 1–2 minutes.

**`output/` directory missing**
It's created automatically on first run. If it doesn't appear, create it manually: `mkdir output`.
