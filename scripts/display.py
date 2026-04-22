"""
Terminal display layer for Financial Researcher.
Prints a clean, readable summary of run_analysis() results.
"""

from datetime import date


# ── ANSI colors (graceful fallback if terminal doesn't support them) ──────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_DIM    = "\033[2m"

SIGNAL_COLOR = {
    "bullish": _GREEN,
    "neutral": _YELLOW,
    "bearish": _RED,
}
ACTION_EMOJI = {
    "buy":      "BUY  ",
    "sell":     "SELL ",
    "hold":     "HOLD ",
    "add":      "ADD  ",
    "trim":     "TRIM ",
    "avoid":    "AVOID",
    "watch":    "WATCH",
}


def _c(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


def _bold(text: str) -> str:
    return f"{_BOLD}{text}{_RESET}"


def _signal_badge(signal: str) -> str:
    color = SIGNAL_COLOR.get(signal, "")
    label = signal.upper().ljust(7)
    return _c(label, color + _BOLD)


def _bar(score: float, max_score: float, width: int = 12) -> str:
    if max_score <= 0:
        return " " * width
    filled = round((score / max_score) * width)
    return "█" * filled + "░" * (width - filled)


def print_summary(result: dict) -> None:
    """Print full analysis summary to stdout."""
    ticker    = result["ticker"]
    consensus = result["consensus"]
    pm        = result.get("portfolio_decision")
    info      = result.get("company_data", {}).get("info", {})
    km        = result.get("company_data", {}).get("key_metrics", {})
    signals   = result.get("agent_signals", {})

    width = 70
    div   = "─" * width

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print(_bold("=" * width))
    company_name = info.get("longName") or info.get("shortName") or ticker
    sector       = info.get("sector", "")
    header_right = f"{sector}  |  {date.today()}" if sector else str(date.today())
    print(_bold(f"  {ticker}  {company_name}"))
    print(_c(f"  {header_right}", _DIM))
    print(_bold("=" * width))

    # ── Key metrics ───────────────────────────────────────────────────────────
    price      = km.get("current_price")
    mktcap     = km.get("market_cap")
    pe         = km.get("pe_ratio")
    fwd_pe     = km.get("forward_pe")
    ps         = km.get("ps_ratio")
    ev_ebitda  = km.get("ev_to_ebitda")

    def _fmt_cap(v):
        if v is None:
            return "N/A"
        if v >= 1e12:
            return f"${v/1e12:.2f}T"
        if v >= 1e9:
            return f"${v/1e9:.1f}B"
        return f"${v/1e6:.0f}M"

    def _fmt(v, fmt=".1f", prefix="", suffix=""):
        return f"{prefix}{v:{fmt}}{suffix}" if v is not None else "N/A"

    metrics = [
        ("Price",      _fmt(price, ".2f", "$")),
        ("Mkt Cap",    _fmt_cap(mktcap)),
        ("P/E",        _fmt(pe,      ".1f", "x")),
        ("Fwd P/E",    _fmt(fwd_pe,  ".1f", "x")),
        ("P/S",        _fmt(ps,      ".1f", "x")),
        ("EV/EBITDA",  _fmt(ev_ebitda, ".1f", "x")),
    ]
    row = "  ".join(f"{k}: {_bold(v)}" for k, v in metrics)
    print(f"  {row}")
    print(_c(f"  {div}", _DIM))

    # ── Consensus ─────────────────────────────────────────────────────────────
    n_bull = consensus["bullish"]
    n_neut = consensus["neutral"]
    n_bear = consensus["bearish"]
    avg    = consensus["avg_score_20"]

    print(f"\n  {_bold('CONSENSUS')}")
    print(
        f"  {_c(f'▲ {n_bull} Bullish', _GREEN)}   "
        f"{_c(f'● {n_neut} Neutral', _YELLOW)}   "
        f"{_c(f'▼ {n_bear} Bearish', _RED)}"
    )
    print(f"  Avg score: {_bold(f'{avg:.1f} / 20')}")

    # ── Portfolio Manager decision ─────────────────────────────────────────────
    if pm:
        action_str = ACTION_EMOJI.get(pm.target_action, pm.target_action.upper())
        pt_str     = f"  |  Target: {_bold(f'${pm.price_target:.2f}')}" if pm.price_target else ""
        conf_str   = f"  |  Conf: {pm.confidence:.0%}"

        print(f"\n  {_bold('PORTFOLIO MANAGER')}")
        print(
            f"  {_signal_badge(pm.signal)}  "
            f"{_bold(action_str)}"
            f"{pt_str}{conf_str}"
        )
        # Wrap reasoning at ~66 chars
        reasoning = pm.reasoning or ""
        words, line, lines = reasoning.split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > 66:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        for ln in lines[:6]:   # cap at 6 lines
            print(f"  {_c(ln, _DIM)}")

    # ── Agent breakdown ───────────────────────────────────────────────────────
    print(f"\n  {_bold('AGENT BREAKDOWN')}")
    print(f"  {'Agent':<22} {'Signal':<10} {'Conf':>5}  {'Score':>8}  {'Action'}")
    print(f"  {div}")

    agent_display_names = {
        "fundamentals":      "Fundamentals",
        "ben_graham":        "Ben Graham",
        "warren_buffett":    "Warren Buffett",
        "aswath_damodaran":  "A. Damodaran",
        "cathie_wood":       "Cathie Wood",
        "michael_burry":     "Michael Burry",
        "technicals":        "Technicals",
        "valuation":         "Valuation",
        "risk_manager":      "Risk Manager",
    }

    for agent_id, signal in signals.items():
        name   = agent_display_names.get(agent_id, agent_id).ljust(22)
        sig    = _signal_badge(signal.signal)
        conf   = f"{signal.confidence:.0%}".rjust(5)
        total  = signal.scores.get("total")
        tmax   = signal.scores.get("total_max", 20)
        if total is not None:
            score_str = f"{total:>4.0f}/{tmax}"
        else:
            score_str = "   —/—"
        action = ACTION_EMOJI.get(signal.target_action, signal.target_action.upper())
        print(f"  {name} {sig}  {conf}  {score_str}  {action}")

    # ── Risk snapshot ─────────────────────────────────────────────────────────
    risk = result.get("risk_metrics", {})
    if risk:
        print(f"\n  {_bold('RISK SNAPSHOT')}")
        beta      = risk.get("beta")
        sharpe    = risk.get("sharpe_proxy")
        max_dd    = risk.get("max_drawdown")
        vol       = risk.get("annualized_volatility")
        debt_eq   = risk.get("debt_to_equity")

        risk_items = [
            ("Beta",        _fmt(beta,    ".2f")),
            ("Sharpe",      _fmt(sharpe,  ".2f")),
            ("Max DD",      _fmt(max_dd,  ".1%") if max_dd is not None else "N/A"),
            ("Vol (ann)",   _fmt(vol,     ".1%") if vol is not None else "N/A"),
            ("Debt/Equity", _fmt(debt_eq, ".2f")),
        ]
        row = "   ".join(f"{k}: {_bold(v)}" for k, v in risk_items)
        print(f"  {row}")

    print()
    print(_bold("=" * width))
    print()
