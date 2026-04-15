"""
Analyst Panel Sheet
All 10 agents: signal, score, pillar breakdown, full reasoning, key risks.

Author: Joaquin Abondano w/ Claude Code
"""

from ..styles import (
    _f, fill, merge, wc, sec_hdr, col_hdr, spacer, hide_gridlines,
    NAVY_DARK, NAVY_MED, BLUE_ACC, GREEN, AMBER, RED,
    BLUE_TINT, GRAY_LIGHT, WHITE, BLACK, DARK_GRAY, MID_GRAY,
    SIGNAL_BG, SIGNAL_TEXT, ACTION_TEXT,
    AL_L, AL_LI, AL_C, AL_R, AL_W,
    BORDER_ALL,
)

_COLS = {
    "A": 2,   # spacer
    "B": 22,  # agent name
    "C": 9,   # signal
    "D": 10,  # confidence
    "E": 9,   # score
    "F": 6,   # /max
    "G": 8,   # norm %
    "H": 9,   # action
    "I": 12,  # price target
    "J": 2,   # spacer
}
C1  = 2   # col B
CE  = 9   # col I


SCORE_METHODOLOGY = {
    "fundamentals":      "0–1 score: valuation (P/E, P/B, EV/EBITDA), profitability (ROE, ROA, margins), growth (revenue, earnings), financial health (D/E, current ratio)",
    "ben_graham":        "30 pts: current ratio ≥2, D/E ≤0.5, EPS stability ×5yr, P/E < 1/(2×RF), P/B < 1.2, dividend history — strict value screens",
    "warren_buffett":    "20 pts: ROIC vs WACC, FCF yield, brand/moat durability, earnings consistency, capital allocation quality",
    "aswath_damodaran":  "20 pts: FCFF-DCF vs current price, ROIC>WACC spread, FCF margin trajectory, payout sustainability, growth quality",
    "cathie_wood":       "20 pts (7/4/4/5): innovation proxy score, P/S vs growth rate, TAM addressable market, ecosystem & platform lock-in",
    "michael_burry":     "20 pts (5/5/5/5): short interest (contrarian — high = bullish), insider activity, balance sheet leverage, macro risk score",
    "technicals":        "20 pts: RSI-14 (4), MACD-12/26/9 (4), Bollinger Bands (4), Moving Averages SMA50/200 (5), Volume trend (3)",
    "valuation":         "20 pts: DCF-FCFF (8), peer EV/EBITDA multiples (8), Graham Number (4) — weighted convergence",
    "risk_manager":      "20 pts: annualized volatility (4), beta vs S&P (4), max drawdown (4), Sharpe ratio (4), Kelly criterion (4)",
    "portfolio_manager": "LLM synthesis: integrates all 9 signals with contextual weighting → final BUY/HOLD/SELL + position sizing",
}

AGENT_ORDER = [
    "fundamentals",
    "ben_graham",
    "warren_buffett",
    "aswath_damodaran",
    "cathie_wood",
    "michael_burry",
    "technicals",
    "valuation",
    "risk_manager",
    "portfolio_manager",
]

AGENT_TYPE = {
    "fundamentals":      "Quant",
    "ben_graham":        "Quant + LLM",
    "warren_buffett":    "Quant + LLM",
    "aswath_damodaran":  "Quant + LLM",
    "cathie_wood":       "Quant + LLM",
    "michael_burry":     "Quant + LLM",
    "technicals":        "Quant",
    "valuation":         "Quant + LLM",
    "risk_manager":      "Quant",
    "portfolio_manager": "LLM",
}

# Pillar keys exposed in .scores for quant agents
PILLAR_LABELS = {
    "fundamentals": {
        "valuation":     "Valuation",
        "profitability": "Profitability",
        "growth":        "Growth",
        "health":        "Financial Health",
    },
    "technicals": {
        "rsi":             "RSI (14-period)",
        "macd":            "MACD (12/26/9)",
        "bollinger":       "Bollinger Bands",
        "moving_averages": "Moving Averages",
        "volume":          "Volume Trend",
    },
    "risk_manager": {
        "volatility": "Annualized Volatility",
        "beta":       "Beta vs S&P 500",
        "drawdown":   "Max Drawdown",
        "sharpe":     "Sharpe Ratio",
        "kelly":      "Kelly Max Position",
    },
}


def _norm(score, max_score):
    if score is None or max_score is None or max_score == 0:
        return None
    return score / max_score


def _score_color(norm):
    if norm is None:
        return GRAY_LIGHT
    if norm >= 0.65:
        return GREEN
    if norm >= 0.45:
        return AMBER
    return RED


def _fmt_score(score, max_score):
    if score is None:
        return "—"
    if max_score and max_score <= 1.0:
        return f"{score:.2f}"
    return f"{score:.1f}"


def _fmt_norm(norm):
    return f"{norm:.0%}" if norm is not None else "—"


ACTION_BG = {
    "buy":   GREEN,
    "hold":  AMBER,
    "sell":  RED,
    "short": RED,
    "cover": GREEN,
}


def build(wb, result):
    ws = wb.create_sheet("Analyst Panel")
    hide_gridlines(ws)
    ws.sheet_properties.tabColor = BLUE_ACC

    for col, w in _COLS.items():
        ws.column_dimensions[col].width = w

    ticker  = result.get("ticker", "")
    signals = result.get("agent_signals") or {}
    pm      = result.get("portfolio_decision")

    all_signals = dict(signals)
    if pm:
        all_signals["portfolio_manager"] = pm

    r = 1
    spacer(ws, r, 8); r += 1

    # Title
    ws.row_dimensions[r].height = 28
    wc(ws, r, C1, f"  ANALYST PANEL  —  {ticker}",
       font=_f(14, True, WHITE), bg=NAVY_DARK, align=AL_L)
    merge(ws, r, C1, r, CE)
    r += 1
    spacer(ws, r, 4); r += 1

    col_hdr(ws, r, ["Agent", "Signal", "Confidence", "Score", "/Max", "Score%", "Action", "Target"], C1)
    ws.freeze_panes = f"B{r + 1}"
    r += 1

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_rows = [
        "How to read this table:",
        "Score: raw points from the quantitative model (varies by agent: /30 Ben Graham, /20 most agents, 0-1 Fundamentals)",
        "Score%: normalized score (Score / Max × 100) — enables cross-agent comparison on a uniform scale",
        "Green ≥ 65%  |  Amber 45–64%  |  Red < 45%",
    ]
    for i, txt in enumerate(legend_rows):
        bold = (i == 0)
        ws.row_dimensions[r].height = 14
        wc(ws, r, C1, f"  {txt}",
           font=_f(8, bold, DARK_GRAY), bg=GRAY_LIGHT, align=AL_L)
        merge(ws, r, C1, r, CE)
        r += 1
    spacer(ws, r, 4); r += 1

    for agent_id in AGENT_ORDER:
        sig = all_signals.get(agent_id)
        if sig is None:
            continue

        scores    = sig.scores or {}
        total     = scores.get("total")
        total_max = scores.get("total_max")

        # Fundamentals uses a 0–1 confidence-style score
        if agent_id == "fundamentals" and total is None:
            total     = sig.confidence
            total_max = 1.0

        norm   = _norm(total, total_max)
        sc_bg  = _score_color(norm)
        sig_bg = SIGNAL_BG.get(sig.signal, NAVY_MED)
        act_bg = ACTION_BG.get(sig.target_action, NAVY_MED)

        # ── Agent header row ─────────────────────────────────────────────────
        ws.row_dimensions[r].height = 20
        wc(ws, r, C1,
           f"  {sig.agent_name}  [{AGENT_TYPE.get(agent_id, '')}]",
           font=_f(10, True, WHITE), bg=NAVY_DARK, align=AL_L, border=BORDER_ALL)
        wc(ws, r, C1 + 1, sig.signal.upper(),
           font=_f(9, True, WHITE), bg=sig_bg, align=AL_C, border=BORDER_ALL)
        wc(ws, r, C1 + 2, f"{sig.confidence:.0%}",
           font=_f(9), bg=GRAY_LIGHT, align=AL_C, border=BORDER_ALL)
        # Show score as "16.0/20" in score cell; /Max col shows max for reference
        score_display = (
            f"{total:.2f}/{total_max:.1f}" if (total is not None and total_max and total_max <= 1.0)
            else f"{total:.1f}/{total_max:.0f}" if (total is not None and total_max)
            else _fmt_score(total, total_max)
        )
        wc(ws, r, C1 + 3, score_display,
           font=_f(9, True, WHITE), bg=sc_bg, align=AL_C, border=BORDER_ALL)
        wc(ws, r, C1 + 4, f"/{total_max:.0f}" if total_max else "—",
           font=_f(8, False, DARK_GRAY), bg=GRAY_LIGHT, align=AL_L, border=BORDER_ALL)
        wc(ws, r, C1 + 5, _fmt_norm(norm),
           font=_f(9, True, WHITE), bg=sc_bg, align=AL_C, border=BORDER_ALL)
        wc(ws, r, C1 + 6, (sig.target_action or "—").upper(),
           font=_f(9, True, WHITE), bg=act_bg, align=AL_C, border=BORDER_ALL)
        pt_s = f"${sig.price_target:,.2f}" if sig.price_target else "—"
        wc(ws, r, C1 + 7, pt_s,
           font=_f(9), bg=GRAY_LIGHT, align=AL_R, border=BORDER_ALL)
        r += 1

        # ── Score methodology sub-row ─────────────────────────────────────────
        criteria = SCORE_METHODOLOGY.get(agent_id)
        if criteria:
            ws.row_dimensions[r].height = 13
            wc(ws, r, C1, f"  Methodology: {criteria}",
               font=_f(7.5, False, MID_GRAY), bg=GRAY_LIGHT, align=AL_W, border=BORDER_ALL)
            merge(ws, r, C1, r, CE)
            r += 1

        # ── Pillar breakdown (quant agents with sub-scores) ──────────────────
        pillar_map = PILLAR_LABELS.get(agent_id)
        if pillar_map:
            for p_key, p_label in pillar_map.items():
                p_data = scores.get(p_key)
                if p_data is None:
                    continue

                if isinstance(p_data, dict):
                    p_score = p_data.get("score")
                    p_max   = p_data.get("max")
                    p_sig   = p_data.get("signal")
                    detail  = p_data.get("detail") or {}
                    detail_str = ""
                    for dk, dv in detail.items():
                        if isinstance(dv, dict) and dv.get("value"):
                            detail_str = str(dv["value"])
                            break
                else:
                    p_score = p_data
                    p_max   = None
                    p_sig   = None
                    detail_str = ""

                p_norm  = _norm(p_score, p_max)
                p_sc_bg = _score_color(p_norm)
                p_sig_bg = SIGNAL_BG.get(p_sig, GRAY_LIGHT) if p_sig else GRAY_LIGHT

                ws.row_dimensions[r].height = 15
                wc(ws, r, C1, f"    ↳ {p_label}",
                   font=_f(8, False, DARK_GRAY), bg=WHITE, align=AL_LI, border=BORDER_ALL)
                if p_sig:
                    wc(ws, r, C1 + 1, p_sig.upper(),
                       font=_f(8, True, WHITE), bg=p_sig_bg, align=AL_C, border=BORDER_ALL)
                else:
                    wc(ws, r, C1 + 1, "", font=_f(8), bg=WHITE, align=AL_C, border=BORDER_ALL)
                wc(ws, r, C1 + 2, "", font=_f(8), bg=WHITE, align=AL_C, border=BORDER_ALL)
                wc(ws, r, C1 + 3, _fmt_score(p_score, p_max),
                   font=_f(8, True, WHITE) if p_sc_bg != GRAY_LIGHT else _f(8),
                   bg=p_sc_bg if p_sc_bg != GRAY_LIGHT else WHITE, align=AL_C, border=BORDER_ALL)
                wc(ws, r, C1 + 4, f"/{p_max}" if p_max else "—",
                   font=_f(8, False, DARK_GRAY), bg=WHITE, align=AL_L, border=BORDER_ALL)
                wc(ws, r, C1 + 5, _fmt_norm(p_norm),
                   font=_f(8), bg=WHITE, align=AL_C, border=BORDER_ALL)
                wc(ws, r, C1 + 6, detail_str,
                   font=_f(8, False, DARK_GRAY), bg=WHITE, align=AL_W, border=BORDER_ALL)
                merge(ws, r, C1 + 6, r, CE)
                r += 1

        # ── Reasoning ────────────────────────────────────────────────────────
        if sig.reasoning:
            ws.row_dimensions[r].height = 55
            wc(ws, r, C1, sig.reasoning,
               font=_f(8.5, False, BLACK), bg=BLUE_TINT, align=AL_W, border=BORDER_ALL)
            merge(ws, r, C1, r, CE)
            r += 1

        # ── Key Risks ─────────────────────────────────────────────────────────
        if sig.key_risks:
            risks_text = "  ⚠  " + "   |   ".join(sig.key_risks[:3])
            ws.row_dimensions[r].height = 20
            wc(ws, r, C1, risks_text,
               font=_f(8, False, RED), bg=GRAY_LIGHT, align=AL_W, border=BORDER_ALL)
            merge(ws, r, C1, r, CE)
            r += 1

        spacer(ws, r, 3); r += 1

    # Footer
    ws.row_dimensions[r].height = 14
    wc(ws, r, C1,
       "  Agent outputs are model-generated and do not constitute investment advice. Quant+LLM agents use deterministic scoring + LLM narrative.",
       font=_f(8, False, MID_GRAY), bg=GRAY_LIGHT, align=AL_L)
    merge(ws, r, C1, r, CE)

    return ws
