"""
Analyst Panel Sheet
All 10 agents: signal, confidence, score, action, price target, reasoning, key risks.

Author: Joaquin Abondano w/ Claude Code
"""

from ..styles import (
    _f, fill, merge, wc, sec_hdr, col_hdr, spacer, hide_gridlines,
    NAVY_DARK, NAVY_MED, BLUE_ACC, GREEN, AMBER, RED,
    BLUE_TINT, GRAY_LIGHT, GRAY_MED, WHITE, BLACK, DARK_GRAY, MID_GRAY,
    SIGNAL_BG, SIGNAL_TEXT, ACTION_TEXT,
    AL_L, AL_LI, AL_C, AL_R, AL_W,
    BORDER_ALL,
)

# Column layout
_COLS = {
    "A": 2,   # spacer
    "B": 22,  # Agent Name
    "C": 11,  # Type
    "D": 13,  # Signal
    "E": 10,  # Confidence
    "F": 8,   # Score
    "G": 6,   # Max
    "H": 10,  # Normalized
    "I": 10,  # Action
    "J": 13,  # Price Target
    "K": 2,   # spacer
}
C1    = 2    # B
CE    = 10   # J
N_COL = CE - C1 + 1   # 9

AGENT_ORDER = [
    "fundamentals", "ben_graham", "warren_buffett", "aswath_damodaran",
    "cathie_wood", "michael_burry", "technicals", "valuation", "risk_manager",
]

AGENT_TYPE = {
    "fundamentals":     "Quant",
    "ben_graham":       "Quant + LLM",
    "warren_buffett":   "Quant + LLM",
    "aswath_damodaran": "Quant + LLM",
    "cathie_wood":      "Quant + LLM",
    "michael_burry":    "Quant + LLM",
    "technicals":       "Quant",
    "valuation":        "Quant + LLM",
    "risk_manager":     "Quant",
    "portfolio_manager": "LLM",
}


def _norm(sc):
    """Normalize a score dict → 0–100 float."""
    total = sc.get("total")
    t_max = sc.get("total_max", 20)
    if total is None or not isinstance(total, (int, float)):
        return None
    if t_max and t_max > 0:
        return round(total / t_max * 100, 1)
    return None


def _score_str(sc):
    total = sc.get("total")
    t_max = sc.get("total_max")
    if total is None: return "—", "—"
    total_s = f"{total:.1f}" if isinstance(total, float) else str(total)
    t_max_s = f"{t_max:.0f}" if t_max is not None else "—"
    return total_s, t_max_s


def _pt_str(signal):
    pt = signal.price_target
    if pt is None: return "—"
    return f"${pt:,.2f}"


def _write_agent_row(ws, r, signal, agent_id, alt=False):
    """Write a single agent data row."""
    row_bg    = GRAY_LIGHT if alt else WHITE
    sig_bg    = SIGNAL_BG.get(signal.signal, NAVY_MED)
    sig_text  = SIGNAL_TEXT.get(signal.signal, signal.signal.upper())
    act_text  = ACTION_TEXT.get(signal.target_action, signal.target_action.upper())
    atype     = AGENT_TYPE.get(agent_id, "—")
    norm      = _norm(signal.scores or {})
    sc_s, mx_s = _score_str(signal.scores or {})
    pt_s      = _pt_str(signal)

    ws.row_dimensions[r].height = 17

    # B: Agent Name
    wc(ws, r, C1,   signal.agent_name,
       font=_f(9, True),  bg=row_bg, align=AL_LI, border=BORDER_ALL)
    # C: Type
    wc(ws, r, C1+1, atype,
       font=_f(8, False, DARK_GRAY), bg=row_bg, align=AL_C, border=BORDER_ALL)
    # D: Signal
    wc(ws, r, C1+2, sig_text,
       font=_f(9, True, WHITE), bg=sig_bg, align=AL_C, border=BORDER_ALL)
    # E: Confidence
    wc(ws, r, C1+3, f"{signal.confidence:.0%}",
       font=_f(9),  bg=row_bg, align=AL_C, border=BORDER_ALL)
    # F: Score
    wc(ws, r, C1+4, sc_s,
       font=_f(9),  bg=row_bg, align=AL_R, border=BORDER_ALL)
    # G: Max
    wc(ws, r, C1+5, f"/{mx_s}",
       font=_f(8, False, DARK_GRAY), bg=row_bg, align=AL_L, border=BORDER_ALL)
    # H: Normalized
    norm_s = f"{norm:.0f}%" if norm is not None else "—"
    norm_bg = _score_color(norm)
    wc(ws, r, C1+6, norm_s,
       font=_f(9, True, WHITE), bg=norm_bg, align=AL_C, border=BORDER_ALL)
    # I: Action
    wc(ws, r, C1+7, act_text,
       font=_f(9), bg=row_bg, align=AL_C, border=BORDER_ALL)
    # J: Price Target
    wc(ws, r, C1+8, pt_s,
       font=_f(9), bg=row_bg, align=AL_R, border=BORDER_ALL)

    return r + 1


def _score_color(norm):
    """Background color for normalized score cell."""
    if norm is None: return GRAY_MED
    if norm >= 65:   return GREEN
    if norm >= 45:   return AMBER
    return RED


def _write_detail_row(ws, r, label, text, alt=False):
    """Collapsible detail row (reasoning or key risks)."""
    bg = GRAY_LIGHT if alt else "F0F4F8"
    ws.row_dimensions[r].height = 14

    wc(ws, r, C1,   f"  {label}",
       font=_f(8, True, DARK_GRAY), bg=bg, align=AL_LI, border=BORDER_ALL)
    wc(ws, r, C1+1, text,
       font=_f(8, False, DARK_GRAY), bg=bg, align=AL_W, border=BORDER_ALL)
    merge(ws, r, C1+1, r, CE)


def build(wb, result):
    ws = wb.create_sheet("Analyst Panel")
    hide_gridlines(ws)
    ws.freeze_panes = "B4"
    ws.sheet_properties.tabColor = BLUE_ACC

    for col, w in _COLS.items():
        ws.column_dimensions[col].width = w

    signals     = result.get("agent_signals") or {}
    pm          = result.get("portfolio_decision")
    ticker      = result.get("ticker", "")
    price       = (result.get("company_data") or {}).get("key_metrics", {}).get("current_price")

    r = 1
    spacer(ws, r, 8); r += 1

    # Title
    ws.row_dimensions[r].height = 28
    wc(ws, r, C1, f"  ANALYST PANEL  —  {ticker}",
       font=_f(14, True, WHITE), bg=NAVY_DARK, align=AL_L)
    merge(ws, r, C1, r, CE)
    r += 1

    # Column headers
    col_hdr(ws, r,
            ["Agent", "Type", "Signal", "Confidence", "Score", "/Max", "% Score", "Action", "Price Target"],
            C1)
    r += 1

    spacer(ws, r, 4); r += 1

    # ── Agents 1–9 ────────────────────────────────────────────────────────────
    sec_hdr(ws, r, "Investment Analysts", C1, CE); r += 1

    for idx, aid in enumerate(AGENT_ORDER):
        sig = signals.get(aid)
        if sig is None:
            continue
        alt = (idx % 2 == 0)
        r = _write_agent_row(ws, r, sig, aid, alt)

        # Reasoning row
        if sig.reasoning:
            reasoning = sig.reasoning[:300] + ("…" if len(sig.reasoning) > 300 else "")
            _write_detail_row(ws, r, "Reasoning", reasoning, alt)
            r += 1

        # Key risks row
        if sig.key_risks:
            risks_text = "  ·  ".join(sig.key_risks[:3])
            _write_detail_row(ws, r, "Key Risks", risks_text, alt)
            r += 1

    spacer(ws, r); r += 1

    # ── Portfolio Manager ─────────────────────────────────────────────────────
    if pm:
        sec_hdr(ws, r, "Portfolio Manager  (Final Decision)", C1, CE, bg=NAVY_MED); r += 1

        r = _write_agent_row(ws, r, pm, "portfolio_manager", alt=False)

        # PM reasoning
        if pm.reasoning:
            ws.row_dimensions[r].height = 70
            wc(ws, r, C1,   "  Analysis",
               font=_f(8, True, DARK_GRAY), bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
            wc(ws, r, C1+1, pm.reasoning,
               font=_f(8, False, DARK_GRAY), bg=BLUE_TINT, align=AL_W, border=BORDER_ALL)
            merge(ws, r, C1+1, r, CE)
            r += 1

        # PM key risks
        if pm.key_risks:
            risks_text = "  ·  ".join(pm.key_risks[:3])
            _write_detail_row(ws, r, "Key Risks", risks_text, alt=False)
            r += 1

    spacer(ws, r); r += 1

    # ── Score legend ──────────────────────────────────────────────────────────
    sec_hdr(ws, r, "Score Legend", C1, CE, bg=NAVY_MED); r += 1
    ws.row_dimensions[r].height = 16

    for c, (label, bg) in enumerate(
        [("≥ 65%  Bullish", GREEN), ("45–64%  Neutral", AMBER), ("< 45%  Bearish", RED)],
        start=C1
    ):
        wc(ws, r, c, label, font=_f(9, True, WHITE), bg=bg, align=AL_C, border=BORDER_ALL)

    note = (
        "Score % = raw score / max × 100.  "
        "Ben Graham scores /30; Fundamentals normalized 0–1; all others /20."
    )
    wc(ws, r, C1+3, note,
       font=_f(8, False, DARK_GRAY), bg=GRAY_LIGHT, align=AL_L, border=BORDER_ALL)
    merge(ws, r, C1+3, r, CE)
    r += 1

    # Footer
    spacer(ws, r); r += 1
    ws.row_dimensions[r].height = 14
    wc(ws, r, C1,
       f"  Financial Researcher  |  {ticker}  |  Confidence = agent's self-reported conviction (0–100%)",
       font=_f(8, False, MID_GRAY), bg=GRAY_LIGHT, align=AL_L)
    merge(ws, r, C1, r, CE)

    return ws
