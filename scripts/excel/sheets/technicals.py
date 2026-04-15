"""
Technical Indicators Sheet
Raw indicator values, scores, and technical signal from the Technicals agent.

Author: Joaquin Abondano w/ Claude Code
"""

from ..styles import (
    _f, fill, merge, wc, sec_hdr, col_hdr, spacer, hide_gridlines,
    NAVY_DARK, NAVY_MED, BLUE_ACC, GREEN, AMBER, RED,
    BLUE_TINT, GRAY_LIGHT, WHITE, BLACK, DARK_GRAY, MID_GRAY,
    SIGNAL_BG, SIGNAL_TEXT,
    AL_L, AL_LI, AL_C, AL_R, AL_W,
    BORDER_ALL,
)

_COLS = {
    "A": 2,   # spacer
    "B": 26,  # indicator name
    "C": 14,  # value
    "D": 8,   # score pts
    "E": 6,   # /max
    "F": 20,  # assessment / detail
    "G": 2,   # spacer
}
C1 = 2
CE = 6   # col F


def _pts(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "—"

def _signal_to_color(signal):
    return {
        "bullish": GREEN,
        "neutral": AMBER,
        "bearish": RED,
    }.get(signal, NAVY_MED)


def build(wb, result):
    ws = wb.create_sheet("Technical Indicators")
    hide_gridlines(ws)
    ws.sheet_properties.tabColor = AMBER

    for col, w in _COLS.items():
        ws.column_dimensions[col].width = w

    ticker  = result.get("ticker", "")
    signals = result.get("agent_signals") or {}
    ta_sig  = signals.get("technicals")
    ta_sc   = (ta_sig.scores if ta_sig else {}) or {}

    total     = ta_sc.get("total",     10.0)
    total_max = ta_sc.get("total_max", 20.0)
    signal    = ta_sig.signal    if ta_sig else "neutral"
    action    = ta_sig.target_action if ta_sig else "hold"
    conf      = ta_sig.confidence    if ta_sig else 0.5

    # Derive signal color
    sig_bg = SIGNAL_BG.get(signal, NAVY_MED)

    r = 1
    spacer(ws, r, 8); r += 1

    # Title
    ws.row_dimensions[r].height = 28
    wc(ws, r, C1, f"  TECHNICAL INDICATORS  —  {ticker}",
       font=_f(14, True, WHITE), bg=NAVY_DARK, align=AL_L)
    merge(ws, r, C1, r, CE)
    r += 1
    spacer(ws, r, 4); r += 1

    # ── Signal Summary Box ────────────────────────────────────────────────────
    ws.row_dimensions[r].height = 32
    wc(ws, r, C1, f"  {signal.upper()}",
       font=_f(15, True, WHITE), bg=sig_bg, align=AL_L)
    merge(ws, r, C1, r, C1 + 1)
    wc(ws, r, C1 + 2,
       f"Score:  {total:.1f} / {total_max:.0f}  ({total/total_max*100:.0f}%)",
       font=_f(12, True, WHITE), bg=sig_bg, align=AL_C)
    wc(ws, r, C1 + 3,
       f"Action: {action.upper()}  |  Confidence: {conf:.0%}",
       font=_f(11, True, WHITE), bg=sig_bg, align=AL_C)
    merge(ws, r, C1 + 3, r, CE)
    r += 1
    spacer(ws, r); r += 1

    # ── Indicator Table ───────────────────────────────────────────────────────
    sec_hdr(ws, r, "Indicator Breakdown  (scored /4 or /5 each)", C1, CE); r += 1
    col_hdr(ws, r, ["Indicator", "Reading", "Score", "/Max", "Assessment"], C1); r += 1

    # Define indicator groups in display order
    INDICATORS = [
        ("RSI  (14-period)",          "rsi",             4),
        ("MACD  (12/26/9)",           "macd",            4),
        ("Bollinger Bands  (20, 2σ)", "bollinger",       4),
        ("Moving Averages  (SMA50/200)", "moving_averages", 5),
        ("Volume Trend",              "volume",          3),
    ]

    for idx, (display_name, key, max_pts) in enumerate(INDICATORS):
        alt    = (idx % 2 == 0)
        bg     = GRAY_LIGHT if alt else WHITE
        grp    = ta_sc.get(key) or {}
        pts    = grp.get("score")
        m_max  = grp.get("max", max_pts)
        detail = grp.get("detail") or {}

        # Build reading / assessment from detail dict
        readings = []
        for dk, dv in detail.items():
            if isinstance(dv, dict) and dv.get("value"):
                readings.append(str(dv["value"]))

        reading_str    = readings[0] if readings else "—"
        assessment_str = " | ".join(readings) if len(readings) > 1 else (readings[0] if readings else "—")

        # Score color
        if pts is not None and m_max > 0:
            ratio = pts / m_max
            sc_bg = GREEN if ratio >= 0.65 else (AMBER if ratio >= 0.40 else RED)
        else:
            sc_bg = GRAY_LIGHT

        ws.row_dimensions[r].height = 16
        wc(ws, r, C1,     display_name, font=_f(9, True),  bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
        wc(ws, r, C1 + 1, reading_str,  font=_f(8.5),      bg=bg,        align=AL_R,  border=BORDER_ALL)
        wc(ws, r, C1 + 2, _pts(pts),    font=_f(9, True, WHITE), bg=sc_bg, align=AL_C, border=BORDER_ALL)
        wc(ws, r, C1 + 3, f"/{m_max}",  font=_f(8, False, DARK_GRAY), bg=bg, align=AL_L, border=BORDER_ALL)
        wc(ws, r, C1 + 4, assessment_str,
           font=_f(8, False, DARK_GRAY), bg=bg, align=AL_W, border=BORDER_ALL)
        r += 1

        # Sub-rows for indicators with multiple sub-metrics (moving averages)
        if key == "moving_averages" and len(readings) > 1:
            for sub_key, sub_val in detail.items():
                if not isinstance(sub_val, dict):
                    continue
                sub_pts  = sub_val.get("pts")
                sub_max  = sub_val.get("max")
                sub_lbl  = sub_key.replace("_", " ").title()
                sub_note = sub_val.get("value", "")
                ws.row_dimensions[r].height = 14
                wc(ws, r, C1,     f"   ↳ {sub_lbl}", font=_f(8, False, DARK_GRAY), bg=WHITE, align=AL_LI, border=BORDER_ALL)
                wc(ws, r, C1 + 1, sub_note,           font=_f(8),                  bg=WHITE, align=AL_R,  border=BORDER_ALL)
                wc(ws, r, C1 + 2, _pts(sub_pts),      font=_f(8),                  bg=WHITE, align=AL_C,  border=BORDER_ALL)
                wc(ws, r, C1 + 3, f"/{sub_max}" if sub_max else "—",
                   font=_f(8, False, DARK_GRAY), bg=WHITE, align=AL_L, border=BORDER_ALL)
                wc(ws, r, C1 + 4, "", font=_f(8), bg=WHITE, align=AL_L, border=BORDER_ALL)
                r += 1

    # Total row
    ws.row_dimensions[r].height = 18
    wc(ws, r, C1, "TOTAL TECHNICAL SCORE",
       font=_f(10, True, WHITE), bg=NAVY_DARK, align=AL_LI, border=BORDER_ALL)
    wc(ws, r, C1 + 1, "—",
       font=_f(10, True, WHITE), bg=NAVY_DARK, align=AL_C, border=BORDER_ALL)
    wc(ws, r, C1 + 2, _pts(total),
       font=_f(10, True, WHITE), bg=NAVY_DARK, align=AL_C, border=BORDER_ALL)
    wc(ws, r, C1 + 3, f"/{total_max:.0f}",
       font=_f(10, True, WHITE), bg=NAVY_DARK, align=AL_L, border=BORDER_ALL)
    wc(ws, r, C1 + 4,
       f"{total/total_max*100:.0f}% — {signal.upper()}",
       font=_f(10, True, WHITE), bg=sig_bg, align=AL_C, border=BORDER_ALL)
    r += 1
    spacer(ws, r); r += 1

    # ── Score Legend ──────────────────────────────────────────────────────────
    sec_hdr(ws, r, "Score Legend", C1, CE); r += 1

    legend = [
        ("≥ 65% of max",  "Bullish",  GREEN, "Indicator supports higher prices"),
        ("40% – 64%",     "Neutral",  AMBER, "Mixed or inconclusive signal"),
        ("< 40% of max",  "Bearish",  RED,   "Indicator suggests downside pressure"),
    ]
    for threshold, label, bg_c, note in legend:
        ws.row_dimensions[r].height = 15
        wc(ws, r, C1,     threshold, font=_f(8.5, True, WHITE), bg=bg_c, align=AL_C, border=BORDER_ALL)
        wc(ws, r, C1 + 1, label,     font=_f(8.5, True, WHITE), bg=bg_c, align=AL_C, border=BORDER_ALL)
        wc(ws, r, C1 + 2, note,      font=_f(8, False, DARK_GRAY), bg=WHITE, align=AL_L, border=BORDER_ALL)
        merge(ws, r, C1 + 2, r, CE)
        r += 1

    spacer(ws, r); r += 1

    # ── Reasoning ─────────────────────────────────────────────────────────────
    if ta_sig and ta_sig.reasoning:
        sec_hdr(ws, r, "Technical Analysis Commentary", C1, CE); r += 1
        ws.row_dimensions[r].height = 70
        wc(ws, r, C1, ta_sig.reasoning,
           font=_f(9, False, BLACK), bg=WHITE, align=AL_W, border=BORDER_ALL)
        merge(ws, r, C1, r, CE)
        r += 1

    if ta_sig and ta_sig.key_risks:
        sec_hdr(ws, r, "Key Risks", C1, CE); r += 1
        for i, risk in enumerate(ta_sig.key_risks[:3], 1):
            ws.row_dimensions[r].height = 18
            wc(ws, r, C1,     f"  {i}.", font=_f(9, True, RED), bg=GRAY_LIGHT, align=AL_C, border=BORDER_ALL)
            wc(ws, r, C1 + 1, risk,      font=_f(9, False, BLACK), bg=GRAY_LIGHT, align=AL_W, border=BORDER_ALL)
            merge(ws, r, C1 + 1, r, CE)
            r += 1

    spacer(ws, r); r += 1

    # Footer
    ws.row_dimensions[r].height = 14
    wc(ws, r, C1,
       "  Technical indicators are backward-looking and do not guarantee future price direction.",
       font=_f(8, False, MID_GRAY), bg=GRAY_LIGHT, align=AL_L)
    merge(ws, r, C1, r, CE)

    return ws
