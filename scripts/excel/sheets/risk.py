"""
Risk Profile Sheet
Risk metrics, scores, signal, position sizing recommendation.

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
    "B": 26,  # metric name
    "C": 14,  # value
    "D": 8,   # score pts
    "E": 6,   # /max
    "F": 18,  # assessment
    "G": 2,   # spacer
}
C1 = 2
CE = 6   # F


RISK_SCORES_MAP = {
    "volatility": ("Annualized Volatility",  "vol"),
    "beta":       ("Beta vs S&P 500",        "beta"),
    "drawdown":   ("Max Drawdown (2Y)",       "mdd"),
    "sharpe":     ("Sharpe Ratio (proxy)",    "sharpe"),
    "kelly":      ("Kelly Max Position Size", "kelly"),
}


def _risk_level(total, total_max=20):
    ratio = total / total_max if total_max else 0
    if ratio >= 0.60: return "LOW RISK",      GREEN
    if ratio >= 0.40: return "MODERATE RISK", AMBER
    return "HIGH RISK", RED


def _fmt_pct(v):  return f"{v:.1%}" if v is not None else "—"
def _fmt_num(v, d=2): return f"{v:.{d}f}" if v is not None else "—"
def _fmt_pts(v):  return f"{v:.1f}" if isinstance(v, (int, float)) else "—"


def build(wb, result):
    ws = wb.create_sheet("Risk Profile")
    hide_gridlines(ws)
    ws.sheet_properties.tabColor = RED

    for col, w in _COLS.items():
        ws.column_dimensions[col].width = w

    ticker      = result.get("ticker", "")
    risk_m      = result.get("risk_metrics") or {}
    signals     = result.get("agent_signals") or {}
    rm_sig      = signals.get("risk_manager")
    rm_scores   = (rm_sig.scores if rm_sig else None) or {}

    total     = rm_scores.get("total",     10.0)
    total_max = rm_scores.get("total_max", 20.0)
    risk_label, risk_color = _risk_level(total or 0, total_max or 20)

    vol     = risk_m.get("annualized_volatility")
    ann_ret = risk_m.get("annualized_return")
    beta    = risk_m.get("beta")
    mdd     = risk_m.get("max_drawdown")
    sharpe  = risk_m.get("sharpe_proxy")
    kelly   = risk_m.get("kelly_fraction")
    max_pos = risk_m.get("max_position_size_pct")

    r = 1
    spacer(ws, r, 8); r += 1

    # Title
    ws.row_dimensions[r].height = 28
    wc(ws, r, C1, f"  RISK PROFILE  —  {ticker}",
       font=_f(14, True, WHITE), bg=NAVY_DARK, align=AL_L)
    merge(ws, r, C1, r, CE)
    r += 1
    spacer(ws, r, 4); r += 1

    # ── Risk Signal box ───────────────────────────────────────────────────────
    ws.row_dimensions[r].height = 32
    wc(ws, r, C1, f"  {risk_label}",
       font=_f(15, True, WHITE), bg=risk_color, align=AL_L)
    merge(ws, r, C1, r, C1+1)
    wc(ws, r, C1+2, f"Risk Score:  {total:.1f} / {total_max:.0f}",
       font=_f(12, True, WHITE), bg=risk_color, align=AL_C)
    wc(ws, r, C1+3, f"Max Position:  {max_pos:.1f}% of portfolio" if max_pos is not None else "Max Position: N/A",
       font=_f(11, True, WHITE), bg=risk_color, align=AL_C)
    merge(ws, r, C1+3, r, CE)
    r += 1
    spacer(ws, r); r += 1

    # ── Risk Metrics table ────────────────────────────────────────────────────
    sec_hdr(ws, r, "Risk Metrics  (scored /4 each)", C1, CE); r += 1

    col_hdr(ws, r, ["Metric", "Value", "Score", "/Max", "Assessment"], C1); r += 1

    metric_rows = [
        ("Annualized Volatility",  _fmt_pct(vol),    "volatility", "vol"),
        ("Beta vs S&P 500",        _fmt_num(beta, 3),"beta",       "beta"),
        ("Max Drawdown (2Y)",      _fmt_pct(mdd),    "drawdown",   "mdd"),
        ("Sharpe Ratio (proxy)",   _fmt_num(sharpe,2),"sharpe",    "sharpe"),
        ("Kelly Max Position",     _fmt_pct(kelly),  "kelly",      "kelly"),
    ]

    for idx, (lbl, val_s, score_key, _) in enumerate(metric_rows):
        alt    = (idx % 2 == 0)
        bg     = GRAY_LIGHT if alt else WHITE
        sc_blk = rm_scores.get(score_key) or {}
        pts    = sc_blk.get("score") if isinstance(sc_blk, dict) else None
        m_max  = sc_blk.get("max",  4)  if isinstance(sc_blk, dict) else 4
        # detail label
        detail_lbl = ""
        if isinstance(sc_blk, dict) and sc_blk.get("detail"):
            for dk, dv in sc_blk["detail"].items():
                if isinstance(dv, dict) and dv.get("value"):
                    detail_lbl = str(dv["value"])
                    break

        ws.row_dimensions[r].height = 16
        wc(ws, r, C1,   lbl,          font=_f(9, True),        bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
        wc(ws, r, C1+1, val_s,        font=_f(9),              bg=bg,        align=AL_R,  border=BORDER_ALL)
        wc(ws, r, C1+2, _fmt_pts(pts),font=_f(9, True),        bg=bg,        align=AL_C,  border=BORDER_ALL)
        wc(ws, r, C1+3, f"/{m_max}",  font=_f(8, False, DARK_GRAY), bg=bg,  align=AL_L,  border=BORDER_ALL)
        wc(ws, r, C1+4, detail_lbl or "—",
           font=_f(8, False, DARK_GRAY), bg=bg, align=AL_L, border=BORDER_ALL)
        r += 1

    # Total row
    ws.row_dimensions[r].height = 18
    wc(ws, r, C1,   "TOTAL RISK SCORE",
       font=_f(10, True, WHITE), bg=NAVY_DARK, align=AL_LI, border=BORDER_ALL)
    wc(ws, r, C1+1, "—",
       font=_f(10, True, WHITE), bg=NAVY_DARK, align=AL_R, border=BORDER_ALL)
    wc(ws, r, C1+2, _fmt_pts(total),
       font=_f(10, True, WHITE), bg=NAVY_DARK, align=AL_C, border=BORDER_ALL)
    wc(ws, r, C1+3, f"/{total_max:.0f}",
       font=_f(10, True, WHITE), bg=NAVY_DARK, align=AL_L, border=BORDER_ALL)
    wc(ws, r, C1+4, f"{total/total_max*100:.0f}% — {risk_label}" if total_max else risk_label,
       font=_f(10, True, WHITE), bg=risk_color, align=AL_C, border=BORDER_ALL)
    r += 1

    spacer(ws, r); r += 1

    # ── Key risk figures ──────────────────────────────────────────────────────
    sec_hdr(ws, r, "Key Risk Figures", C1, CE); r += 1

    figures = [
        ("Annualized Return (2Y)",   _fmt_pct(ann_ret),           "Historical price-based annualized return"),
        ("Annualized Volatility",    _fmt_pct(vol),               "Std dev of daily returns × √252"),
        ("Beta vs S&P 500",          _fmt_num(beta, 3),           "Cov(stock, SPY) / Var(SPY); 2-year daily"),
        ("Max Drawdown (2Y)",        _fmt_pct(mdd),               "Worst peak-to-trough decline"),
        ("Sharpe Ratio",             _fmt_num(sharpe, 3),         "(Ann. return − risk-free rate) / Ann. vol"),
        ("Kelly Fraction",           _fmt_pct(kelly),             "Simplified Kelly: (μ − rf) / σ²; capped at 25%"),
        ("Recommended Max Position", f"{max_pos:.1f}%" if max_pos is not None else "—",
                                                                  "Max % of portfolio via Kelly criterion"),
    ]

    for idx, (lbl, val_s, note) in enumerate(figures):
        alt = (idx % 2 == 0)
        bg  = GRAY_LIGHT if alt else WHITE
        ws.row_dimensions[r].height = 15
        wc(ws, r, C1,   lbl,   font=_f(9, True),        bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
        wc(ws, r, C1+1, val_s, font=_f(9),              bg=bg,        align=AL_R,  border=BORDER_ALL)
        wc(ws, r, C1+2, note,  font=_f(8, False, DARK_GRAY), bg=bg,  align=AL_L,  border=BORDER_ALL)
        merge(ws, r, C1+2, r, CE)
        r += 1

    spacer(ws, r); r += 1

    # ── Risk Manager reasoning ────────────────────────────────────────────────
    if rm_sig and rm_sig.reasoning:
        sec_hdr(ws, r, "Risk Manager Assessment", C1, CE); r += 1
        ws.row_dimensions[r].height = 70
        wc(ws, r, C1, rm_sig.reasoning,
           font=_f(9, False, BLACK), bg=WHITE, align=AL_W, border=BORDER_ALL)
        merge(ws, r, C1, r, CE)
        r += 1

    # Key risks
    if rm_sig and rm_sig.key_risks:
        sec_hdr(ws, r, "Risk Flags", C1, CE); r += 1
        for i, risk in enumerate(rm_sig.key_risks[:3], 1):
            ws.row_dimensions[r].height = 20
            wc(ws, r, C1,   f"  {i}.",
               font=_f(9, True, RED), bg=GRAY_LIGHT, align=AL_C, border=BORDER_ALL)
            wc(ws, r, C1+1, risk,
               font=_f(9, False, BLACK), bg=GRAY_LIGHT, align=AL_W, border=BORDER_ALL)
            merge(ws, r, C1+1, r, CE)
            r += 1

    spacer(ws, r); r += 1

    # Footer
    ws.row_dimensions[r].height = 14
    wc(ws, r, C1,
       "  Past risk metrics do not guarantee future risk levels. Kelly criterion is a sizing guide only.",
       font=_f(8, False, MID_GRAY), bg=GRAY_LIGHT, align=AL_L)
    merge(ws, r, C1, r, CE)

    return ws
