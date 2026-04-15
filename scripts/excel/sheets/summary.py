"""
Summary Sheet
Investment decision, consensus, valuation, risk summary, and PM reasoning.
Placed last — synthesizes all prior sheets.

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
    "B": 26,  # label
    "C": 14,  # value 1
    "D": 14,  # value 2
    "E": 14,  # value 3
    "F": 14,  # value 4
    "G": 2,   # spacer
}
C1 = 2   # col B
CE = 6   # col F


def _price(v):  return f"${v:,.2f}" if v is not None else "—"
def _mcap(v):
    if v is None: return "—"
    return f"${v/1e12:.2f}T" if v >= 1e12 else f"${v/1e9:.1f}B"
def _pct(v):    return f"{v:.1f}%" if v is not None else "—"
def _mult(v):   return f"{v:.1f}x"  if v is not None else "—"
def _num(v, d=2): return f"{v:.{d}f}" if v is not None else "—"


ACTION_BG = {
    "buy":   GREEN,
    "hold":  AMBER,
    "sell":  RED,
    "short": RED,
    "cover": GREEN,
}


def _kv(ws, r, label, value, alt=False, span=None):
    """Simple label → value row, optionally with right-side span."""
    bg_l = BLUE_TINT
    bg_v = GRAY_LIGHT if alt else WHITE
    ws.row_dimensions[r].height = 17
    wc(ws, r, C1,     label, font=_f(9, True),  bg=bg_l, align=AL_LI, border=BORDER_ALL)
    wc(ws, r, C1 + 1, value, font=_f(9, False), bg=bg_v, align=AL_R,  border=BORDER_ALL)
    if span:
        merge(ws, r, C1 + 1, r, CE)
    return r + 1


def build(wb, result):
    ws = wb.create_sheet("Summary")
    hide_gridlines(ws)
    ws.sheet_properties.tabColor = NAVY_DARK

    for col, w in _COLS.items():
        ws.column_dimensions[col].width = w

    ticker  = result.get("ticker", "")
    info    = (result.get("company_data") or {}).get("info") or {}
    km      = (result.get("company_data") or {}).get("key_metrics") or {}
    pm      = result.get("portfolio_decision")
    risk_m  = result.get("risk_metrics") or {}
    signals = result.get("agent_signals") or {}
    cons    = result.get("consensus") or {}
    rm_sig  = signals.get("risk_manager")
    val_sig = signals.get("valuation")

    price   = km.get("current_price")
    mkt_cap = info.get("market_cap")
    fv      = (val_sig.scores.get("fair_value") if val_sig else None)
    upside  = (val_sig.scores.get("upside_pct") if val_sig else None)

    pm_signal = pm.signal if pm else "neutral"
    pm_action = pm.target_action if pm else "hold"
    pm_conf   = pm.confidence if pm else 0.5
    pm_target = pm.price_target if pm else None

    pm_cons   = (pm.scores.get("consensus") or {}) if pm else {}
    pt_low    = pm_cons.get("target_price_low")
    pt_high   = pm_cons.get("target_price_high")
    conviction = pm_cons.get("conviction", "medium")

    sig_bg  = SIGNAL_BG.get(pm_signal, NAVY_MED)
    act_bg  = ACTION_BG.get(pm_action, NAVY_MED)

    max_pos = risk_m.get("max_position_size_pct")

    r = 1
    spacer(ws, r, 8); r += 1

    # ── Company Header ─────────────────────────────────────────────────────
    ws.row_dimensions[r].height = 30
    name = info.get("name", ticker)
    wc(ws, r, C1, f"  {name}  ({ticker})",
       font=_f(14, True, WHITE), bg=NAVY_DARK, align=AL_L)
    merge(ws, r, C1, r, CE)
    r += 1

    ws.row_dimensions[r].height = 16
    sector   = info.get("sector",   "—")
    industry = info.get("industry", "—")
    exchange = info.get("exchange", "—")
    wc(ws, r, C1, f"  {sector}  |  {industry}  |  {exchange}",
       font=_f(9, False, MID_GRAY), bg=GRAY_LIGHT, align=AL_L)
    merge(ws, r, C1, r, CE)
    r += 1
    spacer(ws, r, 4); r += 1

    # ── Investment Decision ────────────────────────────────────────────────
    sec_hdr(ws, r, "Investment Decision", C1, CE); r += 1

    # Big signal row
    ws.row_dimensions[r].height = 36
    wc(ws, r, C1, f"  {pm_signal.upper()}",
       font=_f(18, True, WHITE), bg=sig_bg, align=AL_L)
    merge(ws, r, C1, r, C1 + 1)
    wc(ws, r, C1 + 2, f"Action:  {pm_action.upper()}",
       font=_f(14, True, WHITE), bg=act_bg, align=AL_C)
    wc(ws, r, C1 + 3, f"Conviction:  {conviction.upper()}",
       font=_f(12, True, WHITE), bg=sig_bg, align=AL_C)
    wc(ws, r, C1 + 4, f"Confidence:  {pm_conf:.0%}",
       font=_f(12, True, WHITE), bg=sig_bg, align=AL_C)
    r += 1

    # Price snapshot
    ws.row_dimensions[r].height = 18
    wc(ws, r, C1,     "Current Price",         font=_f(9, True),        bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
    wc(ws, r, C1 + 1, _price(price),           font=_f(9),              bg=WHITE,     align=AL_R,  border=BORDER_ALL)
    wc(ws, r, C1 + 2, "Market Cap",            font=_f(9, True),        bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
    wc(ws, r, C1 + 3, _mcap(mkt_cap),          font=_f(9),              bg=WHITE,     align=AL_R,  border=BORDER_ALL)
    wc(ws, r, C1 + 4, "Max Position Size",     font=_f(9, True),        bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
    wc(ws, r, C1 + 5 if C1 + 5 <= CE else CE,
       f"{max_pos:.1f}% of portfolio" if max_pos else "—",
       font=_f(9),              bg=WHITE,     align=AL_R,  border=BORDER_ALL)
    r += 1

    # Price targets
    ws.row_dimensions[r].height = 18
    wc(ws, r, C1,     "Consensus Fair Value",  font=_f(9, True),        bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
    wc(ws, r, C1 + 1, _price(fv),              font=_f(9),              bg=WHITE,     align=AL_R,  border=BORDER_ALL)
    wc(ws, r, C1 + 2, "Implied Upside",        font=_f(9, True),        bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
    upside_bg = GREEN if (upside and upside >= 10) else (AMBER if (upside and upside >= 0) else RED)
    wc(ws, r, C1 + 3, f"{upside:+.1f}%" if upside is not None else "—",
       font=_f(9, True, WHITE), bg=upside_bg, align=AL_C, border=BORDER_ALL)
    wc(ws, r, C1 + 4, "PM Target Range",       font=_f(9, True),        bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
    range_s = f"{_price(pt_low)} – {_price(pt_high)}" if (pt_low and pt_high) else "—"
    wc(ws, r, C1 + 5 if C1 + 5 <= CE else CE,
       range_s, font=_f(9), bg=WHITE, align=AL_R, border=BORDER_ALL)
    r += 1
    spacer(ws, r); r += 1

    # ── Analyst Consensus ─────────────────────────────────────────────────
    sec_hdr(ws, r, "Analyst Consensus  (9 agents)", C1, CE); r += 1

    n_bull  = cons.get("bullish", 0)
    n_neut  = cons.get("neutral", 0)
    n_bear  = cons.get("bearish", 0)
    avg_sc  = cons.get("avg_score_20")
    n_total = n_bull + n_neut + n_bear or 1

    ws.row_dimensions[r].height = 22
    wc(ws, r, C1,     f"  BULLISH  ({n_bull})",
       font=_f(11, True, WHITE), bg=GREEN, align=AL_C, border=BORDER_ALL)
    merge(ws, r, C1, r, C1 + 1)
    wc(ws, r, C1 + 2, f"  NEUTRAL  ({n_neut})",
       font=_f(11, True, WHITE), bg=AMBER, align=AL_C, border=BORDER_ALL)
    merge(ws, r, C1 + 2, r, C1 + 2)
    wc(ws, r, C1 + 3, f"  BEARISH  ({n_bear})",
       font=_f(11, True, WHITE), bg=RED, align=AL_C, border=BORDER_ALL)
    merge(ws, r, C1 + 3, r, C1 + 3)
    wc(ws, r, C1 + 4, f"Avg Score  {avg_sc:.1f} / 20" if avg_sc else "Avg Score  —",
       font=_f(10, True, WHITE), bg=NAVY_MED, align=AL_C, border=BORDER_ALL)
    merge(ws, r, C1 + 4, r, CE)
    r += 1
    spacer(ws, r); r += 1

    # ── Risk Summary ──────────────────────────────────────────────────────
    sec_hdr(ws, r, "Risk Summary", C1, CE); r += 1

    # Risk level from risk manager
    rm_sc   = (rm_sig.scores if rm_sig else {}) or {}
    rm_total     = rm_sc.get("total",     10.0)
    rm_total_max = rm_sc.get("total_max", 20.0)
    ratio = (rm_total / rm_total_max) if rm_total_max else 0
    if ratio >= 0.60:
        risk_label, risk_bg = "LOW RISK",      GREEN
    elif ratio >= 0.40:
        risk_label, risk_bg = "MODERATE RISK", AMBER
    else:
        risk_label, risk_bg = "HIGH RISK",     RED

    ws.row_dimensions[r].height = 22
    wc(ws, r, C1, f"  {risk_label}",
       font=_f(12, True, WHITE), bg=risk_bg, align=AL_L)
    merge(ws, r, C1, r, C1 + 1)
    wc(ws, r, C1 + 2,
       f"Risk Score:  {rm_total:.1f} / {rm_total_max:.0f}",
       font=_f(10, True, WHITE), bg=risk_bg, align=AL_C)
    wc(ws, r, C1 + 3,
       f"Max Position:  {max_pos:.1f}% of portfolio" if max_pos else "Max Position: N/A",
       font=_f(10, True, WHITE), bg=risk_bg, align=AL_C)
    merge(ws, r, C1 + 3, r, CE)
    r += 1

    risk_rows = [
        ("Annualized Volatility",  _pct(risk_m.get("annualized_volatility", 0) * 100 if risk_m.get("annualized_volatility") else None), False),
        ("Annualized Return (2Y)", _pct(risk_m.get("annualized_return", 0) * 100    if risk_m.get("annualized_return") else None),    True),
        ("Beta vs S&P 500",        _num(risk_m.get("beta"), 3),                                                                        False),
        ("Max Drawdown (2Y)",      _pct(risk_m.get("max_drawdown", 0) * 100         if risk_m.get("max_drawdown") else None),          True),
        ("Sharpe Ratio",           _num(risk_m.get("sharpe_proxy"), 2),                                                                False),
        ("Kelly Fraction",         _pct(risk_m.get("kelly_fraction", 0) * 100       if risk_m.get("kelly_fraction") else None),        True),
    ]
    for lbl, val, alt in risk_rows:
        r = _kv(ws, r, lbl, val, alt=alt)

    spacer(ws, r); r += 1

    # ── Valuation Summary ─────────────────────────────────────────────────
    sec_hdr(ws, r, "Valuation Summary", C1, CE); r += 1

    val_sc = (val_sig.scores if val_sig else {}) or {}
    methods_detail = ((val_sc.get("methods") or {}).get("detail") or {})

    val_items = [
        ("Current Price",           _price(price),  False),
        ("Consensus Fair Value",    _price(fv),      True),
        ("Implied Upside",          f"{upside:+.1f}%" if upside is not None else "—", False),
    ]
    if methods_detail.get("dcf"):
        val_items.append(("DCF Fair Value",   methods_detail["dcf"].get("value", "—"),          True))
    if methods_detail.get("peer_multiples"):
        val_items.append(("Peer Multiples FV", methods_detail["peer_multiples"].get("value", "—"), False))
    if methods_detail.get("graham_number"):
        val_items.append(("Graham Number",    methods_detail["graham_number"].get("value", "—"), True))

    for lbl, val, alt in val_items:
        r = _kv(ws, r, lbl, str(val), alt=alt, span=CE)

    spacer(ws, r); r += 1

    # ── PM Reasoning ──────────────────────────────────────────────────────
    if pm and pm.reasoning:
        sec_hdr(ws, r, "Portfolio Manager Thesis", C1, CE); r += 1
        ws.row_dimensions[r].height = 85
        wc(ws, r, C1, pm.reasoning,
           font=_f(9, False, BLACK), bg=WHITE, align=AL_W, border=BORDER_ALL)
        merge(ws, r, C1, r, CE)
        r += 1

    # ── Key Risks ─────────────────────────────────────────────────────────
    all_risks = []
    if pm and pm.key_risks:
        all_risks = pm.key_risks[:4]
    elif rm_sig and rm_sig.key_risks:
        all_risks = rm_sig.key_risks[:3]

    if all_risks:
        sec_hdr(ws, r, "Key Risks", C1, CE); r += 1
        for i, risk in enumerate(all_risks, 1):
            ws.row_dimensions[r].height = 20
            wc(ws, r, C1,     f"  {i}.",
               font=_f(9, True, RED), bg=GRAY_LIGHT, align=AL_C, border=BORDER_ALL)
            wc(ws, r, C1 + 1, risk,
               font=_f(9, False, BLACK), bg=GRAY_LIGHT, align=AL_W, border=BORDER_ALL)
            merge(ws, r, C1 + 1, r, CE)
            r += 1

    spacer(ws, r); r += 1

    # Footer
    ws.row_dimensions[r].height = 14
    wc(ws, r, C1,
       "  This report is generated by an AI system and does not constitute investment advice.",
       font=_f(8, False, MID_GRAY), bg=GRAY_LIGHT, align=AL_L)
    merge(ws, r, C1, r, CE)

    return ws
