"""
Valuation Sheet
DCF, peer multiples, Graham Number — methods, assumptions, price target bridge.

Author: Joaquin Abondano w/ Claude Code
"""

import re

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
    "B": 24,  # label
    "C": 13,  # value 1
    "D": 13,  # value 2
    "E": 13,  # value 3
    "F": 13,  # value 4
    "G": 13,  # value 5
    "H": 2,   # spacer
}
C1 = 2
CE = 7   # G


# ── Parsers for valuation agent's detail strings ──────────────────────────────

def _parse_fv(s):
    """Extract '$XXX.XX' from a detail string."""
    m = re.match(r'\$([0-9,]+\.?\d*)', s or "")
    return float(m.group(1).replace(",", "")) if m else None


def _parse_dcf(s):
    """Extract WACC, CAGR, FCF margin from DCF detail string."""
    wacc = re.search(r'WACC=([\d.]+)%', s or "")
    cagr = re.search(r'CAGR=([\d.]+)%', s or "")
    fcfm = re.search(r'FCF margin=([\d.]+)%', s or "")
    return (
        float(wacc.group(1)) if wacc else None,
        float(cagr.group(1)) if cagr else None,
        float(fcfm.group(1)) if fcfm else None,
    )


def _parse_peers(s):
    """Extract median EV/EBITDA and peer count."""
    mult = re.search(r'EV/EBITDA ([\d.]+)x', s or "")
    cnt  = re.search(r'(\d+) peers?', s or "")
    return (
        float(mult.group(1)) if mult else None,
        int(cnt.group(1))    if cnt  else None,
    )


def _parse_graham(s):
    """Extract EPS and BVPS."""
    eps  = re.search(r'EPS=\$([\d.]+)', s or "")
    bvps = re.search(r'BVPS=\$([\d.]+)', s or "")
    return (
        float(eps.group(1))  if eps  else None,
        float(bvps.group(1)) if bvps else None,
    )


def _fmt_price(v): return f"${v:,.2f}" if v is not None else "—"
def _fmt_pct(v):   return f"{v:.1f}%"  if v is not None else "—"
def _fmt_mult(v):  return f"{v:.1f}x"  if v is not None else "—"
def _fmt_int(v):   return str(v)        if v is not None else "—"


# ── Label-value helper ────────────────────────────────────────────────────────

def _kv(ws, r, label, value, c_lbl=C1, c_val=None, span=1, alt=False):
    if c_val is None: c_val = c_lbl + 1
    bg_l = BLUE_TINT
    bg_v = GRAY_LIGHT if alt else WHITE
    ws.row_dimensions[r].height = 16
    wc(ws, r, c_lbl, label, font=_f(9, True),  bg=bg_l, align=AL_LI, border=BORDER_ALL)
    wc(ws, r, c_val, value, font=_f(9, False), bg=bg_v, align=AL_R,  border=BORDER_ALL)
    if span > 1:
        merge(ws, r, c_val, r, c_val + span - 1)
    return r + 1


def build(wb, result):
    ws = wb.create_sheet("Valuation")
    hide_gridlines(ws)
    ws.sheet_properties.tabColor = GREEN

    for col, w in _COLS.items():
        ws.column_dimensions[col].width = w

    ticker  = result.get("ticker", "")
    km      = (result.get("company_data") or {}).get("key_metrics") or {}
    price   = km.get("current_price")
    signals = result.get("agent_signals") or {}
    pm      = result.get("portfolio_decision")

    val_sig  = signals.get("valuation")
    val_sc   = (val_sig.scores if val_sig else None) or {}
    fv       = val_sc.get("fair_value")
    upside   = val_sc.get("upside_pct")
    methods  = (val_sc.get("methods") or {}).get("detail") or {}

    dcf_detail    = methods.get("dcf")    or {}
    peer_detail   = methods.get("peer_multiples") or {}
    graham_detail = methods.get("graham_number")  or {}
    consensus_det = methods.get("consensus_fair_value") or {}

    dcf_str    = dcf_detail.get("value",    "")
    peer_str   = peer_detail.get("value",   "")
    graham_str = graham_detail.get("value", "")

    dcf_fv    = _parse_fv(dcf_str)
    peer_fv   = _parse_fv(peer_str)
    graham_fv = _parse_fv(graham_str)
    wacc, cagr, fcf_margin = _parse_dcf(dcf_str)
    ev_ebitda_med, n_peers  = _parse_peers(peer_str)
    eps, bvps               = _parse_graham(graham_str)

    # Weight logic mirrors valuation agent
    has_dcf  = dcf_fv    is not None
    has_peer = peer_fv   is not None
    has_gn   = graham_fv is not None
    w_dcf    = (0.50 if has_peer else 0.65) if has_dcf else 0
    w_peer   = 0.30 if has_peer else 0
    w_gn     = (0.20 if has_dcf else 0.35) if has_gn else 0

    r = 1
    spacer(ws, r, 8); r += 1

    # Title
    ws.row_dimensions[r].height = 28
    wc(ws, r, C1, f"  VALUATION ANALYSIS  —  {ticker}",
       font=_f(14, True, WHITE), bg=NAVY_DARK, align=AL_L)
    merge(ws, r, C1, r, CE)
    r += 1
    spacer(ws, r, 4); r += 1

    # ── Valuation Methods table ───────────────────────────────────────────────
    sec_hdr(ws, r, "Valuation Methods", C1, CE); r += 1

    col_hdr(ws, r, ["Method", "Fair Value", "Weight", "Notes", "", ""], C1); r += 1

    def method_row(label, fv_val, weight, notes, alt=False):
        nonlocal r
        row_bg = GRAY_LIGHT if alt else WHITE
        ws.row_dimensions[r].height = 16
        wc(ws, r, C1,   label,             font=_f(9, True),        bg=row_bg,   align=AL_LI, border=BORDER_ALL)
        wc(ws, r, C1+1, _fmt_price(fv_val),font=_f(9, False),       bg=row_bg,   align=AL_R,  border=BORDER_ALL)
        wc(ws, r, C1+2, _fmt_pct(weight*100) if weight else "—",
           font=_f(9),  bg=row_bg, align=AL_C, border=BORDER_ALL)
        wc(ws, r, C1+3, notes,             font=_f(8, False, DARK_GRAY), bg=row_bg, align=AL_L, border=BORDER_ALL)
        merge(ws, r, C1+3, r, CE)
        r += 1

    method_row("DCF  (Revenue-based)",
               dcf_fv, w_dcf,
               f"WACC = {_fmt_pct(wacc)}  |  Revenue CAGR = {_fmt_pct(cagr)}  |  FCF Margin = {_fmt_pct(fcf_margin)}",
               alt=False)
    method_row("Peer EV/EBITDA Multiples",
               peer_fv, w_peer,
               f"Median EV/EBITDA = {_fmt_mult(ev_ebitda_med)}  |  {_fmt_int(n_peers)} peers",
               alt=True)
    method_row("Graham Number",
               graham_fv, w_gn,
               f"EPS = {_fmt_price(eps)}  |  BVPS = {_fmt_price(bvps)}",
               alt=False)

    # Consensus row (highlighted)
    ws.row_dimensions[r].height = 18
    fv_bg = SIGNAL_BG.get((val_sig.signal if val_sig else "neutral"), NAVY_MED)
    wc(ws, r, C1,   "Weighted Consensus Fair Value",
       font=_f(10, True, WHITE), bg=fv_bg, align=AL_LI, border=BORDER_ALL)
    wc(ws, r, C1+1, _fmt_price(fv),
       font=_f(10, True, WHITE), bg=fv_bg, align=AL_R, border=BORDER_ALL)
    wc(ws, r, C1+2, "100%",
       font=_f(10, True, WHITE), bg=fv_bg, align=AL_C, border=BORDER_ALL)
    upside_s = f"Implied Upside: {upside:+.1f}%" if upside is not None else "Upside: N/A"
    wc(ws, r, C1+3, upside_s,
       font=_f(10, True, WHITE), bg=fv_bg, align=AL_L, border=BORDER_ALL)
    merge(ws, r, C1+3, r, CE)
    r += 1

    spacer(ws, r); r += 1

    # ── Price target bridge ───────────────────────────────────────────────────
    sec_hdr(ws, r, "Price Target Bridge", C1, CE); r += 1

    col_hdr(ws, r, ["", "Current Price", "Fair Value (FRA)", "Upside / Down", "PM Target Low", "PM Target High"], C1)
    r += 1

    ws.row_dimensions[r].height = 18
    cons_sc   = (pm.scores.get("consensus") or {}) if pm else {}
    pt_low    = cons_sc.get("target_price_low")
    pt_high   = cons_sc.get("target_price_high")

    sig_bg = SIGNAL_BG.get((pm.signal if pm else "neutral"), NAVY_MED)

    wc(ws, r, C1,   ticker,              font=_f(9, True),        bg=BLUE_TINT,  align=AL_C, border=BORDER_ALL)
    wc(ws, r, C1+1, _fmt_price(price),   font=_f(9),              bg=WHITE,      align=AL_R, border=BORDER_ALL)
    wc(ws, r, C1+2, _fmt_price(fv),      font=_f(9, True),        bg=WHITE,      align=AL_R, border=BORDER_ALL)
    wc(ws, r, C1+3,
       f"{upside:+.1f}%" if upside is not None else "—",
       font=_f(9, True, WHITE), bg=sig_bg, align=AL_C, border=BORDER_ALL)
    try:
        wc(ws, r, C1+4, f"${float(pt_low):,.2f}" if pt_low else "—",
           font=_f(9), bg=WHITE, align=AL_R, border=BORDER_ALL)
        wc(ws, r, C1+5, f"${float(pt_high):,.2f}" if pt_high else "—",
           font=_f(9), bg=WHITE, align=AL_R, border=BORDER_ALL)
    except (TypeError, ValueError):
        wc(ws, r, C1+4, "—", font=_f(9), bg=WHITE, align=AL_R, border=BORDER_ALL)
        wc(ws, r, C1+5, "—", font=_f(9), bg=WHITE, align=AL_R, border=BORDER_ALL)
    r += 1

    spacer(ws, r); r += 1

    # ── DCF Assumptions ───────────────────────────────────────────────────────
    sec_hdr(ws, r, "DCF Model Assumptions", C1, CE); r += 1

    col_hdr(ws, r, ["Assumption", "Value Used", "Note", "", "", ""], C1); r += 1

    for lbl, val, note, alt in [
        ("Revenue CAGR",     _fmt_pct(cagr),       "Historical CAGR clamped to 0–40%",         False),
        ("FCF Margin",       _fmt_pct(fcf_margin),  "OCF + Capex / Revenue (most recent year)",  True),
        ("WACC",             _fmt_pct(wacc),        "Cost of equity (CAPM) + debt weighted avg", False),
        ("Terminal Growth",  "3.0%",                "Long-run nominal GDP assumption",           True),
        ("Projection Years", "5",                   "5-year explicit forecast period",           False),
        ("Tax Rate",         "21.0%",               "US corporate tax rate",                     True),
        ("ERP",              "5.5%",                "Equity risk premium",                       False),
    ]:
        row_bg = GRAY_LIGHT if alt else WHITE
        ws.row_dimensions[r].height = 15
        wc(ws, r, C1,   lbl,  font=_f(9, True),        bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
        wc(ws, r, C1+1, val,  font=_f(9, False),       bg=row_bg,    align=AL_R,  border=BORDER_ALL)
        wc(ws, r, C1+2, note, font=_f(8, False, DARK_GRAY), bg=row_bg, align=AL_L, border=BORDER_ALL)
        merge(ws, r, C1+2, r, CE)
        r += 1

    spacer(ws, r); r += 1

    # ── Peer Universe ─────────────────────────────────────────────────────────
    peers_data = (result.get("company_data") or {}).get("key_metrics")
    # peers_data is not directly available in result; it's nested in base_data
    # We can try to get it from the valuation agent string or skip
    # Let's show what peers were used if we can extract from the orchestrator result
    # The orchestrator doesn't return peers_data directly, so we'll just show a note
    sec_hdr(ws, r, "Peer Universe  (EV/EBITDA Multiples)", C1, CE); r += 1

    ws.row_dimensions[r].height = 15
    note_text = (
        f"Peer median EV/EBITDA: {_fmt_mult(ev_ebitda_med)}  |  "
        f"{_fmt_int(n_peers)} peers used  |  "
        "Peer tickers configured in templates/peers_mapping.json"
    )
    wc(ws, r, C1, note_text,
       font=_f(8, False, DARK_GRAY), bg=GRAY_LIGHT, align=AL_L, border=BORDER_ALL)
    merge(ws, r, C1, r, CE)
    r += 1

    spacer(ws, r); r += 1

    # ── Valuation Agent reasoning ─────────────────────────────────────────────
    if val_sig and val_sig.reasoning:
        sec_hdr(ws, r, "Valuation Analyst Commentary", C1, CE); r += 1
        ws.row_dimensions[r].height = 80
        wc(ws, r, C1, val_sig.reasoning,
           font=_f(9, False, BLACK), bg=WHITE, align=AL_W, border=BORDER_ALL)
        merge(ws, r, C1, r, CE)
        r += 1

    spacer(ws, r); r += 1

    # Footer
    ws.row_dimensions[r].height = 14
    wc(ws, r, C1,
       "  DCF and valuation estimates are model outputs and do not constitute investment advice.",
       font=_f(8, False, MID_GRAY), bg=GRAY_LIGHT, align=AL_L)
    merge(ws, r, C1, r, CE)

    return ws
