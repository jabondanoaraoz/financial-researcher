"""
Summary Sheet
Company overview, key metrics, final investment decision, consensus, and top risks.

Author: Joaquin Abondano w/ Claude Code
"""

from datetime import date

from ..styles import (
    _f, fill, merge, wc, sec_hdr, spacer, hide_gridlines,
    NAVY_DARK, NAVY_MED, GREEN, AMBER, RED,
    BLUE_TINT, GRAY_LIGHT, WHITE, BLACK, DARK_GRAY, MID_GRAY,
    SIGNAL_BG, SIGNAL_TEXT, ACTION_TEXT,
    AL_L, AL_LI, AL_C, AL_R, AL_W,
    BORDER_ALL,
)

# Column layout: A(spacer) B-H(content) I(spacer)
_COLS = {"A": 2, "B": 24, "C": 14, "D": 14, "E": 14, "F": 14, "G": 14, "H": 14, "I": 2}
C1 = 2   # first content col (B)
CE = 8   # last content col (H)
N_COLS = CE - C1 + 1   # 7


# ── Value formatters ──────────────────────────────────────────────────────────

def _price(v):
    return f"${v:,.2f}" if v is not None else "—"

def _mcap(v):
    if v is None: return "—"
    if v >= 1e12: return f"${v/1e12:.1f}T"
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"

def _mult(v):
    return f"{v:.1f}x" if v is not None else "—"

def _pct(v):
    return f"{v:.1f}%" if v is not None else "—"

def _num(v, d=2):
    return f"{v:.{d}f}" if v is not None else "—"


# ── Sheet builder ─────────────────────────────────────────────────────────────

def build(wb, result):
    ws = wb.create_sheet("Summary")
    hide_gridlines(ws)

    for col, w in _COLS.items():
        ws.column_dimensions[col].width = w
    ws.sheet_properties.tabColor = NAVY_DARK

    # Data extraction
    info  = (result.get("company_data") or {}).get("info")  or {}
    km    = (result.get("company_data") or {}).get("key_metrics") or {}
    cons  = result.get("consensus") or {}
    pm    = result.get("portfolio_decision")
    ticker = result.get("ticker", "")

    name        = info.get("name") or ticker
    sector      = info.get("sector")    or "—"
    industry    = info.get("industry")  or "—"
    exchange    = info.get("exchange")  or "—"
    country     = info.get("country")   or "—"
    description = info.get("description") or ""
    employees   = info.get("employees")

    price     = km.get("current_price")
    mktcap    = info.get("market_cap")
    pe        = km.get("pe_ratio")
    fwd_pe    = km.get("forward_pe")
    pb        = km.get("pb_ratio")
    ps        = km.get("ps_ratio")
    ev_ebitda = km.get("ev_ebitda")
    beta      = km.get("beta")
    w52h      = km.get("52w_high")
    w52l      = km.get("52w_low")
    div_yield = km.get("dividend_yield")

    r = 1

    # ── Top margin ────────────────────────────────────────────────────────────
    spacer(ws, r, 8); r += 1

    # ── Company header band ───────────────────────────────────────────────────
    ws.row_dimensions[r].height = 36
    wc(ws, r, C1, f"  {name}   ·   {ticker}",
       font=_f(20, True, WHITE), bg=NAVY_DARK, align=AL_L)
    merge(ws, r, C1, r, CE)
    r += 1

    ws.row_dimensions[r].height = 16
    subtitle = f"  {sector}  |  {industry}  |  {exchange}  |  {country}"
    if employees:
        subtitle += f"  |  {employees:,} employees"
    wc(ws, r, C1, subtitle, font=_f(9, False, WHITE), bg=NAVY_MED, align=AL_L)
    merge(ws, r, C1, r, CE)
    r += 1

    # ── Business description ──────────────────────────────────────────────────
    if description:
        ws.row_dimensions[r].height = 52
        desc = description[:480] + ("…" if len(description) > 480 else "")
        wc(ws, r, C1, desc, font=_f(8, False, DARK_GRAY), bg=GRAY_LIGHT, align=AL_W)
        merge(ws, r, C1, r, CE)
        r += 1

    spacer(ws, r); r += 1

    # ── Key Metrics ───────────────────────────────────────────────────────────
    sec_hdr(ws, r, "Key Metrics", C1, CE); r += 1

    # 3 pairs per row: (label, value) × 3 → cols B-C, D-E, F-G; H empty
    metric_rows = [
        [("Current Price", _price(price)),  ("Market Cap",  _mcap(mktcap)),  ("EV/EBITDA", _mult(ev_ebitda))],
        [("P/E  (TTM)",    _mult(pe)),       ("P/E  (Fwd)",  _mult(fwd_pe)),  ("P/S Ratio", _mult(ps))],
        [("P/B Ratio",     _mult(pb)),       ("Beta",        _num(beta, 2)),  ("Div Yield", _pct(div_yield))],
        [("52W High",      _price(w52h)),    ("52W Low",     _price(w52l)),   ("",          "")],
    ]
    for mrow in metric_rows:
        ws.row_dimensions[r].height = 16
        for i, (lbl, val) in enumerate(mrow):
            c = C1 + i * 2       # B=2, D=4, F=6
            wc(ws, r, c,   lbl, font=_f(9, True),  bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
            wc(ws, r, c+1, val, font=_f(9, False), bg=WHITE,     align=AL_R,  border=BORDER_ALL)
        wc(ws, r, CE, "", bg=WHITE, border=BORDER_ALL)   # H col placeholder
        r += 1

    spacer(ws, r); r += 1

    # ── Investment Decision ───────────────────────────────────────────────────
    sec_hdr(ws, r, "Investment Decision", C1, CE); r += 1

    if pm:
        signal     = pm.signal
        action     = pm.target_action
        conf       = pm.confidence
        cons_sc    = pm.scores.get("consensus") or {}
        conviction = (cons_sc.get("conviction") or "—").upper()
        pt_low     = cons_sc.get("target_price_low")
        pt_high    = cons_sc.get("target_price_high")
        pt_mid     = pm.price_target
        sig_bg     = SIGNAL_BG.get(signal, NAVY_MED)

        # Signal box row
        ws.row_dimensions[r].height = 34
        wc(ws, r, C1,   SIGNAL_TEXT.get(signal, signal.upper()),
           font=_f(17, True, WHITE), bg=sig_bg, align=AL_C)
        merge(ws, r, C1, r, C1+1)
        wc(ws, r, C1+2, f"Action: {ACTION_TEXT.get(action, action.upper())}",
           font=_f(11, True, WHITE), bg=sig_bg, align=AL_C)
        merge(ws, r, C1+2, r, C1+3)
        wc(ws, r, C1+4, f"Conviction:  {conviction}",
           font=_f(11, True, WHITE), bg=sig_bg, align=AL_C)
        merge(ws, r, C1+4, r, C1+5)
        wc(ws, r, CE, f"Conf:  {conf:.0%}",
           font=_f(10, True, WHITE), bg=sig_bg, align=AL_C)
        r += 1

        # Price target row
        ws.row_dimensions[r].height = 20
        if pt_low is not None and pt_high is not None:
            try:
                pt_text = f"Price Target:   ${float(pt_low):,.2f}  —  ${float(pt_high):,.2f}"
                if price and pt_mid:
                    upside = (pt_mid / price - 1) * 100
                    pt_text += f"     |     Implied Upside:  {upside:+.1f}%"
            except (TypeError, ValueError):
                pt_text = "Price Target: N/A"
        elif pt_mid:
            pt_text = f"Fair Value Estimate:   ${pt_mid:,.2f}"
            if price:
                upside = (pt_mid / price - 1) * 100
                pt_text += f"     |     Implied Upside:  {upside:+.1f}%"
        else:
            pt_text = "Price Target: N/A"

        wc(ws, r, C1, pt_text, font=_f(10, False, DARK_GRAY), bg=GRAY_LIGHT, align=AL_C)
        merge(ws, r, C1, r, CE)
        r += 1

    spacer(ws, r); r += 1

    # ── Analyst Consensus ─────────────────────────────────────────────────────
    sec_hdr(ws, r, "Analyst Consensus  (9 Analysts + Portfolio Manager)", C1, CE); r += 1

    n_bull = cons.get("bullish", 0)
    n_neut = cons.get("neutral", 0)
    n_bear = cons.get("bearish", 0)
    avg    = cons.get("avg_score_20") or 0.0

    ws.row_dimensions[r].height = 24
    wc(ws, r, C1,   "▲  Bullish", font=_f(10, True, WHITE), bg=GREEN, align=AL_C, border=BORDER_ALL)
    wc(ws, r, C1+1, str(n_bull),  font=_f(16, True, GREEN), bg=WHITE, align=AL_C, border=BORDER_ALL)
    wc(ws, r, C1+2, "●  Neutral", font=_f(10, True, WHITE), bg=AMBER, align=AL_C, border=BORDER_ALL)
    wc(ws, r, C1+3, str(n_neut),  font=_f(16, True, AMBER), bg=WHITE, align=AL_C, border=BORDER_ALL)
    wc(ws, r, C1+4, "▼  Bearish", font=_f(10, True, WHITE), bg=RED,   align=AL_C, border=BORDER_ALL)
    wc(ws, r, C1+5, str(n_bear),  font=_f(16, True, RED),   bg=WHITE, align=AL_C, border=BORDER_ALL)
    wc(ws, r, CE,   f"Avg Score:  {avg:.1f} / 20",
       font=_f(10, True), bg=BLUE_TINT, align=AL_C, border=BORDER_ALL)
    r += 1

    spacer(ws, r); r += 1

    # ── Portfolio Manager reasoning ───────────────────────────────────────────
    if pm and pm.reasoning:
        sec_hdr(ws, r, "Portfolio Manager Analysis", C1, CE); r += 1
        ws.row_dimensions[r].height = 90
        wc(ws, r, C1, pm.reasoning, font=_f(9, False, BLACK), bg=WHITE, align=AL_W)
        merge(ws, r, C1, r, CE)
        r += 1

    spacer(ws, r); r += 1

    # ── Key Risks ─────────────────────────────────────────────────────────────
    if pm and pm.key_risks:
        sec_hdr(ws, r, "Key Risks", C1, CE); r += 1
        for i, risk in enumerate(pm.key_risks[:3], 1):
            ws.row_dimensions[r].height = 22
            wc(ws, r, C1,   f"  {i}.",
               font=_f(9, True, RED), bg=GRAY_LIGHT, align=AL_C, border=BORDER_ALL)
            wc(ws, r, C1+1, risk,
               font=_f(9, False, BLACK), bg=GRAY_LIGHT, align=AL_W, border=BORDER_ALL)
            merge(ws, r, C1+1, r, CE)
            r += 1

    spacer(ws, r); r += 1

    # ── Footer ────────────────────────────────────────────────────────────────
    ws.row_dimensions[r].height = 16
    footer = (
        f"  Financial Researcher  |  Analysis Date: {date.today().strftime('%B %d, %Y')}"
        "  |  AI-Powered Multi-Agent Analysis  |  For informational purposes only"
    )
    wc(ws, r, C1, footer, font=_f(8, False, MID_GRAY), bg=GRAY_LIGHT, align=AL_L)
    merge(ws, r, C1, r, CE)

    return ws
