"""
DCF Model Sheet
Year-by-year discounted cash flow model with assumptions, EV bridge, and sensitivity table.

Author: Joaquin Abondano w/ Claude Code
"""

import math

from ..styles import (
    _f, fill, merge, wc, sec_hdr, col_hdr, spacer, hide_gridlines,
    NAVY_DARK, NAVY_MED, BLUE_ACC, GREEN, AMBER, RED,
    BLUE_TINT, GRAY_LIGHT, WHITE, BLACK, DARK_GRAY, MID_GRAY,
    SIGNAL_BG,
    AL_L, AL_LI, AL_C, AL_R, AL_W,
    BORDER_ALL,
)

_COLS = {
    "A": 2,   # spacer
    "B": 22,  # label / year
    "C": 13,  # base / yr 1
    "D": 13,  # yr 2
    "E": 13,  # yr 3
    "F": 13,  # yr 4
    "G": 13,  # yr 5
    "H": 2,   # spacer
}
C1 = 2   # col B
CE = 7   # col G

EQUITY_RISK_PREMIUM = 0.055
TAX_RATE            = 0.21
TERMINAL_GROWTH     = 0.03
PROJECTION_YEARS    = 5


# ── Formatters ────────────────────────────────────────────────────────────────

def _b(v):
    if v is None: return "—"
    return f"${v / 1e9:.1f}B"

def _pct(v, d=1):
    if v is None: return "—"
    return f"{v:.{d}f}%"

def _price(v):
    if v is None: return "—"
    return f"${v:,.2f}"

def _num(v, d=2):
    if v is None: return "—"
    return f"{v:.{d}f}x"


# ── DCF helpers (replicate valuation agent logic) ─────────────────────────────

def _get(df, row, col_idx=0):
    if df is None or df.empty or row not in df.index:
        return None
    try:
        v = float(df.iloc[df.index.get_loc(row), col_idx])
        return None if math.isnan(v) else v
    except (TypeError, ValueError, IndexError):
        return None


def _annual_series(df, row, n=5):
    if df is None or df.empty or row not in df.index:
        return []
    out = []
    for col in list(df.columns)[:n]:
        try:
            v = float(df.loc[row, col])
            if not math.isnan(v):
                out.append(v)
        except (TypeError, ValueError):
            pass
    return out


def _compute_wacc(key_metrics, financials, risk_free):
    beta = key_metrics.get("beta") or 1.0
    ke   = risk_free + beta * EQUITY_RISK_PREMIUM
    bs   = financials.get("balance_sheet")
    inc  = financials.get("income_statement")
    if bs is not None and not bs.empty and inc is not None and not inc.empty:
        equity  = _get(bs, "Stockholders Equity") or _get(bs, "Total Stockholders Equity") or 0
        debt    = _get(bs, "Total Debt") or 0
        int_exp = abs(_get(inc, "Interest Expense") or 0)
        kd_pre  = int_exp / debt if debt > 0 else 0
        kd      = kd_pre * (1 - TAX_RATE)
        V       = equity + debt
        if V > 0:
            return (equity / V) * ke + (debt / V) * kd
    return ke


def _build_dcf(key_metrics, financials, risk_free):
    """
    Returns a dict with all DCF components needed for the sheet,
    including year-by-year projections.
    Returns None if data is insufficient.
    """
    inc = financials.get("income_statement")
    cf  = financials.get("cash_flow")
    bs  = financials.get("balance_sheet")

    revenues = _annual_series(inc, "Total Revenue", n=4)
    if len(revenues) < 2:
        return None

    # Revenue CAGR (most recent to oldest available)
    n_years = len(revenues) - 1
    cagr    = (revenues[0] / revenues[-1]) ** (1 / n_years) - 1
    cagr    = min(max(cagr, 0.0), 0.40)

    # FCF margin from most recent year
    ocf   = _get(cf, "Operating Cash Flow")
    capex = _get(cf, "Capital Expenditure") or _get(cf, "Capital Expenditures")
    rev0  = revenues[0]
    if ocf and capex and rev0 > 0:
        fcf_margin = (ocf + capex) / rev0
    else:
        fcf_margin = 0.05

    wacc = max(_compute_wacc(key_metrics, financials, risk_free), 0.06)

    # Historical data
    hist_years   = []
    for col in list(inc.columns)[:4]:
        try:
            hist_years.append(col.year)
        except AttributeError:
            hist_years.append(str(col)[:4])

    # Year-by-year projections
    projections = []
    rev_proj    = rev0
    pv_fcfs     = 0.0
    for yr in range(1, PROJECTION_YEARS + 1):
        rev_proj  *= (1 + cagr)
        fcf_proj   = rev_proj * fcf_margin
        disc_f     = 1 / ((1 + wacc) ** yr)
        pv_fcf     = fcf_proj * disc_f
        pv_fcfs   += pv_fcf
        projections.append({
            "year":         yr,
            "revenue":      rev_proj,
            "growth_pct":   cagr * 100,
            "fcf":          fcf_proj,
            "disc_factor":  disc_f,
            "pv_fcf":       pv_fcf,
        })

    # Terminal value
    fcf_terminal = rev_proj * fcf_margin * (1 + TERMINAL_GROWTH)
    tv           = fcf_terminal / (wacc - TERMINAL_GROWTH)
    pv_tv        = tv / ((1 + wacc) ** PROJECTION_YEARS)
    tv_pct       = pv_tv / (pv_fcfs + pv_tv) * 100 if (pv_fcfs + pv_tv) > 0 else None

    # EV bridge
    ev_implied = pv_fcfs + pv_tv
    cash       = (_get(bs, "Cash Cash Equivalents And Short Term Investments")
                  or _get(bs, "Cash And Cash Equivalents") or 0)
    debt       = _get(bs, "Total Debt") or 0
    equity_val = ev_implied + cash - debt

    shares = key_metrics.get("shares_outstanding")
    per_share = (equity_val / shares) if (shares and shares > 0) else None

    return {
        "cagr":        cagr * 100,
        "fcf_margin":  fcf_margin * 100,
        "wacc":        wacc * 100,
        "risk_free":   risk_free * 100,
        "rev0":        rev0,
        "hist_years":  hist_years,
        "revenues":    revenues,
        "projections": projections,
        "pv_fcfs":     pv_fcfs,
        "pv_tv":       pv_tv,
        "tv_pct":      tv_pct,
        "ev_implied":  ev_implied,
        "cash":        cash,
        "debt":        debt,
        "equity_val":  equity_val,
        "shares":      shares,
        "per_share":   per_share,
    }


def _sensitivity(key_metrics, financials, risk_free, base_cagr, base_wacc):
    """
    Build a WACC vs CAGR sensitivity grid.
    Returns {(wacc_pct, cagr_pct): per_share_value}
    """
    wacc_range = [base_wacc - 2, base_wacc - 1, base_wacc, base_wacc + 1, base_wacc + 2]
    cagr_range = [base_cagr - 2, base_cagr - 0, base_cagr + 2, base_cagr + 4, base_cagr + 6]
    # Clamp to valid ranges
    wacc_range = [max(4.0, w) for w in wacc_range]
    cagr_range = [max(0.0, min(40.0, c)) for c in cagr_range]

    inc = financials.get("income_statement")
    cf  = financials.get("cash_flow")
    bs  = financials.get("balance_sheet")

    revenues = _annual_series(inc, "Total Revenue", n=4)
    if len(revenues) < 1:
        return {}, wacc_range, cagr_range

    rev0      = revenues[0]
    ocf       = _get(cf, "Operating Cash Flow")
    capex     = _get(cf, "Capital Expenditure") or _get(cf, "Capital Expenditures")
    fcf_margin = (ocf + capex) / rev0 if (ocf and capex and rev0 > 0) else 0.05
    cash  = (_get(bs, "Cash Cash Equivalents And Short Term Investments")
             or _get(bs, "Cash And Cash Equivalents") or 0)
    debt  = _get(bs, "Total Debt") or 0
    shares = key_metrics.get("shares_outstanding") or 1

    grid = {}
    for w_pct in wacc_range:
        for c_pct in cagr_range:
            w = w_pct / 100
            c = c_pct / 100
            rev_p   = rev0
            pv_f    = 0.0
            for yr in range(1, PROJECTION_YEARS + 1):
                rev_p  *= (1 + c)
                fcf_p   = rev_p * fcf_margin
                pv_f   += fcf_p / ((1 + w) ** yr)
            fcf_t = rev_p * fcf_margin * (1 + TERMINAL_GROWTH)
            if w > TERMINAL_GROWTH:
                tv    = fcf_t / (w - TERMINAL_GROWTH)
                pv_tv = tv / ((1 + w) ** PROJECTION_YEARS)
            else:
                pv_tv = 0
            eq  = pv_f + pv_tv + cash - debt
            ps  = eq / shares
            grid[(round(w_pct, 1), round(c_pct, 1))] = round(ps, 2)

    return grid, [round(w, 1) for w in wacc_range], [round(c, 1) for c in cagr_range]


# ── Sheet builder ─────────────────────────────────────────────────────────────

def build(wb, result):
    ws = wb.create_sheet("DCF Model")
    hide_gridlines(ws)
    ws.sheet_properties.tabColor = GREEN

    for col, w in _COLS.items():
        ws.column_dimensions[col].width = w

    ticker    = result.get("ticker", "")
    km        = (result.get("company_data") or {}).get("key_metrics") or {}
    fins      = result.get("financials") or {}
    risk_free = result.get("risk_free_rate") or 0.043
    price     = km.get("current_price")

    dcf = _build_dcf(km, fins, risk_free)

    r = 1
    spacer(ws, r, 8); r += 1

    # Title
    ws.row_dimensions[r].height = 28
    wc(ws, r, C1, f"  DCF MODEL  —  {ticker}",
       font=_f(14, True, WHITE), bg=NAVY_DARK, align=AL_L)
    merge(ws, r, C1, r, CE)
    r += 1
    spacer(ws, r, 4); r += 1

    if dcf is None:
        wc(ws, r, C1, "Insufficient financial data to build DCF model.",
           font=_f(10, False, RED), bg=WHITE, align=AL_C)
        merge(ws, r, C1, r, CE)
        return ws

    # ── Assumptions ───────────────────────────────────────────────────────────
    sec_hdr(ws, r, "Model Assumptions", C1, CE); r += 1

    assumptions = [
        ("Revenue CAGR (historical)",  _pct(dcf["cagr"]),       "Historical CAGR from income statement (clamped 0–40%)", False),
        ("FCF Margin",                 _pct(dcf["fcf_margin"]),  "OCF + CapEx / Revenue (most recent year)",              True),
        ("WACC",                       _pct(dcf["wacc"]),        "CAPM cost of equity blended with after-tax cost of debt",False),
        ("Risk-Free Rate",             _pct(dcf["risk_free"]),   "10Y US Treasury (FRED)",                                True),
        ("Equity Risk Premium",        _pct(EQUITY_RISK_PREMIUM * 100), "Market ERP assumption",                          False),
        ("Terminal Growth Rate",       _pct(TERMINAL_GROWTH * 100),     "Long-run nominal GDP assumption",                True),
        ("Projection Period",          f"{PROJECTION_YEARS} years",      "Explicit forecast horizon",                     False),
        ("Tax Rate",                   _pct(TAX_RATE * 100),             "US corporate tax rate",                          True),
    ]

    col_hdr(ws, r, ["Assumption", "Value", "Note", "", "", ""], C1); r += 1
    for lbl, val, note, alt in assumptions:
        bg = GRAY_LIGHT if alt else WHITE
        ws.row_dimensions[r].height = 15
        wc(ws, r, C1,     lbl,  font=_f(9, True),        bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
        wc(ws, r, C1 + 1, val,  font=_f(9),              bg=bg,        align=AL_R,  border=BORDER_ALL)
        wc(ws, r, C1 + 2, note, font=_f(8, False, DARK_GRAY), bg=bg,  align=AL_L,  border=BORDER_ALL)
        merge(ws, r, C1 + 2, r, CE)
        r += 1

    spacer(ws, r); r += 1

    # ── Historical Revenue ────────────────────────────────────────────────────
    sec_hdr(ws, r, "Historical Revenue", C1, CE); r += 1

    n_hist = min(len(dcf["hist_years"]), len(dcf["revenues"]))
    hist_hdrs = [""] + [str(y) for y in dcf["hist_years"][:n_hist]] + [""] * (5 - n_hist)
    col_hdr(ws, r, hist_hdrs, C1); r += 1

    ws.row_dimensions[r].height = 17
    wc(ws, r, C1, "Revenue", font=_f(9, True), bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
    for i in range(n_hist):
        wc(ws, r, C1 + 1 + i, _b(dcf["revenues"][i]),
           font=_f(9), bg=WHITE if i % 2 == 0 else GRAY_LIGHT, align=AL_R, border=BORDER_ALL)
    for i in range(n_hist, 4):
        wc(ws, r, C1 + 1 + i, "—", font=_f(9), bg=WHITE, align=AL_R, border=BORDER_ALL)
    r += 1
    spacer(ws, r); r += 1

    # ── 5-Year Projection ─────────────────────────────────────────────────────
    sec_hdr(ws, r, "5-Year Projection", C1, CE); r += 1

    yr_labels = ["Metric"] + [f"Year {p['year']}" for p in dcf["projections"]]
    col_hdr(ws, r, yr_labels, C1); r += 1

    proj_rows = [
        ("Revenue",         "revenue",      _b,    True),
        ("Growth Rate",     "growth_pct",   lambda v: _pct(v, 1), False),
        ("Free Cash Flow",  "fcf",          _b,    True),
        ("Discount Factor", "disc_factor",  lambda v: f"{v:.4f}", False),
        ("PV of FCF",       "pv_fcf",       _b,    True),
    ]

    for lbl, key, fmt, alt in proj_rows:
        bg = GRAY_LIGHT if alt else WHITE
        ws.row_dimensions[r].height = 16
        wc(ws, r, C1, lbl, font=_f(9, True), bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
        for i, p in enumerate(dcf["projections"]):
            wc(ws, r, C1 + 1 + i, fmt(p[key]),
               font=_f(9), bg=bg, align=AL_R, border=BORDER_ALL)
        r += 1

    # Total PV row
    ws.row_dimensions[r].height = 18
    wc(ws, r, C1, "Total PV (FCFs)",
       font=_f(10, True, WHITE), bg=NAVY_DARK, align=AL_LI, border=BORDER_ALL)
    wc(ws, r, C1 + 1, _b(dcf["pv_fcfs"]),
       font=_f(10, True, WHITE), bg=NAVY_DARK, align=AL_R, border=BORDER_ALL)
    merge(ws, r, C1 + 1, r, CE)
    r += 1
    spacer(ws, r); r += 1

    # ── Terminal Value ─────────────────────────────────────────────────────────
    sec_hdr(ws, r, "Terminal Value", C1, CE); r += 1

    tv_items = [
        ("Terminal FCF (Yr 6)",    _b(dcf["projections"][-1]["fcf"] * (1 + TERMINAL_GROWTH)),  "FCF × (1 + g)"),
        ("Terminal Growth Rate",   _pct(TERMINAL_GROWTH * 100),  "g = 3.0% (long-run nominal GDP)"),
        ("WACC − g",               _pct(dcf["wacc"] - TERMINAL_GROWTH * 100), "Denominator of Gordon Growth"),
        ("PV of Terminal Value",   _b(dcf["pv_tv"]), f"TV as {_pct(dcf['tv_pct'])} of total EV"),
    ]
    for i, (lbl, val, note) in enumerate(tv_items):
        alt = (i % 2 == 0)
        bg  = GRAY_LIGHT if alt else WHITE
        ws.row_dimensions[r].height = 16
        wc(ws, r, C1,     lbl,  font=_f(9, True),         bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
        wc(ws, r, C1 + 1, val,  font=_f(9),               bg=bg,        align=AL_R,  border=BORDER_ALL)
        wc(ws, r, C1 + 2, note, font=_f(8, False, DARK_GRAY), bg=bg,   align=AL_L,  border=BORDER_ALL)
        merge(ws, r, C1 + 2, r, CE)
        r += 1

    spacer(ws, r); r += 1

    # ── Enterprise Value Bridge ───────────────────────────────────────────────
    sec_hdr(ws, r, "Equity Value Bridge", C1, CE); r += 1

    bridge = [
        ("(+) PV of FCFs",       dcf["pv_fcfs"],   False),
        ("(+) PV of Terminal Value", dcf["pv_tv"],  True),
        ("= Implied Enterprise Value", dcf["ev_implied"], False),
        ("(+) Cash & Equivalents", dcf["cash"],     True),
        ("(−) Total Debt",       -dcf["debt"],       False),
        ("= Equity Value",       dcf["equity_val"], True),
    ]

    for lbl, val, alt in bridge:
        bg    = GRAY_LIGHT if alt else WHITE
        is_ev = lbl.startswith("=")
        ft    = _f(10, True, WHITE) if is_ev else _f(9, True)
        bg_lbl = NAVY_MED if is_ev else BLUE_TINT
        bg_val = NAVY_MED if is_ev else bg
        ws.row_dimensions[r].height = 17
        wc(ws, r, C1,     lbl,   font=ft,        bg=bg_lbl, align=AL_LI, border=BORDER_ALL)
        wc(ws, r, C1 + 1, _b(val), font=ft,      bg=bg_val, align=AL_R,  border=BORDER_ALL)
        merge(ws, r, C1 + 1, r, CE)
        r += 1

    # Per-share value (highlighted)
    ws.row_dimensions[r].height = 22
    ps = dcf.get("per_share")
    upside = ((ps / price - 1) * 100) if (ps and price and price > 0) else None
    wc(ws, r, C1, "= Intrinsic Value Per Share",
       font=_f(12, True, WHITE), bg=GREEN, align=AL_LI, border=BORDER_ALL)
    wc(ws, r, C1 + 1, _price(ps),
       font=_f(12, True, WHITE), bg=GREEN, align=AL_R, border=BORDER_ALL)
    wc(ws, r, C1 + 2, f"vs market ${price:,.2f}  |  implied upside {upside:+.1f}%" if (ps and upside is not None) else "",
       font=_f(10, True, WHITE), bg=GREEN, align=AL_L, border=BORDER_ALL)
    merge(ws, r, C1 + 2, r, CE)
    r += 1
    spacer(ws, r); r += 1

    # ── Sensitivity Table ─────────────────────────────────────────────────────
    sec_hdr(ws, r, "Sensitivity Analysis  —  Fair Value Per Share", C1, CE); r += 1

    grid, wacc_range, cagr_range = _sensitivity(
        km, fins, risk_free, dcf["cagr"], dcf["wacc"]
    )

    if grid:
        # Header row: CAGR values
        ws.row_dimensions[r].height = 16
        wc(ws, r, C1, "WACC \\ CAGR",
           font=_f(8.5, True, WHITE), bg=NAVY_DARK, align=AL_C, border=BORDER_ALL)
        for i, c in enumerate(cagr_range):
            wc(ws, r, C1 + 1 + i, f"{c:.1f}%",
               font=_f(8.5, True, WHITE), bg=NAVY_DARK, align=AL_C, border=BORDER_ALL)
        r += 1

        for w in wacc_range:
            ws.row_dimensions[r].height = 16
            is_base_w = (round(w, 1) == round(dcf["wacc"], 1))
            wc(ws, r, C1, f"{w:.1f}%",
               font=_f(9, True, WHITE if is_base_w else BLACK),
               bg=NAVY_MED if is_base_w else BLUE_TINT, align=AL_C, border=BORDER_ALL)
            for j, c in enumerate(cagr_range):
                is_base_c = (round(c, 1) == round(dcf["cagr"], 1))
                ps_val = grid.get((round(w, 1), round(c, 1)))
                is_base = is_base_w and is_base_c
                # Color: green if above price, amber near, red below
                if ps_val and price:
                    ratio = ps_val / price
                    cell_bg = GREEN if ratio >= 1.10 else (AMBER if ratio >= 0.90 else RED)
                    if is_base:
                        cell_bg = NAVY_MED
                else:
                    cell_bg = WHITE
                wc(ws, r, C1 + 1 + j,
                   _price(ps_val) if ps_val else "—",
                   font=_f(8.5, True if is_base else False, WHITE if cell_bg in (GREEN, AMBER, RED, NAVY_MED) else BLACK),
                   bg=cell_bg, align=AL_C, border=BORDER_ALL)
            r += 1

        spacer(ws, r); r += 1

    # Footer
    ws.row_dimensions[r].height = 14
    wc(ws, r, C1,
       "  Revenue-based DCF. Sensitivity shows fair value per share. Green = >10% upside | Amber = within 10% | Red = downside.",
       font=_f(8, False, MID_GRAY), bg=GRAY_LIGHT, align=AL_L)
    merge(ws, r, C1, r, CE)

    return ws
