"""
Financials Sheet
Company profile, income statement, cash flow, balance sheet, peer comparison.

Author: Joaquin Abondano w/ Claude Code
"""

import math

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
    "B": 24,  # label / ticker
    "C": 13,  # FY1 / metric 1
    "D": 13,  # FY2 / metric 2
    "E": 13,  # FY3 / metric 3
    "F": 13,  # FY4 / metric 4
    "G": 11,  # margin col / extra metric
    "H": 2,   # spacer
}
C1 = 2   # col B
CE = 7   # col G


# ── Formatters ────────────────────────────────────────────────────────────────

def _b(v):
    """Format number in billions."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"${v / 1e9:.1f}B"

def _pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.1f}%"

def _mult(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.1f}x"

def _num(v, d=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{d}f}"

def _price(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"${v:,.2f}"

def _mcap(v):
    if v is None:
        return "—"
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    return f"${v/1e9:.1f}B"


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def _get(df, row, col_idx=0):
    """Safely get a value from a DataFrame by row label and column index."""
    if df is None or df.empty or row not in df.index:
        return None
    try:
        v = float(df.iloc[df.index.get_loc(row), col_idx])
        return None if math.isnan(v) else v
    except (TypeError, ValueError, IndexError):
        return None


def _margin(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return (numerator / denominator) * 100


def _col_labels(df, n=4):
    """Return up to n column labels as year strings."""
    if df is None or df.empty:
        return []
    labels = []
    for col in list(df.columns)[:n]:
        try:
            labels.append(str(col.year))
        except AttributeError:
            labels.append(str(col)[:4])
    return labels


# ── Sheet builder ─────────────────────────────────────────────────────────────

def build(wb, result):
    ws = wb.create_sheet("Financials")
    hide_gridlines(ws)
    ws.sheet_properties.tabColor = NAVY_MED

    for col, w in _COLS.items():
        ws.column_dimensions[col].width = w

    ticker   = result.get("ticker", "")
    info     = (result.get("company_data") or {}).get("info") or {}
    km       = (result.get("company_data") or {}).get("key_metrics") or {}
    fins     = result.get("financials") or {}
    peers    = result.get("peers_data") or {}

    inc = fins.get("income_statement")
    cf  = fins.get("cash_flow")
    bs  = fins.get("balance_sheet")

    years = _col_labels(inc, 4)

    r = 1
    spacer(ws, r, 8); r += 1

    # ── Company Header ────────────────────────────────────────────────────────
    ws.row_dimensions[r].height = 30
    name = info.get("name", ticker)
    wc(ws, r, C1, f"  {name}  ({ticker})",
       font=_f(14, True, WHITE), bg=NAVY_DARK, align=AL_L)
    merge(ws, r, C1, r, CE)
    r += 1

    ws.row_dimensions[r].height = 18
    sector   = info.get("sector",   "—")
    industry = info.get("industry", "—")
    exchange = info.get("exchange", "—")
    country  = info.get("country",  "—")
    wc(ws, r, C1, f"  {sector}  |  {industry}  |  {exchange}  |  {country}",
       font=_f(9, False, MID_GRAY), bg=GRAY_LIGHT, align=AL_L)
    merge(ws, r, C1, r, CE)
    r += 1

    employees = info.get("employees")
    emp_s = f"{employees:,}" if employees else "—"
    website  = info.get("website", "")
    wc(ws, r, C1, f"  Employees: {emp_s}   |   {website}",
       font=_f(8, False, DARK_GRAY), bg=GRAY_LIGHT, align=AL_L)
    merge(ws, r, C1, r, CE)
    r += 1
    spacer(ws, r, 4); r += 1

    # ── Business Description ──────────────────────────────────────────────────
    desc = info.get("description", "")
    if desc:
        sec_hdr(ws, r, "Business Description", C1, CE); r += 1
        ws.row_dimensions[r].height = 60
        wc(ws, r, C1, desc,
           font=_f(8.5, False, BLACK), bg=WHITE, align=AL_W, border=BORDER_ALL)
        merge(ws, r, C1, r, CE)
        r += 1
        spacer(ws, r, 4); r += 1

    # ── Key Metrics ───────────────────────────────────────────────────────────
    sec_hdr(ws, r, "Key Metrics", C1, CE); r += 1

    price   = km.get("current_price")
    mkt_cap = info.get("market_cap") or km.get("market_cap")
    ev      = km.get("enterprise_value")

    metrics_grid = [
        # Row 1
        [("Current Price",   _price(price)),
         ("Market Cap",      _mcap(mkt_cap)),
         ("Enterprise Value",_mcap(ev))],
        # Row 2
        [("P/E Ratio",       _mult(km.get("pe_ratio"))),
         ("Forward P/E",     _mult(km.get("forward_pe"))),
         ("PEG Ratio",       _num(km.get("peg_ratio"), 2))],
        # Row 3
        [("P/B Ratio",       _mult(km.get("pb_ratio"))),
         ("P/S Ratio",       _mult(km.get("ps_ratio"))),
         ("EV / EBITDA",     _mult(km.get("ev_ebitda")))],
        # Row 4
        [("Beta",            _num(km.get("beta"), 3)),
         ("52W High",        _price(km.get("52w_high"))),
         ("52W Low",         _price(km.get("52w_low")))],
        # Row 5
        [("Dividend Yield",  _pct(km.get("dividend_yield"))),
         ("Short % Float",   _pct(km.get("short_percent_of_float"))),
         ("Shares Out.",     _mcap(km.get("shares_outstanding")))],
    ]

    for row_data in metrics_grid:
        ws.row_dimensions[r].height = 17
        for i, (lbl, val) in enumerate(row_data):
            c_lbl = C1 + i * 2
            c_val = c_lbl + 1
            wc(ws, r, c_lbl, lbl, font=_f(8.5, True), bg=BLUE_TINT, align=AL_L, border=BORDER_ALL)
            wc(ws, r, c_val, val, font=_f(9, False),  bg=WHITE,     align=AL_R, border=BORDER_ALL)
        r += 1

    spacer(ws, r); r += 1

    # ── Income Statement ──────────────────────────────────────────────────────
    sec_hdr(ws, r, "Income Statement  (annual)", C1, CE); r += 1

    yr_headers = ["Metric"] + years + ["Margin (MR)"]
    col_hdr(ws, r, yr_headers, C1); r += 1

    def _inc_row(label, row_key, is_margin=False, margin_row=None, alt=False):
        nonlocal r
        bg = GRAY_LIGHT if alt else WHITE
        ws.row_dimensions[r].height = 16

        wc(ws, r, C1, label, font=_f(9, True), bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)

        for i, _ in enumerate(years):
            v = _get(inc, row_key, i)
            wc(ws, r, C1 + 1 + i,
               _b(v) if not is_margin else _pct(v),
               font=_f(9), bg=bg, align=AL_R, border=BORDER_ALL)

        # Margin column (most recent)
        if margin_row and inc is not None:
            num = _get(inc, row_key, 0)
            den = _get(inc, margin_row, 0)
            mgn = _margin(num, den)
            wc(ws, r, C1 + 1 + len(years),
               _pct(mgn), font=_f(9, True), bg=BLUE_TINT, align=AL_C, border=BORDER_ALL)
        else:
            wc(ws, r, C1 + 1 + len(years), "—",
               font=_f(9), bg=bg, align=AL_C, border=BORDER_ALL)
        r += 1

    _inc_row("Revenue",          "Total Revenue",    alt=False)
    _inc_row("Gross Profit",     "Gross Profit",     margin_row="Total Revenue", alt=True)
    _inc_row("EBITDA",           "EBITDA",           margin_row="Total Revenue", alt=False)
    _inc_row("Operating Income", "Operating Income", margin_row="Total Revenue", alt=True)
    _inc_row("Net Income",       "Net Income",       margin_row="Total Revenue", alt=False)

    spacer(ws, r); r += 1

    # ── Cash Flow ─────────────────────────────────────────────────────────────
    sec_hdr(ws, r, "Cash Flow  (annual)", C1, CE); r += 1
    col_hdr(ws, r, ["Metric"] + years + ["FCF Margin"], C1); r += 1

    def _cf_row(label, row_key, margin_row=None, alt=False):
        nonlocal r
        bg = GRAY_LIGHT if alt else WHITE
        ws.row_dimensions[r].height = 16
        wc(ws, r, C1, label, font=_f(9, True), bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
        for i, _ in enumerate(years):
            v = _get(cf, row_key, i)
            wc(ws, r, C1 + 1 + i, _b(v), font=_f(9), bg=bg, align=AL_R, border=BORDER_ALL)
        if margin_row and cf is not None and inc is not None:
            num = _get(cf, row_key, 0)
            den = _get(inc, margin_row, 0)
            wc(ws, r, C1 + 1 + len(years),
               _pct(_margin(num, den)), font=_f(9, True), bg=BLUE_TINT, align=AL_C, border=BORDER_ALL)
        else:
            wc(ws, r, C1 + 1 + len(years), "—",
               font=_f(9), bg=bg, align=AL_C, border=BORDER_ALL)
        r += 1

    _cf_row("Operating Cash Flow", "Operating Cash Flow",  alt=False)
    _cf_row("Capital Expenditure", "Capital Expenditure",  alt=True)
    _cf_row("Free Cash Flow",      "Free Cash Flow",
            margin_row="Total Revenue", alt=False)

    spacer(ws, r); r += 1

    # ── Balance Sheet ─────────────────────────────────────────────────────────
    sec_hdr(ws, r, "Balance Sheet Highlights  (most recent year)", C1, CE); r += 1

    bs_items = [
        ("Cash & Short-Term Investments",  "Cash Cash Equivalents And Short Term Investments", False),
        ("Total Debt",                     "Total Debt",                                       False),
        ("Stockholders Equity",            "Stockholders Equity",                              False),
        ("Current Assets",                 "Current Assets",                                   False),
        ("Current Liabilities",            "Current Liabilities",                              False),
    ]

    for i, (lbl, row_key, _) in enumerate(bs_items):
        alt = (i % 2 == 0)
        bg  = GRAY_LIGHT if alt else WHITE
        v   = _get(bs, row_key, 0)
        ws.row_dimensions[r].height = 16
        wc(ws, r, C1,     lbl,   font=_f(9, True), bg=BLUE_TINT, align=AL_LI, border=BORDER_ALL)
        wc(ws, r, C1 + 1, _b(v), font=_f(9),       bg=bg,        align=AL_R,  border=BORDER_ALL)
        merge(ws, r, C1 + 1, r, CE)
        r += 1

    # Net Debt
    cash_v = _get(bs, "Cash Cash Equivalents And Short Term Investments", 0)
    debt_v = _get(bs, "Total Debt", 0)
    if cash_v is not None and debt_v is not None:
        nd = debt_v - cash_v
        ws.row_dimensions[r].height = 16
        wc(ws, r, C1,     "Net Debt", font=_f(9, True, WHITE), bg=NAVY_MED, align=AL_LI, border=BORDER_ALL)
        wc(ws, r, C1 + 1, _b(nd),    font=_f(9, True, WHITE), bg=NAVY_MED, align=AL_R,  border=BORDER_ALL)
        merge(ws, r, C1 + 1, r, CE)
        r += 1

    spacer(ws, r); r += 1

    # ── Peer Comparison ───────────────────────────────────────────────────────
    if peers:
        sec_hdr(ws, r, "Peer Comparison  (trading multiples)", C1, CE); r += 1

        peer_metrics = ["Mkt Cap", "P/E", "Fwd P/E", "P/B", "P/S", "EV/EBITDA", "Beta", "Yield"]
        col_hdr(ws, r, ["Ticker"] + peer_metrics, C1); r += 1

        def _peer_row(tkr, data, highlight=False):
            nonlocal r
            bg_lbl = NAVY_MED    if highlight else BLUE_TINT
            bg_val = BLUE_ACC    if highlight else WHITE
            ft_lbl = _f(9, True, WHITE) if highlight else _f(9, True)
            ft_val = _f(9, True, WHITE) if highlight else _f(9)
            ws.row_dimensions[r].height = 17
            wc(ws, r, C1, tkr,
               font=ft_lbl, bg=bg_lbl, align=AL_C, border=BORDER_ALL)
            vals = [
                _mcap(data.get("market_cap")),
                _mult(data.get("pe_ratio")),
                _mult(data.get("forward_pe")),
                _mult(data.get("pb_ratio")),
                _mult(data.get("ps_ratio")),
                _mult(data.get("ev_ebitda")),
                _num(data.get("beta"), 2),
                _pct(data.get("dividend_yield")),
            ]
            for i, v in enumerate(vals):
                wc(ws, r, C1 + 1 + i, v,
                   font=ft_val, bg=bg_val, align=AL_R, border=BORDER_ALL)
            r += 1

        # Target company row (highlighted)
        _peer_row(ticker, {
            "market_cap":    mkt_cap,
            "pe_ratio":      km.get("pe_ratio"),
            "forward_pe":    km.get("forward_pe"),
            "pb_ratio":      km.get("pb_ratio"),
            "ps_ratio":      km.get("ps_ratio"),
            "ev_ebitda":     km.get("ev_ebitda"),
            "beta":          km.get("beta"),
            "dividend_yield": km.get("dividend_yield"),
        }, highlight=True)

        # Peer rows
        for i, (pt, pd_) in enumerate(peers.items()):
            alt = (i % 2 == 0)
            bg  = GRAY_LIGHT if alt else WHITE
            ws.row_dimensions[r].height = 17
            wc(ws, r, C1, pt, font=_f(9, True), bg=BLUE_TINT, align=AL_C, border=BORDER_ALL)
            vals = [
                _mcap(pd_.get("market_cap")),
                _mult(pd_.get("pe_ratio")),
                _mult(pd_.get("forward_pe")),
                _mult(pd_.get("pb_ratio")),
                _mult(pd_.get("ps_ratio")),
                _mult(pd_.get("ev_ebitda")),
                _num(pd_.get("beta"), 2),
                _pct(pd_.get("dividend_yield")),
            ]
            for j, v in enumerate(vals):
                wc(ws, r, C1 + 1 + j, v,
                   font=_f(9), bg=bg, align=AL_R, border=BORDER_ALL)
            r += 1

        spacer(ws, r); r += 1

    # Footer
    ws.row_dimensions[r].height = 14
    wc(ws, r, C1,
       "  Source: yfinance / Alpha Vantage. Financial data as of latest reported fiscal year.",
       font=_f(8, False, MID_GRAY), bg=GRAY_LIGHT, align=AL_L)
    merge(ws, r, C1, r, CE)

    return ws
