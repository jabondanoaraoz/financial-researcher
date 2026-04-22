"""
Financial Researcher — Streamlit Web UI
Run with: streamlit run app.py
"""

import sys
import os
import logging
import io
from datetime import date

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from orchestrator import run_analysis
from excel.workbook import generate_report

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Researcher",
    page_icon="📊",
    layout="wide",
)

# ── Signal styling ────────────────────────────────────────────────────────────
SIGNAL_COLOR = {"bullish": "🟢", "neutral": "🟡", "bearish": "🔴"}
ACTION_LABEL  = {"buy": "BUY", "sell": "SELL", "hold": "HOLD",
                 "add": "ADD", "trim": "TRIM", "avoid": "AVOID", "watch": "WATCH"}

def _signal_badge(signal: str) -> str:
    return f"{SIGNAL_COLOR.get(signal, '⚪')} {signal.upper()}"

def _fmt(v, fmt=".1f", prefix="", suffix=""):
    return f"{prefix}{v:{fmt}}{suffix}" if v is not None else "—"

def _fmt_cap(v):
    if v is None: return "—"
    if v >= 1e12: return f"${v/1e12:.2f}T"
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"

def _score_str(scores: dict) -> str:
    total = scores.get("total")
    tmax  = scores.get("total_max", 20)
    if total is None: return "—"
    if tmax <= 1.0:   return f"{total:.0%}"
    return f"{total:.0f} / {tmax:.0f}"


# ── Excel download helper ─────────────────────────────────────────────────────
def _excel_bytes(result: dict) -> bytes:
    buf = io.BytesIO()
    # generate_report expects a file path; use a temp file then read bytes
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    generate_report(result, tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.unlink(tmp_path)
    return data


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Financial Researcher")
    st.caption("10-agent AI investment analysis")
    st.divider()

    ticker_input = st.text_input(
        "Ticker symbol",
        placeholder="e.g. AAPL",
        help="Any publicly traded stock ticker",
    ).upper().strip()

    peers_input = st.text_input(
        "Custom peers (optional)",
        placeholder="e.g. MSFT, GOOGL, META",
        help="Comma-separated tickers. Leave blank to use default peers.",
    )

    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)
    st.divider()
    st.caption("Analysis takes 2–5 min due to LLM rate limits.")
    st.caption("Results are cached in the session — re-run only when changing tickers.")


# ── Trigger analysis ──────────────────────────────────────────────────────────
if analyze_btn and ticker_input:
    peers = None
    if peers_input.strip():
        peers = {t.strip().upper(): None for t in peers_input.split(",") if t.strip()}

    # Only re-run if ticker changed
    if st.session_state.get("last_ticker") != ticker_input:
        with st.spinner(f"Analyzing {ticker_input}… this may take a few minutes"):
            try:
                result = run_analysis(ticker_input, peers=peers)
                st.session_state["result"]       = result
                st.session_state["last_ticker"]  = ticker_input
                st.session_state["excel_bytes"]  = None  # reset cache
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()
    else:
        st.info(f"Showing cached results for {ticker_input}. Change the ticker to re-run.")

elif analyze_btn and not ticker_input:
    st.sidebar.warning("Please enter a ticker symbol.")


# ── Results ───────────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result  = st.session_state["result"]
    ticker  = result["ticker"]
    info    = result.get("company_data", {}).get("info", {})
    km      = result.get("company_data", {}).get("key_metrics", {})
    signals = result.get("agent_signals", {})
    pm      = result.get("portfolio_decision")
    cons    = result.get("consensus", {})
    risk    = result.get("risk_metrics", {})

    # ── Section 1: Company header ─────────────────────────────────────────────
    name    = info.get("longName") or info.get("name") or ticker
    sector  = info.get("sector", "")
    exch    = info.get("exchange", "")

    st.title(f"{ticker} — {name}")
    st.caption(f"{sector}  ·  {exch}  ·  {date.today()}")
    st.divider()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Price",      _fmt(km.get("current_price"), ".2f", "$"))
    col2.metric("Mkt Cap",    _fmt_cap(km.get("market_cap")))
    col3.metric("P/E",        _fmt(km.get("pe_ratio"),      ".1f", "", "x"))
    col4.metric("Fwd P/E",    _fmt(km.get("forward_pe"),    ".1f", "", "x"))
    col5.metric("P/S",        _fmt(km.get("ps_ratio"),      ".1f", "", "x"))
    col6.metric("EV/EBITDA",  _fmt(km.get("ev_ebitda"),     ".1f", "", "x"))
    st.caption("— indicates data not available from the API for this company.")

    st.divider()

    # ── Section 2: Consensus + PM decision ───────────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Consensus")
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Bullish", cons.get("bullish", 0))
        c2.metric("🟡 Neutral", cons.get("neutral", 0))
        c3.metric("🔴 Bearish", cons.get("bearish", 0))
        st.metric("Avg score", f"{cons.get('avg_score_20', 0):.1f} / 20")

    with right:
        st.subheader("Portfolio Manager")
        if pm:
            signal_color = {"bullish": "green", "neutral": "orange", "bearish": "red"}
            color  = signal_color.get(pm.signal, "gray")
            action = ACTION_LABEL.get(pm.target_action, pm.target_action.upper())
            pt_str = f"  ·  Target: ${pm.price_target:.2f}" if pm.price_target else ""
            st.markdown(
                f":{color}[**{pm.signal.upper()}  —  {action}**]"
                f"  ·  Conf: {pm.confidence:.0%}{pt_str}"
            )
            st.write(pm.reasoning)

    st.divider()

    # ── Section 3: Agent breakdown ────────────────────────────────────────────
    st.subheader("Agent Breakdown")

    DISPLAY_NAMES = {
        "fundamentals":     "Fundamentals",
        "ben_graham":       "Ben Graham",
        "warren_buffett":   "Warren Buffett",
        "aswath_damodaran": "A. Damodaran",
        "cathie_wood":      "Cathie Wood",
        "michael_burry":    "Michael Burry",
        "technicals":       "Technicals",
        "valuation":        "Valuation",
        "risk_manager":     "Risk Manager",
    }

    rows = []
    for agent_id, sig in signals.items():
        rows.append({
            "Agent":      DISPLAY_NAMES.get(agent_id, agent_id),
            "Signal":     _signal_badge(sig.signal),
            "Confidence": f"{sig.confidence:.0%}",
            "Score":      _score_str(sig.scores),
            "Action":     ACTION_LABEL.get(sig.target_action, sig.target_action.upper()),
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Agent reasoning expander
    with st.expander("View full agent reasoning & risks"):
        for agent_id, sig in signals.items():
            name = DISPLAY_NAMES.get(agent_id, agent_id)
            st.markdown(f"**{name}** — {_signal_badge(sig.signal)}")
            st.write(sig.reasoning)
            if sig.key_risks:
                st.markdown("**Key risks:** " + " · ".join(sig.key_risks))
            st.divider()

    # ── Section 4: Risk snapshot ──────────────────────────────────────────────
    st.subheader("Risk Snapshot")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Beta",         _fmt(risk.get("beta"),                  ".2f"))
    r2.metric("Sharpe",       _fmt(risk.get("sharpe_proxy"),          ".2f"))
    r3.metric("Max Drawdown", _fmt(risk.get("max_drawdown"),          ".1%") if risk.get("max_drawdown") else "—")
    r4.metric("Vol (ann)",    _fmt(risk.get("annualized_volatility"),  ".1%") if risk.get("annualized_volatility") else "—")
    r5.metric("Kelly sizing", _fmt(risk.get("max_position_size_pct"), ".1f", "", "%") if risk.get("max_position_size_pct") else "—")

    st.divider()

    # ── Section 5: Peer comparables ───────────────────────────────────────────
    peers_data = result.get("peers_data", {})
    if peers_data:
        st.subheader("Peer Comparables")
        peer_rows = []
        for pt, pd_ in peers_data.items():
            def _p(key, decimals=2):
                v = pd_.get(key)
                return f"{v:.{decimals}f}" if v is not None else "—"
            peer_rows.append({
                "Ticker":    pt,
                "Price":     f"${pd_.get('current_price'):.2f}" if pd_.get("current_price") else "—",
                "Mkt Cap":   _fmt_cap(pd_.get("market_cap")),
                "P/S":       _p("ps_ratio"),
                "P/E":       _p("pe_ratio"),
                "Fwd P/E":   _p("forward_pe"),
                "EV/EBITDA": _p("ev_ebitda"),
                "P/FCF":     _p("price_to_fcf"),
                "Beta":      _p("beta", 3),
            })
        st.dataframe(peer_rows, use_container_width=True, hide_index=True)
        st.caption("— indicates data not available from the API for this peer.")
        st.divider()

    # ── Section 6: Excel download ─────────────────────────────────────────────
    st.subheader("Export")

    if st.button("Generate Excel Report"):
        with st.spinner("Generating Excel…"):
            excel_data = _excel_bytes(result)
            st.session_state["excel_bytes"] = excel_data

    if st.session_state.get("excel_bytes"):
        st.download_button(
            label="⬇️ Download Excel Report",
            data=st.session_state["excel_bytes"],
            file_name=f"{ticker}_{date.today()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:
    # Empty state
    st.markdown("## 📊 Financial Researcher")
    st.markdown(
        "Enter a ticker symbol in the sidebar and click **Analyze** to run "
        "a full 10-agent investment analysis."
    )
    st.markdown("""
    **What you'll get:**
    - Key valuation metrics (Price, P/E, P/S, EV/EBITDA)
    - Consensus across 10 AI investment agents
    - Portfolio Manager final recommendation with reasoning
    - Full agent breakdown with scores and signals
    - Risk snapshot (Beta, Sharpe, Max Drawdown, Volatility)
    - Downloadable IB-grade Excel report
    """)
