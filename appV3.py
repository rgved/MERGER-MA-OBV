import urllib3
import warnings
import numpy as np
import pandas as pd
import streamlit as st

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from data.fetcher import (
    get_upstox_instruments, get_instrument_key, fetch_upstox_historical_data,
    map_period_to_dates, get_company_name,
    fetch_nifty50_stocks, fetch_nse500_stocks, fetch_all_nse_stocks,
    fetch_all_bse_stocks, fetch_major_indices, fetch_nse_fno_stocks,
)
from utils.obv import calculate_obv, detect_divergence
from utils.volatility import compute_volatility
from utils.ai_summary import get_ai_summary
from charts.static_chart import plot_static_matplotlib
from charts.plotly_chart import plot_interactive_plotly
from charts.echarts_chart import plot_echarts_synchronized

# -------------------------------------------------
# App Config
# -------------------------------------------------
st.set_page_config(layout="wide", page_title="OBV Divergence Master")
st.title("📊 OBV Divergence Screener (Dual-View)")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    chart_style = st.radio(
        "Chart Style",
        ["Static (Matplotlib)", "Interactive (Plotly)", "ECharts (True Sync Tooltip)"]
    )

    if st.checkbox("Use Preset List", value=True):
        dataset_choice = st.sidebar.selectbox(
            "Select Dataset",
            ["NIFTY 50", "NSE 500", "NSE F&O", "Major Indices", "All NSE Stocks", "All BSE Stocks"],
            index=2,
        )
        is_index_scan = dataset_choice == "Major Indices"

        dataset_loaders = {
            "NIFTY 50":       fetch_nifty50_stocks,
            "NSE 500":        fetch_nse500_stocks,
            "NSE F&O":        fetch_nse_fno_stocks,
            "Major Indices":  fetch_major_indices,
            "All NSE Stocks": fetch_all_nse_stocks,
            "All BSE Stocks": fetch_all_bse_stocks,
        }
        symbols_raw = dataset_loaders[dataset_choice]()

        suffix = "" if is_index_scan else (".BO" if dataset_choice == "All BSE Stocks" else ".NS")
        tickers = [f"{s}{suffix}" for s in symbols_raw]
        num = st.slider("Number of stocks to scan", min(1, len(tickers)), len(tickers), min(100, len(tickers)))
        tickers = tickers[:num]
    else:
        raw = st.text_area("Stocks (comma sep)", "RELIANCE.NS, TCS.NS, HDFCBANK.NS, TATAMOTORS.NS, INFY.NS")
        tickers = [t.strip().upper() for t in raw.split(',')]

    period     = st.selectbox("Lookback Period", ["3mo", "6mo", "1y", "2y"], index=1)
    timeframe  = st.selectbox("Timeframe", ["Daily", "Weekly"], index=0)
    sensitivity = st.slider("Pivot Sensitivity", 3, 20, 5, help="Higher = Fewer, stronger pivots")

    st.markdown("---")
    st.subheader("📉 Volatility Settings")
    vol_method = st.selectbox(
        "Volatility Metric",
        ["GARCH", "Yang-Zhang", "GARCH % Change", "YZ % Change"],
        index=0,
    )
    vol_window = st.slider("Volatility Window", 5, 60, 20) if vol_method in ("Yang-Zhang", "YZ % Change") else 20

    st.markdown("---")
    st.subheader("🔑 Upstox API Settings")
    upstox_access_token = st.text_input("Upstox Access Token", type="password")

    st.markdown("---")
    st.subheader("🤖 AI Settings")
    groq_api_key = st.text_input("Groq API Key", type="password")

# -------------------------------------------------
# Run Screener
# -------------------------------------------------
if st.button("🚀 Run Screener"):
    if not upstox_access_token:
        st.error("❌ Please enter an Upstox Access Token in the sidebar.")
    else:
        progress_bar  = st.progress(0)
        status_text   = st.empty()
        instruments_df = get_upstox_instruments()
        from_date, to_date = map_period_to_dates(period)
        results_container = []
        total = len(tickers)

        for idx, ticker in enumerate(tickers):
            status_text.text(f"🔍 Analyzing {idx + 1}/{total}: {ticker}")
            try:
                ikey = get_instrument_key(instruments_df, ticker, is_index=is_index_scan)
                if not ikey:
                    st.warning(f"Could not find instrument key for {ticker}")
                    continue
                interval = {"Daily": "day", "Weekly": "week"}.get(timeframe, "day")
                df = fetch_upstox_historical_data(ikey, interval, from_date, to_date, upstox_access_token)
                if df.empty or 'Close' not in df.columns:
                    continue
                df = calculate_obv(df)
                divs, ph, pl = detect_divergence(df, order=sensitivity)
                results_container.append((ticker, df, divs, ph, pl))
            except:
                pass
            progress_bar.progress((idx + 1) / total)

        status_text.success(
            f"✅ Done! Analyzed {len(results_container)} stocks. "
            f"Found {len([r for r in results_container if r[2]])} with signals."
        )

        results_container = [r for r in results_container if r[2]]
        results_container.sort(key=lambda x: max(d['P2_Date'] for d in x[2]), reverse=True)

        summary_rows = []
        results_map  = {}
        for ticker, df, divs, ph, pl in results_container:
            last_price = float(df['Close'].iloc[-1]) if not df.empty else np.nan
            name       = get_company_name(ticker, instruments_df)
            sig_type   = divs[0]['Type'] if divs else ""
            summary_rows.append({
                "Symbol": ticker, "Name": name,
                "Price":  round(last_price, 2) if not np.isnan(last_price) else None,
                "Signal": "Yes" if divs else "No", "Type": sig_type,
                "From":   divs[0]['P1_Date'].date() if divs else "",
                "To":     divs[0]['P2_Date'].date() if divs else "",
            })
            results_map[ticker] = (df, divs, ph, pl)

        st.session_state.scan_results = summary_rows
        st.session_state.results_map  = results_map

# -------------------------------------------------
# Results Display
# -------------------------------------------------
if st.session_state.get("scan_results"):
    st.subheader("Scan Results")
    summary_rows = st.session_state["scan_results"]

    for r in summary_rows:
        for k in ("From", "To"):
            if hasattr(r.get(k), "strftime"):
                r[k] = str(r[k])
        for k in ("Name", "Signal", "Type"):
            r[k] = str(r.get(k) or "")

    summary_df = pd.DataFrame(summary_rows)
    if "Price" in summary_df.columns:
        summary_df["Price"] = pd.to_numeric(summary_df["Price"], errors="coerce")
    for col in ["Symbol", "Name", "Signal", "Type", "From", "To"]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].fillna("").astype(str)
    st.dataframe(summary_df, width='stretch')

    if st.button("✨ Generate AI Summary"):
        if not groq_api_key:
            st.warning("⚠️ Enter your Groq API Key in the sidebar first.")
        else:
            with st.spinner("🤖 Analyzing..."):
                try:
                    st.markdown("### AI Market Analysis")
                    st.markdown(get_ai_summary(groq_api_key, summary_rows))
                except Exception as e:
                    st.error(f"Error: {e}")

    st.subheader("Interactive Stock Viewer")
    stock_options = [r["Symbol"] for r in summary_rows]
    if stock_options:
        selected_symbol = st.selectbox("Select a Stock", stock_options)
        results_map = st.session_state.get("results_map", {})
        if selected_symbol in results_map:
            df, divs, ph, pl = results_map[selected_symbol]
            st.subheader(f"{selected_symbol} Chart")
            if chart_style == "Static (Matplotlib)":
                plot_static_matplotlib(df, divs, ph, pl, selected_symbol, vol_method, vol_window)
            elif chart_style == "Interactive (Plotly)":
                plot_interactive_plotly(df, divs, selected_symbol, vol_method, vol_window)
            else:
                plot_echarts_synchronized(df, divs, selected_symbol, vol_method, vol_window)