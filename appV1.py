import streamlit as st
import upstox_client
from upstox_client.rest import ApiException
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import requests
from io import StringIO
import concurrent.futures
import time
import urllib3
import warnings
from groq import Groq
import datetime
import gzip
import io

# -------------------------------------------------
# Network & Session Configuration
# -------------------------------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# -------------------------------------------------
# Upstox Instrument Mapping
# -------------------------------------------------
@st.cache_data(ttl=86400)
def get_upstox_instruments():
    """
    Downloads and caches the Upstox instrument master list.
    """
    url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with gzip.open(io.BytesIO(response.content), 'rt') as f:
                df = pd.read_csv(f)
            # Filter for NSE and BSE equities, indices, and futures
            df = df[df['exchange'].isin(['NSE_EQ', 'BSE_EQ', 'NSE_FO', 'NSE_INDEX'])]
            return df
    except Exception as e:
        st.error(f"Error fetching instrument list: {e}")
    return pd.DataFrame()


def get_instrument_key(instruments_df, ticker, is_index=False):
    """
    Returns the instrument_key for a given ticker.
    If is_index is True, it tries to find the current month FUTURE contract for volume.
    """
    if instruments_df.empty:
        return None

    if is_index:
        # Map Index Names to their derivative symbols in Upstox
        index_map = {
            "NIFTY 50": "NIFTY",
            "NIFTY BANK": "BANKNIFTY",
            "NIFTY FIN SERVICE": "FINNIFTY",
            "NIFTY MIDCAP SELECT": "MIDCPNIFTY"
        }
        search_symbol = index_map.get(ticker, ticker)

        # Filter for NSE_FO and find the future with the nearest expiry
        futures = instruments_df[
            (instruments_df['exchange'] == 'NSE_FO') &
            (instruments_df['instrument_type'] == 'FUTIDX') &
            (instruments_df['tradingsymbol'].str.startswith(search_symbol))
            ].copy()

        if not futures.empty:
            # Sort by expiry and take the nearest one (current month)
            futures['expiry'] = pd.to_datetime(futures['expiry'])
            futures = futures.sort_values('expiry')
            return futures.iloc[0]['instrument_key']

    clean_ticker = ticker.split('.')[0].upper()
    exchange = 'NSE_EQ' if ticker.endswith('.NS') else 'BSE_EQ' if ticker.endswith('.BO') else 'NSE_EQ'

    # Try exact match on tradingsymbol
    match = instruments_df[
        (instruments_df['tradingsymbol'].str.upper() == clean_ticker) & (instruments_df['exchange'] == exchange)]
    if not match.empty:
        return match.iloc[0]['instrument_key']

    return None


def get_session():
    """Creates a requests Session with robust retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_upstox_historical_data(instrument_key, interval, from_date, to_date, access_token):
    """
    Fetches historical candle data from Upstox API v2.
    """
    api_instance = upstox_client.HistoryApi()
    api_instance.api_client.configuration.access_token = access_token

    try:
        # api_response = api_instance.get_historical_candle_data_v3(instrument_key, interval, to_date, from_date)
        # Note: Upstox API v2 Historical Candle Data
        api_response = api_instance.get_historical_candle_data1(instrument_key, interval, to_date, from_date, "v2")

        if api_response.status == "success" and api_response.data.candles:
            # Upstox candles are [timestamp, open, high, low, close, volume, open_interest]
            candles = api_response.data.candles
            df = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            df.sort_index(inplace=True)
            # Upstox returns data in descending order usually, or it depends. Sorting ensures consistency.
            return df
    except ApiException as e:
        st.error(f"Upstox API Error for {instrument_key}: {e}")
    except Exception as e:
        st.error(f"General Error for {instrument_key}: {e}")
    return pd.DataFrame()


def map_period_to_dates(period):
    """Maps yfinance-like periods to from_date and to_date for Upstox."""
    to_date = datetime.date.today()
    if period == "3mo":
        from_date = to_date - datetime.timedelta(days=90)
    elif period == "6mo":
        from_date = to_date - datetime.timedelta(days=180)
    elif period == "1y":
        from_date = to_date - datetime.timedelta(days=365)
    elif period == "2y":
        from_date = to_date - datetime.timedelta(days=730)
    else:
        from_date = to_date - datetime.timedelta(days=180)
    return from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d')


# -------------------------------------------------
# Configuration
# -------------------------------------------------
st.set_page_config(layout="wide", page_title="OBV Divergence Master")
st.title("📊 OBV Divergence Screener (Dual-View)")


# -------------------------------------------------
# 0. Data Fetching Functions
# -------------------------------------------------
@st.cache_data(ttl=86400)
def fetch_nifty50_stocks():
    return ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
            "BHARTIARTL", "KOTAKBANK", "BAJFINANCE", "LT", "ASIANPAINT", "AXISBANK", "MARUTI",
            "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND", "WIPRO", "DMART", "HCLTECH",
            "TECHM", "POWERGRID", "BAJAJFINSV", "NTPC", "TATAMOTORS", "COALINDIA", "ADANIPORTS",
            "ONGC", "HINDALCO", "GRASIM", "JSWSTEEL", "CIPLA", "DRREDDY", "EICHERMOT",
            "BRITANNIA", "APOLLOHOSP", "DIVISLAB", "GODREJCP", "PIDILITIND", "BERGEPAINT",
            "INDUSINDBK", "BANKBARODA", "SIEMENS", "DLF", "BAJAJ-AUTO", "TATASTEEL", "ADANIENT", "VEDL"]


@st.cache_data(ttl=86400)
def fetch_nse500_stocks():
    try:
        url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=30, verify=False)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df['Symbol'].tolist()
    except:
        pass
    return fetch_nifty50_stocks()  # Fallback


@st.cache_data(ttl=86400)
def fetch_all_nse_stocks():
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=30, verify=False)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df['SYMBOL'].tolist()
    except:
        pass
    return fetch_nifty50_stocks()  # Fallback


@st.cache_data(ttl=86400)
def fetch_all_bse_stocks():
    try:
        url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry=&segment=Equity&status=Active"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=30, verify=False)
        if response.status_code == 200:
            data = response.json()
            return [item['scrip_cd'] for item in data.get('Table', [])]
    except:
        pass
    return ["500325", "532540", "500180"]  # Very minimal fallback


@st.cache_data(ttl=86400)
def fetch_major_indices():
    return ["NIFTY 50", "NIFTY BANK", "NIFTY FIN SERVICE", "NIFTY MIDCAP SELECT"]


@st.cache_data(ttl=86400)
def fetch_nse_fno_stocks():
    """
    Filters the Upstox instrument list for stocks in the F&O segment.
    """
    df = get_upstox_instruments()
    if df.empty:
        return []
    # Filter for NSE_FO exchange and FUTSTK (Future Stock)
    # These represent the underlying stocks in F&O
    fno_df = df[(df['exchange'] == 'NSE_FO') & (df['instrument_type'] == 'FUTSTK')]

    # Extract clean symbol (e.g., RELIANCE from RELIANCE24FEB)
    # In Upstox, the 'tradingsymbol' for futures usually starts with the symbol
    # but a better way is to use the 'symbol' or 'name' if available.
    # However, let's look at common F&O stocks or filter by tradingsymbol prefix.
    # Actually, Upstox CSV has a 'tradingsymbol' like 'RELIANCE25JANFUT'
    # We can use a regex or just take the symbols.
    symbols = fno_df['tradingsymbol'].str.replace(r'\d{2}[A-Z]{3}FUT$', '', regex=True).unique().tolist()
    return sorted([s for s in symbols if s])


# -------------------------------------------------
# 1. Logic & Math (The Improved "Synced" Engine)
# -------------------------------------------------
def calculate_obv(df):
    """Vectorized, fast OBV calculation"""
    df = df.copy()
    df['Change'] = df['Close'].diff()
    df['OBV_Vol'] = np.where(df['Change'] > 0, df['Volume'],
                             np.where(df['Change'] < 0, -df['Volume'], 0))
    df['OBV'] = df['OBV_Vol'].cumsum().fillna(0)
    return df


@st.cache_data(ttl=86400)
def get_company_name(ticker, instruments_df):
    try:
        clean_ticker = ticker.split('.')[0]
        match = instruments_df[instruments_df['tradingsymbol'] == clean_ticker]
        if not match.empty:
            return match.iloc[0]['name']
    except Exception:
        pass
    return ""


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_pivots(data, order=5):
    """Finds peaks and valleys using Scipy"""
    highs_idx = argrelextrema(data.values, np.greater, order=order)[0]
    lows_idx = argrelextrema(data.values, np.less, order=order)[0]
    return highs_idx, lows_idx


def detect_divergence(df, order=5):
    """
    Detects divergence by syncing Price Pivots with OBV timestamps.
    Returns a list of divergence dictionaries.
    """
    price_highs_idx, price_lows_idx = get_pivots(df['Close'], order)
    results = []

    # --- Bearish Divergence (Price Higher High, OBV Lower High) ---
    if len(price_highs_idx) >= 2:
        recent_peaks = price_highs_idx[-5:]  # Check recent 5 peaks
        for i in range(len(recent_peaks) - 1, 0, -1):
            curr_idx = recent_peaks[i]
            prev_idx = recent_peaks[i - 1]

            if df['Close'].iloc[curr_idx] > df['Close'].iloc[prev_idx]:  # Price HH
                if df['OBV'].iloc[curr_idx] < df['OBV'].iloc[prev_idx]:  # OBV LH
                    results.append({
                        "Type": "Bearish",
                        "P1_Idx": prev_idx, "P2_Idx": curr_idx,
                        "P1_Date": df.index[prev_idx], "P2_Date": df.index[curr_idx],
                        "P1_Price": df['Close'].iloc[prev_idx], "P2_Price": df['Close'].iloc[curr_idx],
                        "P1_OBV": df['OBV'].iloc[prev_idx], "P2_OBV": df['OBV'].iloc[curr_idx]
                    })
                    break  # Stop after finding the most recent one

    # --- Bullish Divergence (Price Lower Low, OBV Higher Low) ---
    if len(price_lows_idx) >= 2:
        recent_lows = price_lows_idx[-5:]
        for i in range(len(recent_lows) - 1, 0, -1):
            curr_idx = recent_lows[i]
            prev_idx = recent_lows[i - 1]

            if df['Close'].iloc[curr_idx] < df['Close'].iloc[prev_idx]:  # Price LL
                if df['OBV'].iloc[curr_idx] > df['OBV'].iloc[prev_idx]:  # OBV HL
                    results.append({
                        "Type": "Bullish",
                        "P1_Idx": prev_idx, "P2_Idx": curr_idx,
                        "P1_Date": df.index[prev_idx], "P2_Date": df.index[curr_idx],
                        "P1_Price": df['Close'].iloc[prev_idx], "P2_Price": df['Close'].iloc[curr_idx],
                        "P1_OBV": df['OBV'].iloc[prev_idx], "P2_OBV": df['OBV'].iloc[curr_idx]
                    })
                    break

    return results, price_highs_idx, price_lows_idx


# -------------------------------------------------
# 1.5. AI Summary Logic
# -------------------------------------------------
def get_ai_summary(api_key, results_data):
    """
    Generates a summary of the scan results using Groq API.
    """
    client = Groq(api_key=api_key)

    # Prepare data for the prompt
    # We'll take the top 10 results to avoid token limits if list is huge
    top_results = results_data[:15]

    prompt_content = f"""
    You are a financial analyst assistant. 
    Here are the findings from an OBV Divergence Scan on Indian Stocks:

    {top_results}

    Please provide a concise but insightful summary of these findings.
    Highlight the most significant Bullish and Bearish setups.
    Explain what these divergences might indicate for the short-term trend of these specific stocks.
    Format the output in clean Markdown.
    """

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    return completion.choices[0].message.content


# -------------------------------------------------
# 2. Visualization Options
# -------------------------------------------------

def plot_static_matplotlib(df, divs, price_highs, price_lows, ticker):
    """The 'Classic' Matplotlib Look"""
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # 1. Price Chart
    ax1.plot(df.index, df['Close'], color='black', linewidth=1, label='Price')
    # Plot pivots dots
    ax1.scatter(df.index[price_highs], df['Close'].iloc[price_highs], color='red', s=10, alpha=0.5)
    ax1.scatter(df.index[price_lows], df['Close'].iloc[price_lows], color='green', s=10, alpha=0.5)
    ax1.set_title(f"{ticker} - Price Action")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Price")

    # 2. OBV Chart
    ax2.plot(df.index, df['OBV'], color='purple', linewidth=1, label='OBV')
    ax2.set_title("On-Balance Volume (OBV)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Volume")

    # 3. Draw Divergence Lines
    for d in divs:
        color = 'green' if d['Type'] == 'Bullish' else 'red'
        # Draw on Price
        ax1.plot([d['P1_Date'], d['P2_Date']], [d['P1_Price'], d['P2_Price']],
                 color=color, linewidth=2, linestyle='--')
        # Draw on OBV
        ax2.plot([d['P1_Date'], d['P2_Date']], [d['P1_OBV'], d['P2_OBV']],
                 color=color, linewidth=2, linestyle='--')

        # Add Label
        ax1.annotate(d['Type'], (d['P2_Date'], d['P2_Price']),
                     xytext=(10, 0), textcoords='offset points', color=color, weight='bold')

    plt.tight_layout()
    st.pyplot(fig)


def plot_interactive_plotly(df, divs, ticker):
    """The 'Modern' Plotly Look"""
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = compute_rsi(df['Close'], period=14)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.6, 0.25, 0.15],
                        subplot_titles=(f"{ticker} Price", "OBV", "RSI"))

    # Price Candle
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20',
                             line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50',
                             line=dict(color='orange')), row=1, col=1)

    # OBV Line
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV',
                             line=dict(color='purple')), row=2, col=1)

    # RSI Line
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                             line=dict(color='teal')), row=3, col=1)
    fig.add_hline(y=70, line=dict(color='red', dash='dot'), row=3, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dot'), row=3, col=1)

    # Divergence Lines
    for d in divs:
        c = 'green' if d['Type'] == 'Bullish' else 'red'
        fig.add_trace(go.Scatter(x=[d['P1_Date'], d['P2_Date']], y=[d['P1_Price'], d['P2_Price']],
                                 mode='lines', line=dict(color=c, width=2, dash='dot'),
                                 name=f"{d['Type']} Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[d['P1_Date'], d['P2_Date']], y=[d['P1_OBV'], d['P2_OBV']],
                                 mode='lines', line=dict(color=c, width=2, dash='dot'),
                                 name=f"{d['Type']} OBV"), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=750, showlegend=False)
    st.plotly_chart(fig, width="stretch")


with st.sidebar:
    st.header("⚙️ Settings")
    chart_style = st.radio("Chart Style", ["Static (Matplotlib)", "Interactive (Plotly)"])

    if st.checkbox("Use Preset List", value=True):
        dataset_choice = st.sidebar.selectbox(
            "Select Dataset",
            ["NIFTY 50", "NSE 500", "NSE F&O", "Major Indices", "All NSE Stocks", "All BSE Stocks"],
            index=2
        )
        is_index_scan = dataset_choice == "Major Indices"

        if dataset_choice == "NIFTY 50":
            symbols_raw = fetch_nifty50_stocks()
        elif dataset_choice == "NSE 500":
            symbols_raw = fetch_nse500_stocks()
        elif dataset_choice == "NSE F&O":
            symbols_raw = fetch_nse_fno_stocks()
        elif dataset_choice == "Major Indices":
            symbols_raw = fetch_major_indices()
        elif dataset_choice == "All NSE Stocks":
            symbols_raw = fetch_all_nse_stocks()
        else:
            symbols_raw = fetch_all_bse_stocks()

        suffix = "" if is_index_scan else (".BO" if dataset_choice == "All BSE Stocks" else ".NS")
        tickers = [f"{s}{suffix}" for s in symbols_raw]

        num_to_scan = st.slider("Number of stocks to scan", min(1, len(tickers)), len(tickers), min(100, len(tickers)))
        tickers = tickers[:num_to_scan]
    else:
        tickers_input = st.text_area("Stocks (comma sep)", "RELIANCE.NS, TCS.NS, HDFCBANK.NS, TATAMOTORS.NS, INFY.NS")
        tickers = [t.strip().upper() for t in tickers_input.split(',')]

    period = st.selectbox("Lookback Period", ["3mo", "6mo", "1y", "2y"], index=1)
    timeframe = st.selectbox("Timeframe", ["Daily", "Weekly"], index=0)
    sensitivity = st.slider("Pivot Sensitivity", 3, 20, 5, help="Higher = Fewer, stronger pivots")

    st.markdown("---")
    st.subheader("🔑 Upstox API Settings")
    upstox_access_token = st.text_input("Upstox Access Token", type="password",
                                        help="Enter your Upstox Access Token (valid for 1 day).")

    st.markdown("---")
    st.subheader("🤖 AI Settings")
    groq_api_key = st.text_input("Groq API Key", type="password",
                                 help="Enter your Groq API Key to enable AI summaries.")

if st.button("🚀 Run Screener"):
    if not upstox_access_token:
        st.error("❌ Please enter an Upstox Access Token in the sidebar.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        instruments_df = get_upstox_instruments()
        from_date, to_date = map_period_to_dates(period)

        results_container = []
        total_tickers = len(tickers)

        for idx, ticker in enumerate(tickers):
            status_text.text(f"🔍 Analyzing {idx + 1}/{total_tickers}: {ticker}")
            try:
                ikey = get_instrument_key(instruments_df, ticker, is_index=is_index_scan)
                if not ikey:
                    st.warning(f"Could not find instrument key for {ticker}")
                    continue

                interval_map = {"Daily": "day", "Weekly": "week"}
                selected_interval = interval_map.get(timeframe, "day")

                df = fetch_upstox_historical_data(ikey, selected_interval, from_date, to_date, upstox_access_token)

                if df.empty or 'Close' not in df.columns:
                    continue

                df = calculate_obv(df)
                divs, ph, pl = detect_divergence(df, order=sensitivity)
                results_container.append((ticker, df, divs, ph, pl))
            except Exception as e:
                # Silently skip errors for individual tickers
                continue

            progress_bar.progress((idx + 1) / total_tickers)

        status_text.success(
            f"✅ Scanning Complete! Analyzed {len(results_container)} stocks. Found {len([r for r in results_container if r[2]])} stocks with signals.")

        # Only keep results that have signals
        results_container = [r for r in results_container if r[2]]
        results_container.sort(key=lambda x: max([d['P2_Date'] for d in x[2]]), reverse=True)
        summary_rows = []
        results_map = {}
        instruments_df = get_upstox_instruments()
        for ticker, df, divs, ph, pl in results_container:
            last_price = float(df['Close'].iloc[-1]) if not df.empty else np.nan
            name = get_company_name(ticker, instruments_df)
            has_signal = len(divs) > 0
            sig_type = divs[0]['Type'] if has_signal else ""
            from_date = divs[0]['P1_Date'].date() if has_signal else ""
            to_date = divs[0]['P2_Date'].date() if has_signal else ""
            summary_rows.append({
                "Symbol": ticker,
                "Name": name,
                "Price": round(last_price, 2) if not np.isnan(last_price) else None,
                "Signal": "Yes" if has_signal else "No",
                "Type": sig_type,
                "From": from_date,
                "To": to_date
            })
            results_map[ticker] = (df, divs, ph, pl)
        st.session_state.scan_results = summary_rows
        st.session_state.results_map = results_map
        st.session_state.selected_symbol = None

if st.session_state.get("scan_results"):
    st.subheader("Scan Results")
    summary_rows = st.session_state.get("scan_results", [])
    for r in summary_rows:
        if isinstance(r.get("From"), (pd.Timestamp, np.datetime64)) or hasattr(r.get("From"), "strftime"):
            r["From"] = str(r["From"])
        if isinstance(r.get("To"), (pd.Timestamp, np.datetime64)) or hasattr(r.get("To"), "strftime"):
            r["To"] = str(r["To"])
        r["Name"] = str(r.get("Name") or "")
        r["Signal"] = str(r.get("Signal") or "")
        r["Type"] = str(r.get("Type") or "")
    summary_df = pd.DataFrame(summary_rows)
    if "Price" in summary_df.columns:
        summary_df["Price"] = pd.to_numeric(summary_df["Price"], errors="coerce")
    for col in ["Symbol", "Name", "Signal", "Type", "From", "To"]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].fillna("").astype(str)
    st.dataframe(summary_df, width="stretch")

    # --- AI Summary Section ---
    if st.button("✨ Generate AI Summary"):
        if not groq_api_key:
            st.warning("⚠️ Please enter your Groq API Key in the sidebar settings first.")
        elif not summary_rows:
            st.warning("⚠️ No scan results to summarize. Please run the screener first.")
        else:
            with st.spinner("🤖 AI is analyzing market data..."):
                try:
                    # Pass the raw summary rows (list of dicts)
                    ai_summary = get_ai_summary(groq_api_key, summary_rows)
                    st.markdown("### AI Market Analysis")
                    st.markdown(ai_summary)
                except Exception as e:
                    # Avoid emoji in the error message string to prevent UnicodeEncodeError in some consoles
                    st.error(f"Error generating summary: {e}")
    # --------------------------
    st.subheader("Interactive Stock Viewer")
    stock_options = [row["Symbol"] for row in summary_rows]
    if stock_options:
        selected_symbol = st.selectbox("Select a Stock", stock_options)

        results_map = st.session_state.get("results_map", {})
        if selected_symbol in results_map:
            df, divs, ph, pl = results_map[selected_symbol]
            st.subheader(f"{selected_symbol} Chart")
            if chart_style == "Static (Matplotlib)":
                plot_static_matplotlib(df, divs, ph, pl, selected_symbol)
            else:
                plot_interactive_plotly(df, divs, selected_symbol)