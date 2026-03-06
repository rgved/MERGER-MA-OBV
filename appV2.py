import streamlit as st
import joblib
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
import streamlit.components.v1 as components
import urllib3
import warnings
from groq import Groq
import datetime
import gzip
import io
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# -------------------------------------------------
# Volatility Calculation Functions
# -------------------------------------------------

def calc_log_returns(close: np.ndarray) -> np.ndarray:
    c = np.array(close, dtype=float)
    lr = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        if c[i - 1] > 0 and c[i] > 0:
            lr[i] = np.log(c[i] / c[i - 1])
    return lr


def calc_pct_change(values: np.ndarray) -> np.ndarray:
    v = np.array(values, dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(1, len(v)):
        if not np.isnan(v[i - 1]) and v[i - 1] != 0:
            out[i] = (v[i] - v[i - 1]) * 100.0 / v[i - 1]
    return out


def calc_sma(arr: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        window_vals = arr[i - window + 1:i + 1]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) > 0:
            result[i] = np.mean(valid)
    return result


def calc_volume_weighted_garch(
        log_returns, volumes, window=20,
        omega=0.0001, alpha=0.15, beta=0.80, gamma=0.4,
):
    n = len(log_returns)
    if n == 0 or len(volumes) != n:
        return np.array([])

    volatilities = np.full(n, np.nan)
    avg_volume = np.nanmean(np.maximum(volumes, 0))
    if avg_volume <= 0:
        avg_volume = 1.0
    norm_vols = np.maximum(volumes, 0) / avg_volume

    init_slice = log_returns[:min(window, n)]
    valid_init = init_slice[np.isfinite(init_slice)]
    if len(valid_init) == 0:
        return volatilities
    uncond_var = np.clip(np.mean(valid_init ** 2), 0.0001, 1.0)
    prev_variance = uncond_var
    prev_sq_return = uncond_var

    for i in range(n):
        cur_ret = log_returns[i] if np.isfinite(log_returns[i]) else 0.0
        sq_ret = cur_ret ** 2
        vol_ratio = np.clip(norm_vols[i] if np.isfinite(norm_vols[i]) else 1.0, 0.1, 10.0)
        core_var = omega + alpha * prev_sq_return + beta * prev_variance
        vol_term = np.tanh(gamma * np.log(vol_ratio))
        new_var = core_var * (1.0 + vol_term)
        if not np.isfinite(new_var) or new_var < 0:
            new_var = prev_variance
        new_var = np.clip(new_var, 0.0001, 1.0)
        volatilities[i] = np.sqrt(new_var) * np.sqrt(252) * 100.0
        prev_variance = new_var
        prev_sq_return = sq_ret

    return volatilities


def calc_yang_zhang_vol(open_, high, low, close, volumes=None, window=10):
    n = len(close)
    if n < window:
        return np.full(n, np.nan)

    open_ = np.array(open_, dtype=float)
    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)
    close = np.array(close, dtype=float)

    log_oc = np.log(close / open_)
    log_cc = np.log(close / np.roll(close, 1))
    log_co = np.log(open_ / np.roll(close, 1))
    log_oh = np.log(high / open_)
    log_ol = np.log(low / open_)
    log_cc[0] = np.nan
    log_co[0] = np.nan

    rs = log_oh * (log_oh - log_oc) + log_ol * (log_ol - log_oc)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    yz_vol = np.full(n, np.nan)
    avg_vol = calc_sma(volumes, window) if volumes is not None else None

    for i in range(window - 1, n):
        sl = slice(i - window + 1, i + 1)
        cc = log_cc[sl][~np.isnan(log_cc[sl])]
        co = log_co[sl][~np.isnan(log_co[sl])]
        oc = log_oc[sl][~np.isnan(log_oc[sl])]
        r = rs[sl][~np.isnan(rs[sl])]
        if len(cc) < 2:
            continue

        yz_var = np.var(co, ddof=1) + k * np.var(oc, ddof=1) + (1 - k) * np.mean(r)
        base_vol = np.sqrt(max(yz_var, 0)) * np.sqrt(252) * 100.0

        price_move = close[i] - open_[i]
        candle_range = high[i] - low[i]
        flow_pressure = (price_move / candle_range) if candle_range > 0 else 0.0
        if volumes is not None and avg_vol is not None and not np.isnan(avg_vol[i]) and avg_vol[i] > 0:
            flow_pressure *= min(volumes[i] / avg_vol[i], 3.0)

        yz_vol[i] = base_vol * (1.0 + 0.3 * np.tanh(flow_pressure))

    return yz_vol


def compute_volatility(df, method, window=20):
    close = df['Close'].values
    open_ = df['Open'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    log_ret = calc_log_returns(close)

    if method in ("GARCH", "GARCH % Change"):
        raw = calc_volume_weighted_garch(log_ret, volume, window=window)
        label = "GARCH Volatility (%)"
    else:
        raw = calc_yang_zhang_vol(open_, high, low, close, volume, window=window)
        label = "Yang-Zhang Volatility (%)"

    if method in ("GARCH % Change", "YZ % Change"):
        raw = calc_pct_change(raw)
        label = label.replace("(%)", "% Chg")

    return pd.Series(raw, index=df.index, name=label)


# -------------------------------------------------
# Upstox Instrument Mapping
# -------------------------------------------------
@st.cache_data(ttl=86400)
def get_upstox_instruments():
    url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with gzip.open(io.BytesIO(response.content), 'rt') as f:
                df = pd.read_csv(f)
            return df[df['exchange'].isin(['NSE_EQ', 'BSE_EQ', 'NSE_FO', 'NSE_INDEX'])]
    except Exception as e:
        st.error(f"Error fetching instrument list: {e}")
    return pd.DataFrame()


def get_instrument_key(instruments_df, ticker, is_index=False):
    if instruments_df.empty:
        return None

    if is_index:
        index_map = {
            "NIFTY 50": "NIFTY", "NIFTY BANK": "BANKNIFTY",
            "NIFTY FIN SERVICE": "FINNIFTY", "NIFTY MIDCAP SELECT": "MIDCPNIFTY",
        }
        search_symbol = index_map.get(ticker, ticker)
        futures = instruments_df[
            (instruments_df['exchange'] == 'NSE_FO') &
            (instruments_df['instrument_type'] == 'FUTIDX') &
            (instruments_df['tradingsymbol'].str.startswith(search_symbol))
            ].copy()
        if not futures.empty:
            futures['expiry'] = pd.to_datetime(futures['expiry'])
            return futures.sort_values('expiry').iloc[0]['instrument_key']

    clean_ticker = ticker.split('.')[0].upper()
    exchange = 'NSE_EQ' if ticker.endswith('.NS') else 'BSE_EQ' if ticker.endswith('.BO') else 'NSE_EQ'
    match = instruments_df[
        (instruments_df['tradingsymbol'].str.upper() == clean_ticker) &
        (instruments_df['exchange'] == exchange)
        ]
    return match.iloc[0]['instrument_key'] if not match.empty else None


def fetch_upstox_historical_data(instrument_key, interval, from_date, to_date, access_token):
    api_instance = upstox_client.HistoryApi()
    api_instance.api_client.configuration.access_token = access_token
    try:
        api_response = api_instance.get_historical_candle_data1(
            instrument_key, interval, to_date, from_date, "v2"
        )
        if api_response.status == "success" and api_response.data.candles:
            df = pd.DataFrame(
                api_response.data.candles,
                columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']
            )
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            df.sort_index(inplace=True)
            return df
    except ApiException as e:
        st.error(f"Upstox API Error for {instrument_key}: {e}")
    except Exception as e:
        st.error(f"General Error for {instrument_key}: {e}")
    return pd.DataFrame()


def map_period_to_dates(period):
    to_date = datetime.date.today()
    days = {"3mo": 90, "6mo": 180, "1y": 365, "2y": 730}.get(period, 180)
    from_date = to_date - datetime.timedelta(days=days)
    return from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d')


# -------------------------------------------------
# App Config
# -------------------------------------------------
st.set_page_config(layout="wide", page_title="OBV Divergence Master")
st.title("📊 OBV Divergence Screener (Dual-View)")


@st.cache_resource
def load_model():
    return joblib.load("divergence_model.pkl")

model = load_model()


# -------------------------------------------------
# Data Fetching
# -------------------------------------------------
@st.cache_data(ttl=86400)
def fetch_nifty50_stocks():
    return [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
        "BHARTIARTL", "KOTAKBANK", "BAJFINANCE", "LT", "ASIANPAINT", "AXISBANK", "MARUTI",
        "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND", "WIPRO", "DMART", "HCLTECH",
        "TECHM", "POWERGRID", "BAJAJFINSV", "NTPC", "TATAMOTORS", "COALINDIA", "ADANIPORTS",
        "ONGC", "HINDALCO", "GRASIM", "JSWSTEEL", "CIPLA", "DRREDDY", "EICHERMOT",
        "BRITANNIA", "APOLLOHOSP", "DIVISLAB", "GODREJCP", "PIDILITIND", "BERGEPAINT",
        "INDUSINDBK", "BANKBARODA", "SIEMENS", "DLF", "BAJAJ-AUTO", "TATASTEEL",
        "ADANIENT", "VEDL",
    ]


@st.cache_data(ttl=86400)
def fetch_nse500_stocks():
    try:
        r = requests.get(
            "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
            headers={'User-Agent': 'Mozilla/5.0'}, timeout=30, verify=False
        )
        if r.status_code == 200:
            return pd.read_csv(StringIO(r.text))['Symbol'].tolist()
    except:
        pass
    return fetch_nifty50_stocks()


@st.cache_data(ttl=86400)
def fetch_all_nse_stocks():
    try:
        r = requests.get(
            "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
            headers={'User-Agent': 'Mozilla/5.0'}, timeout=30, verify=False
        )
        if r.status_code == 200:
            return pd.read_csv(StringIO(r.text))['SYMBOL'].tolist()
    except:
        pass
    return fetch_nifty50_stocks()


@st.cache_data(ttl=86400)
def fetch_all_bse_stocks():
    try:
        r = requests.get(
            "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry=&segment=Equity&status=Active",
            headers={'User-Agent': 'Mozilla/5.0'}, timeout=30, verify=False
        )
        if r.status_code == 200:
            return [item['scrip_cd'] for item in r.json().get('Table', [])]
    except:
        pass
    return ["500325", "532540", "500180"]


@st.cache_data(ttl=86400)
def fetch_major_indices():
    return ["NIFTY 50", "NIFTY BANK", "NIFTY FIN SERVICE", "NIFTY MIDCAP SELECT"]


@st.cache_data(ttl=86400)
def fetch_nse_fno_stocks():
    df = get_upstox_instruments()
    if df.empty:
        return []
    fno_df = df[(df['exchange'] == 'NSE_FO') & (df['instrument_type'] == 'FUTSTK')]
    symbols = fno_df['tradingsymbol'].str.replace(r'\d{2}[A-Z]{3}FUT$', '', regex=True).unique().tolist()
    return sorted([s for s in symbols if s])


# -------------------------------------------------
# OBV + Divergence Logic
# -------------------------------------------------
def calculate_obv(df):
    df = df.copy()
    df['Change'] = df['Close'].diff()
    df['OBV_Vol'] = np.where(df['Change'] > 0, df['Volume'],
                             np.where(df['Change'] < 0, -df['Volume'], 0))
    df['OBV'] = df['OBV_Vol'].cumsum().fillna(0)
    return df


@st.cache_data(ttl=86400)
def get_company_name(ticker, instruments_df):
    try:
        match = instruments_df[instruments_df['tradingsymbol'] == ticker.split('.')[0]]
        if not match.empty:
            return match.iloc[0]['name']
    except:
        pass
    return ""


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=series.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=series.index)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def get_pivots(data, order=5):
    return (
        argrelextrema(data.values, np.greater, order=order)[0],
        argrelextrema(data.values, np.less, order=order)[0],
    )


def detect_divergence(df, order=5):
    price_highs_idx, price_lows_idx = get_pivots(df['Close'], order)
    results = []

    if len(price_highs_idx) >= 2:
        for i in range(len(price_highs_idx[-5:]) - 1, 0, -1):
            peaks = price_highs_idx[-5:]
            ci, pi = peaks[i], peaks[i - 1]
            if df['Close'].iloc[ci] > df['Close'].iloc[pi] and df['OBV'].iloc[ci] < df['OBV'].iloc[pi]:
                results.append({
                    "Type": "Bearish",
                    "P1_Idx": pi, "P2_Idx": ci,
                    "P1_Date": df.index[pi], "P2_Date": df.index[ci],
                    "P1_Price": df['Close'].iloc[pi], "P2_Price": df['Close'].iloc[ci],
                    "P1_OBV": df['OBV'].iloc[pi], "P2_OBV": df['OBV'].iloc[ci],
                })
                break

    if len(price_lows_idx) >= 2:
        for i in range(len(price_lows_idx[-5:]) - 1, 0, -1):
            lows = price_lows_idx[-5:]
            ci, pi = lows[i], lows[i - 1]
            if df['Close'].iloc[ci] < df['Close'].iloc[pi] and df['OBV'].iloc[ci] > df['OBV'].iloc[pi]:
                results.append({
                    "Type": "Bullish",
                    "P1_Idx": pi, "P2_Idx": ci,
                    "P1_Date": df.index[pi], "P2_Date": df.index[ci],
                    "P1_Price": df['Close'].iloc[pi], "P2_Price": df['Close'].iloc[ci],
                    "P1_OBV": df['OBV'].iloc[pi], "P2_OBV": df['OBV'].iloc[ci],
                })
                break

    return results, price_highs_idx, price_lows_idx


# -------------------------------------------------
# AI Summary
# -------------------------------------------------
def get_ai_summary(api_key, results_data):
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""
You are a financial analyst assistant.
Here are the findings from an OBV Divergence Scan on Indian Stocks:
{results_data[:15]}
Provide a concise but insightful summary. Highlight the most significant Bullish and Bearish setups.
Explain what these divergences might indicate for the short-term trend. Format in clean Markdown.
"""}],
        temperature=0.7, max_tokens=1024,
    )
    return completion.choices[0].message.content


# -------------------------------------------------
# Visualization – Static
# -------------------------------------------------
def plot_static_matplotlib(df, divs, price_highs, price_lows, ticker, vol_method, vol_window):
    vol_series = compute_volatility(df, vol_method, window=vol_window)
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={'height_ratios': [2, 1, 1]}
    )
    ax1.plot(df.index, df['Close'], color='black', linewidth=1)
    ax1.scatter(df.index[price_highs], df['Close'].iloc[price_highs], color='red', s=10, alpha=0.5)
    ax1.scatter(df.index[price_lows], df['Close'].iloc[price_lows], color='green', s=10, alpha=0.5)
    ax1.set_title(f"{ticker} - Price Action");
    ax1.grid(True, alpha=0.3);
    ax1.set_ylabel("Price")

    ax2.plot(df.index, df['OBV'], color='purple', linewidth=1)
    ax2.set_title("OBV");
    ax2.grid(True, alpha=0.3);
    ax2.set_ylabel("OBV")

    vc = '#d62728' if 'GARCH' in vol_method else '#1f77b4'
    ax3.plot(vol_series.index, vol_series.values, color=vc, linewidth=1)
    ax3.fill_between(vol_series.index, vol_series.values, alpha=0.15, color=vc)
    ax3.set_title(vol_series.name);
    ax3.grid(True, alpha=0.3);
    ax3.set_ylabel(vol_series.name)

    for d in divs:
        c = 'green' if d['Type'] == 'Bullish' else 'red'
        ax1.plot([d['P1_Date'], d['P2_Date']], [d['P1_Price'], d['P2_Price']], color=c, lw=2, ls='--')
        ax2.plot([d['P1_Date'], d['P2_Date']], [d['P1_OBV'], d['P2_OBV']], color=c, lw=2, ls='--')
        ax1.annotate(d['Type'], (d['P2_Date'], d['P2_Price']),
                     xytext=(10, 0), textcoords='offset points', color=c, weight='bold')
    plt.tight_layout()
    st.pyplot(fig)


# -------------------------------------------------
# Visualization – Interactive (JS synchronized hover)
# -------------------------------------------------
def plot_interactive_plotly(df, divs, ticker, vol_method, vol_window):
    """
    Renders 4 Plotly subplots inside a single HTML component.
    A small JS block listens to plotly_hover on the unified figure and
    uses Plotly.Fx.hover() to force ALL subplots to show their tooltip
    simultaneously at the same x-position.
    """
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = compute_rsi(df['Close'], period=14)

    vol_series = compute_volatility(df, vol_method, window=vol_window)
    vol_label = vol_series.name
    vol_color_hex = '#e74c3c' if 'GARCH' in vol_method else '#2980b9'

    def hex_rgba(h, a=0.15):
        h = h.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{a})'

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.45, 0.18, 0.15, 0.22],
        subplot_titles=(f"{ticker} Price", "OBV", "RSI", vol_label),
    )

    fig.add_trace(go.Ohlc(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price',
        increasing=dict(line=dict(color='#26a69a')),
        decreasing=dict(line=dict(color='#ef5350')),
        hovertemplate='O:%{open:.2f} H:%{high:.2f} L:%{low:.2f} C:%{close:.2f}<extra>Price</extra>',
        hoverlabel=dict(bgcolor='#1e222d', font_color='white', font_size=12),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA20'], name='SMA20',
        line=dict(color='#4e9af1', width=1),
        hovertemplate='%{y:.2f}<extra>SMA20</extra>',
        hoverlabel=dict(bgcolor='#4e9af1', font_color='white'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA50'], name='SMA50',
        line=dict(color='#f0a500', width=1),
        hovertemplate='%{y:.2f}<extra>SMA50</extra>',
        hoverlabel=dict(bgcolor='#f0a500', font_color='white'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['OBV'], name='OBV',
        line=dict(color='#9b59b6', width=1.2),
        hovertemplate='%{y:,.0f}<extra>OBV</extra>',
        hoverlabel=dict(bgcolor='#9b59b6', font_color='white'),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'], name='RSI',
        line=dict(color='#1abc9c', width=1.2),
        hovertemplate='%{y:.1f}<extra>RSI</extra>',
        hoverlabel=dict(bgcolor='#1abc9c', font_color='white'),
    ), row=3, col=1)
    fig.add_hline(y=70, line=dict(color='rgba(231,76,60,0.6)', dash='dot'), row=3, col=1)
    fig.add_hline(y=30, line=dict(color='rgba(46,204,113,0.6)', dash='dot'), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=vol_series.index, y=vol_series.values,
        name=vol_label,
        line=dict(color=vol_color_hex, width=1.5),
        fill='tozeroy', fillcolor=hex_rgba(vol_color_hex, 0.15),
        hovertemplate='%{y:.3f}%<extra>' + vol_label + '</extra>',
        hoverlabel=dict(bgcolor=vol_color_hex, font_color='white'),
    ), row=4, col=1)

    for d in divs:
        c = '#2ecc71' if d['Type'] == 'Bullish' else '#e74c3c'
        fig.add_trace(go.Scatter(
            x=[d['P1_Date'], d['P2_Date']], y=[d['P1_Price'], d['P2_Price']],
            mode='lines+markers', line=dict(color=c, width=2, dash='dot'),
            marker=dict(size=7, color=c), showlegend=False,
            hovertemplate='%{y:.2f}<extra>' + d['Type'] + ' Price</extra>',
            hoverlabel=dict(bgcolor=c, font_color='white'),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[d['P1_Date'], d['P2_Date']], y=[d['P1_OBV'], d['P2_OBV']],
            mode='lines+markers', line=dict(color=c, width=2, dash='dot'),
            marker=dict(size=7, color=c), showlegend=False,
            hovertemplate='%{y:,.0f}<extra>' + d['Type'] + ' OBV</extra>',
            hoverlabel=dict(bgcolor=c, font_color='white'),
        ), row=2, col=1)

    fig.update_xaxes(
        showspikes=True, spikemode='across', spikesnap='cursor',
        spikecolor='rgba(200,200,200,0.5)', spikethickness=1, spikedash='dot',
        gridcolor='rgba(255,255,255,0.06)',
    )
    fig.update_yaxes(showspikes=False, gridcolor='rgba(255,255,255,0.06)',
                     zerolinecolor='rgba(255,255,255,0.1)')

    fig.update_layout(
        hovermode='x',
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=False,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', size=11),
        margin=dict(l=60, r=20, t=50, b=40),
    )

    fig_json = fig.to_json()

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #0e1117; }}
    #plt {{ width: 100%; }}
  </style>
</head>
<body>
<div id="plt"></div>
<script>
(function () {{
  const fig = {fig_json};

  Plotly.newPlot('plt', fig.data, fig.layout, {{
    responsive: true,
    displayModeBar: false,
    scrollZoom: true,
  }}).then(function (gd) {{
    const xaxisForRow = ['x', 'x2', 'x3', 'x4'];
    const anchorCurve = {{}};

    gd.data.forEach(function (trace, ci) {{
      const xref = trace.xaxis || 'x';
      const idx  = xaxisForRow.indexOf(xref);
      if (idx !== -1 && !(idx in anchorCurve)) {{
        anchorCurve[idx] = ci;
      }}
    }});

    function nearestIdx(traceX, xVal) {{
      if (!traceX || traceX.length === 0) return 0;
      const isDate = typeof traceX[0] === 'string';
      const target = isDate ? new Date(xVal).getTime() : Number(xVal);
      let best = 0, bestDist = Infinity;
      traceX.forEach(function (v, i) {{
        const vn   = isDate ? new Date(v).getTime() : Number(v);
        const dist = Math.abs(vn - target);
        if (dist < bestDist) {{ bestDist = dist; best = i; }}
      }});
      return best;
    }}

    let busy = false;

    gd.on('plotly_hover', function (evt) {{
      if (busy || !evt.points || evt.points.length === 0) return;
      busy = true;
      const xVal  = evt.points[0].x;
      const hpts  = [];
      Object.keys(anchorCurve).forEach(function (subIdx) {{
        const ci    = anchorCurve[subIdx];
        const trace = gd.data[ci];
        const ptIdx = nearestIdx(trace.x, xVal);
        hpts.push({{ curveNumber: ci, pointNumber: ptIdx }});
      }});
      Plotly.Fx.hover(gd, hpts);
      setTimeout(function () {{ busy = false; }}, 16);
    }});

    gd.on('plotly_unhover', function () {{
      if (busy) return;
      busy = true;
      Plotly.Fx.unhover(gd);
      setTimeout(function () {{ busy = false; }}, 16);
    }});
  }});
}})();
</script>
</body>
</html>
"""
    components.html(html, height=920, scrolling=False)


# -------------------------------------------------
# Visualization – ECharts (TRUE synchronized tooltips)
# -------------------------------------------------
def plot_echarts_synchronized(df, divs, ticker, vol_method, vol_window):
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = compute_rsi(df['Close'], period=14)
    vol_series = compute_volatility(df, vol_method, window=vol_window)

    dates = [str(ts)[:10] for ts in df.index]

    def to_list(arr):
        return [
            None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4)
            for v in arr
        ]

    # FIX 1: ECharts candlestick with category xAxis needs data as
    # [open, close, low, high] — NO date in the array; the index
    # position maps to the category (dates[i]).
    ohlc_data = [
        [
            round(float(df['Open'].iloc[i]), 2),  # 0 = open
            round(float(df['Close'].iloc[i]), 2),  # 1 = close
            round(float(df['Low'].iloc[i]), 2),  # 2 = low
            round(float(df['High'].iloc[i]), 2),  # 3 = high
        ]
        for i in range(len(df))
    ]

    sma20_data = to_list(df['SMA20'].values)
    sma50_data = to_list(df['SMA50'].values)
    obv_data = to_list(df['OBV'].values)
    rsi_data = to_list(df['RSI'].values)
    vol_data = to_list(vol_series.values)
    vol_label = vol_series.name
    vol_color = '#e74c3c' if 'GARCH' in vol_method else '#3498db'
    # FIX 2: pre-compute the rgba string instead of fragile string replacement
    vol_color_rgba = 'rgba(231,76,60,0.12)' if 'GARCH' in vol_method else 'rgba(52,152,219,0.12)'

    # --- Divergence mark lines ---
    price_mark_lines = []
    obv_mark_lines = []
    for d in divs:
        d1 = str(d['P1_Date'])[:10]
        d2 = str(d['P2_Date'])[:10]
        color = '#2ecc71' if d['Type'] == 'Bullish' else '#e74c3c'
        label = d['Type']
        price_mark_lines.append([
            {"name": label, "xAxis": d1, "yAxis": round(float(d['P1_Price']), 2),
             "itemStyle": {"color": color}, "label": {"show": False}},
            {"xAxis": d2, "yAxis": round(float(d['P2_Price']), 2)},
        ])
        obv_mark_lines.append([
            {"name": label, "xAxis": d1, "yAxis": round(float(d['P1_OBV']), 0),
             "itemStyle": {"color": color},
             "label": {"show": True, "formatter": label, "color": color}},
            {"xAxis": d2, "yAxis": round(float(d['P2_OBV']), 0)},
        ])

    price_mark_lines_json = json.dumps(price_mark_lines)
    obv_mark_lines_json = json.dumps(obv_mark_lines)
    ohlc_json = json.dumps(ohlc_data)
    sma20_json = json.dumps(sma20_data)
    sma50_json = json.dumps(sma50_data)
    obv_json = json.dumps(obv_data)
    rsi_json = json.dumps(rsi_data)
    vol_json = json.dumps(vol_data)
    dates_json = json.dumps(dates)

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html, body {{ background: #0e1117; width: 100%; }}
    .chart-wrap {{ width: 100%; }}
    .chart-panel {{ width: 100%; }}
    #chart-price {{ height: 360px; }}
    #chart-obv   {{ height: 200px; }}
    #chart-rsi   {{ height: 160px; }}
    #chart-vol   {{ height: 190px; }}
    .panel-title {{
      color: #8b9dc3;
      font-family: 'Segoe UI', sans-serif;
      font-size: 11px;
      letter-spacing: 1px;
      text-transform: uppercase;
      padding: 6px 12px 0;
      background: #0e1117;
    }}
  </style>
</head>
<body>
<div class="chart-wrap">
  <div class="chart-panel">
    <div class="panel-title">{ticker} — Price / OHLC</div>
    <div id="chart-price"></div>
  </div>
  <div class="chart-panel">
    <div class="panel-title">OBV</div>
    <div id="chart-obv"></div>
  </div>
  <div class="chart-panel">
    <div class="panel-title">RSI (14)</div>
    <div id="chart-rsi"></div>
  </div>
  <div class="chart-panel">
    <div class="panel-title">{vol_label}</div>
    <div id="chart-vol"></div>
  </div>
</div>

<script>
(function() {{
  const dates       = {dates_json};
  const ohlcData    = {ohlc_json};
  const sma20Data   = {sma20_json};
  const sma50Data   = {sma50_json};
  const obvData     = {obv_json};
  const rsiData     = {rsi_json};
  const volData     = {vol_json};
  const priceMark   = {price_mark_lines_json};
  const obvMark     = {obv_mark_lines_json};

  const BG        = '#0e1117';
  const GRID_BG   = '#161b22';
  const GRID_LINE = 'rgba(255,255,255,0.06)';
  const AXIS_TICK = '#5a6a8a';
  const AXIS_LBL  = '#8b9dc3';
  const TOOLTIP_BG= '#1e2535';

  const sharedAxisPointer = {{
    link: [{{ xAxisIndex: 'all' }}],
    label: {{ backgroundColor: '#2c3e50' }}
  }};

  const sharedTooltip = {{
    trigger: 'axis',
    axisPointer: {{
      type: 'cross',
      crossStyle: {{ color: 'rgba(180,180,200,0.4)', width: 1 }},
      lineStyle:  {{ color: 'rgba(180,180,200,0.3)', width: 1, type: 'dashed' }},
      label: {{ show: true, backgroundColor: '#2c3e50', color: '#e0e6f0', fontSize: 11 }}
    }},
    backgroundColor: TOOLTIP_BG,
    borderColor: '#2c3e50',
    borderWidth: 1,
    textStyle: {{ color: '#e0e6f0', fontSize: 11, fontFamily: 'Segoe UI, sans-serif' }},
    extraCssText: 'box-shadow: 0 4px 16px rgba(0,0,0,0.5); border-radius: 6px;',
  }};

  const sharedGrid = {{
    left: '60px', right: '20px', top: '8px', bottom: '28px',
    backgroundColor: GRID_BG,
    show: true,
    borderColor: 'transparent',
  }};

  const xAxisBase = {{
    type: 'category',
    data: dates,
    boundaryGap: true,
    axisLine:   {{ lineStyle: {{ color: AXIS_TICK }} }},
    axisTick:   {{ lineStyle: {{ color: AXIS_TICK }} }},
    axisLabel:  {{ color: AXIS_LBL, fontSize: 10 }},
    splitLine:  {{ show: true, lineStyle: {{ color: GRID_LINE }} }},
  }};

  const yAxisBase = {{
    type: 'value',
    scale: true,
    axisLine:   {{ lineStyle: {{ color: AXIS_TICK }} }},
    axisTick:   {{ lineStyle: {{ color: AXIS_TICK }} }},
    axisLabel:  {{ color: AXIS_LBL, fontSize: 10 }},
    splitLine:  {{ show: true, lineStyle: {{ color: GRID_LINE }} }},
  }};

  // ── Chart 1: OHLC + SMA ─────────────────────────────────────────────────
  const chartPrice = echarts.init(document.getElementById('chart-price'), null,
    {{ renderer: 'canvas', useDirtyRect: true }});

  chartPrice.setOption({{
    backgroundColor: BG,
    axisPointer: sharedAxisPointer,
    tooltip: {{
      ...sharedTooltip,
      // FIX 3: candlestick params[i].value is [open,close,low,high]
      formatter: function(params) {{
        let html = `<div style="font-weight:600;margin-bottom:4px">${{params[0]?.axisValue}}</div>`;
        params.forEach(p => {{
          if (p.seriesType === 'candlestick') {{
            const v = p.value;   // [open, close, low, high]
            html += `<div style="margin:2px 0">
              O:<b>${{v[0]}}</b> &nbsp;
              H:<b>${{v[3]}}</b> &nbsp;
              L:<b>${{v[2]}}</b> &nbsp;
              C:<b>${{v[1]}}</b>
            </div>`;
          }} else {{
            if (p.value !== null && p.value !== undefined)
              html += `<div><span style="display:inline-block;width:10px;height:10px;
                border-radius:50%;background:${{p.color}};margin-right:5px"></span>
                ${{p.seriesName}}: <b>${{typeof p.value === 'number' ? p.value.toFixed(2) : p.value}}</b></div>`;
          }}
        }});
        return html;
      }}
    }},
    grid: sharedGrid,
    xAxis: {{ ...xAxisBase }},
    yAxis: {{ ...yAxisBase }},
    dataZoom: [
      {{ type: 'inside', xAxisIndex: 0, start: 0, end: 100 }},
    ],
    series: [
      {{
        name: 'OHLC',
        type: 'candlestick',
        data: ohlcData,
        itemStyle: {{
          color:        '#26a69a',
          color0:       '#ef5350',
          borderColor:  '#26a69a',
          borderColor0: '#ef5350',
        }},
        markLine: priceMark.length > 0 ? {{
          symbol: ['circle', 'arrow'],
          symbolSize: 6,
          lineStyle: {{ width: 2, type: 'dashed' }},
          data: priceMark,
          label: {{ show: false }},
        }} : undefined,
      }},
      {{
        name: 'SMA20',
        type: 'line',
        data: sma20Data,
        smooth: false,
        symbol: 'none',
        lineStyle: {{ color: '#4e9af1', width: 1.2 }},
        itemStyle: {{ color: '#4e9af1' }},
      }},
      {{
        name: 'SMA50',
        type: 'line',
        data: sma50Data,
        smooth: false,
        symbol: 'none',
        lineStyle: {{ color: '#f0a500', width: 1.2 }},
        itemStyle: {{ color: '#f0a500' }},
      }},
    ],
  }});

  // ── Chart 2: OBV ────────────────────────────────────────────────────────
  const chartObv = echarts.init(document.getElementById('chart-obv'), null,
    {{ renderer: 'canvas', useDirtyRect: true }});

  chartObv.setOption({{
    backgroundColor: BG,
    axisPointer: sharedAxisPointer,
    tooltip: {{
      ...sharedTooltip,
      formatter: function(params) {{
        let html = `<div style="font-weight:600;margin-bottom:4px">${{params[0]?.axisValue}}</div>`;
        params.forEach(p => {{
          if (p.value !== null && p.value !== undefined)
            html += `<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
              background:${{p.color}};margin-right:5px"></span>
              ${{p.seriesName}}: <b>${{typeof p.value === 'number' ? (+p.value).toLocaleString() : p.value}}</b></div>`;
        }});
        return html;
      }}
    }},
    grid: {{ ...sharedGrid }},
    xAxis: {{ ...xAxisBase, axisLabel: {{ show: false }} }},
    yAxis: {{ ...yAxisBase }},
    dataZoom: [{{ type: 'inside', xAxisIndex: 0, start: 0, end: 100 }}],
    series: [
      {{
        name: 'OBV',
        type: 'line',
        data: obvData,
        smooth: false,
        symbol: 'none',
        lineStyle: {{ color: '#9b59b6', width: 1.5 }},
        itemStyle: {{ color: '#9b59b6' }},
        areaStyle: {{ color: 'rgba(155,89,182,0.08)' }},
        markLine: obvMark.length > 0 ? {{
          symbol: ['circle', 'arrow'],
          symbolSize: 6,
          lineStyle: {{ width: 2, type: 'dashed' }},
          data: obvMark,
        }} : undefined,
      }},
    ],
  }});

  // ── Chart 3: RSI ────────────────────────────────────────────────────────
  const chartRsi = echarts.init(document.getElementById('chart-rsi'), null,
    {{ renderer: 'canvas', useDirtyRect: true }});

  chartRsi.setOption({{
    backgroundColor: BG,
    axisPointer: sharedAxisPointer,
    tooltip: {{
      ...sharedTooltip,
      formatter: function(params) {{
        let html = `<div style="font-weight:600;margin-bottom:4px">${{params[0]?.axisValue}}</div>`;
        params.forEach(p => {{
          if (p.value !== null && p.value !== undefined && p.seriesName !== 'OB' && p.seriesName !== 'OS')
            html += `<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
              background:${{p.color}};margin-right:5px"></span>
              ${{p.seriesName}}: <b>${{typeof p.value === 'number' ? p.value.toFixed(1) : p.value}}</b></div>`;
        }});
        return html;
      }}
    }},
    grid: {{ ...sharedGrid }},
    xAxis: {{ ...xAxisBase, axisLabel: {{ show: false }} }},
    yAxis: {{ ...yAxisBase, min: 0, max: 100, splitNumber: 2 }},
    dataZoom: [{{ type: 'inside', xAxisIndex: 0, start: 0, end: 100 }}],
    series: [
      {{
        name: 'RSI',
        type: 'line',
        data: rsiData,
        smooth: false,
        symbol: 'none',
        lineStyle: {{ color: '#1abc9c', width: 1.5 }},
        itemStyle: {{ color: '#1abc9c' }},
      }},
      {{
        name: 'OB',
        type: 'line',
        data: dates.map(() => 70),
        symbol: 'none',
        lineStyle: {{ color: 'rgba(231,76,60,0.5)', width: 1, type: 'dashed' }},
        tooltip: {{ show: false }},
      }},
      {{
        name: 'OS',
        type: 'line',
        data: dates.map(() => 30),
        symbol: 'none',
        lineStyle: {{ color: 'rgba(46,204,113,0.5)', width: 1, type: 'dashed' }},
        tooltip: {{ show: false }},
      }},
    ],
  }});

  // ── Chart 4: Volatility ─────────────────────────────────────────────────
  const chartVol = echarts.init(document.getElementById('chart-vol'), null,
    {{ renderer: 'canvas', useDirtyRect: true }});

  chartVol.setOption({{
    backgroundColor: BG,
    axisPointer: sharedAxisPointer,
    tooltip: {{
      ...sharedTooltip,
      formatter: function(params) {{
        let html = `<div style="font-weight:600;margin-bottom:4px">${{params[0]?.axisValue}}</div>`;
        params.forEach(p => {{
          if (p.value !== null && p.value !== undefined)
            html += `<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
              background:${{p.color}};margin-right:5px"></span>
              ${{p.seriesName}}: <b>${{typeof p.value === 'number' ? p.value.toFixed(3) + '%' : p.value}}</b></div>`;
        }});
        return html;
      }}
    }},
    grid: {{ ...sharedGrid }},
    xAxis: {{ ...xAxisBase }},
    yAxis: {{ ...yAxisBase }},
    dataZoom: [{{ type: 'inside', xAxisIndex: 0, start: 0, end: 100 }}],
    series: [
      {{
        name: '{vol_label}',
        type: 'line',
        data: volData,
        smooth: false,
        symbol: 'none',
        lineStyle: {{ color: '{vol_color}', width: 1.5 }},
        itemStyle: {{ color: '{vol_color}' }},
        // FIX 2: use pre-computed rgba string, not brittle string replacement
        areaStyle: {{ color: '{vol_color_rgba}' }},
      }},
    ],
  }});

  // ── TRUE SYNCHRONIZATION ─────────────────────────────────────────────────
  echarts.connect([chartPrice, chartObv, chartRsi, chartVol]);

  // ── Sync dataZoom across all charts ─────────────────────────────────────
  function syncZoom(source, targets) {{
    source.on('dataZoom', function() {{
      const opt = source.getOption();
      const dz  = opt.dataZoom[0];
      targets.forEach(t => {{
        t.dispatchAction({{ type: 'dataZoom', start: dz.start, end: dz.end }});
      }});
    }});
  }}

  syncZoom(chartPrice, [chartObv, chartRsi, chartVol]);
  syncZoom(chartObv,   [chartPrice, chartRsi, chartVol]);
  syncZoom(chartRsi,   [chartPrice, chartObv, chartVol]);
  syncZoom(chartVol,   [chartPrice, chartObv, chartRsi]);

  window.addEventListener('resize', function() {{
    [chartPrice, chartObv, chartRsi, chartVol].forEach(c => c.resize());
  }});
}})();
</script>
</body>
</html>
"""
    components.html(html, height=950, scrolling=False)


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
        num = st.slider("Number of stocks to scan", min(1, len(tickers)), len(tickers), min(100, len(tickers)))
        tickers = tickers[:num]
    else:
        raw = st.text_area("Stocks (comma sep)", "RELIANCE.NS, TCS.NS, HDFCBANK.NS, TATAMOTORS.NS, INFY.NS")
        tickers = [t.strip().upper() for t in raw.split(',')]

    period = st.selectbox("Lookback Period", ["3mo", "6mo", "1y", "2y"], index=1)
    timeframe = st.selectbox("Timeframe", ["Daily", "Weekly"], index=0)
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
# ML Feature Builder (matches training exactly)
# -------------------------------------------------
def build_ml_features(df, div):
    idx = div["P2_Idx"]
    if idx < 50:
        return None

    try:
        df["RSI"] = compute_rsi(df["Close"])

        rsi = df["RSI"].iloc[idx]
        rsi_slope = df["RSI"].iloc[idx] - df["RSI"].iloc[idx - 3]
        obv_slope = df["OBV"].iloc[idx] - df["OBV"].iloc[idx - 5]

        ret_5d = df["Close"].pct_change(5).iloc[idx]
        ret_10d = df["Close"].pct_change(10).iloc[idx]

        volatility = df["Close"].pct_change().rolling(20).std().iloc[idx]

        volume_spike = df["Volume"].iloc[idx] / df["Volume"].rolling(20).mean().iloc[idx]

        sma20 = df["Close"].rolling(20).mean().iloc[idx]
        sma50 = df["Close"].rolling(50).mean().iloc[idx]

        sma20_dist = (df["Close"].iloc[idx] - sma20) / df["Close"].iloc[idx]
        sma50_dist = (df["Close"].iloc[idx] - sma50) / df["Close"].iloc[idx]

        div_binary = 1 if div["Type"] == "Bullish" else 0

        features = {
            "rsi": rsi,
            "rsi_slope": rsi_slope,
            "obv_slope": obv_slope,
            "ret_5d": ret_5d,
            "ret_10d": ret_10d,
            "volatility": volatility,
            "volume_spike": volume_spike,
            "sma20_dist": sma20_dist,
            "sma50_dist": sma50_dist,
            "div_type": div_binary,
        }

        return pd.DataFrame([features])

    except:
        return None


# -------------------------------------------------
# Run Screener
# -------------------------------------------------
if st.button("🚀 Run Screener"):
    if not upstox_access_token:
        st.error("❌ Please enter an Upstox Access Token in the sidebar.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
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
        results_map = {}
        for ticker, df, divs, ph, pl in results_container:
            last_price = float(df['Close'].iloc[-1]) if not df.empty else np.nan
            name = get_company_name(ticker, instruments_df)
            sig_type = divs[0]['Type'] if divs else ""
            probability = None
            if divs:
                features_df = build_ml_features(df, divs[0])
                if features_df is not None:
                    prob = model.predict_proba(features_df)[0][1]
                    probability = round(prob * 100, 2)

            summary_rows.append({
                "Symbol": ticker, "Name": name,
                "Price": round(last_price, 2) if not np.isnan(last_price) else None,
                "Signal": "Yes" if divs else "No", "Type": sig_type,
                "From": divs[0]['P1_Date'].date() if divs else "",
                "To": divs[0]['P2_Date'].date() if divs else "",
                "Probability (%)": probability,
            })
            results_map[ticker] = (df, divs, ph, pl)

        st.session_state.scan_results = summary_rows
        st.session_state.results_map = results_map

        # --- Build Date-wise Divergence Summary ---
        # Fetch Nifty 50 closing data for the same period
        nifty_close_map = {}
        try:
            nifty_ikey = get_instrument_key(instruments_df, "NIFTY 50", is_index=True)
            if nifty_ikey:
                nifty_df = fetch_upstox_historical_data(
                    nifty_ikey, "day", from_date, to_date, upstox_access_token
                )
                if not nifty_df.empty:
                    for ts, row in nifty_df.iterrows():
                        nifty_close_map[str(ts.date())] = round(float(row['Close']), 2)
        except Exception:
            pass

        # Group divergences by completion date (P2_Date)
        datewise = {}  # date_str -> {bullish: count, bearish: count, stocks: [...]}
        for ticker, df_stock, divs, ph, pl in results_container:
            for d in divs:
                completion_date = str(d['P2_Date'].date()) if hasattr(d['P2_Date'], 'date') else str(d['P2_Date'])[:10]
                stock_close = round(float(d['P2_Price']), 2)
                if completion_date not in datewise:
                    datewise[completion_date] = {'bullish': 0, 'bearish': 0, 'stocks': []}
                if d['Type'] == 'Bullish':
                    datewise[completion_date]['bullish'] += 1
                else:
                    datewise[completion_date]['bearish'] += 1
                datewise[completion_date]['stocks'].append({
                    'Symbol': ticker, 'Close': stock_close, 'Type': d['Type']
                })

        datewise_rows = []
        for date_str in sorted(datewise.keys(), reverse=True):
            info = datewise[date_str]
            datewise_rows.append({
                'Date': date_str,
                'Nifty Close': nifty_close_map.get(date_str, None),
                '# Bullish OBV': info['bullish'],
                '# Bearish OBV': info['bearish'],
                'Total Divergences': info['bullish'] + info['bearish'],
            })

        st.session_state.datewise_summary = datewise_rows
        st.session_state.datewise_detail = datewise

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
    if "Probability (%)" in summary_df.columns:
        summary_df["Probability (%)"] = pd.to_numeric(
            summary_df["Probability (%)"], errors="coerce"
        )
    for col in ["Symbol", "Name", "Signal", "Type", "From", "To"]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].fillna("").astype(str)
    st.caption("👆 Click on a row to view that stock's chart below")
    event = st.dataframe(
        summary_df, use_container_width=True,
        on_select="rerun", selection_mode="single-rows",
    )

    # --- CSV Download for Scan Results ---
    scan_csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Scan Results as CSV",
        data=scan_csv,
        file_name="obv_scan_results.csv",
        mime="text/csv",
    )

    # --- Auto-show chart for selected row ---
    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        row_idx = selected_rows[0]
        selected_symbol = summary_df.iloc[row_idx]["Symbol"]
        results_map = st.session_state.get("results_map", {})
        if selected_symbol in results_map:
            df, divs, ph, pl = results_map[selected_symbol]

            st.markdown("---")
            selected_row = summary_df[summary_df["Symbol"] == selected_symbol]
            if not selected_row.empty and "Probability (%)" in summary_df.columns:
                prob_val = selected_row["Probability (%)"].values[0]
                if prob_val is not None and not pd.isna(prob_val):
                    st.metric("ML Success Probability", f"{prob_val}%")

            st.subheader(f"{selected_symbol} Chart")
            if chart_style == "Static (Matplotlib)":
                plot_static_matplotlib(df, divs, ph, pl, selected_symbol, vol_method, vol_window)
            elif chart_style == "Interactive (Plotly)":
                plot_interactive_plotly(df, divs, selected_symbol, vol_method, vol_window)
            else:  # ECharts (True Sync Tooltip)
                plot_echarts_synchronized(df, divs, selected_symbol, vol_method, vol_window)

    # --- Date-wise Divergence Summary ---
    if st.session_state.get("datewise_summary"):
        st.subheader("📅 Date-wise Divergence Summary")
        datewise_df = pd.DataFrame(st.session_state["datewise_summary"])
        datewise_df['Nifty Close'] = pd.to_numeric(datewise_df['Nifty Close'], errors='coerce')
        st.dataframe(datewise_df, use_container_width=True)

        datewise_csv = datewise_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Date-wise Summary as CSV",
            data=datewise_csv,
            file_name="obv_datewise_summary.csv",
            mime="text/csv",
        )

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
