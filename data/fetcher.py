import datetime
import gzip
import io

import pandas as pd
import requests
import streamlit as st
import upstox_client
from upstox_client.rest import ApiException
from io import StringIO


# -------------------------------------------------
# Upstox Instruments
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
# Stock Universe Fetchers
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


@st.cache_data(ttl=86400)
def get_company_name(ticker, instruments_df):
    try:
        match = instruments_df[instruments_df['tradingsymbol'] == ticker.split('.')[0]]
        if not match.empty:
            return match.iloc[0]['name']
    except:
        pass
    return ""