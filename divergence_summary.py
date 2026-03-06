import upstox_client
from upstox_client.rest import ApiException
import pandas as pd
import numpy as np
import datetime
import requests
import gzip
import io
import json
from io import StringIO
from scipy.signal import argrelextrema
import urllib3
import time

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# CONFIGURATION - PLEASE PROVIDE TOKEN HERE
# ==========================================
ACCESS_TOKEN = "PASTE_YOUR_ACCESS_TOKEN_HERE"
LOOKBACK_PERIOD = "6mo"  # Options: 3mo, 6mo, 1y, 2y
SENSITIVITY = 5          # Pivot sensitivity
# ==========================================

def get_upstox_instruments():
    url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with gzip.open(io.BytesIO(response.content), 'rt') as f:
                df = pd.read_csv(f)
            return df[df['exchange'].isin(['NSE_EQ', 'BSE_EQ', 'NSE_FO', 'NSE_INDEX'])]
    except Exception as e:
        print(f"Error fetching instrument list: {e}")
    return pd.DataFrame()

def get_instrument_key(instruments_df, ticker, is_index=False):
    if instruments_df.empty:
        return None
    if is_index:
        index_map = {"NIFTY 50": "NIFTY"}
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
    match = instruments_df[
        (instruments_df['tradingsymbol'].str.upper() == clean_ticker) &
        (instruments_df['exchange'] == 'NSE_EQ')
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
    except Exception as e:
        pass
    return pd.DataFrame()

def map_period_to_dates(period):
    to_date = datetime.date.today()
    days = {"3mo": 90, "6mo": 180, "1y": 365, "2y": 730}.get(period, 180)
    from_date = to_date - datetime.timedelta(days=days)
    return from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d')

def fetch_nse_fno_stocks(instruments_df):
    if instruments_df.empty:
        return []
    fno_df = instruments_df[(instruments_df['exchange'] == 'NSE_FO') & (instruments_df['instrument_type'] == 'FUTSTK')]
    symbols = fno_df['tradingsymbol'].str.replace(r'\d{2}[A-Z]{3}FUT$', '', regex=True).unique().tolist()
    return sorted([s for s in symbols if s])

def calculate_obv(df):
    df = df.copy()
    df['Change'] = df['Close'].diff()
    df['OBV_Vol'] = np.where(df['Change'] > 0, df['Volume'],
                             np.where(df['Change'] < 0, -df['Volume'], 0))
    df['OBV'] = df['OBV_Vol'].cumsum().fillna(0)
    return df

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
                results.append({"Type": "Bearish", "To": df.index[ci]})
                break
    if len(price_lows_idx) >= 2:
        for i in range(len(price_lows_idx[-5:]) - 1, 0, -1):
            lows = price_lows_idx[-5:]
            ci, pi = lows[i], lows[i - 1]
            if df['Close'].iloc[ci] < df['Close'].iloc[pi] and df['OBV'].iloc[ci] > df['OBV'].iloc[pi]:
                results.append({"Type": "Bullish", "To": df.index[ci]})
                break
    return results

def main():
    if ACCESS_TOKEN == "PASTE_YOUR_ACCESS_TOKEN_HERE":
        print("❌ Error: Please provide an Upstox Access Token in the script.")
        return

    print("🚀 Starting Divergence Summary Generation...")
    instruments_df = get_upstox_instruments()
    from_date, to_date = map_period_to_dates(LOOKBACK_PERIOD)
    
    # Fetch Nifty 50 Index Data
    print("📈 Fetching Nifty 50 Index Data...")
    nifty_ikey = get_instrument_key(instruments_df, "NIFTY 50", is_index=True)
    nifty_df = fetch_upstox_historical_data(nifty_ikey, "day", from_date, to_date, ACCESS_TOKEN)
    if nifty_df.empty:
        print("❌ Error: Could not fetch Nifty 50 data.")
        return
    
    nifty_closes = nifty_df['Close'].to_dict()
    
    # Fetch F&O Stocks
    stocks = fetch_nse_fno_stocks(instruments_df)
    print(f"🔍 Found {len(stocks)} F&O stocks. Analyzing...")
    
    all_divergences = []
    
    for idx, ticker in enumerate(stocks):
        if idx % 20 == 0:
            print(f"Analyzing {idx}/{len(stocks)}: {ticker}...")
        
        ikey = get_instrument_key(instruments_df, ticker)
        if not ikey: continue
        
        df = fetch_upstox_historical_data(ikey, "day", from_date, to_date, ACCESS_TOKEN)
        if df.empty: continue
        
        df = calculate_obv(df)
        divs = detect_divergence(df, order=SENSITIVITY)
        
        for d in divs:
            all_divergences.append({
                "Symbol": ticker,
                "Type": d["Type"],
                "Date": d["To"].normalize()  # Formation completion date
            })
        
        # Avoid rate limiting
        time.sleep(0.05)

    if not all_divergences:
        print("❌ No divergences found.")
        return

    # Aggregate by Date
    df_divs = pd.DataFrame(all_divergences)
    summary = df_divs.groupby(['Date', 'Type']).size().unstack(fill_value=0).reset_index()
    
    if 'Bullish' not in summary.columns: summary['Bullish'] = 0
    if 'Bearish' not in summary.columns: summary['Bearish'] = 0
    
    # Merge with Nifty Close
    summary['Nifty Close'] = summary['Date'].map(lambda x: nifty_closes.get(x, nifty_closes.get(pd.Timestamp(x), np.nan)))
    
    # Format and Display
    summary = summary[['Date', 'Nifty Close', 'Bullish', 'Bearish']]
    summary = summary.sort_values('Date', ascending=False)
    
    print("\n✅ DATE-WISE DIVERGENCE SUMMARY")
    print("====================================")
    print(summary.to_string(index=False))
    
    # Save to CSV
    output_file = f"divergence_summary_{datetime.date.today()}.csv"
    summary.to_csv(output_file, index=False)
    print(f"\n📁 Summary saved to {output_file}")

if __name__ == "__main__":
    main()
