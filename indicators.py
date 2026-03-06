import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
def calculate_obv(df):
    df = df.copy()
    df['Change'] = df['Close'].diff()
    df['OBV_Vol'] = np.where(df['Change'] > 0, df['Volume'], np.where(df['Change'] < 0, -df['Volume'], 0))
    df['OBV'] = df['OBV_Vol'].cumsum().fillna(0)
    return df

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

def detect_divergence_all(df, order=5, min_gap=10):

    if "OBV" not in df.columns:
        df = calculate_obv(df)

    price_highs_idx, price_lows_idx = get_pivots(df['Close'], order)
    results = []
    last_signal_idx = -min_gap

    # Bearish
    for i in range(1, len(price_highs_idx)):
        pi = price_highs_idx[i - 1]
        ci = price_highs_idx[i]

        if df['Close'].iloc[ci] > df['Close'].iloc[pi] and \
           df['OBV'].iloc[ci] < df['OBV'].iloc[pi]:

            if ci - last_signal_idx >= min_gap:
                results.append(("Bearish", ci))
                last_signal_idx = ci

    # Bullish
    for i in range(1, len(price_lows_idx)):
        pi = price_lows_idx[i - 1]
        ci = price_lows_idx[i]

        if df['Close'].iloc[ci] < df['Close'].iloc[pi] and \
           df['OBV'].iloc[ci] > df['OBV'].iloc[pi]:

            if ci - last_signal_idx >= min_gap:
                results.append(("Bullish", ci))
                last_signal_idx = ci

    return results