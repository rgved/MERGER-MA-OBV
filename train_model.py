# train_model.py

import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
from indicators import calculate_obv, compute_rsi, detect_divergence_all
import requests
from io import StringIO
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib


# --------------------------------------------
# 1️⃣ Fetch NSE 500 and select 200 stocks
# --------------------------------------------

def fetch_nse500():
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    df = pd.read_csv(StringIO(r.text))
    return df["Symbol"].tolist()

symbols = fetch_nse500()
stocks = [s + ".NS" for s in symbols[:200]]


# --------------------------------------------
# 2️⃣ Download data
# --------------------------------------------

def download_data(stocks, period="2y"):
    all_data = {}

    for symbol in tqdm(stocks):
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df is not None and len(df) > 200:
            df.dropna(inplace=True)
            all_data[symbol] = df

    return all_data


# --------------------------------------------
# 3️⃣ Label function (7% in 20 days)
# --------------------------------------------

def create_label(df, idx, div_type, horizon=20, threshold=0.07):
    if idx + horizon >= len(df):
        return None

    entry_price = df["Close"].iloc[idx]
    future_prices = df["Close"].iloc[idx + 1: idx + horizon + 1]

    if div_type == "Bullish":
        return 1 if (future_prices.max() - entry_price) / entry_price >= threshold else 0
    else:
        return 1 if (entry_price - future_prices.min()) / entry_price >= threshold else 0


# --------------------------------------------
# 4️⃣ Feature extraction (10 features)
# --------------------------------------------

def extract_features(df, idx, div_type):

    if idx < 50:
        return None  # not enough history

    try:
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

        div_binary = 1 if div_type == "Bullish" else 0

        return {
            "rsi": rsi,
            "rsi_slope": rsi_slope,
            "obv_slope": obv_slope,
            "ret_5d": ret_5d,
            "ret_10d": ret_10d,
            "volatility": volatility,
            "volume_spike": volume_spike,
            "sma20_dist": sma20_dist,
            "sma50_dist": sma50_dist,
            "div_type": div_binary
        }

    except:
        return None


# --------------------------------------------
# 5️⃣ Main Execution
# --------------------------------------------

if __name__ == "__main__":

    print("Downloading data...")
    data = download_data(stocks)
    print(f"Downloaded data for {len(data)} stocks.")

    X = []
    y = []

    for symbol, df in data.items():

        df = calculate_obv(df)
        df["RSI"] = compute_rsi(df["Close"])

        divergences = detect_divergence_all(df)

        for div_type, idx in divergences:

            label = create_label(df, idx, div_type)
            if label is None:
                continue

            features = extract_features(df, idx, div_type)
            if features is None:
                continue

            X.append(features)
            y.append(label)

    X = pd.DataFrame(X)
    y = pd.Series(y)

    print("Total samples:", len(X))
    print("Success rate:", y.mean())

    # ----------------------------------------
    # 6️⃣ Time-based train/test split
    # ----------------------------------------

    split_index = int(len(X) * 0.7)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

# ----------------------------------------
    # 7️⃣ Train & Save Model
    # ----------------------------------------
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'divergence_model.pkl')
    print("\n✅ Model saved as divergence_model.pkl")

    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")