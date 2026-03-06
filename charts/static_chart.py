import matplotlib.pyplot as plt
import streamlit as st

from utils.volatility import compute_volatility


def plot_static_matplotlib(df, divs, price_highs, price_lows, ticker, vol_method, vol_window):
    vol_series = compute_volatility(df, vol_method, window=vol_window)
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={'height_ratios': [2, 1, 1]}
    )
    ax1.plot(df.index, df['Close'], color='black', linewidth=1)
    ax1.scatter(df.index[price_highs], df['Close'].iloc[price_highs], color='red', s=10, alpha=0.5)
    ax1.scatter(df.index[price_lows], df['Close'].iloc[price_lows], color='green', s=10, alpha=0.5)
    ax1.set_title(f"{ticker} - Price Action")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Price")

    ax2.plot(df.index, df['OBV'], color='purple', linewidth=1)
    ax2.set_title("OBV")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("OBV")

    vc = '#d62728' if 'GARCH' in vol_method else '#1f77b4'
    ax3.plot(vol_series.index, vol_series.values, color=vc, linewidth=1)
    ax3.fill_between(vol_series.index, vol_series.values, alpha=0.15, color=vc)
    ax3.set_title(vol_series.name)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel(vol_series.name)

    for d in divs:
        c = 'green' if d['Type'] == 'Bullish' else 'red'
        ax1.plot([d['P1_Date'], d['P2_Date']], [d['P1_Price'], d['P2_Price']], color=c, lw=2, ls='--')
        ax2.plot([d['P1_Date'], d['P2_Date']], [d['P1_OBV'], d['P2_OBV']], color=c, lw=2, ls='--')
        ax1.annotate(d['Type'], (d['P2_Date'], d['P2_Price']),
                     xytext=(10, 0), textcoords='offset points', color=c, weight='bold')
    plt.tight_layout()
    st.pyplot(fig)