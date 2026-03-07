from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from moving_average.features import compute_ema, compute_sma, generate_signals


@dataclass(frozen=True)
class MAAnalysisResult:
    symbol: str
    ma_type: str
    fast_window: int
    slow_window: int
    latest_signal: str
    latest_crossover_date: str


def compute_volatility(df: pd.DataFrame, lookback: int = 20) -> float:
    returns = df["Close"].pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.rolling(lookback).std().iloc[-1] or 0.0)


def compute_trend_strength(df: pd.DataFrame, lookback: int = 30) -> float:
    closes = df["Close"].dropna().tail(lookback)
    if len(closes) < 2:
        return 0.0
    x = np.arange(len(closes), dtype=float)
    slope, _ = np.polyfit(x, closes.values.astype(float), 1)
    norm = float(closes.mean()) if closes.mean() else 1.0
    return abs(float(slope)) / norm


def compute_noise_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    closes = df["Close"].dropna().tail(lookback + 1)
    if len(closes) < 3:
        return 1.0
    path = closes.diff().abs().sum()
    displacement = abs(closes.iloc[-1] - closes.iloc[0])
    if displacement == 0:
        return float(path)
    return float(path / displacement)


def select_ma_type(volatility: float, trend_strength: float, noise_ratio: float) -> str:
    """Rule-based adaptive MA selector inspired by the MA project."""
    if noise_ratio > 3.0 or trend_strength < 0.0008:
        return "SMA"
    if volatility > 0.03:
        return "EMA"
    return "EMA"



def prepare_ma_overlay(
    price_df: pd.DataFrame,
    ma_type: str,
    fast_window: int,
    slow_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return dataframe with MA columns and a filtered crossover dataframe."""
    work = price_df.copy()
    if ma_type == "SMA":
        work["MA_Fast"] = compute_sma(work, "Close", fast_window)
        work["MA_Slow"] = compute_sma(work, "Close", slow_window)
    else:
        work["MA_Fast"] = compute_ema(work, "Close", fast_window)
        work["MA_Slow"] = compute_ema(work, "Close", slow_window)

    work = generate_signals(work, "MA_Fast", "MA_Slow")
    crossovers = work[work["MA_Crossover"] != ""]
    return work, crossovers



def apply_moving_average_analysis(
    symbol: str,
    price_df: pd.DataFrame,
    ma_pairs: Iterable[tuple[int, int]] = ((5, 20), (9, 21), (10, 50)),
) -> MAAnalysisResult:
    if price_df.empty or "Close" not in price_df.columns:
        return MAAnalysisResult(symbol, "EMA", 5, 20, "No Data", "")

    vol = compute_volatility(price_df)
    trend = compute_trend_strength(price_df)
    noise = compute_noise_ratio(price_df)
    ma_type = select_ma_type(vol, trend, noise)

    fast_window, slow_window = next(iter(ma_pairs))

    _, crossovers = prepare_ma_overlay(price_df, ma_type, fast_window, slow_window)
=======

    work = price_df.copy()
    if ma_type == "SMA":
        work["MA_Fast"] = compute_sma(work, "Close", fast_window)
        work["MA_Slow"] = compute_sma(work, "Close", slow_window)
    else:
        work["MA_Fast"] = compute_ema(work, "Close", fast_window)
        work["MA_Slow"] = compute_ema(work, "Close", slow_window)

    work = generate_signals(work, "MA_Fast", "MA_Slow")
    crossovers = work[work["MA_Crossover"] != ""]


    if crossovers.empty:
        latest_signal = "No Crossover"
        latest_date = ""
    else:
        latest = crossovers.iloc[-1]
        latest_signal = f"{latest['MA_Crossover']} Crossover"
        ts = crossovers.index[-1]
        latest_date = str(ts.date()) if hasattr(ts, "date") else str(ts)

    return MAAnalysisResult(
        symbol=symbol,
        ma_type=ma_type,
        fast_window=fast_window,
        slow_window=slow_window,
        latest_signal=latest_signal,
        latest_crossover_date=latest_date,
    )


def analyze_obv_filtered_stocks(
    filtered_results: list[tuple[str, pd.DataFrame, list, list, list]],
) -> dict[str, MAAnalysisResult]:
    """Run MA analysis only for stocks that already passed OBV filtering."""
    output: dict[str, MAAnalysisResult] = {}
    for symbol, df, *_ in filtered_results:
        output[symbol] = apply_moving_average_analysis(symbol=symbol, price_df=df)
    return output
