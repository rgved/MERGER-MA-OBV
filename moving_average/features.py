import pandas as pd


def compute_sma(df: pd.DataFrame, price_col: str, window: int) -> pd.Series:
    """Compute simple moving average for a given window."""
    return df[price_col].rolling(window=window, min_periods=window).mean()


def compute_ema(df: pd.DataFrame, price_col: str, window: int) -> pd.Series:
    """Compute exponential moving average for a given window."""
    return df[price_col].ewm(span=window, adjust=False, min_periods=window).mean()


def generate_signals(df: pd.DataFrame, fast_col: str, slow_col: str) -> pd.DataFrame:
    """Generate crossover direction (+1 bullish, -1 bearish, 0 no crossover)."""
    out = df.copy()
    crossover_state = (out[fast_col] > out[slow_col]).astype(int)
    out["MA_Signal"] = crossover_state.diff().fillna(0)
    out["MA_Crossover"] = out["MA_Signal"].map({1.0: "Bullish", -1.0: "Bearish"}).fillna("")
    return out
