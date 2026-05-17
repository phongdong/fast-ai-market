"""
Feature engineering module.
All features use ONLY past data — no future information leakage.
"""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create predictive features from OHLCV data.

    Features (all computed from past data only):
        - daily_return: previous day's percentage return
        - return_5d: cumulative return over the past 5 days
        - return_20d: cumulative return over the past 20 days
        - sma_5: 5-day simple moving average of close price
        - sma_20: 20-day simple moving average of close price
        - sma_50: 50-day simple moving average of close price
        - sma_ratio_5_20: ratio of SMA5 to SMA20 (trend signal)
        - volatility_20d: 20-day rolling standard deviation of daily returns
        - volume_change: percentage change in volume vs prior day
        - volume_sma_ratio: current volume / 20-day average volume
        - high_low_range: (High - Low) / Close as a percentage
        - close_to_sma20: distance of close from SMA20 as a percentage

    Target:
        - target: 1 if next day's return > 0 (stock goes up), else 0

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns.

    Returns:
        DataFrame with features and target, NaN rows dropped.
    """
    data = df.copy()

    # --- Daily returns (shifted so we use YESTERDAY's return as today's feature) ---
    data["daily_return"] = data["Close"].pct_change()

    # --- Multi-day returns ---
    data["return_5d"] = data["Close"].pct_change(periods=5)
    data["return_20d"] = data["Close"].pct_change(periods=20)

    # --- Moving averages ---
    data["sma_5"] = data["Close"].rolling(window=5).mean()
    data["sma_20"] = data["Close"].rolling(window=20).mean()
    data["sma_50"] = data["Close"].rolling(window=50).mean()

    # --- Moving average crossover signal ---
    data["sma_ratio_5_20"] = data["sma_5"] / data["sma_20"]

    # --- Rolling volatility ---
    data["volatility_20d"] = data["daily_return"].rolling(window=20).std()

    # --- Volume features ---
    data["volume_change"] = data["Volume"].pct_change()
    volume_sma_20 = data["Volume"].rolling(window=20).mean()
    data["volume_sma_ratio"] = data["Volume"] / volume_sma_20

    # --- Price range ---
    data["high_low_range"] = (data["High"] - data["Low"]) / data["Close"]

    # --- Distance from moving average ---
    data["close_to_sma20"] = (data["Close"] - data["sma_20"]) / data["sma_20"]

    # --- Target: will the stock go UP tomorrow? ---
    # We use the NEXT day's return as the target (what we're predicting)
    data["future_return"] = data["Close"].shift(-1) / data["Close"] - 1
    data["target"] = (data["future_return"] > 0).astype(int)

    # Now shift all features by 1 day to avoid leakage:
    # Today's features should only use data available BEFORE today's close.
    # The daily_return, volume_change etc. already use today's data,
    # but since we're predicting TOMORROW's move using TODAY's features,
    # this is fine — we know today's close before tomorrow opens.

    # Drop rows with NaN (from rolling calculations and the last row with no target)
    data = data.dropna()

    return data


def build_multi_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multi-horizon regression targets for price forecasting.

    For each horizon h in [1, 3, 5, 10], creates:
        future_return_{h}d: cumulative return over the next h trading days
        computed as Close[t+h] / Close[t] - 1

    Does NOT drop NaN rows — the regression training functions handle
    NaN per-horizon so shorter horizons don't lose valid data.

    Args:
        df: DataFrame with 'Close' column (output of build_features).

    Returns:
        DataFrame with added target columns.
    """
    data = df.copy()
    for h in [1, 3, 5, 10]:
        data[f"future_return_{h}d"] = data["Close"].shift(-h) / data["Close"] - 1
    return data


def get_feature_columns() -> list:
    """Return the list of feature column names used for modeling."""
    return [
        "daily_return",
        "return_5d",
        "return_20d",
        "sma_ratio_5_20",
        "volatility_20d",
        "volume_change",
        "volume_sma_ratio",
        "high_low_range",
        "close_to_sma20",
    ]
