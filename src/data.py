"""
Data downloading and caching module.
Uses yfinance to fetch historical daily stock data.
"""

import yfinance as yf
import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Download historical daily OHLCV data for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        period: How far back to fetch data (default "5y" = 5 years).

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, indexed by Date.

    Raises:
        ValueError: If the ticker is invalid or no data is returned.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, auto_adjust=True)

    if df.empty:
        raise ValueError(
            f"No data found for ticker '{ticker}'. "
            "Check the symbol and try again."
        )

    # Drop unnecessary columns if present
    drop_cols = [c for c in ["Dividends", "Stock Splits"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Ensure the index is a proper DatetimeIndex
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df


def validate_ticker(ticker: str) -> bool:
    """Quick check that a ticker string is plausible."""
    if not ticker or not ticker.isalpha() or len(ticker) > 10:
        return False
    return True
