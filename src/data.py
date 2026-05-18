"""
Data downloading and caching module.
Uses yfinance to fetch historical daily stock data.
Includes retry logic for Yahoo Finance rate limiting.
"""

import time
import yfinance as yf
import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(
    ticker: str, period: str = "5y", max_retries: int = 3
) -> pd.DataFrame:
    """
    Download historical daily OHLCV data for a given ticker.
    Retries on rate limiting with exponential backoff.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        period: How far back to fetch data (default "5y" = 5 years).
        max_retries: Number of retry attempts on rate limiting.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, indexed by Date.

    Raises:
        ValueError: If the ticker is invalid or no data is returned.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
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

        except ValueError:
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                time.sleep(wait)

    raise ValueError(
        f"Could not download data for '{ticker}' after {max_retries} attempts. "
        f"Yahoo Finance may be rate-limiting. Try again in a minute. "
        f"Error: {last_error}"
    )


def validate_ticker(ticker: str) -> bool:
    """Quick check that a ticker string is plausible."""
    if not ticker or not ticker.isalpha() or len(ticker) > 10:
        return False
    return True
