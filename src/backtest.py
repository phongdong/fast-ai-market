"""
Simple backtest module.
Compares model strategy returns against buy-and-hold on the test period.
"""

import pandas as pd
import numpy as np


def run_backtest(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Run a simple long-only backtest on the test period.

    Strategy rules:
        - If model predicts UP (1): hold the stock (earn the actual return).
        - If model predicts DOWN (0): stay in cash (earn 0%).

    Comparison:
        - Buy-and-hold: invest on day 1 of test period and hold throughout.

    Args:
        test_df: Test DataFrame with 'future_return' column (actual next-day returns).
        predictions: Model predictions (0 or 1) aligned with test_df rows.

    Returns:
        DataFrame with columns: date, strategy_return, buyhold_return,
        strategy_cumulative, buyhold_cumulative.
    """
    results = pd.DataFrame(index=test_df.index)

    # Actual next-day returns
    results["actual_return"] = test_df["future_return"].values

    # Strategy: only invested when model predicts UP
    results["strategy_return"] = results["actual_return"] * predictions

    # Buy-and-hold: always invested
    results["buyhold_return"] = results["actual_return"]

    # Cumulative returns (compounded)
    results["strategy_cumulative"] = (1 + results["strategy_return"]).cumprod()
    results["buyhold_cumulative"] = (1 + results["buyhold_return"]).cumprod()

    return results


def backtest_summary(results: pd.DataFrame) -> dict:
    """
    Compute summary statistics for the backtest.

    Returns:
        Dict with total return, annualized return, max drawdown, and Sharpe ratio
        for both strategy and buy-and-hold.
    """
    def _stats(cumulative: pd.Series, daily_returns: pd.Series, label: str) -> dict:
        total_return = cumulative.iloc[-1] - 1
        n_days = len(cumulative)
        ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # Max drawdown
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()

        # Sharpe ratio (annualized, assuming 0% risk-free rate)
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        return {
            f"{label}_total_return": total_return,
            f"{label}_annual_return": ann_return,
            f"{label}_max_drawdown": max_dd,
            f"{label}_sharpe": sharpe,
        }

    stats = {}
    stats.update(
        _stats(results["strategy_cumulative"], results["strategy_return"], "strategy")
    )
    stats.update(
        _stats(results["buyhold_cumulative"], results["buyhold_return"], "buyhold")
    )
    stats["test_days"] = len(results)

    return stats
