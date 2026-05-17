"""
Standalone training script.
Run this to train and evaluate the model from the command line.

Usage:
    python train_model.py --ticker AAPL --period 5y
"""

import argparse

from src.data import download_stock_data, validate_ticker
from src.features import build_features, get_feature_columns
from src.model import train_model, evaluate_model, get_feature_importance, time_split
from src.backtest import run_backtest, backtest_summary


def main():
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument(
        "--ticker", type=str, default="AAPL", help="Stock ticker symbol"
    )
    parser.add_argument(
        "--period", type=str, default="5y", help="Data period (e.g., 1y, 2y, 5y, max)"
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    if not validate_ticker(ticker):
        print(f"Invalid ticker: {ticker}")
        return

    # Download data
    print(f"Downloading {ticker} data ({args.period})...")
    # Use yfinance directly since we can't use st.cache outside Streamlit
    import yfinance as yf
    stock = yf.Ticker(ticker)
    df = stock.history(period=args.period, auto_adjust=True)
    if df.empty:
        print(f"No data found for {ticker}")
        return
    print(f"  Downloaded {len(df)} rows")

    # Build features
    print("Building features...")
    data = build_features(df)
    print(f"  {len(data)} rows after feature engineering")

    # Split
    feature_cols = get_feature_columns()
    train_df, test_df = time_split(data)
    print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    # Train
    print("Training XGBoost model...")
    model = train_model(train_df, feature_cols)

    # Evaluate
    print("Evaluating...")
    metrics = evaluate_model(model, test_df, feature_cols)

    print(f"\n{'='*40}")
    print(f"  Results for {ticker}")
    print(f"{'='*40}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    # Feature importance
    importance = get_feature_importance(model, feature_cols)
    print(f"\nFeature Importance:")
    for _, row in importance.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:>20s}: {row['importance']:.4f} {bar}")

    # Backtest
    print("\nBacktest Results (Test Period):")
    results = run_backtest(test_df, metrics["predictions"])
    summary = backtest_summary(results)
    print(f"  Strategy Total Return:  {summary['strategy_total_return']:>8.2%}")
    print(f"  Buy&Hold Total Return:  {summary['buyhold_total_return']:>8.2%}")
    print(f"  Strategy Sharpe Ratio:  {summary['strategy_sharpe']:>8.2f}")
    print(f"  Buy&Hold Sharpe Ratio:  {summary['buyhold_sharpe']:>8.2f}")
    print(f"  Strategy Max Drawdown:  {summary['strategy_max_drawdown']:>8.2%}")
    print(f"  Buy&Hold Max Drawdown:  {summary['buyhold_max_drawdown']:>8.2%}")

    print("\nDisclaimer: This is for educational purposes only. Not financial advice.")


if __name__ == "__main__":
    main()
