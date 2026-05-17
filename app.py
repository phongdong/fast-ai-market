"""
Stock Price Prediction App
==========================
A Streamlit web app that predicts short-term stock movements using machine learning.
Supports single-ticker analysis, multi-ticker comparison, and batch scanning.

This is for educational purposes only — not financial advice.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data import download_stock_data, validate_ticker
from src.features import build_features, build_multi_horizon_targets, get_feature_columns
from src.model import (
    train_model,
    evaluate_model,
    get_feature_importance,
    get_model_name,
    time_split,
    train_regression_model,
    evaluate_regression_model,
    compute_prediction_interval,
    train_multi_horizon_models,
    predict_multi_horizon,
)
from src.backtest import run_backtest, backtest_summary
from src.plots import (
    plot_price_history,
    plot_predictions,
    plot_feature_importance,
    plot_backtest,
    plot_confusion_matrix,
    plot_price_range,
    plot_forecast_path,
    plot_backtest_comparison,
    plot_forecast_comparison,
)


# ──────────────────────────────────────────────
# Pipeline helpers (no Streamlit calls)
# ──────────────────────────────────────────────

def run_single_ticker_pipeline(
    ticker: str,
    period: str,
    model_type: str,
    train_ratio: float,
    feature_cols: list,
) -> dict:
    """
    Run the full ML pipeline for a single ticker.
    Returns all results as a dict — no Streamlit rendering.
    """
    raw_data = download_stock_data(ticker, period)

    data = build_features(raw_data)
    data = build_multi_horizon_targets(data)

    train_df, test_df = time_split(data, train_ratio)

    # Classification model
    clf_model = train_model(train_df, feature_cols, model_type=model_type)

    # Regression model (next-day return)
    reg_model = train_regression_model(
        train_df, target_col="future_return",
        feature_cols=feature_cols, model_type=model_type,
    )
    reg_metrics = evaluate_regression_model(
        reg_model, test_df, target_col="future_return", feature_cols=feature_cols,
    )

    # Multi-horizon models
    horizon_models = train_multi_horizon_models(
        train_df, feature_cols=feature_cols, model_type=model_type,
    )
    horizon_residuals = {}
    for h, h_model in horizon_models.items():
        h_metrics = evaluate_regression_model(
            h_model, test_df,
            target_col=f"future_return_{h}d", feature_cols=feature_cols,
        )
        horizon_residuals[h] = h_metrics["residuals"]

    # Evaluate classifier
    metrics = evaluate_model(clf_model, test_df, feature_cols)

    # Backtest
    bt_results = run_backtest(test_df, metrics["predictions"])
    bt_summary = backtest_summary(bt_results)

    # Latest predictions
    latest = data.iloc[[-1]]
    current_close = float(latest["Close"].iloc[0])
    latest_pred = int(clf_model.predict(latest[feature_cols])[0])
    latest_prob = float(clf_model.predict_proba(latest[feature_cols])[0, 1])

    predicted_return = float(reg_model.predict(latest[feature_cols])[0])
    price_interval = compute_prediction_interval(
        reg_metrics["residuals"], predicted_return, current_close, confidence=0.80,
    )

    forecast_df = None
    if horizon_models:
        forecast_df = predict_multi_horizon(
            horizon_models, latest, current_close,
            horizon_residuals, feature_cols=feature_cols, confidence=0.80,
        )

    importance = get_feature_importance(clf_model, feature_cols)

    return {
        "ticker": ticker,
        "raw_data": raw_data,
        "data": data,
        "train_df": train_df,
        "test_df": test_df,
        "model": clf_model,
        "model_name": get_model_name(clf_model),
        "metrics": metrics,
        "reg_model": reg_model,
        "reg_metrics": reg_metrics,
        "horizon_models": horizon_models,
        "horizon_residuals": horizon_residuals,
        "backtest_results": bt_results,
        "backtest_summary": bt_summary,
        "latest": latest,
        "latest_pred": latest_pred,
        "latest_prob": latest_prob,
        "current_close": current_close,
        "predicted_return": predicted_return,
        "price_interval": price_interval,
        "forecast_df": forecast_df,
        "importance": importance,
    }


def run_batch_ticker_pipeline(
    ticker: str,
    period: str,
    model_type: str,
    train_ratio: float,
    feature_cols: list,
) -> dict:
    """
    Run a lighter pipeline for batch ranking.
    Skips multi-horizon, feature importance, and price intervals.
    """
    raw_data = download_stock_data(ticker, period)

    data = build_features(raw_data)
    train_df, test_df = time_split(data, train_ratio)

    # Classification model
    clf_model = train_model(train_df, feature_cols, model_type=model_type)
    metrics = evaluate_model(clf_model, test_df, feature_cols)

    # Regression model (next-day return)
    reg_model = train_regression_model(
        train_df, target_col="future_return",
        feature_cols=feature_cols, model_type=model_type,
    )

    # Backtest
    bt_results = run_backtest(test_df, metrics["predictions"])
    bt_summary = backtest_summary(bt_results)

    # Latest predictions
    latest = data.iloc[[-1]]
    current_close = float(latest["Close"].iloc[0])
    latest_pred = int(clf_model.predict(latest[feature_cols])[0])
    latest_prob = float(clf_model.predict_proba(latest[feature_cols])[0, 1])
    predicted_return = float(reg_model.predict(latest[feature_cols])[0])

    return {
        "ticker": ticker,
        "metrics": metrics,
        "backtest_summary": bt_summary,
        "latest_pred": latest_pred,
        "latest_prob": latest_prob,
        "predicted_return": predicted_return,
        "current_close": current_close,
    }


# ──────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────

def render_single_ticker_results(ticker: str, r: dict):
    """Render the full analysis UI for a single ticker from pipeline results."""
    feature_cols = get_feature_columns()

    # Price chart
    st.subheader("Historical Price")
    st.plotly_chart(plot_price_history(r["raw_data"], ticker), use_container_width=True)

    # Data info
    st.info(
        f"Created **{len(feature_cols)}** features from "
        f"**{len(r['data']):,}** valid data points. "
        f"Model: **{r['model_name']}**"
    )

    # Split info
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", f"{len(r['data']):,}")
    col2.metric("Training Set", f"{len(r['train_df']):,}")
    col3.metric("Test Set", f"{len(r['test_df']):,}")

    st.caption(
        f"Train: {r['train_df'].index[0].strftime('%Y-%m-%d')} to "
        f"{r['train_df'].index[-1].strftime('%Y-%m-%d')} | "
        f"Test: {r['test_df'].index[0].strftime('%Y-%m-%d')} to "
        f"{r['test_df'].index[-1].strftime('%Y-%m-%d')}"
    )

    # Classification metrics
    st.subheader("Model Performance")
    metrics = r["metrics"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    col2.metric("Precision", f"{metrics['precision']:.1%}")
    col3.metric("Recall", f"{metrics['recall']:.1%}")
    col4.metric("F1 Score", f"{metrics['f1']:.1%}")

    # Predictions + confusion matrix
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("Predictions")
        st.plotly_chart(
            plot_predictions(r["test_df"], metrics["predictions"], metrics["probabilities"], ticker),
            use_container_width=True,
        )
        st.caption("Green dots = model predicts UP | Red dots = model predicts DOWN")
    with col_right:
        st.subheader("Confusion Matrix")
        st.plotly_chart(
            plot_confusion_matrix(metrics["confusion_matrix"]),
            use_container_width=True,
        )

    # Feature importance
    st.subheader("Feature Importance")
    st.plotly_chart(plot_feature_importance(r["importance"]), use_container_width=True)

    # Backtest
    st.subheader("Backtest: Strategy vs Buy & Hold")
    summary = r["backtest_summary"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model Strategy** (long only when predicting UP)")
        st.metric("Total Return", f"{summary['strategy_total_return']:.2%}")
        st.metric("Annualized Return", f"{summary['strategy_annual_return']:.2%}")
        st.metric("Max Drawdown", f"{summary['strategy_max_drawdown']:.2%}")
        st.metric("Sharpe Ratio", f"{summary['strategy_sharpe']:.2f}")
    with col2:
        st.markdown("**Buy & Hold**")
        st.metric("Total Return", f"{summary['buyhold_total_return']:.2%}")
        st.metric("Annualized Return", f"{summary['buyhold_annual_return']:.2%}")
        st.metric("Max Drawdown", f"{summary['buyhold_max_drawdown']:.2%}")
        st.metric("Sharpe Ratio", f"{summary['buyhold_sharpe']:.2f}")
    st.plotly_chart(plot_backtest(r["backtest_results"], ticker), use_container_width=True)

    # Latest signal + price range
    st.subheader("Latest Signal & Estimated Price Range")

    latest_date = r["latest"].index[0].strftime("%Y-%m-%d")
    if r["latest_pred"] == 1:
        st.success(
            f"**{latest_date}**: Model predicts **UP** for next trading day "
            f"(confidence: {r['latest_prob']:.1%})"
        )
    else:
        st.error(
            f"**{latest_date}**: Model predicts **DOWN** for next trading day "
            f"(confidence: {1 - r['latest_prob']:.1%})"
        )

    pi = r["price_interval"]
    st.markdown("**Estimated Closing Price (Next Trading Day)**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Close", f"${r['current_close']:.2f}")
    col2.metric("Point Estimate", f"${pi['point_estimate']:.2f}", delta=f"{r['predicted_return']:+.2%}")
    col3.metric("Low (80% CI)", f"${pi['low']:.2f}")
    col4.metric("High (80% CI)", f"${pi['high']:.2f}")
    st.plotly_chart(
        plot_price_range(r["current_close"], pi["point_estimate"], pi["low"], pi["high"], ticker),
        use_container_width=True,
    )
    st.caption(
        f"Regression model MAE: {r['reg_metrics']['mae']:.4f} "
        f"({r['reg_metrics']['mae']*100:.2f}% avg error). "
        "80% CI from historical prediction residuals."
    )

    # Multi-day forecast
    if r["forecast_df"] is not None:
        st.subheader("Multi-Day Forecast")
        st.plotly_chart(
            plot_forecast_path(r["forecast_df"], r["current_close"], r["latest"].index[0], ticker),
            use_container_width=True,
        )
        display_df = r["forecast_df"][
            ["horizon", "direction", "predicted_return", "point_estimate", "low", "high"]
        ].copy()
        display_df.columns = ["Horizon", "Direction", "Est. Return", "Price Estimate", "Low (80%)", "High (80%)"]
        display_df["Est. Return"] = display_df["Est. Return"].apply(lambda x: f"{x:+.2%}")
        display_df["Price Estimate"] = display_df["Price Estimate"].apply(lambda x: f"${x:.2f}")
        display_df["Low (80%)"] = display_df["Low (80%)"].apply(lambda x: f"${x:.2f}")
        display_df["High (80%)"] = display_df["High (80%)"].apply(lambda x: f"${x:.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(
            "Each horizon uses a separately trained model. "
            "Longer horizons have wider confidence bands."
        )

    st.caption(
        "All signals and estimates are based on historical patterns — **not** trading advice."
    )

    # Raw data
    with st.expander("View Raw Feature Data"):
        st.dataframe(
            r["data"][feature_cols + ["target"]].tail(20).style.format("{:.4f}"),
            use_container_width=True,
        )


def render_comparison_view(results: dict):
    """Render the Comparison tab for multi-ticker compare mode."""
    # Metrics table
    st.subheader("Metrics Comparison")
    rows = []
    for t, r in results.items():
        m = r["metrics"]
        s = r["backtest_summary"]
        rows.append({
            "Ticker": t,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "F1": m["f1"],
            "Strategy Return": s["strategy_total_return"],
            "Strategy Sharpe": s["strategy_sharpe"],
            "B&H Return": s["buyhold_total_return"],
            "Max Drawdown": s["strategy_max_drawdown"],
        })
    metrics_df = pd.DataFrame(rows)
    st.dataframe(
        metrics_df.style.format({
            "Accuracy": "{:.1%}",
            "Precision": "{:.1%}",
            "F1": "{:.1%}",
            "Strategy Return": "{:+.2%}",
            "Strategy Sharpe": "{:.2f}",
            "B&H Return": "{:+.2%}",
            "Max Drawdown": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Backtest overlay
    st.subheader("Backtest Comparison")
    all_bt = {t: r["backtest_results"] for t, r in results.items()}
    st.plotly_chart(plot_backtest_comparison(all_bt), use_container_width=True)

    # Forecast comparison
    all_forecasts = {t: r["forecast_df"] for t, r in results.items() if r["forecast_df"] is not None}
    all_closes = {t: r["current_close"] for t, r in results.items() if r["forecast_df"] is not None}
    if all_forecasts:
        st.subheader("Forecast Comparison")
        sample_date = next(iter(results.values()))["latest"].index[0]
        st.plotly_chart(
            plot_forecast_comparison(all_forecasts, all_closes, sample_date),
            use_container_width=True,
        )

    # Signal summary
    st.subheader("Latest Signals")
    cols = st.columns(len(results))
    for col, (t, r) in zip(cols, results.items()):
        with col:
            direction = "UP" if r["latest_pred"] == 1 else "DOWN"
            confidence = r["latest_prob"] if r["latest_pred"] == 1 else 1 - r["latest_prob"]
            st.metric(t, direction, f"{confidence:.1%}")
            st.caption(f"${r['current_close']:.2f}")


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Price Prediction")
st.caption(
    "Predict stock movements using machine learning. "
    "Single analysis, side-by-side comparison, or batch scanning."
)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    mode = st.radio(
        "Mode",
        options=["Single Ticker", "Compare (2-4)", "Batch Scan"],
        index=0,
        help=(
            "Single: full analysis of one ticker. "
            "Compare: side-by-side for 2-4 tickers. "
            "Batch: scan up to 20 tickers and rank them."
        ),
    )

    # Ticker input varies by mode
    if mode == "Single Ticker":
        ticker_input = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            max_chars=10,
            help="Enter a valid stock ticker, e.g. AAPL, TSLA, NVDA, MSFT",
        ).upper().strip()
        tickers = [ticker_input] if ticker_input else []

    elif mode == "Compare (2-4)":
        ticker_input = st.text_input(
            "Tickers (comma-separated)",
            value="AAPL, MSFT",
            help="Enter 2-4 tickers separated by commas.",
        )
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        if len(tickers) > 4:
            st.warning("Compare mode supports up to 4 tickers. Using first 4.")
            tickers = tickers[:4]

    else:  # Batch Scan
        ticker_input = st.text_area(
            "Tickers (one per line or comma-separated)",
            value="AAPL\nMSFT\nGOOGL\nAMZN\nNVDA\nTSLA\nMETA\nJPM\nV\nJNJ",
            height=200,
            help="Enter up to 20 tickers.",
        )
        raw = [t for line in ticker_input.split("\n") for t in line.split(",")]
        tickers = [t.strip().upper() for t in raw if t.strip()]
        tickers = list(dict.fromkeys(tickers))  # deduplicate
        if len(tickers) > 20:
            st.warning("Batch mode supports up to 20 tickers. Using first 20.")
            tickers = tickers[:20]
        st.caption(f"{len(tickers)} ticker(s)")

    period = st.selectbox(
        "Historical Period",
        options=["2y", "3y", "5y", "10y", "max"],
        index=2,
        help="How many years of historical data to use.",
    )

    model_choice = st.selectbox(
        "Model",
        options=["Auto (XGBoost preferred)", "XGBoost", "Random Forest"],
        index=0,
    )

    train_ratio = st.slider(
        "Train/Test Split",
        min_value=0.5, max_value=0.9, value=0.8, step=0.05,
        help="Fraction of data for training. Uses chronological split.",
    )

    run_button = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

    st.divider()
    st.warning(
        "⚠️ **Disclaimer**\n\n"
        "This app is for **educational and research purposes only**. "
        "It is **not financial advice**. Past performance does not "
        "guarantee future results. Always do your own research."
    )


# ──────────────────────────────────────────────
# Main logic
# ──────────────────────────────────────────────
if run_button:
    model_type_map = {
        "Auto (XGBoost preferred)": "auto",
        "XGBoost": "xgboost",
        "Random Forest": "randomforest",
    }
    selected_model_type = model_type_map[model_choice]
    feature_cols = get_feature_columns()

    # Validate tickers
    valid_tickers = []
    for t in tickers:
        if validate_ticker(t):
            valid_tickers.append(t)
        else:
            st.warning(f"'{t}' is not a valid ticker format. Skipping.")
    tickers = valid_tickers

    if not tickers:
        st.error("No valid tickers entered.")
        st.stop()

    # ────────────────────────────────────
    # SINGLE TICKER MODE
    # ────────────────────────────────────
    if mode == "Single Ticker":
        t = tickers[0]
        with st.spinner(f"Running full analysis for {t}..."):
            try:
                result = run_single_ticker_pipeline(
                    t, period, selected_model_type, train_ratio, feature_cols,
                )
            except ValueError as e:
                st.error(str(e))
                st.stop()

        st.success(f"Downloaded **{len(result['raw_data']):,}** days of data for **{t}**")
        render_single_ticker_results(t, result)

    # ────────────────────────────────────
    # COMPARE MODE (2-4 tickers)
    # ────────────────────────────────────
    elif mode == "Compare (2-4)":
        if len(tickers) < 2:
            st.error("Enter at least 2 tickers for comparison.")
            st.stop()

        results = {}
        errors = []
        progress = st.progress(0, text="Processing tickers...")

        for i, t in enumerate(tickers):
            progress.progress(
                (i + 1) / len(tickers),
                text=f"Processing {t} ({i+1}/{len(tickers)})...",
            )
            try:
                results[t] = run_single_ticker_pipeline(
                    t, period, selected_model_type, train_ratio, feature_cols,
                )
            except Exception as e:
                errors.append((t, str(e)))

        progress.empty()

        if errors:
            for t, err in errors:
                st.warning(f"Skipped {t}: {err}")

        if not results:
            st.error("No valid tickers could be processed.")
            st.stop()

        # Build tabs: one per ticker + Comparison
        tab_names = list(results.keys()) + ["📊 Comparison"]
        tabs = st.tabs(tab_names)

        for i, (t, r) in enumerate(results.items()):
            with tabs[i]:
                render_single_ticker_results(t, r)

        with tabs[-1]:
            render_comparison_view(results)

    # ────────────────────────────────────
    # BATCH SCAN MODE (up to 20 tickers)
    # ────────────────────────────────────
    else:
        results = {}
        errors = []
        progress = st.progress(0, text="Scanning tickers...")

        for i, t in enumerate(tickers):
            progress.progress(
                (i + 1) / len(tickers),
                text=f"Scanning {t} ({i+1}/{len(tickers)})...",
            )
            try:
                results[t] = run_batch_ticker_pipeline(
                    t, period, selected_model_type, train_ratio, feature_cols,
                )
            except Exception as e:
                errors.append((t, str(e)))

        progress.empty()

        if errors:
            with st.expander(f"Warnings ({len(errors)} tickers skipped)"):
                for t, err in errors:
                    st.warning(f"{t}: {err}")

        if not results:
            st.error("No valid tickers could be processed.")
            st.stop()

        # Build ranking table
        rows = []
        for t, r in results.items():
            direction = "Up" if r["latest_pred"] == 1 else "Down"
            confidence = r["latest_prob"] if r["latest_pred"] == 1 else 1 - r["latest_prob"]
            rows.append({
                "Ticker": t,
                "Price": r["current_close"],
                "Direction": direction,
                "Confidence": confidence,
                "Est. Return": r["predicted_return"],
                "Accuracy": r["metrics"]["accuracy"],
                "Sharpe": r["backtest_summary"]["strategy_sharpe"],
            })

        ranking_df = pd.DataFrame(rows).sort_values("Confidence", ascending=False)

        st.subheader(f"Batch Scan Results — {len(results)} Tickers")

        def color_direction(row):
            color = (
                "background-color: rgba(76, 175, 80, 0.15)"
                if row["Direction"] == "Up"
                else "background-color: rgba(244, 67, 54, 0.15)"
            )
            return [color] * len(row)

        styled = ranking_df.style.apply(color_direction, axis=1).format({
            "Price": "${:.2f}",
            "Confidence": "{:.1%}",
            "Est. Return": "{:+.2%}",
            "Accuracy": "{:.1%}",
            "Sharpe": "{:.2f}",
        })

        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Summary
        up_count = sum(1 for r in rows if r["Direction"] == "Up")
        down_count = len(rows) - up_count
        avg_conf = ranking_df["Confidence"].mean()
        avg_acc = ranking_df["Accuracy"].mean()
        st.caption(
            f"**Summary:** {up_count} bullish, {down_count} bearish. "
            f"Avg confidence: {avg_conf:.1%}. Avg accuracy: {avg_acc:.1%}."
        )

        st.caption(
            "All signals are based on historical patterns — **not** trading advice."
        )

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Streamlit, XGBoost, yfinance, and Plotly. "
    "This is for educational purposes only — not financial advice."
)
