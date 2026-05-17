"""
Plotting module using Plotly for interactive charts.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_price_history(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Plot historical closing price with volume as a secondary axis.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} — Closing Price", "Volume"),
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#2196F3", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color="rgba(100,100,100,0.3)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=20),
        xaxis2_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def plot_predictions(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    ticker: str,
) -> go.Figure:
    """
    Plot test period prices colored by model prediction (green=up, red=down),
    with prediction probability on a secondary axis.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.4],
        subplot_titles=(
            f"{ticker} — Test Period with Predictions",
            "Prediction Probability (Up)",
        ),
    )

    # Color each day by prediction
    colors = ["#4CAF50" if p == 1 else "#F44336" for p in predictions]

    fig.add_trace(
        go.Scatter(
            x=test_df.index,
            y=test_df["Close"],
            mode="markers+lines",
            marker=dict(color=colors, size=4),
            line=dict(color="gray", width=1),
            name="Close",
        ),
        row=1,
        col=1,
    )

    # Probability chart
    fig.add_trace(
        go.Scatter(
            x=test_df.index,
            y=probabilities,
            mode="lines",
            name="P(Up)",
            line=dict(color="#FF9800", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,152,0,0.1)",
        ),
        row=2,
        col=1,
    )

    # 50% threshold line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="gray",
        row=2,
        col=1,
    )

    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=20),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=1, range=[0, 1])

    return fig


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of feature importance.
    """
    # Reverse so highest importance is at top
    df = importance_df.sort_values("importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker_color="#2196F3",
        )
    )

    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="",
        height=350,
        margin=dict(l=130, r=20, t=40, b=40),
    )

    return fig


def plot_backtest(results: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Plot cumulative returns: model strategy vs buy-and-hold.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=(results["strategy_cumulative"] - 1) * 100,
            mode="lines",
            name="Model Strategy",
            line=dict(color="#4CAF50", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=(results["buyhold_cumulative"] - 1) * 100,
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#2196F3", width=2),
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"{ticker} — Strategy vs Buy & Hold (Test Period)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """
    Plot confusion matrix as a heatmap.
    """
    labels = ["Down (0)", "Up (1)"]

    # Annotate with counts and percentages
    total = cm.sum()
    text = [
        [f"{cm[i][j]}<br>({cm[i][j]/total*100:.1f}%)" for j in range(2)]
        for i in range(2)
    ]

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=350,
        width=400,
        margin=dict(l=80, r=20, t=40, b=40),
    )

    return fig


def plot_price_range(
    current_close: float,
    point_estimate: float,
    low: float,
    high: float,
    ticker: str,
) -> go.Figure:
    """
    Horizontal range bar showing current close, point estimate, and CI band.
    """
    fig = go.Figure()

    # Confidence interval as a thick bar
    fig.add_trace(go.Scatter(
        x=[low, high],
        y=[0, 0],
        mode="lines",
        line=dict(color="rgba(33,150,243,0.3)", width=20),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Point estimate marker
    fig.add_trace(go.Scatter(
        x=[point_estimate],
        y=[0],
        mode="markers",
        marker=dict(color="#2196F3", size=16, symbol="diamond"),
        name=f"Estimate: ${point_estimate:.2f}",
    ))

    # Current close marker
    fig.add_trace(go.Scatter(
        x=[current_close],
        y=[0],
        mode="markers",
        marker=dict(color="#FF9800", size=14, symbol="circle"),
        name=f"Current: ${current_close:.2f}",
    ))

    fig.add_annotation(x=low, y=0, text=f"${low:.2f}", showarrow=False, yshift=-20)
    fig.add_annotation(x=high, y=0, text=f"${high:.2f}", showarrow=False, yshift=-20)

    fig.update_layout(
        height=120,
        margin=dict(l=20, r=20, t=10, b=40),
        xaxis=dict(title="Price ($)", showgrid=True),
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        legend=dict(orientation="h", yanchor="top", y=1.3, xanchor="center", x=0.5),
        showlegend=True,
    )

    return fig


def plot_forecast_path(
    forecast_df: pd.DataFrame,
    current_close: float,
    current_date: pd.Timestamp,
    ticker: str,
) -> go.Figure:
    """
    Multi-horizon forecast: predicted price path with confidence band.

    Args:
        forecast_df: DataFrame from predict_multi_horizon() with columns:
            horizon_days, point_estimate, low, high, confidence_level.
        current_close: The most recent closing price (day 0).
        current_date: The date of the most recent close.
        ticker: Ticker symbol for the title.
    """
    # Build x-axis as business-day offsets from current_date
    days = [0] + forecast_df["horizon_days"].tolist()
    dates = [current_date + pd.tseries.offsets.BDay(d) for d in days]

    estimates = [current_close] + forecast_df["point_estimate"].tolist()
    lows = [current_close] + forecast_df["low"].tolist()
    highs = [current_close] + forecast_df["high"].tolist()

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=highs + lows[::-1],
        fill="toself",
        fillcolor="rgba(33,150,243,0.15)",
        line=dict(color="rgba(33,150,243,0)"),
        name=f"{int(forecast_df['confidence_level'].iloc[0]*100)}% Confidence",
        hoverinfo="skip",
    ))

    # Point estimate line
    fig.add_trace(go.Scatter(
        x=dates,
        y=estimates,
        mode="lines+markers",
        name="Predicted Price",
        line=dict(color="#2196F3", width=2),
        marker=dict(size=8),
    ))

    # Current close marker
    fig.add_trace(go.Scatter(
        x=[dates[0]],
        y=[current_close],
        mode="markers",
        name=f"Current Close: ${current_close:.2f}",
        marker=dict(color="#FF9800", size=12, symbol="star"),
    ))

    fig.update_layout(
        title=f"{ticker} — Multi-Day Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
    )

    return fig


# ──────────────────────────────────────────────
# Multi-ticker comparison charts
# ──────────────────────────────────────────────

COLORS = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0", "#00BCD4"]


def plot_backtest_comparison(
    all_results: dict[str, pd.DataFrame],
) -> go.Figure:
    """
    Overlay strategy cumulative return curves for multiple tickers.

    Args:
        all_results: Dict mapping ticker -> backtest results DataFrame
                     (output of run_backtest, with 'strategy_cumulative' column).
    """
    fig = go.Figure()

    for i, (ticker, results) in enumerate(all_results.items()):
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=results.index,
            y=(results["strategy_cumulative"] - 1) * 100,
            mode="lines",
            name=ticker,
            line=dict(color=color, width=2),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Strategy Returns Comparison (Test Period)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def plot_forecast_comparison(
    all_forecasts: dict[str, pd.DataFrame],
    all_closes: dict[str, float],
    current_date: pd.Timestamp,
) -> go.Figure:
    """
    Overlay multi-day forecast paths for multiple tickers.
    Normalized to % return from current close so different-priced stocks
    are comparable on the same y-axis.

    Args:
        all_forecasts: Dict mapping ticker -> forecast_df from predict_multi_horizon().
        all_closes: Dict mapping ticker -> current close price.
        current_date: Most recent trading date.
    """
    fig = go.Figure()

    for i, (ticker, forecast_df) in enumerate(all_forecasts.items()):
        color = COLORS[i % len(COLORS)]
        current_close = all_closes[ticker]

        days = [0] + forecast_df["horizon_days"].tolist()
        dates = [current_date + pd.tseries.offsets.BDay(d) for d in days]

        # Normalize to % return
        estimates = [0.0] + ((forecast_df["point_estimate"] / current_close - 1) * 100).tolist()
        lows = [0.0] + ((forecast_df["low"] / current_close - 1) * 100).tolist()
        highs = [0.0] + ((forecast_df["high"] / current_close - 1) * 100).tolist()

        # Confidence band
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=highs + lows[::-1],
            fill="toself",
            fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba").replace("#", "rgba(")
            if color.startswith("rgb") else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Estimate line
        fig.add_trace(go.Scatter(
            x=dates,
            y=estimates,
            mode="lines+markers",
            name=ticker,
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Forecast Comparison (% Return from Current Close)",
        xaxis_title="Date",
        yaxis_title="Predicted Return (%)",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
    )

    return fig
