"""
Model training and evaluation module.
Supports XGBoost (primary) with RandomForest fallback.
Includes both classification (up/down) and regression (return prediction).
Uses time-based train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from src.features import get_feature_columns

# Try importing XGBoost; fall back to sklearn if unavailable
try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# ──────────────────────────────────────────────
# Data splitting
# ──────────────────────────────────────────────

def time_split(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically — no random shuffling.

    Args:
        df: Feature DataFrame sorted by date.
        train_ratio: Fraction of data for training (default 80%).

    Returns:
        (train_df, test_df) tuple.
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


# ──────────────────────────────────────────────
# Classification (up/down prediction)
# ──────────────────────────────────────────────

def train_model(
    train_df: pd.DataFrame,
    feature_cols: list | None = None,
    model_type: str = "auto",
):
    """
    Train a classifier on the training data.

    Args:
        train_df: Training DataFrame with features and 'target' column.
        feature_cols: List of feature column names. Uses defaults if None.
        model_type: "xgboost", "randomforest", or "auto" (XGBoost if available).

    Returns:
        Fitted classifier.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    if model_type == "auto":
        model_type = "xgboost" if XGBOOST_AVAILABLE else "randomforest"

    if model_type == "xgboost" and XGBOOST_AVAILABLE:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    elif model_type == "randomforest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    test_df: pd.DataFrame,
    feature_cols: list | None = None,
) -> dict:
    """
    Evaluate the classifier on test data and return metrics.

    Returns:
        Dict with accuracy, precision, recall, f1, predictions, probabilities,
        confusion matrix, and classification report.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "predictions": predictions,
        "probabilities": probabilities,
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "report": classification_report(y_test, predictions, output_dict=True),
    }
    return metrics


# ──────────────────────────────────────────────
# Regression (return prediction)
# ──────────────────────────────────────────────

def train_regression_model(
    train_df: pd.DataFrame,
    target_col: str = "future_return",
    feature_cols: list | None = None,
    model_type: str = "auto",
):
    """
    Train a regression model to predict continuous returns.

    Args:
        train_df: Training DataFrame with features and target column.
        target_col: Name of the continuous target column.
        feature_cols: Feature column names.
        model_type: "xgboost", "randomforest", or "auto".

    Returns:
        Fitted regressor.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    # Drop rows where this specific target is NaN
    valid = train_df.dropna(subset=[target_col])
    X_train = valid[feature_cols]
    y_train = valid[target_col]

    if model_type == "auto":
        model_type = "xgboost" if XGBOOST_AVAILABLE else "randomforest"

    if model_type == "xgboost" and XGBOOST_AVAILABLE:
        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    elif model_type == "randomforest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    model.fit(X_train, y_train)
    return model


def evaluate_regression_model(
    model,
    test_df: pd.DataFrame,
    target_col: str = "future_return",
    feature_cols: list | None = None,
) -> dict:
    """
    Evaluate a regression model and return metrics + predictions.

    Returns:
        Dict with mae, rmse, r2, predictions, actuals, residuals.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    valid = test_df.dropna(subset=[target_col])
    X_test = valid[feature_cols]
    y_test = valid[target_col]

    predictions = model.predict(X_test)
    residuals = y_test.values - predictions

    return {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
        "predictions": predictions,
        "actuals": y_test.values,
        "residuals": residuals,
    }


def compute_prediction_interval(
    residuals: np.ndarray,
    predicted_return: float,
    current_close: float,
    confidence: float = 0.80,
) -> dict:
    """
    Compute a point estimate and confidence interval for a future price.

    Uses the empirical distribution of residuals (actual - predicted) from the
    test set to build the interval — more robust than assuming normality.

    Args:
        residuals: Array of (actual - predicted) residuals from test evaluation.
        predicted_return: The model's predicted return for the period.
        current_close: The most recent closing price.
        confidence: Confidence level (default 0.80 for 80% interval).

    Returns:
        Dict with point_estimate, low, high, predicted_return,
        low_return, high_return, confidence_level.
    """
    alpha = (1 - confidence) / 2

    lower_residual = np.percentile(residuals, alpha * 100)
    upper_residual = np.percentile(residuals, (1 - alpha) * 100)

    low_return = predicted_return + lower_residual
    high_return = predicted_return + upper_residual

    point_estimate = current_close * (1 + predicted_return)
    low_price = max(0.01, current_close * (1 + low_return))
    high_price = current_close * (1 + high_return)

    return {
        "point_estimate": point_estimate,
        "low": low_price,
        "high": high_price,
        "predicted_return": predicted_return,
        "low_return": low_return,
        "high_return": high_return,
        "confidence_level": confidence,
    }


# ──────────────────────────────────────────────
# Multi-horizon forecasting
# ──────────────────────────────────────────────

def train_multi_horizon_models(
    train_df: pd.DataFrame,
    feature_cols: list | None = None,
    model_type: str = "auto",
    horizons: list[int] | None = None,
) -> dict:
    """
    Train separate regression models for multiple forecast horizons.

    Args:
        train_df: Training DataFrame with multi-horizon target columns.
        feature_cols: Feature column names.
        model_type: Model type string.
        horizons: List of horizon days (default [1, 3, 5, 10]).

    Returns:
        Dict mapping horizon (int) -> fitted regressor.
        Skips horizons with fewer than 30 valid training samples.
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    models = {}
    for h in horizons:
        target_col = f"future_return_{h}d"
        valid_count = train_df[target_col].notna().sum()
        if valid_count < 30:
            continue
        models[h] = train_regression_model(
            train_df,
            target_col=target_col,
            feature_cols=feature_cols,
            model_type=model_type,
        )
    return models


def predict_multi_horizon(
    models: dict,
    latest_row: pd.DataFrame,
    current_close: float,
    test_residuals: dict,
    feature_cols: list | None = None,
    confidence: float = 0.80,
) -> pd.DataFrame:
    """
    Generate multi-horizon forecasts from the latest data point.

    Args:
        models: Dict mapping horizon -> fitted regressor.
        latest_row: Single-row DataFrame of features for the most recent date.
        current_close: Most recent closing price.
        test_residuals: Dict mapping horizon -> residuals array from test eval.
        feature_cols: Feature column names.
        confidence: Confidence level for intervals.

    Returns:
        DataFrame with columns: horizon, horizon_days, predicted_return,
        direction, point_estimate, low, high, confidence_level.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    X = latest_row[feature_cols]
    rows = []

    for h in sorted(models.keys()):
        pred_return = float(models[h].predict(X)[0])
        interval = compute_prediction_interval(
            test_residuals[h], pred_return, current_close, confidence
        )
        rows.append({
            "horizon": f"{h}d",
            "horizon_days": h,
            "predicted_return": pred_return,
            "direction": "Up" if pred_return > 0 else "Down",
            "point_estimate": interval["point_estimate"],
            "low": interval["low"],
            "high": interval["high"],
            "confidence_level": confidence,
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────

def get_model_name(model) -> str:
    """Return a human-readable name for the model."""
    class_name = type(model).__name__
    names = {
        "XGBClassifier": "XGBoost",
        "RandomForestClassifier": "Random Forest",
        "GradientBoostingClassifier": "Gradient Boosting",
        "XGBRegressor": "XGBoost",
        "RandomForestRegressor": "Random Forest",
        "GradientBoostingRegressor": "Gradient Boosting",
    }
    return names.get(class_name, class_name)


def get_feature_importance(
    model,
    feature_cols: list | None = None,
) -> pd.DataFrame:
    """
    Extract feature importance from the trained model.

    Returns:
        DataFrame with 'feature' and 'importance' columns, sorted descending.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    )
    importance = importance.sort_values("importance", ascending=False).reset_index(
        drop=True
    )
    return importance
