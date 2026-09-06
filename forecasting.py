"""
forecasting.py
Core ML pipeline for the Sales & Demand Forecasting task.

Covers every required deliverable:
  - Data cleaning & time-based feature engineering
  - Forecasting using regression / time-series methods
  - Model evaluation and error analysis
  - Business-friendly outputs (forecast table + plain-English summary)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


# ---------------------------------------------------------------------------
# 1. DATA CLEANING
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Basic, robust cleaning: parse dates, drop bad rows, fill small gaps."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)

    # Coerce target to numeric, drop rows that are entirely unusable
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    # Remove negative sales (data entry errors) and extreme outliers (winsorize at 1st/99th pct)
    df = df[df[target_col] >= 0]
    lower, upper = df[target_col].quantile([0.01, 0.99])
    df[target_col] = df[target_col].clip(lower, upper)

    df = df.drop_duplicates(subset=[date_col] if df[date_col].is_unique else None)
    return df.reset_index(drop=True)


def aggregate_daily(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Aggregate to a single daily total series (sum across stores/categories)."""
    daily = df.groupby(date_col, as_index=False)[target_col].sum()
    daily = daily.set_index(date_col).asfreq("D")
    daily[target_col] = daily[target_col].interpolate(method="linear").ffill().bfill()
    return daily.reset_index()


# ---------------------------------------------------------------------------
# 2. TIME-BASED FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def make_features(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Create calendar, trend, lag, and rolling-window features."""
    df = df.copy()
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_month"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["year"] = df[date_col].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df[date_col].dt.dayofyear

    # Cyclical encodings so the model understands "December is near January"
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Linear time trend
    df["time_idx"] = (df[date_col] - df[date_col].min()).dt.days

    # Lag & rolling features (helps capture recent momentum)
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = df[target_col].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df[target_col].shift(1).rolling(window).std()

    return df


FEATURE_COLUMNS = [
    "day_of_week", "day_of_month", "month", "quarter", "year", "is_weekend",
    "day_of_year", "month_sin", "month_cos", "dow_sin", "dow_cos", "time_idx",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_14", "rolling_std_28",
]


# ---------------------------------------------------------------------------
# 3. MODEL TRAINING & FORECASTING
# ---------------------------------------------------------------------------
def train_test_split_by_time(df: pd.DataFrame, test_size: float = 0.15):
    n_test = max(7, int(len(df) * test_size))
    train, test = df.iloc[:-n_test], df.iloc[-n_test:]
    return train, test


def train_model(train_df: pd.DataFrame, target_col: str, model_type: str = "random_forest"):
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[target_col]

    if model_type == "linear_regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, test_df: pd.DataFrame, target_col: str) -> dict:
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[target_col]
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    accuracy = max(0, 100 - mape)

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "Approx. Accuracy (%)": round(accuracy, 2),
        "predictions": preds,
        "actuals": y_test.values,
        "dates": test_df["date"].values,
    }


def forecast_future(model, history_df: pd.DataFrame, date_col: str, target_col: str, horizon: int = 30) -> pd.DataFrame:
    """Recursive multi-step forecast: predict one day, feed it back in, repeat."""
    history = history_df.copy()
    future_rows = []
    last_date = history[date_col].max()

    for step in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=step)
        temp = pd.concat(
            [history, pd.DataFrame({date_col: [next_date], target_col: [np.nan]})],
            ignore_index=True,
        )
        temp_feat = make_features(temp, date_col, target_col)
        row = temp_feat.iloc[[-1]][FEATURE_COLUMNS]
        pred = float(model.predict(row)[0])
        pred = max(0, pred)

        future_rows.append({date_col: next_date, target_col: pred})
        history = pd.concat(
            [history, pd.DataFrame({date_col: [next_date], target_col: [pred]})],
            ignore_index=True,
        )

    return pd.DataFrame(future_rows)


def business_summary(forecast_df: pd.DataFrame, target_col: str, history_df: pd.DataFrame) -> str:
    """Plain-English takeaway a non-technical stakeholder can act on."""
    avg_recent = history_df[target_col].tail(28).mean()
    avg_future = forecast_df[target_col].mean()
    change = (avg_future - avg_recent) / avg_recent * 100 if avg_recent else 0
    peak_idx = forecast_df[target_col].idxmax()
    peak_date = forecast_df.loc[peak_idx, forecast_df.columns[0]]
    peak_value = forecast_df.loc[peak_idx, target_col]
    direction = "increase" if change > 1 else ("decrease" if change < -1 else "stay roughly flat")

    return (
        f"Based on recent patterns, average daily sales are expected to **{direction}** "
        f"by about **{abs(change):.1f}%** over the forecast period "
        f"(from ~{avg_recent:,.0f}/day to ~{avg_future:,.0f}/day). "
        f"The highest-demand day is projected to be **{pd.to_datetime(peak_date).strftime('%b %d, %Y')}** "
        f"at ~{peak_value:,.0f} in sales. "
        f"Use this to plan inventory and staffing ahead of that peak, and avoid overstocking on slower days."
    )
