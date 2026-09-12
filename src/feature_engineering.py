"""
feature_engineering.py
======================
Builds a supervised tabular dataset from a single daily target series for a
DIRECT multi-horizon forecasting task (a separate model per horizon).

There is NO target leakage:
  - LAG / ROLLING features use only values strictly BEFORE time t (shift).
  - CALENDAR features are known-future covariates (no look-ahead needed).
  - The target for horizon h is value at t+h, and rows are aligned to time t.

The feature set mirrors the grant proposal (section 3.2 / 3.3):
  temporal markers (day-of-week, month, cyclic encodings),
  holiday / Lent / Ramadan flags,
  lagged target values (t-1, -2, -7, -14, -28),
  7- and 30-day rolling means & standard deviations.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .calendar_features import build_calendar_frame


# The columns handed to the model (calendar + lags + rolling).
CALENDAR_COLS = [
    "week_of_year", "weekend",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_lent", "is_ramadan", "is_eid", "is_public_holiday",
    "days_until_easter",
]


def build_feature_frame(series: pd.Series) -> pd.DataFrame:
    """Turn a daily target Series (indexed by date) into a feature DataFrame.

    The returned frame is indexed by date and has a column `target` plus the
    engineered features (calendar, lags, rolling stats).
    """
    series = series.astype(float)
    cal = build_calendar_frame(series.index.min().strftime("%Y-%m-%d"),
                               series.index.max().strftime("%Y-%m-%d"))
    df = cal.copy()
    df["target"] = series.reindex(df.index)

    # Most recent observed value at the forecast origin t (this is the info the
    # naive "persistence" baseline also gets - a fair comparison). No FUTURE
    # information is used: t+h is only ever the supervised target, never a feature.
    df["lag_0"] = df["target"]

    # Lagged target values (info up to and including t, shifted back by L).
    for lag in config.LAGS:
        df[f"lag_{lag}"] = df["target"].shift(lag)

    # Rolling statistics - window ENDING at t (inclusive of the current day).
    for w in config.ROLLING_WINDOWS:
        df[f"roll_mean_{w}"] = df["target"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["target"].rolling(w).std()

    return df


def make_xy(frame: pd.DataFrame, horizon: int):
    """Return (X, y) for DIRECT forecasting `horizon` days ahead.

    At each row t: X = features known at t; y = target at t+horizon.
    NaN rows (warm-up and horizon tail) are dropped.
    """
    feats = CALENDAR_COLS + ["lag_0"] + [f"lag_{l}" for l in config.LAGS]
    for w in config.ROLLING_WINDOWS:
        feats += [f"roll_mean_{w}", f"roll_std_{w}"]

    y = frame["target"].shift(-horizon)          # value at t + h
    X = frame[feats]
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]
    # Keep the date index aligned so we can split chronologically later.
    X.index = frame.index[valid]
    y.index = frame.index[valid]
    return X, y


def chronological_split(dates: pd.DatetimeIndex, horizon: int,
                        data_end: pd.Timestamp | None = None):
    """Return (train_mask, val_mask, test_mask) using the proposal's 48/6/6 split.

    The split is by CALENDAR time, never by shuffling, so the test set is a
    strict out-of-sample hold-out. `data_end` is the last available date in the
    FULL series (so the test window is identical across horizons); by default it
    falls back to the max of `dates`.
    """
    if data_end is None:
        data_end = dates.max()
    test_start = data_end - pd.DateOffset(months=config.TEST_MONTHS)
    val_start = test_start - pd.DateOffset(months=config.VAL_MONTHS)

    train = dates < val_start
    val = (dates >= val_start) & (dates < test_start)
    test = dates >= test_start
    return train, val, test
