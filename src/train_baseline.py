"""
train_baseline.py
=================
Wires up the FIRST runnable model: XGBoost with engineered features (Obj 3 &
Obj 4). It trains one DIRECT model per forecast horizon (7 / 30 / 90 days),
evaluates with MAE, RMSE and MAPE on a strict chronological hold-out test set,
and also reports a persistence (naive "predict the last known value") baseline
so you can see whether XGBoost actually beats doing nothing.

Design choice - DIRECT multi-horizon: a separate model per horizon is the
simplest, leakage-free way to get per-horizon metrics (the proposal evaluates
each horizon separately). Not to be confused with recursive (one-step) models.

Outputs (written to `output/`):
  baseline_metrics.json   -> per-horizon MAE / RMSE / MAPE + naive baseline
  baseline_plot.png        -> XGBoost vs actual on the test set (30-day horizon)

Run:
  python -m src.train_baseline
  python -m src.train_baseline --target issuance --region national
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

# Plotting is optional (matplotlib is a soft dependency).
try:
    import matplotlib
    matplotlib.use("Agg")            # headless plotting
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# XGBoost is preferred; if it is not installed (e.g. offline sandbox) we fall
# back to scikit-learn's HistGradientBoostingRegressor, which is the same
# gradient-boosted-tree family. The metrics are directly comparable.
try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False

from . import config
from .data_source import SyntheticSource
from .feature_engineering import (CALENDAR_COLS, build_feature_frame,
                                  chronological_split, make_xy)
from . import config as cfg


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y, yhat):
    # Guard against divide-by-zero on zero demand days.
    denom = np.where(np.abs(y) < 1e-9, np.nan, np.abs(y))
    return float(np.nanmean(np.abs((y - yhat) / denom)) * 100)


def _persistence_mae(y_true, y_naive):
    return float(np.mean(np.abs(y_true - y_naive)))


# ---------------------------------------------------------------------------
# Data loading: pick a daily series (by default national total collection)
# ---------------------------------------------------------------------------
def _load_target(measure: str, region: str) -> pd.Series:
    source = SyntheticSource()
    if region == "national":
        df = source.national_daily()
        return df.set_index("date")[measure].rename("target")
    # Region-specific: aggregate the long table up to a per-day series.
    long = source.load_frame(measure)
    sub = long[long.region == region].groupby("date")["units"].sum()
    return sub.rename("target")


def _build_model(params):
    """Return (model, is_xgboost). xgboost if available, else sklearn HGB."""
    if _HAS_XGBOOST:
        model = xgb.XGBRegressor(**params, random_state=cfg.RNG_SEED)
        return model, True
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        random_state=cfg.RNG_SEED,
    )
    return model, False


def _fit(model, is_xgb, Xtr, ytr, Xv, yv):
    if is_xgb:
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    else:
        model.fit(Xtr, ytr)
    return model


# ---------------------------------------------------------------------------
# Train + evaluate one horizon
# ---------------------------------------------------------------------------
def _run_horizon(frame, horizon, params):
    X, y = make_xy(frame, horizon)
    dates = y.index
    # Target value *known at time t* for each valid row: this is the naive
    # "persistence" forecast (predict the last observed value).
    current = frame.loc[dates, "target"]
    train_m, val_m, test_m = chronological_split(dates, horizon,
                                                 data_end=frame.index.max())

    Xtr, ytr = X[train_m], y[train_m]
    Xv, yv = X[val_m], y[val_m]
    Xte, yte = X[test_m], y[test_m]

    model, is_xgb = _build_model(params)
    model = _fit(model, is_xgb, Xtr, ytr, Xv, yv)

    yhat = model.predict(Xte)
    y_naive = current[test_m].values

    return {
        "horizon": horizon,
        "n_train": int(len(ytr)), "n_val": int(len(yv)), "n_test": int(len(yte)),
        "xgboost": {"mae": mae(yte, yhat), "rmse": rmse(yte, yhat),
                    "mape": mape(yte, yhat)},
        "persistence": {"mae": _persistence_mae(yte, y_naive)},
        "dates": {"test_start": str(dates[test_m].min().date()),
                  "test_end": str(dates[test_m].max().date())},
        "test_y": yte.values, "test_yhat": yhat,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_baseline(target: str = "collection", region: str = "national",
                 n_estimators: int = 800, max_depth: int = 6,
                 learning_rate: float = 0.05):
    """Train and evaluate the baseline, write metrics + plot. Returns the JSON dict."""
    series = _load_target(target, region)
    frame = build_feature_frame(series)
    print(f"[data] target = {target} ({region}) | "
          f"{frame.index.min().date()} -> {frame.index.max().date()} | "
          f"{len(frame)} daily rows")
    print(f"[model] backend = {'xgboost' if _HAS_XGBOOST else 'sklearn HistGradientBoosting (fallback)'}")

    params = {"n_estimators": n_estimators, "max_depth": max_depth,
              "learning_rate": learning_rate, "subsample": 0.9,
              "colsample_bytree": 0.9, "early_stopping_rounds": 30}

    raw_results = []
    for horizon in cfg.FORECAST_HORIZONS:
        res = _run_horizon(frame, horizon, params)
        xg = res["xgboost"]
        print(f"\n[horizon {horizon:2d}d] "
              f"MAE={xg['mae']:8.2f}  RMSE={xg['rmse']:8.2f}  MAPE={xg['mape']:6.2f}% "
              f"| naive MAE={res['persistence']['mae']:8.2f} "
              f"(test {res['n_test']} rows, {res['dates']['test_start']}..{res['dates']['test_end']})")
        raw_results.append(res)

    # Persist the metrics (drop the raw test arrays from the JSON).
    results = [{k: v for k, v in r.items() if k not in ("test_y", "test_yhat")}
               for r in raw_results]
    out = {"target": target, "region": region,
           "feature_columns": CALENDAR_COLS + ["lag_0"] + [f"lag_{l}" for l in cfg.LAGS] + [
               f"{s}_{w}" for w in cfg.ROLLING_WINDOWS for s in ("roll_mean", "roll_std")],
           "windows": {"start": str(frame.index.min().date()),
                       "end": str(frame.index.max().date())},
           "horizons": results}
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.OUTPUT_DIR / "baseline_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)

    _plot(raw_results, target, region)
    print(f"\n[ok] metrics -> {cfg.OUTPUT_DIR / 'baseline_metrics.json'}")
    if _HAS_MPL:
        print(f"[ok] plot    -> {cfg.OUTPUT_DIR / 'baseline_plot.png'}")
    return out


def main():
    ap = argparse.ArgumentParser(description="XGBoost engineering-features baseline")
    ap.add_argument("--target", default="collection", choices=["collection", "issuance"])
    ap.add_argument("--region", default="national")
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()
    run_baseline(args.target, args.region, args.n_estimators,
                 args.max_depth, args.learning_rate)


def _plot(results, target, region):
    if not _HAS_MPL:
        print(f"[warn] matplotlib not installed - skipping plot")
        return
    plt.figure(figsize=(11, 4.5))
    # Show the 30-day horizon (medium-term) as the representative figure.
    pick = next(r for r in results if r["horizon"] == 30)
    x = range(len(pick["test_y"]))
    plt.plot(x, pick["test_y"], label="actual", color="#2c3e50", lw=1.6)
    plt.plot(x, pick["test_yhat"], label="XGBoost forecast", color="#e74c3c", lw=1.6, alpha=0.85)
    plt.title(f"National {target} - XGBoost (30-day direct horizon, test set)")
    plt.ylabel("units per day")
    plt.xlabel(f"test days ({pick['dates']['test_start']} .. {pick['dates']['test_end']})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.OUTPUT_DIR / "baseline_plot.png", dpi=110)


if __name__ == "__main__":
    main()
