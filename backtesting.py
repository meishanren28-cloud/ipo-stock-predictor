from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from features import FEATURES
from modeling import _fit_quantile_model, _numeric_matrix


@dataclass
class BacktestResult:
    horizon: int
    predictions: pd.DataFrame
    quantile_summary: pd.DataFrame
    fold_summary: pd.DataFrame
    touch_calibration: pd.DataFrame
    touch_raw: pd.DataFrame


def _year_folds(samples: pd.DataFrame, min_train: int = 250, min_test: int = 30, max_folds: int = 5):
    """Strict IPO-group/time folds.

    For test year Y:
      * training rows must have asof_date < Jan 1 of Y
      * test rows belong to IPOs listed in Y

    This prevents the same IPO from appearing in train and test and also prevents
    future dates from leaking into the training set. It is slightly conservative:
    later-in-year tests do not get to use rows learned during that same year.
    """
    s = samples.copy()
    s["asof_date"] = pd.to_datetime(s["asof_date"], errors="coerce")
    s["listing_date"] = pd.to_datetime(s["listing_date"], errors="coerce")
    s = s.dropna(subset=["asof_date", "listing_date"])
    years = sorted(int(y) for y in s["listing_date"].dt.year.dropna().unique())
    candidates = []
    for y in years:
        cutoff = pd.Timestamp(y, 1, 1)
        train = s[s["asof_date"] < cutoff].copy()
        test = s[s["listing_date"].dt.year.eq(y)].copy()
        # The training cutoff automatically excludes all IPOs listed in y, but
        # enforce code separation explicitly as a guardrail.
        test_codes = set(test["code"].astype(str))
        train = train[~train["code"].astype(str).isin(test_codes)].copy()
        if len(train) >= min_train and len(test) >= min_test:
            candidates.append((y, train, test))
    return candidates[-max_folds:]


def _enforce_monotone(pred_matrix: np.ndarray) -> np.ndarray:
    """Sort each row's q10/q50/q90 to prevent finite-sample quantile crossing."""
    return np.sort(pred_matrix, axis=1)


def _quantile_metrics(pred: pd.DataFrame, prefix: str, actual_col: str) -> list[dict]:
    rows = []
    actual = pred[actual_col].to_numpy(float)
    for q in (0.1, 0.5, 0.9):
        p = pred[f"{prefix}_q{int(q*100)}"].to_numpy(float)
        coverage = float(np.mean(actual <= p))
        rows.append({
            "target": prefix,
            "metric": f"P{int(q*100)} actual<=prediction",
            "expected": q,
            "actual": coverage,
            "error": coverage - q,
        })
    p10 = pred[f"{prefix}_q10"].to_numpy(float)
    p90 = pred[f"{prefix}_q90"].to_numpy(float)
    inside = float(np.mean((actual >= p10) & (actual <= p90)))
    rows.append({
        "target": prefix,
        "metric": "P10-P90 interval coverage",
        "expected": 0.8,
        "actual": inside,
        "error": inside - 0.8,
    })
    width = float(np.mean(p90 - p10))
    rows.append({
        "target": prefix,
        "metric": "Mean P10-P90 width",
        "expected": np.nan,
        "actual": width,
        "error": np.nan,
    })
    return rows


def _neighbor_touch_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
    n_neighbors: int,
    relative_levels: tuple[float, ...],
) -> pd.DataFrame:
    """Predict touch probabilities for relative price levels using train-only neighbors."""
    if train.empty or test.empty:
        return pd.DataFrame()
    xt = _numeric_matrix(train)
    xv = _numeric_matrix(test)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_z = scaler.fit_transform(imp.fit_transform(xt))
    test_z = scaler.transform(imp.transform(xv))
    n = min(int(n_neighbors), len(train))
    nn = NearestNeighbors(n_neighbors=n, metric="euclidean")
    nn.fit(train_z)
    dist, idx = nn.kneighbors(test_z)

    max_ret = train[f"future_max_ret_{horizon}d"].to_numpy(float)
    min_ret = train[f"future_min_ret_{horizon}d"].to_numpy(float)
    actual_max = test[f"future_max_ret_{horizon}d"].to_numpy(float)
    actual_min = test[f"future_min_ret_{horizon}d"].to_numpy(float)

    rows: list[dict] = []
    for i in range(len(test)):
        d = dist[i]
        ids = idx[i]
        scale = max(float(np.median(d)), 1e-6)
        w = np.exp(-0.5 * (d / scale) ** 2)
        if w.sum() <= 0:
            w = np.ones_like(w)
        for r in relative_levels:
            if r >= 0:
                neighbor_hit = max_ret[ids] >= r
                actual_hit = actual_max[i] >= r
                direction = "up"
            else:
                neighbor_hit = min_ret[ids] <= r
                actual_hit = actual_min[i] <= r
                direction = "down"
            p = float(np.average(neighbor_hit.astype(float), weights=w))
            rows.append({
                "code": str(test.iloc[i]["code"]),
                "asof_date": pd.Timestamp(test.iloc[i]["asof_date"]),
                "relative_level": float(r),
                "direction": direction,
                "predicted_probability": p,
                "actual_hit": int(actual_hit),
            })
    return pd.DataFrame(rows)


def _calibration_bins(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["probability_bin", "count", "predicted_mean", "actual_rate", "gap"])
    edges = np.linspace(0, 1, 11)
    labels = [f"{int(edges[i]*100)}-{int(edges[i+1]*100)}%" for i in range(10)]
    x = raw.copy()
    # Include p==1.0 in the last bin.
    x["probability_bin"] = pd.cut(
        x["predicted_probability"].clip(0, 1),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    g = x.groupby("probability_bin", observed=True)
    out = g.agg(
        count=("actual_hit", "size"),
        predicted_mean=("predicted_probability", "mean"),
        actual_rate=("actual_hit", "mean"),
    ).reset_index()
    out["gap"] = out["actual_rate"] - out["predicted_mean"]
    return out


def walk_forward_backtest(
    samples: pd.DataFrame,
    horizon: int = 1,
    n_neighbors: int = 40,
    max_folds: int = 5,
    relative_levels: tuple[float, ...] = (-0.10, -0.075, -0.05, -0.025, 0.025, 0.05, 0.075, 0.10),
) -> BacktestResult:
    """Run strict out-of-sample walk-forward validation.

    The same model family used for live forecasts is fitted only on information
    that would have been available before each test year.
    """
    horizon = int(horizon)
    required = [
        f"future_max_ret_{horizon}d",
        f"future_min_ret_{horizon}d",
        f"future_close_ret_{horizon}d",
        "asof_date",
        "listing_date",
        "code",
        *FEATURES,
    ]
    missing = [c for c in required if c not in samples.columns]
    if missing:
        raise ValueError(f"Backtest missing columns: {missing[:6]}")
    s = samples.dropna(subset=[
        f"future_max_ret_{horizon}d",
        f"future_min_ret_{horizon}d",
        f"future_close_ret_{horizon}d",
        "asof_date",
        "listing_date",
    ]).copy()
    folds = _year_folds(s, max_folds=max_folds)
    if not folds:
        raise ValueError("Not enough dated IPO samples to form a strict out-of-sample fold.")

    pred_frames: list[pd.DataFrame] = []
    touch_frames: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    q_levels = (0.1, 0.5, 0.9)

    for test_year, train, test in folds:
        x_train = _numeric_matrix(train)
        x_test = _numeric_matrix(test)
        fold_pred = test[["code", "asof_date", "listing_date"]].copy().reset_index(drop=True)

        for prefix, target in [
            ("high", f"future_max_ret_{horizon}d"),
            ("low", f"future_min_ret_{horizon}d"),
        ]:
            pmat = []
            for q in q_levels:
                model = _fit_quantile_model(x_train, train[target], q)
                pmat.append(model.predict(x_test))
            pmat = _enforce_monotone(np.column_stack(pmat))
            for j, q in enumerate(q_levels):
                fold_pred[f"{prefix}_q{int(q*100)}"] = pmat[:, j]

        close_target = f"future_close_ret_{horizon}d"
        close_model = _fit_quantile_model(x_train, train[close_target], 0.5)
        fold_pred["close_q50"] = close_model.predict(x_test)
        fold_pred["actual_high"] = test[f"future_max_ret_{horizon}d"].to_numpy(float)
        fold_pred["actual_low"] = test[f"future_min_ret_{horizon}d"].to_numpy(float)
        fold_pred["actual_close"] = test[f"future_close_ret_{horizon}d"].to_numpy(float)
        fold_pred["test_year"] = int(test_year)
        pred_frames.append(fold_pred)

        touch = _neighbor_touch_predictions(train, test.reset_index(drop=True), horizon, n_neighbors, relative_levels)
        if not touch.empty:
            touch["test_year"] = int(test_year)
            touch_frames.append(touch)

        fold_rows.append({
            "test_year": int(test_year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_ipos": int(test["code"].astype(str).nunique()),
            "train_last_date": pd.Timestamp(train["asof_date"].max()),
            "test_first_listing": pd.Timestamp(test["listing_date"].min()),
        })

    pred = pd.concat(pred_frames, ignore_index=True)
    quantile_rows = []
    quantile_rows.extend(_quantile_metrics(pred, "high", "actual_high"))
    quantile_rows.extend(_quantile_metrics(pred, "low", "actual_low"))
    # Close median calibration and MAE.
    close_cov = float(np.mean(pred["actual_close"].to_numpy(float) <= pred["close_q50"].to_numpy(float)))
    close_mae = float(np.mean(np.abs(pred["actual_close"] - pred["close_q50"])))
    quantile_rows.extend([
        {"target": "close", "metric": "P50 actual<=prediction", "expected": 0.5, "actual": close_cov, "error": close_cov - 0.5},
        {"target": "close", "metric": "Median prediction MAE", "expected": np.nan, "actual": close_mae, "error": np.nan},
    ])
    quantile_summary = pd.DataFrame(quantile_rows)

    touch_raw = pd.concat(touch_frames, ignore_index=True) if touch_frames else pd.DataFrame()
    touch_cal = _calibration_bins(touch_raw)
    fold_summary = pd.DataFrame(fold_rows)
    return BacktestResult(
        horizon=horizon,
        predictions=pred,
        quantile_summary=quantile_summary,
        fold_summary=fold_summary,
        touch_calibration=touch_cal,
        touch_raw=touch_raw,
    )
