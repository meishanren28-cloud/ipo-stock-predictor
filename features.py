from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "day_index",
    "close_to_first_open",
    "close_to_first_close",
    "close_to_offer",
    "max_gain_since_list",
    "drawdown_from_peak",
    "days_since_peak",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "gap_pct",
    "range_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_pos_in_range",
    "volume_to_max",
    "volume_to_mean3",
    "volume_to_mean5",
    "volatility_3d",
    "volatility_5d",
    "down_days_3",
    "up_days_3",
    "peak_volume_age",
]


def _safe_div(a, b, default=np.nan):
    try:
        if b is None or pd.isna(b) or b == 0:
            return default
        return a / b
    except Exception:
        return default


def compute_state_features(history: pd.DataFrame, offer_price: float | None = None) -> dict:
    """Create features using only observations available at the final row."""
    g = history.copy().sort_values("Date").reset_index(drop=True)
    if len(g) < 2:
        raise ValueError("At least 2 daily bars are required")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    if g[["Open", "High", "Low", "Close"]].tail(1).isna().any(axis=None):
        raise ValueError("Latest OHLC contains missing values")

    cur = g.iloc[-1]
    prev = g.iloc[-2]
    first = g.iloc[0]
    closes = g["Close"]
    highs = g["High"]
    volumes = g["Volume"].fillna(0)
    rets = closes.pct_change()

    peak_idx = int(highs.values.argmax())
    peak_vol_idx = int(volumes.values.argmax()) if len(volumes) else 0
    day_index = len(g)
    day_range = float(cur["High"] - cur["Low"])
    body_hi = max(float(cur["Open"]), float(cur["Close"]))
    body_lo = min(float(cur["Open"]), float(cur["Close"]))

    out = {
        "day_index": float(day_index),
        "close_to_first_open": _safe_div(cur["Close"], first["Open"]) - 1,
        "close_to_first_close": _safe_div(cur["Close"], first["Close"]) - 1,
        "close_to_offer": _safe_div(cur["Close"], offer_price) - 1 if offer_price else np.nan,
        "max_gain_since_list": _safe_div(highs.max(), first["Open"]) - 1,
        "drawdown_from_peak": _safe_div(cur["Close"], highs.max()) - 1,
        "days_since_peak": float(len(g) - 1 - peak_idx),
        "ret_1d": _safe_div(cur["Close"], prev["Close"]) - 1,
        "ret_3d": _safe_div(cur["Close"], g["Close"].iloc[-4]) - 1 if len(g) >= 4 else np.nan,
        "ret_5d": _safe_div(cur["Close"], g["Close"].iloc[-6]) - 1 if len(g) >= 6 else np.nan,
        "gap_pct": _safe_div(cur["Open"], prev["Close"]) - 1,
        "range_pct": _safe_div(day_range, prev["Close"]),
        "body_pct": _safe_div(cur["Close"] - cur["Open"], prev["Close"]),
        "upper_wick_pct": _safe_div(cur["High"] - body_hi, prev["Close"]),
        "lower_wick_pct": _safe_div(body_lo - cur["Low"], prev["Close"]),
        "close_pos_in_range": _safe_div(cur["Close"] - cur["Low"], day_range, default=0.5),
        "volume_to_max": _safe_div(cur["Volume"], volumes.max(), default=0.0),
        "volume_to_mean3": _safe_div(cur["Volume"], volumes.tail(3).mean(), default=0.0),
        "volume_to_mean5": _safe_div(cur["Volume"], volumes.tail(5).mean(), default=0.0),
        "volatility_3d": float(rets.tail(3).std(ddof=0)) if len(g) >= 4 else np.nan,
        "volatility_5d": float(rets.tail(5).std(ddof=0)) if len(g) >= 6 else np.nan,
        "down_days_3": float((rets.tail(3) < 0).sum()),
        "up_days_3": float((rets.tail(3) > 0).sum()),
        "peak_volume_age": float(len(g) - 1 - peak_vol_idx),
    }
    return out


def build_training_samples(
    prices: pd.DataFrame,
    master: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
    min_history_days: int = 4,
    max_state_day: int = 35,
) -> pd.DataFrame:
    master_idx = master.drop_duplicates("code").set_index("code")
    rows: list[dict] = []
    for code, g in prices.groupby("code"):
        code = str(code)
        if code not in master_idx.index:
            continue
        meta = master_idx.loc[code]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[-1]
        offer = meta.get("offer_price")
        offer = None if pd.isna(offer) else float(offer)
        g = g.sort_values("Date").reset_index(drop=True)
        max_h = max(horizons)
        last_t = min(len(g) - max_h - 1, max_state_day - 1)
        if last_t < min_history_days - 1:
            continue

        for t in range(min_history_days - 1, last_t + 1):
            hist = g.iloc[: t + 1]
            try:
                feats = compute_state_features(hist, offer_price=offer)
            except Exception:
                continue
            cur_close = float(hist["Close"].iloc[-1])
            row = {
                "code": code,
                "asof_date": pd.Timestamp(hist["Date"].iloc[-1]),
                "listing_date": pd.Timestamp(meta["listing_date"]),
                "market": str(meta.get("market", "")),
                "offer_price": offer,
                **feats,
            }
            for h in horizons:
                fut = g.iloc[t + 1 : t + 1 + h]
                row[f"future_max_ret_{h}d"] = float(fut["High"].max() / cur_close - 1)
                row[f"future_min_ret_{h}d"] = float(fut["Low"].min() / cur_close - 1)
                row[f"future_close_ret_{h}d"] = float(fut["Close"].iloc[-1] / cur_close - 1)
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["asof_date", "code"]).reset_index(drop=True)


def state_row_from_inputs(
    history: pd.DataFrame,
    offer_price: float | None,
    code: str,
    listing_date,
    market: str,
) -> pd.DataFrame:
    feats = compute_state_features(history, offer_price=offer_price)
    row = {
        "code": str(code),
        "asof_date": pd.Timestamp(history["Date"].iloc[-1]),
        "listing_date": pd.Timestamp(listing_date),
        "market": market,
        "offer_price": offer_price,
        **feats,
    }
    return pd.DataFrame([row])
