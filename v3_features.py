from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = [
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

# Cross-sectional context among other newly-listed stocks on the same trading date.
# These are computed only from information available on that date; no future prices are used.
PEER_FEATURES = [
    "peer_count_log",
    "peer_median_ret_1d",
    "peer_up_share_1d",
    "rel_ret_1d_peer",
    "peer_median_ret_3d",
    "rel_ret_3d_peer",
    "peer_median_drawdown",
    "rel_drawdown_peer",
    "peer_median_volume5",
    "rel_volume5_peer",
    "peer_median_range",
    "rel_range_peer",
]

FEATURES = BASE_FEATURES + PEER_FEATURES

# Extra feature known only once the next session opening price (or a pre-open scenario) is available.
# It is intentionally NOT part of FEATURES so the ordinary close-to-next-day model stays unchanged.
OPEN_CONTEXT_FEATURE = "known_open_gap"
OPEN_FEATURES = FEATURES + [OPEN_CONTEXT_FEATURE]


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


def add_peer_context_features(samples: pd.DataFrame, min_group_size: int = 2) -> pd.DataFrame:
    """Add same-day IPO cohort context to historical state rows.

    The comparison set is every IPO state available on the same `asof_date`
    (normally the first <=35 trading days after listing). This is a contemporaneous
    cross-sectional feature and therefore does not use future market information.

    If only one IPO state exists on a date, peer-derived values are left NaN so the
    model's imputer can explicitly treat peer context as unavailable.
    """
    if samples.empty:
        return samples.copy()
    out = samples.copy()
    if "asof_date" not in out.columns:
        for c in PEER_FEATURES:
            if c not in out.columns:
                out[c] = np.nan
        return out

    out["asof_date"] = pd.to_datetime(out["asof_date"], errors="coerce").dt.normalize()
    grp = out.groupby("asof_date", dropna=False)
    counts = grp["code"].transform("count") if "code" in out.columns else grp["ret_1d"].transform("count")
    out["peer_count_log"] = np.log1p(counts.astype(float))

    metrics = {
        "ret_1d": ("peer_median_ret_1d", "rel_ret_1d_peer"),
        "ret_3d": ("peer_median_ret_3d", "rel_ret_3d_peer"),
        "drawdown_from_peak": ("peer_median_drawdown", "rel_drawdown_peer"),
        "volume_to_mean5": ("peer_median_volume5", "rel_volume5_peer"),
        "range_pct": ("peer_median_range", "rel_range_peer"),
    }
    for source, (median_col, rel_col) in metrics.items():
        if source not in out.columns:
            out[median_col] = np.nan
            out[rel_col] = np.nan
            continue
        med = grp[source].transform("median")
        med = med.where(counts >= min_group_size)
        out[median_col] = med
        out[rel_col] = out[source] - med

    if "ret_1d" in out.columns:
        up_share = grp["ret_1d"].transform(lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean()))
        out["peer_up_share_1d"] = up_share.where(counts >= min_group_size)
    else:
        out["peer_up_share_1d"] = np.nan

    for c in PEER_FEATURES:
        if c not in out.columns:
            out[c] = np.nan
    return out


def apply_live_peer_context(state: pd.DataFrame, peer_states: pd.DataFrame) -> pd.DataFrame:
    """Apply the same peer feature definitions to one live target state.

    `peer_states` should contain the target plus other current IPO states for the
    same trading date when possible. When peer data are unavailable, NaNs are used
    and the trained model falls back via its median imputer/indicator.
    """
    s = state.copy()
    for c in PEER_FEATURES:
        s[c] = np.nan
    if s.empty:
        return s

    pool = peer_states.copy() if peer_states is not None else pd.DataFrame()
    if pool.empty:
        return s

    # Ensure the live target is represented exactly once in the comparison pool.
    target_code = str(s.iloc[0].get("code", ""))
    if "code" in pool.columns and target_code:
        pool = pool[pool["code"].astype(str) != target_code].copy()
    pool = pd.concat([pool, s], ignore_index=True, sort=False)
    if len(pool) < 2:
        return s

    enriched = add_peer_context_features(pool, min_group_size=2)
    if target_code and "code" in enriched.columns:
        row = enriched[enriched["code"].astype(str).eq(target_code)].tail(1)
    else:
        row = enriched.tail(1)
    if row.empty:
        return s
    for c in PEER_FEATURES:
        s.loc[s.index[0], c] = row.iloc[0].get(c, np.nan)
    return s


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
            # Next-session opening-conditioned targets. These let a second model answer:
            # “given the opening price we now know, how far does the session usually travel?”
            # They are computed from the very next daily bar and therefore can be used in strict
            # walk-forward backtests without minute data.
            nxt = g.iloc[t + 1]
            nxt_open = float(nxt["Open"])
            if nxt_open > 0:
                row["next_open_gap_1d"] = float(nxt_open / cur_close - 1)
                row["next_high_from_open_1d"] = float(float(nxt["High"]) / nxt_open - 1)
                row["next_low_from_open_1d"] = float(float(nxt["Low"]) / nxt_open - 1)
                row["next_close_from_open_1d"] = float(float(nxt["Close"]) / nxt_open - 1)
            else:
                row["next_open_gap_1d"] = np.nan
                row["next_high_from_open_1d"] = np.nan
                row["next_low_from_open_1d"] = np.nan
                row["next_close_from_open_1d"] = np.nan

            for h in horizons:
                fut = g.iloc[t + 1 : t + 1 + h]
                row[f"future_max_ret_{h}d"] = float(fut["High"].max() / cur_close - 1)
                row[f"future_min_ret_{h}d"] = float(fut["Low"].min() / cur_close - 1)
                row[f"future_close_ret_{h}d"] = float(fut["Close"].iloc[-1] / cur_close - 1)
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["asof_date", "code"]).reset_index(drop=True)
    return add_peer_context_features(out)


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
    for c in PEER_FEATURES:
        row[c] = np.nan
    return pd.DataFrame([row])


def add_open_conditioned_targets(samples: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible upgrade for an existing training_samples.parquet.

    Older V2 caches contain the state features and close-anchored future targets but not
    the next session's opening-conditioned targets. If ipo_daily.parquet is available,
    derive those columns without re-downloading any data.
    """
    if samples.empty:
        return samples.copy()
    needed = [
        "next_open_gap_1d",
        "next_high_from_open_1d",
        "next_low_from_open_1d",
        "next_close_from_open_1d",
    ]
    if all(c in samples.columns for c in needed):
        return samples.copy()
    out = samples.copy()
    for c in needed:
        if c not in out.columns:
            out[c] = np.nan
    if prices is None or prices.empty or not {"code", "Date", "Open", "High", "Low", "Close"}.issubset(prices.columns):
        return out

    p = prices.copy()
    p["code"] = p["code"].astype(str).str.upper()
    p["Date"] = pd.to_datetime(p["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    for c in ["Open", "High", "Low", "Close"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna(subset=["code", "Date", "Open", "High", "Low", "Close"]).sort_values(["code", "Date"])
    p["next_Date"] = p.groupby("code")["Date"].shift(-1)
    for c in ["Open", "High", "Low", "Close"]:
        p[f"next_{c}"] = p.groupby("code")[c].shift(-1)
    lookup = p[["code", "Date", "Close", "next_Open", "next_High", "next_Low", "next_Close"]].copy()
    lookup = lookup.rename(columns={"Date": "asof_date", "Close": "state_close"})

    out["code"] = out["code"].astype(str).str.upper()
    out["asof_date"] = pd.to_datetime(out["asof_date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    merged = out[["code", "asof_date"]].merge(lookup, on=["code", "asof_date"], how="left")
    state_close = pd.to_numeric(merged["state_close"], errors="coerce")
    nxt_open = pd.to_numeric(merged["next_Open"], errors="coerce")
    valid = (state_close > 0) & (nxt_open > 0)
    vals = {
        "next_open_gap_1d": nxt_open / state_close - 1,
        "next_high_from_open_1d": pd.to_numeric(merged["next_High"], errors="coerce") / nxt_open - 1,
        "next_low_from_open_1d": pd.to_numeric(merged["next_Low"], errors="coerce") / nxt_open - 1,
        "next_close_from_open_1d": pd.to_numeric(merged["next_Close"], errors="coerce") / nxt_open - 1,
    }
    for c, v in vals.items():
        arr = v.where(valid).to_numpy()
        existing = pd.to_numeric(out[c], errors="coerce").to_numpy()
        fill = pd.isna(existing) & pd.notna(arr)
        existing[fill] = arr[fill]
        out[c] = existing
    return out
