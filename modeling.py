from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import FEATURES, OPEN_FEATURES

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class ForecastBundle:
    horizon: int
    current_price: float
    low_quantiles: dict[float, float]
    high_quantiles: dict[float, float]
    close_median: float
    neighbors: pd.DataFrame
    metrics: dict[str, float]
    sample_count: int


@dataclass
class OpenForecastBundle:
    """One-session forecast after an opening price (or pre-open scenario) is known."""
    open_price: float
    previous_close: float
    low_quantiles: dict[float, float]
    high_quantiles: dict[float, float]
    close_median: float
    neighbors: pd.DataFrame
    metrics: dict[str, float]
    sample_count: int
    open_gap: float
    mode: str = "actual_open"


def _matrix_for(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    x = df.reindex(columns=feature_names).copy()
    return x.replace([np.inf, -np.inf], np.nan)


def _numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return _matrix_for(df, FEATURES)


def _fit_quantile_model(x: pd.DataFrame, y: pd.Series, q: float):
    if LGBMRegressor is not None:
        model = LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=220,
            learning_rate=0.035,
            num_leaves=15,
            min_child_samples=18,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.2,
            random_state=42,
            verbosity=-1,
        )
    else:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=180,
            learning_rate=0.04,
            max_depth=2,
            random_state=42,
        )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", model),
    ])
    pipe.fit(x, y)
    return pipe


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    if weights.sum() <= 0:
        return float(np.quantile(values, q))
    cdf = np.cumsum(weights) / weights.sum()
    return float(np.interp(q, cdf, values))


def find_similar_cases(samples: pd.DataFrame, state: pd.DataFrame, n_neighbors: int = 40) -> pd.DataFrame:
    if len(samples) < 5:
        return pd.DataFrame()
    x = _numeric_matrix(samples)
    s = _numeric_matrix(state)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x2 = scaler.fit_transform(imp.fit_transform(x))
    s2 = scaler.transform(imp.transform(s))
    n = min(n_neighbors, len(samples))
    nn = NearestNeighbors(n_neighbors=n, metric="euclidean")
    nn.fit(x2)
    dist, idx = nn.kneighbors(s2)
    out = samples.iloc[idx[0]].copy()
    out.insert(0, "distance", dist[0])
    # Smooth, monotonic weights. Very close cases dominate but no single case can
    # completely overwhelm the empirical distribution.
    scale = max(float(np.median(dist[0])), 1e-6)
    out.insert(1, "weight", np.exp(-0.5 * (out["distance"] / scale) ** 2))
    return out.reset_index(drop=True)


def _temporal_train_test(samples: pd.DataFrame, min_test: int = 100):
    s = samples.sort_values("asof_date").reset_index(drop=True)
    if len(s) < 250:
        cut = max(int(len(s) * 0.8), len(s) - min_test)
    else:
        cut = int(len(s) * 0.8)
    cut = min(max(cut, 20), len(s) - max(20, min_test if len(s) > min_test + 20 else 20))
    return s.iloc[:cut], s.iloc[cut:]


def forecast(
    samples: pd.DataFrame,
    state: pd.DataFrame,
    current_price: float,
    horizon: int = 1,
    n_neighbors: int = 40,
) -> ForecastBundle:
    required = [f"future_max_ret_{horizon}d", f"future_min_ret_{horizon}d", f"future_close_ret_{horizon}d"]
    s = samples.dropna(subset=required).copy()
    target_code = str(state["code"].iloc[0]) if "code" in state.columns else ""
    # Leave the target stock itself out. For a just-listed stock, its earlier
    # states would otherwise be trivially closest to its current state and
    # make the "historical analog" result circular.
    if target_code:
        s_other = s[s["code"].astype(str) != target_code].copy()
        if len(s_other) >= 80:
            s = s_other
    if len(s) < 80:
        raise ValueError(f"Usable historical samples are too few ({len(s)}); at least 80 are required.")

    x = _numeric_matrix(s)
    x_state = _numeric_matrix(state)
    q_levels = (0.1, 0.5, 0.9)
    high_ret: dict[float, float] = {}
    low_ret: dict[float, float] = {}

    for q in q_levels:
        mh = _fit_quantile_model(x, s[f"future_max_ret_{horizon}d"], q)
        ml = _fit_quantile_model(x, s[f"future_min_ret_{horizon}d"], q)
        high_ret[q] = float(mh.predict(x_state)[0])
        low_ret[q] = float(ml.predict(x_state)[0])

    # Guard against quantile crossing caused by finite-sample model error.
    high_sorted = np.sort([high_ret[q] for q in q_levels])
    low_sorted = np.sort([low_ret[q] for q in q_levels])
    high_ret = dict(zip(q_levels, high_sorted))
    low_ret = dict(zip(q_levels, low_sorted))

    mc = _fit_quantile_model(x, s[f"future_close_ret_{horizon}d"], 0.5)
    close_ret = float(mc.predict(x_state)[0])

    neighbors = find_similar_cases(s, state, n_neighbors=n_neighbors)

    # Lightweight, time-ordered holdout metrics. This is deliberately not random
    # split, which would leak later IPO regimes into earlier validation.
    train, test = _temporal_train_test(s)
    metrics: dict[str, float] = {}
    if len(test) >= 20 and len(train) >= 50:
        xt = _numeric_matrix(train)
        xv = _numeric_matrix(test)
        for label, target in [
            ("high", f"future_max_ret_{horizon}d"),
            ("low", f"future_min_ret_{horizon}d"),
            ("close", f"future_close_ret_{horizon}d"),
        ]:
            m = _fit_quantile_model(xt, train[target], 0.5)
            pred = m.predict(xv)
            metrics[f"mae_{label}_ret"] = float(mean_absolute_error(test[target], pred))

    high_prices = {q: current_price * (1 + r) for q, r in high_ret.items()}
    low_prices = {q: current_price * (1 + r) for q, r in low_ret.items()}
    return ForecastBundle(
        horizon=horizon,
        current_price=current_price,
        low_quantiles=low_prices,
        high_quantiles=high_prices,
        close_median=current_price * (1 + close_ret),
        neighbors=neighbors,
        metrics=metrics,
        sample_count=len(s),
    )


def empirical_touch_probabilities(
    neighbors: pd.DataFrame,
    current_price: float,
    horizon: int,
    levels: list[float],
) -> pd.DataFrame:
    if neighbors.empty:
        return pd.DataFrame(columns=["level", "direction", "probability"])
    weights = neighbors["weight"].to_numpy(float)
    max_ret = neighbors[f"future_max_ret_{horizon}d"].to_numpy(float)
    min_ret = neighbors[f"future_min_ret_{horizon}d"].to_numpy(float)
    rows = []
    for level in levels:
        r = level / current_price - 1
        if level >= current_price:
            hit = max_ret >= r
            direction = "上触"
        else:
            hit = min_ret <= r
            direction = "下触"
        p = float(np.average(hit.astype(float), weights=weights)) if weights.sum() else float(hit.mean())
        rows.append({"level": float(level), "direction": direction, "probability": p})
    return pd.DataFrame(rows)


def similar_case_distribution(neighbors: pd.DataFrame, current_price: float, horizon: int) -> dict[str, float]:
    if neighbors.empty:
        return {}
    w = neighbors["weight"].to_numpy(float)
    hi = neighbors[f"future_max_ret_{horizon}d"].to_numpy(float)
    lo = neighbors[f"future_min_ret_{horizon}d"].to_numpy(float)
    out = {}
    for q in [0.1, 0.5, 0.9]:
        out[f"high_q{int(q*100)}"] = current_price * (1 + _weighted_quantile(hi, w, q))
        out[f"low_q{int(q*100)}"] = current_price * (1 + _weighted_quantile(lo, w, q))
    return out


def model_action(
    bundle: ForecastBundle,
    sell_level: float | None,
    buyback_level: float | None,
    shares_for_t: int,
    probability_calibrator=None,
) -> dict[str, Any]:
    """Translate model output into a transparent, non-execution decision aid.

    Daily OHLC cannot establish whether a day's high happened before its low.
    Therefore the engine never claims a same-day sell-then-buy sequence probability.
    """
    neighbors = bundle.neighbors
    if neighbors.empty:
        return {"action": "历史相似样本不足", "confidence": "低", "reasons": []}

    levels = [x for x in [sell_level, buyback_level] if x and x > 0]
    probs = empirical_touch_probabilities(neighbors, bundle.current_price, bundle.horizon, levels)
    p_sell = None
    p_buy = None
    if sell_level:
        r = probs[probs["level"].eq(float(sell_level))]
        p_sell = float(r["probability"].iloc[0]) if not r.empty else None
        if p_sell is not None and probability_calibrator is not None:
            p_sell = float(probability_calibrator(p_sell))
    if buyback_level:
        r = probs[probs["level"].eq(float(buyback_level))]
        p_buy = float(r["probability"].iloc[0]) if not r.empty else None
        if p_buy is not None and probability_calibrator is not None:
            p_buy = float(probability_calibrator(p_buy))

    # Additional upside threshold: +8% above the user's sell level, used to
    # quantify sell-too-early risk rather than hiding it.
    p_blowthrough = None
    if sell_level and sell_level >= bundle.current_price:
        extra = sell_level * 1.08
        tmp = empirical_touch_probabilities(neighbors, bundle.current_price, bundle.horizon, [extra])
        p_blowthrough = float(tmp["probability"].iloc[0])
        if probability_calibrator is not None:
            p_blowthrough = float(probability_calibrator(p_blowthrough))

    reasons = []
    if p_sell is not None:
        reasons.append(f"历史相似案例触及卖价的加权概率约 {p_sell:.0%}")
    if p_buy is not None:
        reasons.append(f"历史相似案例触及回补价的加权概率约 {p_buy:.0%}")
    if p_blowthrough is not None:
        reasons.append(f"卖价再上方约8%的触及概率约 {p_blowthrough:.0%}（卖飞风险代理）")

    if sell_level and sell_level >= bundle.current_price and p_sell is not None:
        if p_sell >= 0.58 and (p_blowthrough is None or p_blowthrough <= 0.38):
            action = f"模型倾向：到 {sell_level:.0f} 附近分批减 {shares_for_t} 股"
            confidence = "中高" if p_sell >= 0.68 else "中"
        elif p_sell >= 0.42:
            action = f"模型倾向：{sell_level:.0f} 可作为条件卖价，但不追求机械成交"
            confidence = "中"
        else:
            action = f"模型倾向：当前不把 {sell_level:.0f} 当作高概率成交目标"
            confidence = "中低"
    else:
        # No explicit sell target: use median high as a reference, not a magic target.
        action = f"模型参考：{bundle.horizon}日内最高中位区约 {bundle.high_quantiles[0.5]:.0f}"
        confidence = "中"

    return {
        "action": action,
        "confidence": confidence,
        "p_sell": p_sell,
        "p_buy": p_buy,
        "p_blowthrough": p_blowthrough,
        "reasons": reasons,
        "sequence_warning": "仅凭日线无法判断同一天内‘先卖价后回补价’的先后顺序；该顺序必须用分钟线或盘中截图更新。",
    }


def _prepare_open_samples(samples: pd.DataFrame, target_code: str = "") -> pd.DataFrame:
    required = [
        "next_open_gap_1d",
        "next_high_from_open_1d",
        "next_low_from_open_1d",
        "next_close_from_open_1d",
    ]
    missing = [c for c in required if c not in samples.columns]
    if missing:
        raise ValueError("开盘大师需要新版训练目标；请先重新建立历史库，或确保 ipo_daily.parquet 可用于自动升级。")
    s = samples.dropna(subset=required).copy()
    s["known_open_gap"] = pd.to_numeric(s["next_open_gap_1d"], errors="coerce")
    s = s.dropna(subset=["known_open_gap"])
    if target_code:
        other = s[s["code"].astype(str) != str(target_code)].copy()
        if len(other) >= 80:
            s = other
    return s


def _open_state(state: pd.DataFrame, previous_close: float, open_price: float) -> pd.DataFrame:
    if previous_close <= 0 or open_price <= 0:
        raise ValueError("前收与开盘/气配价格必须大于0")
    x = state.copy()
    x["known_open_gap"] = float(open_price / previous_close - 1)
    return x


def find_similar_open_cases(
    samples: pd.DataFrame,
    state: pd.DataFrame,
    previous_close: float,
    open_price: float,
    n_neighbors: int = 40,
) -> pd.DataFrame:
    """KNN analogs conditioned on the next session's known/scenario opening gap."""
    target_code = str(state["code"].iloc[0]) if "code" in state.columns and len(state) else ""
    s = _prepare_open_samples(samples, target_code=target_code)
    if len(s) < 5:
        return pd.DataFrame()
    live = _open_state(state, previous_close, open_price)
    x = _matrix_for(s, OPEN_FEATURES)
    q = _matrix_for(live, OPEN_FEATURES)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x2 = scaler.fit_transform(imp.fit_transform(x))
    q2 = scaler.transform(imp.transform(q))
    n = min(int(n_neighbors), len(s))
    nn = NearestNeighbors(n_neighbors=n, metric="euclidean")
    nn.fit(x2)
    dist, idx = nn.kneighbors(q2)
    out = s.iloc[idx[0]].copy()
    out.insert(0, "distance", dist[0])
    scale = max(float(np.median(dist[0])), 1e-6)
    out.insert(1, "weight", np.exp(-0.5 * (out["distance"] / scale) ** 2))
    return out.reset_index(drop=True)


def open_forecast(
    samples: pd.DataFrame,
    state: pd.DataFrame,
    previous_close: float,
    open_price: float,
    n_neighbors: int = 40,
    mode: str = "actual_open",
) -> OpenForecastBundle:
    """Forecast the current session after an opening price is known.

    Historical training uses the real next-session opening gap as an input, so the
    ``actual_open`` mode is strictly backtestable with free daily OHLC data. A pre-open
    indicative quote can be passed as a scenario, but that quote itself is not a
    historically backtested 08:55 feature.
    """
    target_code = str(state["code"].iloc[0]) if "code" in state.columns and len(state) else ""
    s = _prepare_open_samples(samples, target_code=target_code)
    if len(s) < 80:
        raise ValueError(f"开盘大师可用历史样本过少（{len(s)}）；至少需要80条。")
    live = _open_state(state, previous_close, open_price)
    x = _matrix_for(s, OPEN_FEATURES)
    xv = _matrix_for(live, OPEN_FEATURES)
    q_levels = (0.1, 0.5, 0.9)
    preds: dict[str, dict[float, float]] = {"high": {}, "low": {}}
    for prefix, target in [("high", "next_high_from_open_1d"), ("low", "next_low_from_open_1d")]:
        raw = []
        for q in q_levels:
            m = _fit_quantile_model(x, s[target], q)
            raw.append(float(m.predict(xv)[0]))
        raw = np.sort(raw)
        preds[prefix] = dict(zip(q_levels, raw))
    mc = _fit_quantile_model(x, s["next_close_from_open_1d"], 0.5)
    close_ret = float(mc.predict(xv)[0])
    neighbors = find_similar_open_cases(s, live, previous_close, open_price, n_neighbors=n_neighbors)

    train, test = _temporal_train_test(s)
    metrics: dict[str, float] = {}
    if len(test) >= 20 and len(train) >= 50:
        train = train.copy(); test = test.copy()
        train["known_open_gap"] = train["next_open_gap_1d"]
        test["known_open_gap"] = test["next_open_gap_1d"]
        xt = _matrix_for(train, OPEN_FEATURES)
        xv2 = _matrix_for(test, OPEN_FEATURES)
        for label, target in [
            ("high", "next_high_from_open_1d"),
            ("low", "next_low_from_open_1d"),
            ("close", "next_close_from_open_1d"),
        ]:
            m = _fit_quantile_model(xt, train[target], 0.5)
            metrics[f"mae_{label}_ret"] = float(mean_absolute_error(test[target], m.predict(xv2)))

    return OpenForecastBundle(
        open_price=float(open_price),
        previous_close=float(previous_close),
        low_quantiles={q: float(open_price) * (1 + preds["low"][q]) for q in q_levels},
        high_quantiles={q: float(open_price) * (1 + preds["high"][q]) for q in q_levels},
        close_median=float(open_price) * (1 + close_ret),
        neighbors=neighbors,
        metrics=metrics,
        sample_count=len(s),
        open_gap=float(open_price / previous_close - 1),
        mode=mode,
    )


def open_touch_probabilities(bundle: OpenForecastBundle, levels: list[float], probability_calibrator=None) -> pd.DataFrame:
    """Weighted next-session touch probabilities, anchored to the opening price."""
    n = bundle.neighbors
    if n.empty:
        return pd.DataFrame(columns=["level", "direction", "probability"])
    w = n["weight"].to_numpy(float)
    hi = n["next_high_from_open_1d"].to_numpy(float)
    lo = n["next_low_from_open_1d"].to_numpy(float)
    rows = []
    for level in levels:
        r = float(level) / bundle.open_price - 1
        if level >= bundle.open_price:
            hit = hi >= r
            direction = "上触"
        else:
            hit = lo <= r
            direction = "下触"
        p = float(np.average(hit.astype(float), weights=w)) if w.sum() else float(hit.mean())
        raw = p
        if probability_calibrator is not None:
            p = float(probability_calibrator(p))
        rows.append({"level": float(level), "direction": direction, "probability": p, "raw_probability": raw})
    return pd.DataFrame(rows)


def neighbor_quality(neighbors: pd.DataFrame, feature_count: int | None = None) -> dict[str, Any]:
    """Transparent diagnostics instead of pretending every KNN result is equally similar."""
    if neighbors is None or neighbors.empty:
        return {"count": 0, "effective_n": 0.0, "grade": "低", "nearest_rms_z": np.nan, "median_rms_z": np.nan, "strong_count": 0, "moderate_count": 0}
    d = pd.to_numeric(neighbors["distance"], errors="coerce").dropna().to_numpy(float)
    w = pd.to_numeric(neighbors["weight"], errors="coerce").fillna(0).to_numpy(float)
    eff = float((w.sum() ** 2) / np.square(w).sum()) if np.square(w).sum() > 0 else 0.0
    dim = max(int(feature_count or len(OPEN_FEATURES)), 1)
    rms = d / np.sqrt(dim)
    nearest = float(np.min(rms)) if len(rms) else np.nan
    median = float(np.median(rms)) if len(rms) else np.nan
    strong = int(np.sum(rms <= 0.75))
    moderate = int(np.sum(rms <= 1.0))
    if strong >= 5 and eff >= 8 and nearest <= 0.65:
        grade = "高"
    elif moderate >= 8 and eff >= 6 and nearest <= 0.95:
        grade = "中"
    else:
        grade = "低"
    return {
        "count": int(len(neighbors)),
        "effective_n": eff,
        "grade": grade,
        "nearest_rms_z": nearest,
        "median_rms_z": median,
        "strong_count": strong,
        "moderate_count": moderate,
    }


def _weighted_rate(mask: np.ndarray, weights: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=float)
    return float(np.average(mask, weights=weights)) if weights.sum() > 0 else float(mask.mean())


def optimize_trade_grid(
    bundle: OpenForecastBundle,
    shares: int = 100,
    price_step: float = 10.0,
    risk_pct: float = 0.05,
    min_spread_pct: float = 0.025,
    max_rows: int = 2000,
) -> pd.DataFrame:
    """Search buy→sell plans using only facts daily OHLC can support.

    ``success_lower`` is a conservative rate that daily bars can actually prove.
    For a buy below the open, the daily low must reach the buy and the close must
    finish at/above the sell, which guarantees the sell occurred after the buy.
    ``success_upper`` only requires both prices to have appeared, so order may be unknown.
    """
    n = bundle.neighbors
    if n.empty or bundle.open_price <= 0:
        return pd.DataFrame()
    step = max(float(price_step), 1.0)
    op = float(bundle.open_price)
    w = n["weight"].to_numpy(float)
    hi = n["next_high_from_open_1d"].to_numpy(float)
    lo = n["next_low_from_open_1d"].to_numpy(float)
    cl = n["next_close_from_open_1d"].to_numpy(float)
    low_floor = max(step, min(bundle.low_quantiles.values()) * 0.98)
    buy_floor = min(low_floor, op * 0.96)
    sell_floor = op * (1 + min_spread_pct)
    sell_ceiling = max(bundle.high_quantiles.values()) * 1.02
    buy_values = np.arange(np.floor(buy_floor / step) * step, np.floor(op / step) * step + step / 2, step)
    sell_values = np.arange(np.ceil(sell_floor / step) * step, np.ceil(sell_ceiling / step) * step + step / 2, step)
    rows = []
    for buy in buy_values:
        if buy <= 0 or buy > op + 1e-9:
            continue
        buy_r = buy / op - 1
        buy_hit = lo <= buy_r
        p_buy = _weighted_rate(buy_hit, w)
        if p_buy < 0.12:
            continue
        stop = buy * (1 - risk_pct)
        stop_r = stop / op - 1
        p_stop = _weighted_rate(lo <= stop_r, w)
        for sell in sell_values:
            if sell <= buy or (sell / buy - 1) < min_spread_pct:
                continue
            sell_r = sell / op - 1
            sell_hit = hi >= sell_r
            both = buy_hit & sell_hit
            definite = sell_hit if buy >= op - 1e-9 else (buy_hit & (cl >= sell_r))
            p_sell = _weighted_rate(sell_hit, w)
            p_upper = _weighted_rate(both, w)
            p_lower = _weighted_rate(definite, w)
            if p_upper < 0.10:
                continue
            p_mid = 0.5 * (p_lower + p_upper)
            spread = float(sell - buy)
            loss_band = float(buy - stop)
            expected_conservative_per_share = p_lower * spread - p_stop * loss_band
            expected_mid_per_share = p_mid * spread - p_stop * loss_band
            rows.append({
                "buy": float(buy), "sell": float(sell),
                "spread": spread, "spread_pct": spread / buy,
                "p_buy": p_buy, "p_sell": p_sell,
                "success_lower": p_lower, "success_upper": p_upper,
                "sequence_unknown": max(p_upper - p_lower, 0.0),
                "p_downside_5pct": p_stop,
                "expected_conservative_yen": expected_conservative_per_share * int(shares),
                "expected_mid_yen": expected_mid_per_share * int(shares),
                "shares": int(shares),
            })
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["score"] = (
        out["expected_conservative_yen"]
        + 0.20 * out["expected_mid_yen"]
        + 1000.0 * out["success_lower"]
        - 600.0 * out["sequence_unknown"]
    )
    return out.sort_values(["score", "success_lower", "expected_conservative_yen"], ascending=False).reset_index(drop=True)


def select_master_plans(grid: pd.DataFrame) -> dict[str, dict[str, Any] | None]:
    if grid is None or grid.empty:
        return {"稳健": None, "首选": None, "激进": None}

    # A "master" should be allowed to say no-trade. Do not label a negative
    # conservative expectancy row as the preferred plan just to fill the table.
    positive = grid[grid["expected_conservative_yen"] > 0].copy()
    best = positive.iloc[0] if not positive.empty else None

    conservative_pool = grid[(grid["expected_conservative_yen"] > 0) & (grid["p_buy"] >= 0.45) & (grid["success_upper"] >= 0.35)].copy()
    if conservative_pool.empty:
        conservative_pool = positive.copy()
    conservative = (conservative_pool.sort_values(
        ["success_lower", "p_buy", "expected_conservative_yen"], ascending=False
    ).iloc[0] if not conservative_pool.empty else None)

    if best is not None:
        aggressive_pool = grid[(grid["expected_mid_yen"] > 0) & (grid["spread_pct"] >= max(float(best["spread_pct"]) * 1.15, 0.05)) & (grid["success_upper"] >= 0.20)].copy()
    else:
        aggressive_pool = grid[(grid["expected_mid_yen"] > 0) & (grid["success_upper"] >= 0.20)].copy()
    aggressive = (aggressive_pool.sort_values(["expected_mid_yen", "spread"], ascending=False).iloc[0]
                  if not aggressive_pool.empty else None)

    def pack(r):
        if r is None:
            return None
        out = {}
        for k in r.index:
            v = r[k]
            if isinstance(v, (np.integer, int)):
                out[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                out[k] = float(v)
            else:
                out[k] = v
        return out
    return {"稳健": pack(conservative), "首选": pack(best), "激进": pack(aggressive)}


def sell_then_buy_grid(
    bundle: OpenForecastBundle,
    shares: int = 100,
    price_step: float = 10.0,
    min_spread_pct: float = 0.025,
) -> pd.DataFrame:
    """For an existing holding, search sell-high → buy-back-low plans with bounds."""
    n = bundle.neighbors
    if n.empty:
        return pd.DataFrame()
    op = float(bundle.open_price); step = max(float(price_step), 1.0)
    w = n["weight"].to_numpy(float); hi = n["next_high_from_open_1d"].to_numpy(float)
    lo = n["next_low_from_open_1d"].to_numpy(float); cl = n["next_close_from_open_1d"].to_numpy(float)
    sell_lo = op; sell_hi = max(bundle.high_quantiles.values()) * 1.02
    buy_lo = min(bundle.low_quantiles.values()) * 0.98; buy_hi = op * (1 - min_spread_pct)
    sells = np.arange(np.ceil(sell_lo / step) * step, np.ceil(sell_hi / step) * step + step / 2, step)
    buys = np.arange(np.floor(buy_lo / step) * step, np.floor(buy_hi / step) * step + step / 2, step)
    rows = []
    for sell in sells:
        sr = sell / op - 1; sell_hit = hi >= sr; ps = _weighted_rate(sell_hit, w)
        if ps < 0.12:
            continue
        for buy in buys:
            if buy >= sell or (sell / buy - 1) < min_spread_pct:
                continue
            br = buy / op - 1; buy_hit = lo <= br; both = sell_hit & buy_hit
            definite = buy_hit if sell <= op + 1e-9 else (sell_hit & (cl <= br))
            pu = _weighted_rate(both, w); pl = _weighted_rate(definite, w)
            if pu < 0.10:
                continue
            spread = sell - buy
            p_blow = _weighted_rate(hi >= (sell * 1.05 / op - 1), w)
            score = pl * spread * shares + 0.25 * ((pl + pu) / 2) * spread * shares - 0.20 * p_blow * spread * shares
            rows.append({
                "sell": float(sell), "buyback": float(buy), "spread": float(spread),
                "p_sell": ps, "p_buyback": _weighted_rate(buy_hit, w),
                "success_lower": pl, "success_upper": pu,
                "sequence_unknown": max(pu - pl, 0.0), "p_blowthrough_5pct": p_blow,
                "score": score, "shares": int(shares),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["score", "success_lower"], ascending=False).reset_index(drop=True)


def condition_open_bundle_intraday(
    bundle: OpenForecastBundle,
    current_price: float,
    observed_high: float,
    observed_low: float,
    min_neighbors: int = 8,
) -> OpenForecastBundle:
    """Condition opening-model analogs on facts already observed intraday.

    This does NOT pretend to have historical 09:15/09:30 snapshots. It only applies
    necessary daily-bar constraints: a compatible historical day must eventually have
    a daily high at least as high as today's observed high/current price and a daily low
    at least as low as today's observed low/current price. If too few analogs survive,
    the original neighbor set is retained.
    """
    if bundle.neighbors.empty or bundle.open_price <= 0:
        return bundle
    op = float(bundle.open_price)
    cur = float(current_price)
    hi_seen = max(float(observed_high), cur)
    lo_seen = min(float(observed_low), cur)
    hi_req = hi_seen / op - 1
    lo_req = lo_seen / op - 1
    n = bundle.neighbors.copy()
    mask = (
        pd.to_numeric(n["next_high_from_open_1d"], errors="coerce") >= hi_req
    ) & (
        pd.to_numeric(n["next_low_from_open_1d"], errors="coerce") <= lo_req
    )
    filtered = n.loc[mask].copy()
    if len(filtered) < int(min_neighbors):
        # Keep the broader distribution rather than letting 2-3 survivors create
        # a fake sense of precision.
        return replace(
            bundle,
            high_quantiles={q: max(v, hi_seen) for q, v in bundle.high_quantiles.items()},
            low_quantiles={q: min(v, lo_seen) for q, v in bundle.low_quantiles.items()},
            mode="intraday_constraints_sparse",
        )
    # Preserve original similarity weights; filtering itself is the conditioning step.
    return replace(
        bundle,
        neighbors=filtered.reset_index(drop=True),
        high_quantiles={q: max(v, hi_seen) for q, v in bundle.high_quantiles.items()},
        low_quantiles={q: min(v, lo_seen) for q, v in bundle.low_quantiles.items()},
        mode="intraday_constrained",
    )
