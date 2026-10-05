from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import FEATURES

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


def _numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURES].copy()
    return x.replace([np.inf, -np.inf], np.nan)


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
