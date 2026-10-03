from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data_sources import (
    fetch_jpx_ipo_master,
    fetch_yahoo_history_for_ipos,
    load_verified_627a,
    validate_ohlcv,
)
from features import (
    PEER_FEATURES,
    add_peer_context_features,
    apply_live_peer_context,
    build_training_samples,
    compute_state_features,
    state_row_from_inputs,
)
from modeling import (
    empirical_touch_probabilities,
    forecast,
    model_action,
    similar_case_distribution,
)
from ocr_utils import extract_fields, run_ocr
from backtesting import (
    build_touch_calibrator,
    quantile_bias_corrections,
    walk_forward_backtest,
)

DATA = ROOT / "data"


# Built-in verified akippa fallback. This prevents the app from crashing if
# data/verified_627A.csv or data/akippa_metadata.json are missing after a cloud redeploy.
_BUILTIN_AKIPPA_ROWS = [
    ("2026-09-18", 1244, 1315, 1019, 1037, 12872700),
    ("2026-09-24", 995, 1337, 989, 1337, 10680200),
    ("2026-09-25", 1367, 1637, 1350, 1637, 27485000),
    ("2026-09-28", 1837, 2037, 1800, 2037, 8642700),
    ("2026-09-29", 2238, 2507, 2024, 2105, 40989800),
    ("2026-09-30", 2250, 2379, 2016, 2041, 35229300),
    ("2026-10-01", 1911, 1953, 1682, 1745, 10505600),
    ("2026-10-02", 1750, 1967, 1750, 1778, 11148400),
]
_BUILTIN_AKIPPA_META = {
    "code": "627A",
    "company": "akippa株式会社",
    "listing_date": "2026-09-18",
    "market": "スタンダード",
    "offer_price": 570,
    "listing_shares": 6262140,
    "industry": "情報・通信業",
}

def safe_load_verified_627a(data_dir):
    try:
        return load_verified_627a(data_dir)
    except (FileNotFoundError, OSError):
        hist = pd.DataFrame(
            _BUILTIN_AKIPPA_ROWS,
            columns=["Date", "Open", "High", "Low", "Close", "Volume"],
        )
        hist["Date"] = pd.to_datetime(hist["Date"])
        return hist, dict(_BUILTIN_AKIPPA_META)

st.set_page_config(page_title="日本IPO相似案例预测", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load_cache():
    master_path = DATA / "ipo_master.csv"
    samples_path = DATA / "training_samples.parquet"
    quality_path = DATA / "quality_report.csv"
    manifest_path = DATA / "manifest.json"
    master = pd.read_csv(master_path, parse_dates=["listing_date"]) if master_path.exists() else pd.DataFrame()
    samples = pd.read_parquet(samples_path) if samples_path.exists() else pd.DataFrame()
    if not samples.empty:
        # Backward-compatible upgrade: old training parquet files do not contain
        # peer-context columns. They can be derived from the same-date state rows
        # without re-downloading market data.
        samples = add_peer_context_features(samples)
    quality = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    for c in ["asof_date", "listing_date"]:
        if c in samples.columns:
            samples[c] = pd.to_datetime(samples[c])
    return master, samples, quality, manifest


@st.cache_data(show_spinner=False)
def load_daily_cache():
    path = DATA / "ipo_daily.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.upper()
    return df


def live_peer_states(daily: pd.DataFrame, master: pd.DataFrame, asof_date, max_state_day: int = 35) -> pd.DataFrame:
    """Build contemporaneous IPO states for the target trading date.

    Only data on or before `asof_date` are used. We keep IPOs within their first
    `max_state_day` trading days so the peer environment matches the training cohort.
    """
    if daily.empty or master.empty:
        return pd.DataFrame()
    asof = pd.Timestamp(asof_date).normalize()
    d = daily.copy()
    d = d[pd.to_datetime(d["Date"], errors="coerce").dt.normalize() <= asof]
    if d.empty:
        return pd.DataFrame()
    m = master.copy()
    m["code"] = m["code"].astype(str).str.upper()
    m["listing_date"] = pd.to_datetime(m["listing_date"], errors="coerce").dt.normalize()
    # A generous calendar window; trading-day count below is the real filter.
    m = m[(m["listing_date"] <= asof) & (m["listing_date"] >= asof - pd.Timedelta(days=90))]
    m = m.sort_values("listing_date").drop_duplicates("code", keep="last")
    meta = m.set_index("code") if not m.empty else pd.DataFrame()
    rows = []
    for code, g in d[d["code"].isin(set(m["code"]))].groupby("code"):
        g = g.sort_values("Date").reset_index(drop=True)
        if len(g) < 4 or len(g) > max_state_day:
            continue
        # The peer must have a bar on the same target date, otherwise we would be
        # comparing stale states from a different session.
        if pd.Timestamp(g["Date"].iloc[-1]).normalize() != asof:
            continue
        if code not in meta.index:
            continue
        mr = meta.loc[code]
        if isinstance(mr, pd.DataFrame):
            mr = mr.iloc[-1]
        offer = mr.get("offer_price")
        offer = None if pd.isna(offer) or float(offer or 0) <= 0 else float(offer)
        try:
            feats = compute_state_features(g[["Date", "Open", "High", "Low", "Close", "Volume"]], offer_price=offer)
        except Exception:
            continue
        rows.append({
            "code": str(code),
            "asof_date": asof,
            "listing_date": pd.Timestamp(mr["listing_date"]),
            "market": str(mr.get("market", "")),
            "offer_price": offer,
            **feats,
        })
    return pd.DataFrame(rows)


def apply_backtest_calibration(bundle, bt):
    """Bias-correct live quantiles using strict out-of-sample residuals."""
    if bt is None or getattr(bt, "horizon", None) != bundle.horizon:
        return bundle
    corr = quantile_bias_corrections(bt)
    if not corr:
        return bundle
    highs = {}
    lows = {}
    for q in (0.1, 0.5, 0.9):
        hret = bundle.high_quantiles[q] / bundle.current_price - 1
        lret = bundle.low_quantiles[q] / bundle.current_price - 1
        hret += corr.get(f"high_q{int(q*100)}", 0.0)
        lret += corr.get(f"low_q{int(q*100)}", 0.0)
        highs[q] = bundle.current_price * (1 + hret)
        lows[q] = bundle.current_price * (1 + lret)
    hs = np.sort([highs[q] for q in (0.1, 0.5, 0.9)])
    ls = np.sort([lows[q] for q in (0.1, 0.5, 0.9)])
    highs = dict(zip((0.1, 0.5, 0.9), hs))
    lows = dict(zip((0.1, 0.5, 0.9), ls))
    cret = bundle.close_median / bundle.current_price - 1
    cret += corr.get("close_q50", 0.0)
    return replace(bundle, high_quantiles=highs, low_quantiles=lows, close_median=bundle.current_price * (1 + cret))


@st.cache_data(show_spinner=False)
def run_strict_backtest_cached(samples_df: pd.DataFrame, horizon: int, n_neighbors: int):
    return walk_forward_backtest(
        samples_df,
        horizon=int(horizon),
        n_neighbors=int(n_neighbors),
        max_folds=5,
    )


def render_backtest(result):
    st.markdown("**它怎么考试：** 每个测试年份只能使用该年1月1日以前的数据训练；测试IPO整只留出，绝不把同一只IPO的早期状态塞回训练集。")
    folds = result.fold_summary.copy()
    if not folds.empty:
        folds["train_last_date"] = pd.to_datetime(folds["train_last_date"]).dt.strftime("%Y-%m-%d")
        folds["test_first_listing"] = pd.to_datetime(folds["test_first_listing"]).dt.strftime("%Y-%m-%d")
        folds = folds.rename(columns={
            "test_year": "测试年份", "train_rows": "训练样本", "test_rows": "测试样本",
            "test_ipos": "测试IPO数", "train_last_date": "训练数据最晚日期",
            "test_first_listing": "测试IPO最早上市日",
        })
        st.dataframe(folds, use_container_width=True, hide_index=True)

    q = result.quantile_summary.copy()
    label_map = {"high": "未来最高", "low": "未来最低", "close": "未来收盘"}
    q["对象"] = q["target"].map(label_map).fillna(q["target"])
    q["指标"] = q["metric"].replace({
        "P10 actual<=prediction": "P10 实际≤预测",
        "P50 actual<=prediction": "P50 实际≤预测",
        "P90 actual<=prediction": "P90 实际≤预测",
        "P10-P90 interval coverage": "P10-P90 区间覆盖率",
        "Mean P10-P90 width": "P10-P90 平均宽度（收益率）",
        "Median prediction MAE": "中位预测 MAE（收益率）",
    })
    def fmt_expected(x):
        return "—" if pd.isna(x) else f"{x:.1%}"
    def fmt_actual(row):
        if "宽度" in row["指标"] or "MAE" in row["指标"]:
            return f"{row['actual']:.2%}"
        return f"{row['actual']:.1%}"
    q["理论值"] = q["expected"].map(fmt_expected)
    q["实测值"] = q.apply(fmt_actual, axis=1)
    q["偏差"] = q["error"].map(lambda x: "—" if pd.isna(x) else f"{x:+.1%}")
    st.subheader("分位数校准")
    st.dataframe(q[["对象", "指标", "理论值", "实测值", "偏差"]], use_container_width=True, hide_index=True)

    cal = result.touch_calibration.copy()
    st.subheader("触及概率校准")
    if cal.empty:
        st.write("没有足够的触及概率回测结果。")
    else:
        cal["预测概率均值"] = cal["predicted_mean"].map(lambda x: f"{x:.1%}")
        cal["实际发生率"] = cal["actual_rate"].map(lambda x: f"{x:.1%}")
        cal["偏差"] = cal["gap"].map(lambda x: f"{x:+.1%}")
        cal = cal.rename(columns={"probability_bin": "模型概率档", "count": "事件数"})
        st.dataframe(cal[["模型概率档", "事件数", "预测概率均值", "实际发生率", "偏差"]], use_container_width=True, hide_index=True)
        st.caption("例如模型经常报50%-60%的事件，如果实际发生率也接近50%-60%，说明‘55%’这个数字比较可信。")

    # A compact verdict based only on calibration errors; no claim of profitability.
    core = result.quantile_summary[result.quantile_summary["expected"].notna()].copy()
    mean_abs_gap = float(core["error"].abs().mean()) if not core.empty else np.nan
    if pd.notna(mean_abs_gap):
        if mean_abs_gap <= 0.04:
            st.success(f"分位数平均校准偏差约 {mean_abs_gap:.1%}：目前看校准较好。")
        elif mean_abs_gap <= 0.08:
            st.info(f"分位数平均校准偏差约 {mean_abs_gap:.1%}：可用，但仍有明显误差。")
        else:
            st.warning(f"分位数平均校准偏差约 {mean_abs_gap:.1%}：偏差较大，当前概率数字不宜过度相信。")


def build_public_cache(start_year: int = 2018):
    status = st.status("正在建立历史IPO数据库", expanded=True)
    status.write("读取 JPX 新股上市档案…")
    master = fetch_jpx_ipo_master(start_year=start_year)
    DATA.mkdir(exist_ok=True)
    master.to_csv(DATA / "ipo_master.csv", index=False)
    status.write(f"JPX母表：{len(master)} 条上市记录")

    status.write("下载上市后日线 OHLCV…")
    prices, report = fetch_yahoo_history_for_ipos(master)
    prices.to_parquet(DATA / "ipo_daily.parquet", index=False)
    status.write(f"取得 {prices['code'].nunique()} 只股票，共 {len(prices):,} 根日线")

    status.write("进行OHLC和上市日期质量检查…")
    quality = validate_ohlcv(prices)
    quality.to_csv(DATA / "quality_report.csv", index=False)
    good = set(quality.loc[quality["usable"], "code"].astype(str))
    p2 = prices[prices["code"].astype(str).isin(good)]
    m2 = master[master["code"].astype(str).isin(good)]
    status.write(f"通过严格检查：{len(good)} 只")

    status.write("生成无未来泄漏的训练样本…")
    samples = build_training_samples(p2, m2, horizons=(1, 3, 5), min_history_days=4, max_state_day=35)
    samples.to_parquet(DATA / "training_samples.parquet", index=False)
    manifest = {
        "built_at": pd.Timestamp.utcnow().isoformat(),
        "metadata_source": "JPX new-listing archive",
        "price_source": report.provider,
        "listings": int(len(master)),
        "downloaded_symbols": int(report.succeeded),
        "usable_symbols": int(len(good)),
        "training_samples": int(len(samples)),
        "failed_symbols": report.failed,
    }
    with open(DATA / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    load_cache.clear()
    load_daily_cache.clear()
    status.update(label="历史库建立完成", state="complete", expanded=False)
    return manifest


def fetch_target_history(code: str, listing_date: pd.Timestamp, lookahead_days=180) -> pd.DataFrame:
    if code.upper() == "627A":
        h, _ = safe_load_verified_627a(DATA)
        return h[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    try:
        import yfinance as yf
        end = min(pd.Timestamp.today().normalize() + pd.Timedelta(days=1), listing_date + pd.Timedelta(days=lookahead_days))
        t = yf.Ticker(f"{code}.T")
        h = t.history(start=listing_date.date().isoformat(), end=end.date().isoformat(), auto_adjust=False)
        if h.empty:
            return pd.DataFrame()
        h = h.rename_axis("Date").reset_index()
        h["Date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None)
        return h[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
    except Exception:
        return pd.DataFrame()


def merge_snapshot(history: pd.DataFrame, snapshot: dict) -> pd.DataFrame:
    h = history.copy()
    d = pd.Timestamp(snapshot["Date"]).normalize()
    row = {
        "Date": d,
        "Open": float(snapshot["Open"]),
        "High": float(snapshot["High"]),
        "Low": float(snapshot["Low"]),
        "Close": float(snapshot["Close"]),
        "Volume": float(snapshot["Volume"]),
    }
    if not h.empty:
        h["Date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None).dt.normalize()
        h = h[h["Date"] != d]
    return pd.concat([h, pd.DataFrame([row])], ignore_index=True).sort_values("Date").reset_index(drop=True)


def price_chart(history: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=history["Date"],
            open=history["Open"], high=history["High"], low=history["Low"], close=history["Close"],
            name="OHLC",
        )
    )
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
    return fig


def forecast_band_chart(bundle):
    labels = ["P10", "P50", "P90"]
    low = [bundle.low_quantiles[q] for q in (0.1, 0.5, 0.9)]
    high = [bundle.high_quantiles[q] for q in (0.1, 0.5, 0.9)]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="最低价分位", x=labels, y=low))
    fig.add_trace(go.Bar(name="最高价分位", x=labels, y=high))
    fig.add_hline(y=bundle.current_price, line_dash="dash", annotation_text="当前价")
    fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=30, b=10))
    return fig


st.title("📈 日本IPO相似案例 + 概率预测")
st.caption("不是“猜一个神奇最高点”：用历史相似IPO + 分位数机器学习，输出未来1/3/5个交易日的价格区间与关键价位触及概率。")

master, samples, quality, manifest = load_cache()
daily_prices = load_daily_cache()

with st.sidebar:
    st.header("数据状态")
    if samples.empty:
        st.warning("历史训练库尚未建立。")
        start_year = st.number_input("历史起始年份", min_value=2015, max_value=date.today().year, value=2018, step=1)
        if st.button("建立/更新公开历史库", type="primary", use_container_width=True):
            try:
                build_public_cache(int(start_year))
                st.rerun()
            except Exception as e:
                st.error(f"历史库建立失败：{e}")
    else:
        st.success(f"训练样本：{len(samples):,}")
        if manifest:
            st.write(f"可用IPO：{manifest.get('usable_symbols', '—')}")
            st.write(f"数据构建：{str(manifest.get('built_at', manifest.get('built_at_utc', '—')))[:19]}")
        if st.button("重新建立历史库", use_container_width=True):
            try:
                build_public_cache(2018)
                st.rerun()
            except Exception as e:
                st.error(f"更新失败：{e}")

    st.divider()
    st.caption("数据原则")
    st.write("• 上市日期/发行价：JPX官方")
    st.write("• 历史OHLCV：公开Yahoo Finance，逐只做OHLC一致性检查")
    st.write("• 627A内置8日数据已用株探/みんかぶ交叉核验")
    st.write("• 日线无法知道当天最高/最低发生顺序")
    st.write("• 模型加入同一交易日其他新股的相对强弱/量价环境")

if samples.empty:
    st.info("先在左侧点击“建立/更新公开历史库”。部署到有互联网的 Streamlit Cloud 后可直接建立。项目也附带命令行构建脚本。")
    st.stop()

left, right = st.columns([1.05, 0.95])
with left:
    st.subheader("1. 输入今天的状态")
    upload = st.file_uploader("可选：上传券商/行情截图（OCR只作为预填，必须人工确认）", type=["png", "jpg", "jpeg", "webp"])
    parsed = {}
    ocr_text = ""
    if upload is not None:
        img = Image.open(upload)
        st.image(img, caption="上传截图", use_container_width=True)
        ocr_text = run_ocr(img)
        parsed = extract_fields(ocr_text)
        with st.expander("查看OCR原文/识别结果"):
            st.code(ocr_text)
            st.json(parsed)

    code_default = str(parsed.get("code", "627A"))
    code = st.text_input("股票代码", value=code_default).strip().upper()

    meta_row = None
    if not master.empty and code in set(master["code"].astype(str)):
        meta_row = master[master["code"].astype(str).eq(code)].sort_values("listing_date").iloc[-1]
    if code == "627A":
        _, ak_meta = safe_load_verified_627a(DATA)
        listing_default = pd.Timestamp(ak_meta["listing_date"]).date()
        offer_default = float(ak_meta["offer_price"])
        market_default = ak_meta["market"]
    elif meta_row is not None:
        listing_default = pd.Timestamp(meta_row["listing_date"]).date()
        offer_default = float(meta_row["offer_price"]) if pd.notna(meta_row.get("offer_price")) else 0.0
        market_default = str(meta_row.get("market", ""))
    else:
        listing_default = date.today()
        offer_default = 0.0
        market_default = ""

    c1, c2, c3 = st.columns(3)
    listing_date = c1.date_input("上市日", value=listing_default)
    offer_price = c2.number_input("发行/公募价（没有就0）", min_value=0.0, value=float(offer_default), step=1.0)
    market = c3.text_input("市场", value=market_default)

    hist = fetch_target_history(code, pd.Timestamp(listing_date))
    if hist.empty and code != "627A":
        st.warning("未自动取得该股上市后日线。你仍可用下面的截图/手工状态，但至少要有2天历史才能生成状态特征。")

    if not hist.empty:
        last = hist.iloc[-1]
        default_date = pd.Timestamp(last["Date"]).date()
        default_open = float(last["Open"])
        default_high = float(last["High"])
        default_low = float(last["Low"])
        default_close = float(last["Close"])
        default_volume = float(last["Volume"])
    else:
        default_date = date.today()
        default_open = default_high = default_low = default_close = 0.0
        default_volume = 0.0

    snap_date = st.date_input("截图/行情日期", value=default_date)
    a, b, c = st.columns(3)
    open_v = a.number_input("开盘", min_value=0.0, value=float(parsed.get("open", default_open)), step=1.0)
    high_v = b.number_input("截至目前最高", min_value=0.0, value=float(parsed.get("high", default_high)), step=1.0)
    low_v = c.number_input("截至目前最低", min_value=0.0, value=float(parsed.get("low", default_low)), step=1.0)
    d, e = st.columns(2)
    close_v = d.number_input("当前价/收盘价", min_value=0.0, value=float(parsed.get("current", default_close)), step=1.0)
    volume_v = e.number_input("截至目前成交量", min_value=0.0, value=float(parsed.get("volume", default_volume)), step=1000.0)

    intraday = st.checkbox("这是盘中截图（不是收盘数据）", value=(snap_date == date.today()))
    if intraday:
        st.warning("盘中模式：当前K线和成交量尚未走完，模型置信度会打折；历史训练样本仍是完整日线。")

    if all(x > 0 for x in [open_v, high_v, low_v, close_v]) and high_v >= max(open_v, close_v) and low_v <= min(open_v, close_v):
        hist2 = merge_snapshot(hist, {"Date": snap_date, "Open": open_v, "High": high_v, "Low": low_v, "Close": close_v, "Volume": volume_v})
    else:
        hist2 = hist.copy()
        st.error("OHLC关系不成立：最高必须≥开盘/当前价，最低必须≤开盘/当前价。")

with right:
    st.subheader("2. 你的持仓/关键价位")
    h1, h2 = st.columns(2)
    holding = h1.number_input("持仓股数", min_value=0, value=300, step=100)
    avg_cost = h2.number_input("平均成本", min_value=0.0, value=0.0, step=1.0)
    t1, t2 = st.columns(2)
    sell_level = t1.number_input("想测试的卖价", min_value=0.0, value=1950.0, step=10.0)
    buyback_level = t2.number_input("想测试的回补价", min_value=0.0, value=1750.0, step=10.0)
    shares_for_t = st.number_input("做T股数", min_value=0, max_value=max(int(holding), 100), value=min(100, int(holding)) if holding else 0, step=100)
    horizon = st.radio("预测窗口", [1, 3, 5], horizontal=True, format_func=lambda x: f"未来{x}个交易日")
    n_neighbors = st.slider("参考最相似案例数", 15, 80, 40, 5)
    extra_levels = st.text_input("其他关键价位（逗号分隔）", value="1650,1700,1900,2000,2100,2300")
    compare_codes_text = st.text_input("额外对比股票代码（可选，逗号分隔）", value="", help="只用于结果页横向检查；真正进入模型的是自动计算的同期IPO整体环境，避免人为挑选样本。")

    run = st.button("开始预测", type="primary", use_container_width=True)

st.divider()
with st.expander("🧪 模型体检：严格样本外回测 / 分位数校准", expanded=False):
    st.write("这不会改历史数据，也不会改627A当前预测。它只是拿历史IPO做‘闭卷考试’，检查P10/P50/P90和触及概率是否名副其实。")
    st.caption(f"当前将检查：未来 {horizon} 个交易日；相似案例数 {n_neighbors}。首次运行会训练多个历史年度折叠。")
    if st.button("运行严格回测", key="run_strict_backtest", use_container_width=True):
        try:
            with st.spinner("正在做严格样本外回测…"):
                bt = run_strict_backtest_cached(samples, int(horizon), int(n_neighbors))
            st.session_state["last_backtest"] = bt
        except Exception as e:
            st.error(f"严格回测失败：{e}")
    bt = st.session_state.get("last_backtest")
    if bt is not None and getattr(bt, "horizon", None) == int(horizon):
        render_backtest(bt)
        use_calibration = st.checkbox(
            "用这次样本外回测结果校准最终区间与触及概率",
            value=True,
            help="用历史闭卷考试暴露出的系统偏差修正当前预测；不是用周一结果反改周一预测。",
        )
    else:
        use_calibration = False
        if bt is not None:
            st.info("你改了预测窗口，请重新运行一次严格回测。")
        st.caption("回测尚未运行时，最终结果使用原始模型概率；跑完体检后可开启自动校准。")

if run:
    if len(hist2) < 4:
        st.error("目标股至少需要4根历史日线。")
        st.stop()
    if close_v <= 0:
        st.error("当前价必须大于0。")
        st.stop()

    try:
        state = state_row_from_inputs(
            hist2,
            offer_price=offer_price if offer_price > 0 else None,
            code=code,
            listing_date=pd.Timestamp(listing_date),
            market=market,
        )
        peer_pool = live_peer_states(daily_prices, master, snap_date, max_state_day=35)
        state = apply_live_peer_context(state, peer_pool)
        raw_bundle = forecast(samples, state, current_price=float(close_v), horizon=int(horizon), n_neighbors=int(n_neighbors))

        calibration_bt = st.session_state.get("last_backtest")
        calibration_ok = bool(
            use_calibration
            and calibration_bt is not None
            and getattr(calibration_bt, "horizon", None) == int(horizon)
        )
        probability_calibrator = build_touch_calibrator(calibration_bt) if calibration_ok else None
        bundle = apply_backtest_calibration(raw_bundle, calibration_bt) if calibration_ok else raw_bundle
    except Exception as e:
        st.error(f"模型无法运行：{e}")
        st.stop()

    st.divider()
    st.header(f"{code} 模型结果 — 未来 {horizon} 个交易日")
    if calibration_ok:
        st.success("当前结果已使用严格样本外回测做历史偏差校准。")
    else:
        st.caption("当前结果为原始模型输出；运行上方严格回测后，可以启用历史偏差校准。")

    peer_count = max(len(peer_pool), 0) if 'peer_pool' in locals() else 0
    if peer_count >= 1 and pd.notna(state.iloc[0].get("peer_median_ret_1d", np.nan)):
        target_rel = state.iloc[0].get("rel_ret_1d_peer", np.nan)
        peer_med = state.iloc[0].get("peer_median_ret_1d", np.nan)
        peer_up = state.iloc[0].get("peer_up_share_1d", np.nan)
        st.caption(
            f"同期IPO环境已进入模型：约 {peer_count + 1} 个当日新股状态；"
            f"同期1日收益中位数 {peer_med:+.1%}，目标股相对同期 {target_rel:+.1%}，"
            f"同期上涨占比 {peer_up:.0%}。"
        )
    else:
        st.caption("当前未取得足够的同日IPO横向状态；横向特征将按缺失值处理，基础日线/量价模型仍可运行。")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("最低价 P50", f"{bundle.low_quantiles[0.5]:.0f}")
    q2.metric("最高价 P50", f"{bundle.high_quantiles[0.5]:.0f}")
    q3.metric("收盘/末日 P50", f"{bundle.close_median:.0f}")
    q4.metric("训练样本", f"{bundle.sample_count:,}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(price_chart(hist2), use_container_width=True)
    with c2:
        st.plotly_chart(forecast_band_chart(bundle), use_container_width=True)

    st.subheader("概率区间")
    range_df = pd.DataFrame({
        "分位": ["P10", "P50", "P90"],
        "未来最低价": [bundle.low_quantiles[q] for q in (0.1, 0.5, 0.9)],
        "未来最高价": [bundle.high_quantiles[q] for q in (0.1, 0.5, 0.9)],
    }).round(0)
    st.dataframe(range_df, use_container_width=True, hide_index=True)

    levels = [sell_level, buyback_level]
    for token in extra_levels.replace("，", ",").split(","):
        try:
            x = float(token.strip())
            if x > 0:
                levels.append(x)
        except Exception:
            pass
    levels = sorted(set(float(x) for x in levels if x and x > 0))
    probs = empirical_touch_probabilities(bundle.neighbors, float(close_v), int(horizon), levels)
    probs["价位"] = probs["level"].round(0).astype(int)
    if probability_calibrator is not None:
        probs["raw_probability"] = probs["probability"]
        probs["probability"] = probs["probability"].map(probability_calibrator)
        probs["校准后概率"] = probs["probability"].map(lambda x: f"{x:.1%}")
        probs["原始概率"] = probs["raw_probability"].map(lambda x: f"{x:.1%}")
        st.subheader("关键价位触及概率（相似历史案例 + 样本外校准）")
        st.dataframe(
            probs[["价位", "direction", "校准后概率", "原始概率"]].rename(columns={"direction": "方向"}),
            use_container_width=True, hide_index=True
        )
    else:
        probs["概率"] = probs["probability"].map(lambda x: f"{x:.1%}")
        st.subheader("关键价位触及概率（相似历史案例加权）")
        st.dataframe(probs[["价位", "direction", "概率"]].rename(columns={"direction": "方向"}), use_container_width=True, hide_index=True)

    action = model_action(
        bundle, sell_level, buyback_level, int(shares_for_t),
        probability_calibrator=probability_calibrator,
    )
    st.subheader("模型动作")
    st.info(f"**{action['action']}**  ｜  置信度：{action['confidence']}")
    for reason in action["reasons"]:
        st.write("• " + reason)
    st.caption(action["sequence_warning"])

    if avg_cost > 0 and holding > 0:
        pnl = (close_v - avg_cost) * holding
        st.write(f"当前按输入价计算的持仓浮动：**{pnl:,.0f} 円**（未计费用/税）")
        if sell_level > 0 and buyback_level > 0 and shares_for_t > 0:
            spread = (sell_level - buyback_level) * shares_for_t
            st.write(f"若实际完成 {sell_level:.0f} → {buyback_level:.0f} 的完整价差，理论价差贡献约 **{spread:,.0f} 円**（未计费用/税，且日线无法证明先后顺序）。")

    tab1, tab2, tab3, tab4 = st.tabs(["最相似历史案例", "模型质量", "横向量价对比", "数据来源/限制"])
    with tab1:
        n = bundle.neighbors.copy()
        if not master.empty:
            names = master[["code", "company", "market", "listing_date"]].drop_duplicates("code", keep="last")
            n = n.merge(names, on="code", how="left", suffixes=("", "_master"))
        cols = [c for c in ["code", "company", "listing_date", "asof_date", "distance", "weight", f"future_max_ret_{horizon}d", f"future_min_ret_{horizon}d", f"future_close_ret_{horizon}d"] if c in n.columns]
        show = n[cols].head(30).copy()
        for c in show.columns:
            if c.startswith("future_"):
                show[c] = show[c].map(lambda x: f"{x:.1%}")
        if "distance" in show:
            show["distance"] = show["distance"].round(2)
        if "weight" in show:
            show["weight"] = show["weight"].round(3)
        st.dataframe(show, use_container_width=True, hide_index=True)
        dist = similar_case_distribution(bundle.neighbors, float(close_v), int(horizon))
        if dist:
            st.caption(
                f"相似案例非参数分布：最低P50≈{dist['low_q50']:.0f}，最高P50≈{dist['high_q50']:.0f}。"
                "它与机器学习分位数是两套独立估计，可用来检查模型是否离谱。"
            )

    with tab2:
        if bundle.metrics:
            m = pd.DataFrame([
                {"指标": "未来最高收益率中位预测 MAE", "值": bundle.metrics.get("mae_high_ret")},
                {"指标": "未来最低收益率中位预测 MAE", "值": bundle.metrics.get("mae_low_ret")},
                {"指标": "未来收盘收益率中位预测 MAE", "值": bundle.metrics.get("mae_close_ret")},
            ]).dropna()
            m["值"] = m["值"].map(lambda x: f"{x:.2%}")
            st.dataframe(m, use_container_width=True, hide_index=True)
        else:
            st.write("样本量不足以生成稳定的时间顺序留出集指标。")
        st.caption("验证按时间顺序切分，不随机打乱；这样避免把未来IPO环境泄漏给过去。")
        if intraday:
            st.warning("你当前使用盘中未完成K线；这里的历史验证是收盘日线，因此实盘置信度应低于表内回测。")

    with tab3:
        st.write("**自动进入模型的横向信息：** 同一交易日、上市初期IPO的涨跌、回撤、振幅和量能相对强弱。不是简单比较绝对成交量。")
        peer_cols = [
            ("peer_median_ret_1d", "同期1日收益中位数"),
            ("rel_ret_1d_peer", "目标相对同期1日强弱"),
            ("peer_up_share_1d", "同期上涨占比"),
            ("peer_median_drawdown", "同期距高点回撤中位数"),
            ("rel_drawdown_peer", "目标相对同期回撤强弱"),
            ("peer_median_volume5", "同期量/5日均量中位数"),
            ("rel_volume5_peer", "目标相对同期量能"),
        ]
        diag = []
        for col, label in peer_cols:
            val = state.iloc[0].get(col, np.nan)
            if pd.notna(val):
                diag.append({"横向特征": label, "数值": float(val)})
        if diag:
            dd = pd.DataFrame(diag)
            def _fmt_peer(row):
                if "量/5日均量" in row["横向特征"] or "相对同期量能" in row["横向特征"]:
                    return f"{row['数值']:.2f}"
                return f"{row['数值']:.1%}"
            dd["值"] = dd.apply(_fmt_peer, axis=1)
            st.dataframe(dd[["横向特征", "值"]], use_container_width=True, hide_index=True)
        else:
            st.info("这次没有足够的同期IPO数据，模型已自动回退到基础特征。")

        compare_codes = []
        for tok in compare_codes_text.replace("，", ",").split(","):
            ccode = tok.strip().upper()
            if ccode and ccode != code and ccode not in compare_codes:
                compare_codes.append(ccode)
        if compare_codes:
            rows = []
            target_feats = compute_state_features(hist2, offer_price=offer_price if offer_price > 0 else None)
            rows.append({"代码": code, "1日涨跌": target_feats.get("ret_1d"), "3日涨跌": target_feats.get("ret_3d"), "距高点回撤": target_feats.get("drawdown_from_peak"), "量/5日均量": target_feats.get("volume_to_mean5"), "日内振幅": target_feats.get("range_pct")})
            for ccode in compare_codes:
                mr = master[master["code"].astype(str).str.upper().eq(ccode)] if not master.empty else pd.DataFrame()
                if mr.empty:
                    continue
                mr = mr.sort_values("listing_date").iloc[-1]
                ch = fetch_target_history(ccode, pd.Timestamp(mr["listing_date"]))
                if ch.empty:
                    continue
                ch = ch[pd.to_datetime(ch["Date"]).dt.tz_localize(None).dt.normalize() <= pd.Timestamp(snap_date).normalize()]
                if len(ch) < 2:
                    continue
                off = mr.get("offer_price")
                off = None if pd.isna(off) or float(off or 0) <= 0 else float(off)
                try:
                    cf = compute_state_features(ch, offer_price=off)
                except Exception:
                    continue
                rows.append({"代码": ccode, "1日涨跌": cf.get("ret_1d"), "3日涨跌": cf.get("ret_3d"), "距高点回撤": cf.get("drawdown_from_peak"), "量/5日均量": cf.get("volume_to_mean5"), "日内振幅": cf.get("range_pct")})
            if len(rows) > 1:
                comp = pd.DataFrame(rows)
                for c in ["1日涨跌", "3日涨跌", "距高点回撤", "日内振幅"]:
                    comp[c] = comp[c].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}")
                comp["量/5日均量"] = comp["量/5日均量"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}x")
                st.markdown("**你指定的股票横向检查（不直接改模型权重）：**")
                st.dataframe(comp, use_container_width=True, hide_index=True)
            else:
                st.caption("指定的对比股票未取得足够日线数据。")

    with tab4:
        st.write("**上市元数据：** JPX 新规上场公司档案。")
        st.write("**历史价格：** Yahoo Finance（通过 yfinance）公开日线；每只股票需通过OHLC包络、正价格、成交量、上市日期滞后等检查才进入训练。")
        st.write("**627A：** 工具内置截至2026-10-02的8根已交叉核验日线。")
        st.write("**关键限制：** 概率是历史条件统计，不是确定预言；日线无法恢复日内高低先后；停牌、涨跌停、重大公告会让历史相似性突然失效。")
        if manifest:
            st.json(manifest)
        if not quality.empty:
            with st.expander("查看历史数据质量报告"):
                st.dataframe(quality, use_container_width=True, hide_index=True)

st.divider()
st.caption("研究/决策辅助工具，不自动下单。模型输出应结合最新公告、市场状态和你能承受的仓位风险。")
