from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_sources import fetch_jpx_ipo_master, fetch_yahoo_history_for_ipos, validate_ohlcv
from src.features import build_training_samples


def main():
    p = argparse.ArgumentParser(description="Build Japanese IPO historical model cache")
    p.add_argument("--start-year", type=int, default=2018)
    p.add_argument("--end-year", type=int, default=None)
    p.add_argument("--max-state-day", type=int, default=35)
    p.add_argument("--out", type=Path, default=ROOT / "data")
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] JPX IPO master")
    master = fetch_jpx_ipo_master(args.start_year, args.end_year)
    master.to_csv(out / "ipo_master.csv", index=False)
    print(f"  {len(master)} listings")

    print("[2/4] Post-IPO daily prices")
    prices, report = fetch_yahoo_history_for_ipos(master)
    prices.to_parquet(out / "ipo_daily.parquet", index=False)
    print(f"  {prices['code'].nunique()} symbols, {len(prices)} daily bars")

    print("[3/4] Quality checks")
    quality = validate_ohlcv(prices)
    quality.to_csv(out / "quality_report.csv", index=False)
    usable_codes = set(quality.loc[quality["usable"], "code"].astype(str))
    usable_prices = prices[prices["code"].astype(str).isin(usable_codes)].copy()
    usable_master = master[master["code"].astype(str).isin(usable_codes)].copy()
    print(f"  usable: {len(usable_codes)} symbols")

    print("[4/4] Leakage-safe training samples")
    samples = build_training_samples(
        usable_prices,
        usable_master,
        horizons=(1, 3, 5),
        min_history_days=4,
        max_state_day=args.max_state_day,
    )
    samples.to_parquet(out / "training_samples.parquet", index=False)
    print(f"  {len(samples)} state/target samples")

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "metadata_source": "JPX new listing archive",
        "price_source": report.provider,
        "master_rows": int(len(master)),
        "requested_price_symbols": int(report.requested),
        "price_symbols_succeeded": int(report.succeeded),
        "price_symbols_failed": report.failed,
        "usable_symbols": int(len(usable_codes)),
        "daily_bars": int(len(usable_prices)),
        "training_samples": int(len(samples)),
        "notes": [
            "JPX metadata is the authority for listing date / market / offer price where available.",
            "Yahoo Finance daily OHLCV is used as the public no-auth price source.",
            "Each OHLC row is sanity-checked; symbols failing checks are excluded.",
            "Training features only use information available as of each historical date.",
            "Daily bars cannot reveal the intraday order of the day's high and low.",
        ],
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
