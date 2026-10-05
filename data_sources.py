from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from io import StringIO
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

JPX_NEW = "https://www.jpx.co.jp/listing/stocks/new/"
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124 Safari/537.36"
    )
}


@dataclass
class FetchReport:
    provider: str
    requested: int
    succeeded: int
    failed: list[str]


def normalize_code(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().upper()
    m = re.search(r"(?<![0-9A-Z])([0-9]{4}|[0-9]{3}[A-Z])(?![0-9A-Z])", s)
    return m.group(1) if m else None


def parse_number(value: object) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).replace(",", "").strip()
    if not s or s in {"-", "—", "－", "nan"}:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def parse_listing_date(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    s = str(value)
    m = re.search(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})", s)
    if not m:
        return None
    try:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def flatten_columns(columns) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        out = []
        for tup in columns:
            parts = []
            for x in tup:
                x = str(x).strip()
                if x and x.lower() != "nan" and x not in parts:
                    parts.append(x)
            out.append(" | ".join(parts))
        return out
    return [str(c).strip() for c in columns]


def _pick_col(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = list(columns)
    for needle in candidates:
        for c in cols:
            if needle in c:
                return c
    return None


def discover_jpx_archive_urls(timeout: int = 20) -> list[str]:
    """Discover current + archived JPX new-listing pages.

    JPX archive URLs historically use 00-archives-NN.html. We first parse the
    live page and then add a conservative fallback range so the builder keeps
    working when the archive selector is rendered differently.
    """
    urls = {JPX_NEW, urljoin(JPX_NEW, "index.html")}
    try:
        r = requests.get(JPX_NEW, headers=UA, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup.find_all(["a", "option"]):
            href = tag.get("href") or tag.get("value")
            if href and "00-archives-" in href:
                urls.add(urljoin(JPX_NEW, href))
    except Exception:
        pass

    # Fallback: enough to cover roughly the last decade. Non-existing pages are
    # harmless because callers skip HTTP/parse failures.
    for i in range(1, 13):
        urls.add(urljoin(JPX_NEW, f"00-archives-{i:02d}.html"))
    return sorted(urls)


def _extract_ipo_table(url: str, timeout: int = 25) -> pd.DataFrame:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text), flavor="bs4")
    best = None
    best_score = -1
    for t in tables:
        t = t.copy()
        t.columns = flatten_columns(t.columns)
        cols = list(t.columns)
        score = sum(
            int(any(k in c for c in cols))
            for k in ["上場日", "会社名", "コード", "市場区分", "公募・売出価格"]
        )
        if score > best_score:
            best_score = score
            best = t
    if best is None or best_score < 2:
        raise ValueError(f"No JPX IPO table recognized at {url}")
    return best



def _extract_code_from_cells(cells: list[str], listing_date: pd.Timestamp | None = None) -> tuple[str | None, int | None]:
    """Extract a JPX security code from individual table cells.

    Never scan the whole row at once: listing dates such as 2022-01-04 would
    otherwise be misread as the security code ``2022``. Prefer cells whose
    entire content is a 4-digit / 3-digit+letter code, then fall back to a
    code-like token in a non-date cell.
    """
    listing_year = str(listing_date.year) if listing_date is not None else None

    # First pass: exact code cell.
    for i, cell in enumerate(cells):
        raw = str(cell).strip().upper()
        if not raw or parse_listing_date(raw) is not None:
            continue
        compact = re.sub(r"\s+", "", raw)
        if re.fullmatch(r"(?:[0-9]{4}|[0-9]{3}[A-Z])", compact):
            # In fallback parsing, a bare year is far more likely to be part of
            # the listing date/header than a real security code.
            if listing_year is not None and compact == listing_year:
                continue
            return compact, i

    # Second pass: code token embedded in a non-date cell.
    for i, cell in enumerate(cells):
        raw = str(cell).strip()
        if not raw or parse_listing_date(raw) is not None:
            continue
        code = normalize_code(raw)
        if code is None:
            continue
        if listing_year is not None and code == listing_year:
            continue
        return code, i
    return None, None


def _fallback_parse_jpx_rows(url: str, start_year: int, end_year: int, timeout: int = 25) -> list[dict]:
    """Best-effort HTML row parser used if pandas cannot understand JPX headers.

    It prioritizes date/code/company/market. Offer price is intentionally left
    blank unless it can be identified unambiguously; missing metadata is safer
    than a wrongly assigned number.
    """
    out = []
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if not cells:
            continue
        full = " | ".join(cells)
        dt = parse_listing_date(full)
        if dt is None or not (start_year <= dt.year <= end_year):
            continue
        if dt.date() > date.today():
            continue
        code, code_idx = _extract_code_from_cells(cells, dt)
        if code is None:
            continue
        company = cells[code_idx-1] if code_idx is not None and code_idx > 0 else ""
        market = next((m for m in ["グロース", "スタンダード", "プライム", "Growth", "Standard", "Prime"] if m in full), "")
        out.append({
            "listing_date": dt.normalize(),
            "company": company,
            "code": code,
            "market": market,
            "offer_price": None,
            "public_offering_k": None,
            "secondary_distribution_k": None,
            "is_technical": "*" in company or "テクニカル" in full,
            "metadata_source": url + "#fallback",
        })
    return out

def fetch_jpx_ipo_master(start_year: int = 2018, end_year: int | None = None) -> pd.DataFrame:
    end_year = end_year or date.today().year
    rows: list[dict] = []
    seen_pages: set[str] = set()

    for url in discover_jpx_archive_urls():
        if url in seen_pages:
            continue
        seen_pages.add(url)
        before = len(rows)
        try:
            t = _extract_ipo_table(url)
        except Exception:
            t = None

        if t is None:
            try:
                rows.extend(_fallback_parse_jpx_rows(url, start_year, end_year))
            except Exception:
                pass
            continue

        date_col = _pick_col(t.columns, ["上場日"])
        name_col = _pick_col(t.columns, ["会社名", "銘柄名"])
        code_col = _pick_col(t.columns, ["コード"])
        market_col = _pick_col(t.columns, ["市場区分"])
        offer_col = _pick_col(t.columns, ["公募・売出価格", "売出価格"])
        public_col = _pick_col(t.columns, ["公募（千株）", "公募"])
        secondary_col = _pick_col(t.columns, ["売出（千株）", "売出"])

        if not date_col or not code_col:
            continue

        for _, row in t.iterrows():
            listing_date = parse_listing_date(row.get(date_col))
            code = normalize_code(row.get(code_col))
            if listing_date is None or code is None:
                continue
            if not (start_year <= listing_date.year <= end_year):
                continue
            if listing_date.date() > date.today():
                continue

            company = str(row.get(name_col, "")).strip() if name_col else ""
            market = str(row.get(market_col, "")).strip() if market_col else ""
            offer = parse_number(row.get(offer_col)) if offer_col else None
            public = parse_number(row.get(public_col)) if public_col else None
            secondary = parse_number(row.get(secondary_col)) if secondary_col else None
            is_technical = "*" in company or "テクニカル" in company

            rows.append(
                {
                    "listing_date": listing_date.normalize(),
                    "company": company,
                    "code": code,
                    "market": market,
                    "offer_price": offer,
                    "public_offering_k": public,
                    "secondary_distribution_k": secondary,
                    "is_technical": bool(is_technical),
                    "metadata_source": url,
                }
            )

        if len(rows) == before:
            try:
                rows.extend(_fallback_parse_jpx_rows(url, start_year, end_year))
            except Exception:
                pass

    if not rows:
        raise RuntimeError("JPX IPO master could not be built. Check internet access / JPX page structure.")

    df = pd.DataFrame(rows)
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
    df = df.dropna(subset=["listing_date", "code"]).copy()
    df["code"] = df["code"].astype(str).str.strip().str.upper()

    # Guard against a historical fallback-parser bug that could interpret the
    # year in a date (e.g. 2022-01-04) as security code 2022. Only apply this
    # filter to fallback-derived rows so a genuine code equal to a year is not
    # discarded from a correctly parsed JPX table.
    fallback = df["metadata_source"].astype(str).str.endswith("#fallback")
    same_as_year = df["code"].eq(df["listing_date"].dt.year.astype(str))
    df = df[~(fallback & same_as_year)].copy()

    df = df.sort_values(["listing_date", "code"]).drop_duplicates(["listing_date", "code"], keep="last")
    # Technical/transfer listings are poor comparables for an ordinary IPO. Keep
    # them in the master for transparency, but mark them for downstream filtering.
    return df.reset_index(drop=True)


def yahoo_symbol(code: str) -> str:
    return f"{str(code).upper()}.T"


def _normalize_yf_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    df = frame.copy()
    df = df.rename_axis("Date").reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep]
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_yahoo_history_for_ipos(
    master: pd.DataFrame,
    first_trading_days: int = 80,
    extra_calendar_days: int = 160,
    batch_size: int = 30,
    sleep_seconds: float = 0.2,
) -> tuple[pd.DataFrame, FetchReport]:
    """Download a compact post-IPO OHLCV window using yfinance.

    Data is grouped by listing year, which keeps network payload reasonable.
    The result is intended as a research cache, not as a real-time feed.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance is required to build the public history cache") from e

    all_frames: list[pd.DataFrame] = []
    failed: list[str] = []
    requested = len(master)

    work = master.copy()
    work = work[(~work["is_technical"].fillna(False))].copy()
    work["listing_date"] = pd.to_datetime(work["listing_date"], errors="coerce")
    work = work.dropna(subset=["listing_date", "code"]).copy()
    work["code"] = work["code"].astype(str).str.strip().str.upper()
    work["listing_year"] = work["listing_date"].dt.year

    # Also protect callers that pass an old cached master produced by the buggy
    # fallback parser. A row where fallback code == listing year is invalid.
    if "metadata_source" in work.columns:
        bad_fallback_year = (
            work["metadata_source"].astype(str).str.endswith("#fallback")
            & work["code"].eq(work["listing_year"].astype("Int64").astype(str))
        )
        work = work[~bad_fallback_year].copy()

    for year, g in work.groupby("listing_year"):
        year = int(year)
        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year + 1, month=6, day=30)

        # Archive pages can overlap, and malformed source rows should never make
        # ``.loc[code]`` return a Series. Keep one clean listing date per code.
        g = g.copy()
        g["code"] = g["code"].astype(str).str.strip().str.upper()
        g["listing_date"] = pd.to_datetime(g["listing_date"], errors="coerce")
        g = g.dropna(subset=["code", "listing_date"])
        if "metadata_source" in g.columns:
            g["_fallback"] = g["metadata_source"].astype(str).str.endswith("#fallback")
            g = g.sort_values(["code", "_fallback", "listing_date"], ascending=[True, True, False])
        else:
            g = g.sort_values(["code", "listing_date"], ascending=[True, False])
        g = g.drop_duplicates("code", keep="first")
        codes = g["code"].tolist()
        listing_by_code = g.set_index("code")["listing_date"].to_dict()

        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]
            tickers = [yahoo_symbol(c) for c in batch_codes]
            try:
                raw = yf.download(
                    tickers=tickers,
                    start=year_start.date().isoformat(),
                    end=year_end.date().isoformat(),
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                    actions=False,
                )
            except Exception:
                raw = pd.DataFrame()

            for code, ticker in zip(batch_codes, tickers):
                listing_date = pd.Timestamp(listing_by_code[code]).normalize()
                try:
                    if raw.empty:
                        raise ValueError("empty batch")
                    if isinstance(raw.columns, pd.MultiIndex):
                        if ticker not in raw.columns.get_level_values(0):
                            raise KeyError(ticker)
                        sub = raw[ticker].copy()
                    else:
                        # yfinance returns single-level columns for a one-ticker batch.
                        sub = raw.copy()
                    sub = _normalize_yf_frame(sub)
                    sub = sub[(sub["Date"] >= listing_date) & (sub["Date"] <= listing_date + pd.Timedelta(days=extra_calendar_days))]
                    sub = sub.dropna(subset=["Open", "High", "Low", "Close"]).head(first_trading_days)
                    if sub.empty:
                        raise ValueError("no post-listing bars")
                    sub.insert(0, "code", code)
                    sub.insert(1, "listing_date", listing_date)
                    sub["price_source"] = "Yahoo Finance via yfinance"
                    all_frames.append(sub)
                except Exception:
                    failed.append(code)
            time.sleep(sleep_seconds)

    if not all_frames:
        raise RuntimeError("No Yahoo Finance IPO price histories could be downloaded.")
    prices = pd.concat(all_frames, ignore_index=True)
    prices = prices.sort_values(["code", "Date"]).reset_index(drop=True)
    succeeded = prices["code"].nunique()
    return prices, FetchReport("Yahoo Finance via yfinance", requested, succeeded, sorted(set(failed)))


def validate_ohlcv(prices: pd.DataFrame, min_days: int = 8) -> pd.DataFrame:
    rows = []
    for code, g in prices.groupby("code"):
        g = g.sort_values("Date")
        numeric_ok = g[["Open", "High", "Low", "Close", "Volume"]].notna().all(axis=1)
        positive_ok = (g[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
        envelope_ok = (
            (g["Low"] <= g[["Open", "Close"]].min(axis=1))
            & (g["High"] >= g[["Open", "Close"]].max(axis=1))
            & (g["High"] >= g["Low"])
        )
        volume_ok = g["Volume"].fillna(0).ge(0)
        first_lag = (pd.Timestamp(g["Date"].iloc[0]) - pd.Timestamp(g["listing_date"].iloc[0])).days
        rows.append(
            {
                "code": code,
                "rows": len(g),
                "first_trade_lag_days": first_lag,
                "numeric_ok_rate": float(numeric_ok.mean()),
                "positive_ok_rate": float(positive_ok.mean()),
                "ohlc_envelope_ok_rate": float(envelope_ok.mean()),
                "volume_ok_rate": float(volume_ok.mean()),
                "usable": bool(
                    len(g) >= min_days
                    and numeric_ok.mean() == 1
                    and positive_ok.mean() == 1
                    and envelope_ok.mean() == 1
                    and volume_ok.mean() == 1
                    and 0 <= first_lag <= 14
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["usable", "rows"], ascending=[False, False]).reset_index(drop=True)


def load_verified_627a(data_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    data_dir = Path(data_dir)
    hist = pd.read_csv(data_dir / "verified_627A.csv", parse_dates=["Date"])
    with open(data_dir / "akippa_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return hist, meta
