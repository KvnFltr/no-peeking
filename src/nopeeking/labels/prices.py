"""Daily adjusted close prices from Yahoo Finance (via yfinance), cached per ticker.

Raw prices are never redistributed: only derived labels are shipped. Yahoo data is for research use;
check its terms before any other use.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-").upper()


def _cache_ok(path: Path, end: pd.Timestamp) -> bool:
    if not path.exists():
        return False
    s = pd.read_parquet(path)
    if s.empty:
        return False
    fetched_today = (time.time() - path.stat().st_mtime) < 86400
    return fetched_today or s.index.max() >= end - pd.Timedelta(days=7)


def _download(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.Series]:
    import yfinance as yf

    raw = yf.download(
        symbols,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=min(8, len(symbols)),
    )
    close = raw["Close"] if "Close" in raw else pd.DataFrame()
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    out = {}
    for sym in symbols:
        s = close[sym].dropna() if sym in close else pd.Series(dtype=float)
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        out[sym] = s.astype(float)
    return out


def load_close(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str | Path,
    batch: int = 50,
    retries: int = 2,
) -> pd.DataFrame:
    """Adjusted close, one column per ticker (original spelling), tz-naive trading-date index.

    Empty downloads (Yahoo throttling, real delistings) are retried one ticker at a time and never cached,
    so a transient failure cannot freeze a hole in the data. Tickers that stay empty get an all-NaN column,
    and their filings are dropped (and counted) by the label-window filter.
    """
    cache = Path(cache_dir) / "prices"
    cache.mkdir(parents=True, exist_ok=True)
    tickers = sorted(set(tickers))
    need = [yahoo_symbol(t) for t in tickers if not _cache_ok(cache / f"{yahoo_symbol(t)}.parquet", end)]
    got: dict[str, pd.Series] = {}
    for i in range(0, len(need), batch):
        log.info("yfinance: downloading %d tickers (%d/%d)", len(need[i : i + batch]), i + batch, len(need))
        got.update(_download(need[i : i + batch], start, end))
    for attempt in range(retries):
        missing = [sym for sym in need if got.get(sym) is None or got[sym].empty]
        if not missing:
            break
        log.warning("yfinance: retrying %d empty tickers one by one (attempt %d)", len(missing), attempt + 1)
        for sym in missing:
            time.sleep(1.0)
            got.update(_download([sym], start, end))
    for sym, s in got.items():
        if s.empty:
            log.warning("yfinance: no prices for %s after retries (not cached)", sym)
        else:
            pd.DataFrame({"close": s}).to_parquet(cache / f"{sym}.parquet")
    out = {}
    for t in tickers:
        path = cache / f"{yahoo_symbol(t)}.parquet"
        s = pd.read_parquet(path)["close"] if path.exists() else pd.Series(dtype=float)
        if s.empty:
            out[t] = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            continue
        out[t] = s[(s.index >= start) & (s.index <= end)]
    return pd.DataFrame(out).sort_index()
