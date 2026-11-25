"""Event day and market-adjusted returns.

Convention (documented in the README):
* t = first trading session whose 16:00 ET close is strictly after the filing's EDGAR acceptance time.
  A filing accepted at 10:00 ET on a trading day has t = that day; one accepted at 16:30 ET has t = next day.
* h-day return over sessions [t, t+h-1] = close(t+h-1) / close(t-1) - 1 (includes the full reaction on day t).
* market-adjusted (abnormal) return = stock return - benchmark (SPY) return over the same window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_TZ = "America/New_York"
CLOSE_HOUR = 16


def event_days(acceptance_utc: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """First trading date whose close is after each acceptance timestamp (NaT if beyond the calendar)."""
    calendar = pd.DatetimeIndex(calendar).sort_values()
    closes = (calendar + pd.Timedelta(hours=CLOSE_HOUR)).tz_localize(MARKET_TZ).tz_convert("UTC")
    close_ns = closes.as_unit("ns").asi8
    acc = pd.DatetimeIndex(acceptance_utc).tz_convert("UTC").as_unit("ns").asi8
    pos = np.searchsorted(close_ns, acc, side="right")
    valid = pos < len(calendar)
    out = pd.Series(pd.NaT, index=acceptance_utc.index, dtype="datetime64[ns]")
    out[valid] = calendar.as_unit("ns")[pos[valid]]
    return out


def market_adjusted_returns(
    tickers: pd.Series, days: pd.Series, close: pd.DataFrame, benchmark: pd.Series, horizon: int
) -> pd.DataFrame:
    """Per-event raw and benchmark-adjusted returns for 1 day and ``horizon`` days.

    Rows with any missing price in the window (or a window running past the data) get NaN.
    """
    cal = pd.DatetimeIndex(benchmark.dropna().index).sort_values()
    px = close.reindex(cal).to_numpy(dtype=float)
    bx = benchmark.reindex(cal).to_numpy(dtype=float)
    n = len(cal)
    idx = cal.get_indexer(pd.DatetimeIndex(days))
    col = close.columns.get_indexer(tickers)
    ok = (idx >= 1) & (col >= 0)
    end_h = idx + horizon - 1
    ok_h = ok & (end_h < n)

    def take(arr2d: np.ndarray, rows: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.full(len(rows), np.nan)
        out[mask] = arr2d[rows[mask], col[mask]]
        return out

    def take_b(rows: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.full(len(rows), np.nan)
        out[mask] = bx[rows[mask]]
        return out

    p_prev = take(px, idx - 1, ok)
    p_t = take(px, idx, ok)
    p_h = take(px, end_h, ok_h)
    b_prev, b_t, b_h = take_b(idx - 1, ok), take_b(idx, ok), take_b(end_h, ok_h)
    ret_1d = p_t / p_prev - 1
    ret_h = p_h / p_prev - 1
    bench_1d = b_t / b_prev - 1
    bench_h = b_h / b_prev - 1
    return pd.DataFrame(
        {
            "ret_1d": ret_1d,
            "ar_1d": ret_1d - bench_1d,
            f"ret_{horizon}d": ret_h,
            f"ar_{horizon}d": ret_h - bench_h,
            "window_end": pd.Series(cal[np.clip(end_h, 0, n - 1)], index=tickers.index).where(ok_h),
        },
        index=tickers.index,
    )


def add_targets(df: pd.DataFrame, horizon: int, winsor: tuple[float, float], train_mask: pd.Series) -> dict:
    """Add sign/tercile targets and winsorized returns. Cut-offs come from the train split only."""
    col = f"ar_{horizon}d"
    train = df.loc[train_mask, col].dropna()
    lo, hi = train.quantile(winsor[0]), train.quantile(winsor[1])
    q1, q2 = train.quantile(1 / 3), train.quantile(2 / 3)
    df[f"{col}_w"] = df[col].clip(lo, hi)
    df["ar_1d_w"] = df["ar_1d"].clip(
        df.loc[train_mask, "ar_1d"].quantile(winsor[0]), df.loc[train_mask, "ar_1d"].quantile(winsor[1])
    )
    df["y_up"] = (df[col] > 0).astype(int)
    df["y_up_1d"] = (df["ar_1d"] > 0).astype(int)
    df["y_tercile"] = np.digitize(df[col].to_numpy(), [q1, q2]).astype(int)
    return {"winsor_lo": float(lo), "winsor_hi": float(hi), "tercile_q1": float(q1), "tercile_q2": float(q2)}
