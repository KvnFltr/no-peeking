"""Event-driven daily long-short portfolio and Sharpe ratio with a block-bootstrap CI.

Tradable timing: the event day t is the first session whose close follows the filing, so the earliest close at
which a position can be opened is close(t). A position therefore earns sessions t+1 .. t+h-1 (``entry_lag=1``);
the day-t reaction, which the prediction target includes, is NOT captured by the portfolio. (Entering at close(t-1)
would trade on information that was not public yet: a look-ahead error.)

Long if the score is in the top tercile, short if in the bottom tercile; tercile cut-offs come from *validation*
scores (never the test set). Positions earn the stock's daily return minus SPY's (market-hedged). Each leg is
equally weighted across its open positions; the portfolio return is long-leg minus short-leg (or a single leg when
only one is open). A round-trip cost of ``2 * cost_bps`` is charged on each position's first holding day.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def long_short_returns(
    events: pd.DataFrame,
    lo_cut: float,
    hi_cut: float,
    close: pd.DataFrame,
    bench: pd.Series,
    horizon: int,
    cost_bps: float = 5.0,
    entry_lag: int = 1,
) -> pd.Series:
    """``events`` needs columns ticker, event_day, score. Returns daily portfolio returns (0 when flat)."""
    cal = pd.DatetimeIndex(bench.dropna().index).sort_values()
    stock_ret = close.reindex(cal).pct_change(fill_method=None)
    bench_ret = bench.reindex(cal).pct_change(fill_method=None)
    legs: dict[str, dict[pd.Timestamp, list[float]]] = {"long": defaultdict(list), "short": defaultdict(list)}
    entries: dict[str, dict[pd.Timestamp, int]] = {"long": defaultdict(int), "short": defaultdict(int)}
    first, last = None, None
    for ev in events.itertuples(index=False):
        side = "long" if ev.score >= hi_cut else "short" if ev.score <= lo_cut else None
        i = cal.get_indexer([pd.Timestamp(ev.event_day)])[0]
        if side is None or i < 1 or ev.ticker not in stock_ret.columns:
            continue
        days = cal[i + entry_lag : i + horizon]
        if len(days) == 0:
            continue
        first = days[0] if first is None else min(first, days[0])
        last = days[-1] if last is None else max(last, days[-1])
        entries[side][days[0]] += 1
        for d in days:
            r = stock_ret.at[d, ev.ticker] - bench_ret.at[d]
            if np.isfinite(r):
                legs[side][d].append(float(r))
    if first is None:
        return pd.Series(dtype=float)
    days = cal[(cal >= first) & (cal <= last)]
    out = pd.Series(0.0, index=days)
    cost = 2 * cost_bps / 1e4
    for d in days:
        parts = []
        for side, sign in (("long", 1.0), ("short", -1.0)):
            rets = legs[side].get(d, [])
            if rets:
                gross = sign * float(np.mean(rets))
                parts.append(gross - cost * entries[side].get(d, 0) / len(rets))
        if len(parts) == 2:
            out[d] = parts[0] + parts[1]
        elif parts:
            out[d] = parts[0]
    return out


def sharpe(daily: np.ndarray, periods: int = 252) -> float:
    daily = np.asarray(daily, dtype=float)
    sd = daily.std(ddof=1) if len(daily) > 1 else 0.0
    return float(daily.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def sharpe_ci(daily: np.ndarray, block: int = 5, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Circular block bootstrap percentile CI for the annualized Sharpe ratio."""
    daily = np.asarray(daily, dtype=float)
    n = len(daily)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        stats[b] = sharpe(daily[idx])
    stats = stats[~np.isnan(stats)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def portfolio_summary(daily: pd.Series, n_positions: int) -> dict[str, float]:
    if daily.empty:
        return {"n_days": 0, "n_positions": n_positions}
    lo, hi = sharpe_ci(daily.to_numpy())
    return {
        "n_days": len(daily),
        "n_active_days": int((daily != 0).sum()),
        "n_positions": n_positions,
        "ann_return": float(daily.mean() * 252),
        "ann_vol": float(daily.std(ddof=1) * np.sqrt(252)),
        "sharpe": sharpe(daily.to_numpy()),
        "sharpe_lo": lo,
        "sharpe_hi": hi,
    }
