"""Turn a predictions table into the standard metric block (per split, CIs, long-short portfolio)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nopeeking.config import DataConfig
from nopeeking.eval.metrics import metrics_with_ci
from nopeeking.eval.portfolio import long_short_returns, portfolio_summary
from nopeeking.labels.prices import load_close


def load_market(cfg: DataConfig, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    start = pd.Timestamp(cfg.start) - pd.Timedelta(days=15)
    close = load_close([*dataset["ticker"].unique(), cfg.benchmark], start, cfg.end_ts, cfg.cache_dir)
    bench = close.pop(cfg.benchmark).dropna()
    return close, bench


def evaluate(
    preds: pd.DataFrame,
    dataset: pd.DataFrame,
    cfg: DataConfig,
    market: tuple[pd.DataFrame, pd.Series] | None = None,
    cost_bps: float = 5.0,
    n_boot: int = 2000,
) -> dict:
    """``preds`` has columns id, p_up (one row per filing). Returns {split: metrics, "portfolio": {...}}."""
    hcol = f"ar_{cfg.horizon}d"
    df = dataset.merge(preds[["id", "p_up"]], on="id", how="inner")
    out: dict = {}
    for split, sub in df.groupby("split"):
        out[str(split)] = metrics_with_ci(
            sub["y_up"].to_numpy(),
            sub["p_up"].to_numpy(),
            sub[f"{hcol}_w"].to_numpy(),
            n_boot=n_boot,
            seed=cfg.seed,
        )
    val, test = df[df["split"] == "val"], df[df["split"] == "test_post"]
    if market is not None and len(val) >= 10 and len(test) > 0:
        lo_cut, hi_cut = np.quantile(val["p_up"], [1 / 3, 2 / 3])
        if hi_cut > lo_cut:
            close, bench = market
            events = test[["ticker", "event_day"]].assign(score=test["p_up"].to_numpy())
            daily = long_short_returns(events, lo_cut, hi_cut, close, bench, cfg.horizon, cost_bps)
            n_pos = int(((test["p_up"] >= hi_cut) | (test["p_up"] <= lo_cut)).sum())
            out["portfolio"] = {
                **portfolio_summary(daily, n_pos),
                "cost_bps_one_way": cost_bps,
                "cuts_from": "val terciles",
                "lo_cut": float(lo_cut),
                "hi_cut": float(hi_cut),
            }
    return out
