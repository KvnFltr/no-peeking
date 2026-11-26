"""Build the NoPeek-8K dataset: universe -> 8-K list -> labels -> (sample) -> texts -> dedup -> parquet.

Every filter is logged with before/after counts in ``filter_log.json`` next to the dataset.
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from nopeeking.config import SPLIT_ORDER, UNUSED, DataConfig, assign_split
from nopeeking.data.filings import EVENT_PRIORITY, classify_event, combine_text, fetch_filing_text, list_8k
from nopeeking.data.sec import SecClient
from nopeeking.data.universe import load_universe
from nopeeking.labels.events import add_targets, event_days, market_adjusted_returns
from nopeeking.labels.prices import load_close

log = logging.getLogger(__name__)


@dataclass
class FilterLog:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, n_after: int, n_before: int | None = None, note: str = "") -> None:
        self.steps.append({"step": step, "n_before": n_before, "n_after": n_after, "note": note})
        log.info("%-55s %8s -> %8d %s", step, "" if n_before is None else n_before, n_after, note)


def _split_counts(df: pd.DataFrame) -> dict[str, int]:
    return {s: int((df["split"] == s).sum()) for s in [*SPLIT_ORDER, UNUSED] if (df["split"] == s).any()}


def build_dataset(cfg: DataConfig) -> pd.DataFrame:
    flog = FilterLog()
    client = SecClient(cfg.cache_dir, max_per_second=cfg.sec_max_rps)
    end = cfg.end_ts

    uni = load_universe(cfg.indexes, client, cfg.universe_dir)
    uni = uni.rename(columns={"index": "index_name"})
    flog.add("universe firms (unique CIK)", len(uni), note="+".join(cfg.indexes))
    if cfg.n_firms and cfg.n_firms < len(uni):
        uni = uni.sample(n=cfg.n_firms, random_state=cfg.seed).reset_index(drop=True)
        flog.add("sampled firms", len(uni), note=f"seed={cfg.seed}")

    frames = []
    for i, firm in enumerate(uni.itertuples(index=False), 1):
        try:
            meta, filings = list_8k(client, int(firm.cik), cfg.start, end)
        except (FileNotFoundError, RuntimeError) as err:
            log.warning("skip %s (cik %s): %s", firm.ticker, firm.cik, err)
            continue
        if filings.empty:
            continue
        filings["ticker"] = firm.ticker
        filings["index_name"] = firm.index_name
        filings["company"] = meta["name"] or firm.name
        filings["aliases"] = json.dumps(sorted({firm.name, meta["name"], *meta["former_names"]}))
        filings["tickers_all"] = json.dumps(sorted({firm.ticker, *meta["tickers"]}))
        frames.append(filings)
        if i % 25 == 0:
            log.info("listed filings for %d/%d firms", i, len(uni))
    df = pd.concat(frames, ignore_index=True)
    flog.add("8-K filings in date range", len(df), note=f"{cfg.start}..{end.date()}")

    n = len(df)
    df["event_type"] = df["items"].map(classify_event)
    df = df[df["event_type"].isin(cfg.event_types)].copy()
    flog.add("event type in scope (items 2.02/1.01/2.01/5.02/8.01)", len(df), n)

    df["acceptance_et"] = df["acceptance_utc"].dt.tz_convert("America/New_York")
    df["filing_date"] = df["acceptance_et"].dt.tz_localize(None).dt.normalize()
    df["split"] = assign_split(df["filing_date"], cfg.splits)
    n = len(df)
    df = df[df["split"] != UNUSED].copy()
    flog.add("inside a split window", len(df), n, note="drops the embargo gap between val and test_post")

    start_px = pd.Timestamp(cfg.start) - pd.Timedelta(days=15)
    close = load_close([*df["ticker"].unique(), cfg.benchmark], start_px, end, cfg.cache_dir)
    bench = close.pop(cfg.benchmark).dropna()
    df["event_day"] = event_days(df["acceptance_utc"], bench.index)
    rets = market_adjusted_returns(df["ticker"], df["event_day"], close, bench, cfg.horizon)
    df = pd.concat([df, rets], axis=1)
    hcol = f"ar_{cfg.horizon}d"
    n = len(df)
    df = df.dropna(subset=["event_day", "ar_1d", hcol]).copy()
    flog.add(
        "prices available for full label window",
        len(df),
        n,
        note=f"needs close(t-1)..close(t+{cfg.horizon - 1}); recent filings wait for their window",
    )

    n = len(df)
    df["_prio"] = df["event_type"].map(EVENT_PRIORITY)
    df = df.sort_values(["cik", "event_day", "_prio", "acceptance_utc"])
    df = df.drop_duplicates(subset=["cik", "event_day"], keep="first").drop(columns="_prio")
    flog.add("one filing per firm and event day", len(df), n, note="keeps highest-priority event type")

    if cfg.quota:
        n = len(df)
        parts = []
        for split, k in cfg.quota.items():
            sub = df[df["split"] == split]
            parts.append(sub.sample(n=min(k, len(sub)), random_state=cfg.seed))
        df = pd.concat(parts)
        flog.add("quota sample per split", len(df), n, note=json.dumps(cfg.quota))

    text_cache = Path(cfg.cache_dir) / "texts"
    texts: dict[str, dict] = {}
    failures = 0
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futs = {
            pool.submit(fetch_filing_text, client, int(r.cik), r.accession, text_cache): r.accession
            for r in df.itertuples(index=False)
        }
        for j, fut in enumerate(as_completed(futs), 1):
            acc = futs[fut]
            try:
                texts[acc] = fut.result()
            except Exception as err:  # keep going; count failures in the filter log
                failures += 1
                log.warning("text fetch failed for %s: %r", acc, err)
            if j % 200 == 0:
                log.info("fetched texts %d/%d (SEC requests so far: %d)", j, len(futs), client.n_requests)
    n = len(df)
    df = df[df["accession"].isin(texts)].copy()
    flog.add("filing text fetched", len(df), n, note=f"{failures} fetch failures")

    df["text_8k"] = df["accession"].map(lambda a: texts[a]["text_8k"])
    df["text_ex99"] = df["accession"].map(lambda a: texts[a]["text_ex99"])
    df["has_ex99"] = df["accession"].map(lambda a: bool(texts[a]["has_ex99"]))
    df["n_chars_8k"] = df["text_8k"].str.len()
    df["n_chars_ex99"] = df["text_ex99"].str.len()
    df["text"] = [
        combine_text(a, b, cfg.max_text_chars) for a, b in zip(df["text_8k"], df["text_ex99"], strict=True)
    ]
    n = len(df)
    df = df[df["text"].str.len() >= cfg.min_text_chars].copy()
    flog.add(f"text length >= {cfg.min_text_chars} chars", len(df), n)

    n = len(df)
    df["text_sha1"] = df["text"].map(lambda t: hashlib.sha1(t.encode()).hexdigest())
    df = df.drop_duplicates(subset="text_sha1", keep="first")
    flog.add("dedup identical texts", len(df), n)

    cuts = add_targets(df, cfg.horizon, tuple(cfg.winsor), df["split"] == "train")
    df = df.sort_values("acceptance_utc").reset_index(drop=True)
    df["id"] = df["accession"]
    keep = [
        "id",
        "accession",
        "cik",
        "ticker",
        "index_name",
        "company",
        "aliases",
        "tickers_all",
        "items",
        "event_type",
        "acceptance_utc",
        "filing_date",
        "event_day",
        "window_end",
        "split",
        "has_ex99",
        "text",
        "n_chars_8k",
        "n_chars_ex99",
        "ret_1d",
        "ar_1d",
        f"ret_{cfg.horizon}d",
        hcol,
        f"{hcol}_w",
        "ar_1d_w",
        "y_up",
        "y_up_1d",
        "y_tercile",
    ]
    df = df[keep]

    out = cfg.dataset_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.dataset_path, index=False, compression="zstd")
    stats = {
        "config": asdict(cfg),
        "n": len(df),
        "splits": _split_counts(df),
        "event_types": df["event_type"].value_counts().to_dict(),
        "up_rate_by_split": df.groupby("split")["y_up"].mean().round(4).to_dict(),
        "label_cutoffs_from_train": cuts,
        "has_ex99_rate": round(float(df["has_ex99"].mean()), 4),
        "median_text_chars": int(df["text"].str.len().median()),
        "first_filing": str(df["filing_date"].min().date()),
        "last_filing": str(df["filing_date"].max().date()),
        "sec_requests": client.n_requests,
    }
    (out / "filter_log.json").write_text(json.dumps(flog.steps, indent=2), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s (%d rows)", cfg.dataset_path, len(df))
    return df
