"""Dataset configuration (YAML -> dataclass) and time-based split assignment."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import pandas as pd
import yaml

SPLIT_ORDER = ("train", "test_pre", "val", "test_post")
UNUSED = "unused"


def _default_splits() -> dict[str, list[str | None]]:
    return {
        "train": ["2018-01-01", "2022-12-31"],
        "test_pre": ["2023-01-01", "2023-12-31"],
        "val": ["2024-01-01", "2024-12-31"],
        "test_post": ["2025-07-01", None],
    }


@dataclass
class DataConfig:
    name: str = "pilot"
    seed: int = 0
    indexes: list[str] = field(default_factory=lambda: ["sp500"])
    n_firms: int | None = None
    start: str = "2018-01-01"
    end: str | None = None
    event_types: list[str] = field(default_factory=lambda: ["earnings", "deal", "exec_change", "other"])
    quota: dict[str, int] | None = None
    splits: dict[str, list[str | None]] = field(default_factory=_default_splits)
    horizon: int = 3
    benchmark: str = "SPY"
    winsor: list[float] = field(default_factory=lambda: [0.01, 0.99])
    min_text_chars: int = 300
    max_text_chars: int = 20000
    cache_dir: str = "data/cache"
    out_dir: str = "data/processed"
    results_dir: str = "results"
    universe_dir: str = "data/universe"
    workers: int = 4
    sec_max_rps: float = 8.0

    @property
    def dataset_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    @property
    def dataset_path(self) -> Path:
        return self.dataset_dir / "nopeek8k.parquet"

    @property
    def run_results_dir(self) -> Path:
        return Path(self.results_dir) / self.name

    @property
    def end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.end) if self.end else pd.Timestamp.today().normalize()


def load_config(path: str | Path) -> DataConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(DataConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return DataConfig(**raw)


def assign_split(dates: pd.Series, splits: dict[str, list[str | None]]) -> pd.Series:
    """Map tz-naive dates to split names; dates outside every window get ``UNUSED``.

    Windows are inclusive on both ends; an open upper bound (None) means "up to today".
    """
    out = pd.Series(UNUSED, index=dates.index, dtype=object)
    for name, (lo, hi) in splits.items():
        mask = dates >= pd.Timestamp(lo)
        if hi is not None:
            mask &= dates <= pd.Timestamp(hi)
        out[mask & (out == UNUSED)] = name
    return out
