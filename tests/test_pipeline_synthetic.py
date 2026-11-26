"""End-to-end baselines + report on a synthetic dataset (no network: prices come from a pre-filled cache)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nopeeking.config import DataConfig
from nopeeking.eval.baselines import run_baselines
from nopeeking.eval.report import END, START, build_report

GOOD = "record revenue raised guidance strong demand margin expansion"
BAD = "impairment charge lowered outlook weak demand restructuring"


def _make(tmp_path):
    rng = np.random.default_rng(0)
    cal = pd.bdate_range("2018-01-01", "2026-06-30")
    tickers = [f"T{i}" for i in range(8)]
    cache = tmp_path / "cache" / "prices"
    cache.mkdir(parents=True)
    for t in [*tickers, "SPY"]:
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(cal))))
        pd.DataFrame({"close": px}, index=cal).to_parquet(cache / f"{t}.parquet")
    rows = []
    windows = {
        "train": ("2018-01-01", "2022-12-31", 160),
        "test_pre": ("2023-01-01", "2023-12-31", 60),
        "val": ("2024-01-01", "2024-12-31", 40),
        "test_post": ("2025-07-01", "2026-06-01", 60),
    }
    for split, (lo, hi, n) in windows.items():
        days = rng.choice(pd.bdate_range(lo, hi), n)
        for k, d in enumerate(days):
            up = int(rng.random() < 0.5)
            words = (GOOD if up else BAD) if rng.random() < 0.8 else (BAD if up else GOOD)
            t = tickers[k % len(tickers)]
            rows.append(
                {
                    "id": f"{split}-{k}",
                    "split": split,
                    "ticker": t,
                    "cik": 1000 + k % 8,
                    "company": f"Company {t} Inc.",
                    "aliases": json.dumps([f"Company {t} Inc."]),
                    "tickers_all": json.dumps([t]),
                    "event_day": pd.Timestamp(d),
                    "filing_date": pd.Timestamp(d),
                    "items": "2.02",
                    "event_type": "earnings",
                    "text": f"Company {t} Inc. reported {words}.",
                    "y_up": up,
                    "ar_3d_w": (0.02 if up else -0.02) + rng.normal(0, 0.01),
                }
            )
    cfg = DataConfig(
        name="synthetic",
        start="2018-01-01",
        end="2026-06-30",
        cache_dir=str(tmp_path / "cache"),
        out_dir=str(tmp_path / "processed"),
        results_dir=str(tmp_path / "results"),
    )
    cfg.dataset_dir.mkdir(parents=True)
    df = pd.DataFrame(rows)
    df.to_parquet(cfg.dataset_path)
    stats = {
        "n": len(df),
        "first_filing": "2018-01-02",
        "last_filing": "2026-06-01",
        "splits": df["split"].value_counts().to_dict(),
        "up_rate_by_split": df.groupby("split")["y_up"].mean().to_dict(),
    }
    (cfg.dataset_dir / "stats.json").write_text(json.dumps(stats))
    (cfg.dataset_dir / "filter_log.json").write_text(
        json.dumps([{"step": "synthetic", "n_before": None, "n_after": len(df), "note": ""}])
    )
    return cfg


def test_baselines_and_report(tmp_path):
    cfg = _make(tmp_path)
    res = json.loads(Path(run_baselines(cfg)).read_text(encoding="utf-8"))
    tfidf = res["methods"]["tfidf_lr"]["test_post"]
    assert tfidf["auc"] > 0.7  # words carry signal by construction
    assert res["methods"]["majority"]["test_post"]["auc"] == 0.5
    assert "portfolio" in res["methods"]["tfidf_lr"]
    assert "did" in res["tfidf_lr_gap"]["auc"]

    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nold\n{END}\noutro\n", encoding="utf-8")
    written = build_report(cfg, readme=readme)
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and "tfidf_lr" in text and text.endswith("outro\n")
    assert any(str(p).endswith("hero.png") for p in written)
    assert "\\newcommand" in (cfg.run_results_dir / "macros.tex").read_text(encoding="utf-8")


class _FakeScorer:
    """Stands in for the language model: scores the filing text, so no download or GPU is needed."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, build):
        msgs = build(4000)
        return msgs[-1]["content"]

    def score(self, texts, batch_size=4):
        p = np.array([0.7 if "record revenue" in t else 0.3 for t in texts])
        return p, np.ones(len(texts))


def test_probe_shards_merge_to_the_unsharded_result(tmp_path, monkeypatch):
    import nopeeking.eval.llm as llm
    from nopeeking.eval.probe import run_probe

    monkeypatch.setattr(llm, "YesNoScorer", _FakeScorer)
    cfg = _make(tmp_path)
    pred = cfg.run_results_dir / "predictions" / "probe__fake__512.parquet"
    parts = []
    for i in range(3):
        assert run_probe(cfg, "fake", max_input_tokens=512, with_market=False, shard=(i, 3)) == str(pred)
        parts.append(pd.read_parquet(pred))
        pred.unlink()
    merged = pd.concat(parts, ignore_index=True)
    n_jobs = 60 + 60 + 40 + 60 + 60  # original for 3 splits + anonymized for the 2 probe splits
    assert len(merged) == n_jobs and not merged.duplicated(["id", "variant"]).any()

    merged.to_parquet(pred, index=False)
    sharded = json.loads(Path(run_probe(cfg, "fake", max_input_tokens=512, with_market=False)).read_text())
    pred.unlink()
    direct = json.loads(Path(run_probe(cfg, "fake", max_input_tokens=512, with_market=False)).read_text())
    assert sharded["gap"]["auc"]["did"] == direct["gap"]["auc"]["did"]
    assert sharded["metrics"]["original"]["test_post"]["auc"] > 0.6
