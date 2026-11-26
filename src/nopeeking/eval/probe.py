"""``nopeeking probe <model>``: does a model look skilled only where it could remember the outcome?

Design (2 x 2): pre-cutoff (test_pre, 2023) vs post-cutoff (test_post) filings, original vs anonymized
text. Memory can only help on original pre-cutoff filings, so the headline estimate is the
difference-in-differences of AUC:

    DiD = (AUC_pre,orig - AUC_pre,anon) - (AUC_post,orig - AUC_post,anon)

Anonymization also removes legitimate information; the post-cutoff term and the TF-IDF control
(trained on <= 2022) measure that loss, so a DiD > 0 points to recall rather than reasoning.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from nopeeking.config import DataConfig
from nopeeking.eval.metrics import auc

log = logging.getLogger(__name__)

PROBE_SPLITS = ("test_pre", "test_post")


def _acc(y: np.ndarray, p: np.ndarray) -> float:
    return float(((p >= 0.5).astype(int) == y).mean())


def gap_estimates(both: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """``both``: id, split, y_up, p_original, p_anonymized. Paired within split, independent across splits."""
    pre = both[both["split"] == "test_pre"]
    post = both[both["split"] == "test_post"]
    if len(pre) < 10 or len(post) < 10:
        return {}
    arrays = {
        name: (sub["y_up"].to_numpy(), sub["p_original"].to_numpy(), sub["p_anonymized"].to_numpy())
        for name, sub in (("pre", pre), ("post", post))
    }

    def stats(fn, a_pre, a_post) -> dict[str, float]:
        y0, o0, n0 = a_pre
        y1, o1, n1 = a_post
        pre_o, pre_a, post_o, post_a = fn(y0, o0), fn(y0, n0), fn(y1, o1), fn(y1, n1)
        return {
            "pre_orig": pre_o,
            "pre_anon": pre_a,
            "post_orig": post_o,
            "post_anon": post_a,
            "pre_minus_post_orig": pre_o - post_o,
            "anon_effect_pre": pre_o - pre_a,
            "anon_effect_post": post_o - post_a,
            "did": (pre_o - pre_a) - (post_o - post_a),
        }

    rng = np.random.default_rng(seed)
    out: dict = {"n_pre": len(pre), "n_post": len(post)}
    for metric, fn in (("auc", auc), ("acc", _acc)):
        point = stats(fn, arrays["pre"], arrays["post"])
        boots: dict[str, list[float]] = {k: [] for k in point}
        for _ in range(n_boot):
            ip = rng.integers(0, len(pre), len(pre))
            iq = rng.integers(0, len(post), len(post))
            s = stats(fn, tuple(a[ip] for a in arrays["pre"]), tuple(a[iq] for a in arrays["post"]))
            for k, v in s.items():
                boots[k].append(v)
        block = {}
        for k, v in point.items():
            arr = np.array(boots[k])
            arr = arr[~np.isnan(arr)]
            block[k] = v
            block[f"{k}_lo"] = float(np.percentile(arr, 2.5)) if len(arr) else float("nan")
            block[f"{k}_hi"] = float(np.percentile(arr, 97.5)) if len(arr) else float("nan")
        out[metric] = block
    return out


def model_slug(model_id: str, adapter: str | None = None) -> str:
    slug = model_id.replace("/", "__")
    if adapter:
        slug += "__" + adapter.rstrip("/\\").replace("\\", "/").split("/")[-1]
    return slug


def run_probe(
    cfg: DataConfig,
    model_id: str,
    splits: list[str] | None = None,
    limit: int | None = None,
    device: str | None = None,
    max_input_tokens: int = 768,
    batch_size: int = 4,
    dtype: str = "auto",
    adapter: str | None = None,
) -> str:
    from nopeeking.eval.evaluate import evaluate, load_market
    from nopeeking.eval.llm import YesNoScorer
    from nopeeking.prompts import direct_messages

    splits = splits or ["test_pre", "test_post", "val"]
    df = pd.read_parquet(cfg.dataset_path)
    df = df[df["split"].isin(splits)]
    if limit:
        df = pd.concat(g.sample(n=min(limit, len(g)), random_state=cfg.seed) for _, g in df.groupby("split"))
    df = df.reset_index(drop=True)
    jobs = [(r["id"], "original") for _, r in df.iterrows()]
    jobs += [(r["id"], "anonymized") for _, r in df.iterrows() if r["split"] in PROBE_SPLITS]

    slug = model_slug(model_id, adapter)
    pred_dir = cfg.run_results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    cache_path = pred_dir / f"probe__{slug}__{max_input_tokens}.parquet"
    done = (
        pd.read_parquet(cache_path)
        if cache_path.exists()
        else pd.DataFrame(columns=["id", "variant", "p_up", "mass"])
    )
    seen = set(zip(done["id"], done["variant"], strict=True))
    pending = [j for j in jobs if j not in seen]
    log.info("probe %s: %d jobs, %d cached, %d pending", slug, len(jobs), len(seen), len(pending))

    if pending:
        scorer = YesNoScorer(
            model_id, device=device, dtype=dtype, max_input_tokens=max_input_tokens, adapter=adapter
        )
        rows = df.set_index("id")
        t0 = time.time()
        chunk = 32
        for i in range(0, len(pending), chunk):
            part = pending[i : i + chunk]
            ids = [
                scorer.encode(
                    lambda c, r=rows.loc[pid], a=(var == "anonymized"): direct_messages(
                        r, anonymized=a, max_chars=c, horizon=cfg.horizon
                    )
                )
                for pid, var in part
            ]
            p, mass = scorer.score(ids, batch_size=batch_size)
            new = pd.DataFrame(
                {"id": [j[0] for j in part], "variant": [j[1] for j in part], "p_up": p, "mass": mass}
            )
            done = pd.concat([done, new], ignore_index=True) if len(done) else new
            done.to_parquet(cache_path, index=False)
            elapsed = time.time() - t0
            n_done = i + len(part)
            log.info(
                "probe %s: %d/%d scored, %.1fs/item, ETA %.0f min",
                slug,
                n_done,
                len(pending),
                elapsed / n_done,
                elapsed / n_done * (len(pending) - n_done) / 60,
            )

    done = done[done["id"].isin(df["id"])]
    wide = done.pivot_table(index="id", columns="variant", values="p_up").reset_index()
    try:
        market = load_market(cfg, df)
    except Exception as err:  # e.g. Yahoo blocking a cloud IP: metrics still computed, portfolio skipped
        log.warning("prices unavailable, long-short portfolio skipped: %r", err)
        market = None
    result: dict = {
        "model": model_id,
        "adapter": adapter,
        "dataset": cfg.name,
        "max_input_tokens": max_input_tokens,
        "answer_mass_mean": float(done["mass"].mean()),
        "metrics": {
            "original": evaluate(
                wide[["id", "original"]].rename(columns={"original": "p_up"}), df, cfg, market
            ),
        },
    }
    if "anonymized" in wide:
        anon = wide.dropna(subset=["anonymized"])[["id", "anonymized"]].rename(columns={"anonymized": "p_up"})
        result["metrics"]["anonymized"] = evaluate(anon, df, cfg, None)
        both = df[["id", "split", "y_up"]].merge(
            wide.rename(columns={"original": "p_original", "anonymized": "p_anonymized"}), on="id"
        )
        result["gap"] = gap_estimates(both.dropna(subset=["p_anonymized"]), seed=cfg.seed)
    out_dir = cfg.run_results_dir / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(path)
