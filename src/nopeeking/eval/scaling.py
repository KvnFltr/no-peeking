"""Does memorization grow with model size? One joint bootstrap over every probed model.

All zero-shot probes score the same seeded sample of pre- and post-cutoff filings, so the per-model DiDs are not
independent: one unlucky draw of filings moves them together. Resampling the filings once per bootstrap draw and
recomputing every model's DiD on that draw keeps this dependence, and gives intervals for

* the mean DiD over models, a pooled test of recall with more power than any single model, and
* the OLS slope of DiD on log10(parameters): memorization per tenfold increase in size.

Only base models without an adapter and with a known parameter count (``configs/models.yaml``) enter.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nopeeking.config import DataConfig
from nopeeking.eval.probe import model_slug

log = logging.getLogger(__name__)


def _auc_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Row-wise rank-sum AUC for (draws x filings) matrices, ties at average rank (same as ``metrics.auc``)."""
    from scipy.stats import rankdata

    r = rankdata(p, axis=1)
    n1 = y.sum(axis=1)
    n0 = y.shape[1] - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((r * y).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def _slopes(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    """OLS slope of each row of ``d`` (draws x models) on ``x``."""
    xc = x - x.mean()
    return ((d - d.mean(axis=1, keepdims=True)) * xc).sum(axis=1) / (xc**2).sum()


def _interval(a: np.ndarray) -> tuple[float, float]:
    a = a[~np.isnan(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (float("nan"),) * 2


def scaling_analysis(
    cfg: DataConfig, probes: dict[str, dict], meta: dict[str, dict], n_boot: int = 2000, seed: int = 0
) -> dict | None:
    """Joint-bootstrap mean DiD and DiD-vs-size slope; ``None`` with fewer than two usable models."""
    labels = pd.read_parquet(cfg.dataset_path, columns=["id", "split", "y_up"])
    labels = labels[labels["split"].isin(["test_pre", "test_post"])]
    models = []
    for p in probes.values():
        m = meta.get(p["model"], {})
        if p.get("adapter") or not p.get("gap") or "params_b" not in m:
            continue
        path = (
            cfg.run_results_dir
            / "predictions"
            / f"probe__{model_slug(p['model'])}__{p['max_input_tokens']}.parquet"
        )
        if not path.exists():
            log.warning("scaling: no predictions for %s at %s", p["model"], path)
            continue
        wide = pd.read_parquet(path).pivot_table(index="id", columns="variant", values="p_up")
        if not {"original", "anonymized"} <= set(wide.columns):
            continue
        models.append((p["model"], float(m["params_b"]), m.get("family", ""), wide.dropna()))
    if len(models) < 2:
        return None
    common = set(labels["id"])
    for *_, wide in models:
        common &= set(wide.index)
    data = labels[labels["id"].isin(common)].sort_values("id")
    pre, post = data[data["split"] == "test_pre"], data[data["split"] == "test_post"]
    if len(pre) < 10 or len(post) < 10:
        return None
    y0, y1 = pre["y_up"].to_numpy(), post["y_up"].to_numpy()
    arrays = []
    for *_, wide in models:
        w0, w1 = wide.loc[pre["id"]], wide.loc[post["id"]]
        arrays.append(
            (
                w0["original"].to_numpy(),
                w0["anonymized"].to_numpy(),
                w1["original"].to_numpy(),
                w1["anonymized"].to_numpy(),
            )
        )
    x = np.log10([m[1] for m in models])
    rng = np.random.default_rng(seed)
    i0 = np.vstack(
        [np.arange(len(y0)), rng.integers(0, len(y0), (n_boot, len(y0)))]
    )  # row 0 = point estimate
    i1 = np.vstack([np.arange(len(y1)), rng.integers(0, len(y1), (n_boot, len(y1)))])
    yy0, yy1 = y0[i0], y1[i1]
    anon_pre = np.column_stack([_auc_rows(yy0, o0[i0]) - _auc_rows(yy0, a0[i0]) for o0, a0, _, _ in arrays])
    anon_post = np.column_stack([_auc_rows(yy1, o1[i1]) - _auc_rows(yy1, a1[i1]) for _, _, o1, a1 in arrays])
    did = anon_pre - anon_post  # (1 + n_boot) x models
    out = {
        "n_models": len(models),
        "n_pre": len(pre),
        "n_post": len(post),
        "n_boot": n_boot,
        "models": [
            {
                "model": name,
                "params_b": size,
                "family": fam,
                "did": float(did[0, k]),
                "did_lo": _interval(did[1:, k])[0],
                "did_hi": _interval(did[1:, k])[1],
            }
            for k, (name, size, fam, _) in enumerate(models)
        ],
    }
    blocks = {"mean_did": did.mean(axis=1)}
    if len(models) >= 3 and np.ptp(x) > 0:
        # slope of the DiD = slope of the masking effect before the cutoff - slope of the masking effect after it
        blocks |= {
            "slope_per_decade": _slopes(x, did),
            "slope_anon_effect_pre": _slopes(x, anon_pre),
            "slope_anon_effect_post": _slopes(x, anon_post),
        }
    for key, values in blocks.items():
        out[key] = float(values[0])
        out[f"{key}_lo"], out[f"{key}_hi"] = _interval(values[1:])
    if "slope_per_decade" in blocks:
        out["share_boot_slope_le_0"] = float(np.mean(blocks["slope_per_decade"][1:] <= 0))
    if len(models) >= 4:
        # leave one model out: how much does the slope rest on any single model?
        loo = []
        for k, (name, *_) in enumerate(models):
            keep = np.arange(len(models)) != k
            if np.ptp(x[keep]) == 0:
                continue
            s = _slopes(x[keep], did[:, keep])
            lo, hi = _interval(s[1:])
            loo.append({"dropped": name, "slope": float(s[0]), "slope_lo": lo, "slope_hi": hi})
        out["loo"] = loo
        out["loo_n_excluding_zero"] = sum(r["slope_lo"] > 0 or r["slope_hi"] < 0 for r in loo)
    return out
