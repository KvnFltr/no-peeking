"""Classification / calibration / ranking metrics with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import rankdata, spearmanr


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC via the rank-sum formula (ties get average ranks); NaN if only one class."""
    y = np.asarray(y, dtype=int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if y.min() == y.max():
        return float("nan")
    tpr = (pred[y == 1] == 1).mean()
    tnr = (pred[y == 0] == 0).mean()
    return float((tpr + tnr) / 2)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on P(up)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def calibration_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return [
        {
            "bin": b,
            "n": int((idx == b).sum()),
            "mean_p": float(p[idx == b].mean()),
            "frac_up": float(y[idx == b].mean()),
        }
        for b in range(n_bins)
        if (idx == b).any()
    ]


def rank_ic(p: np.ndarray, ret: np.ndarray) -> float:
    if len(p) < 3 or np.allclose(p, p[0]):
        return float("nan")
    return float(spearmanr(p, ret).statistic)


def point_metrics(y: np.ndarray, p: np.ndarray, ret: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    out = {
        "n": len(y),
        "acc": float((pred == y).mean()),
        "bal_acc": balanced_accuracy(y, pred),
        "auc": auc(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece(y, p),
        "up_rate": float(y.mean()),
        "pred_up_rate": float(pred.mean()),
    }
    if ret is not None:
        out["rank_ic"] = rank_ic(p, np.asarray(ret, dtype=float))
    return out


def bootstrap_ci(
    fn: Callable[..., float], arrays: list[np.ndarray], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = fn(*(a[idx] for a in arrays))
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def metrics_with_ci(
    y: np.ndarray, p: np.ndarray, ret: np.ndarray | None = None, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    out: dict[str, float] = point_metrics(y, p, ret)
    fns: dict[str, tuple[Callable[..., float], list[np.ndarray]]] = {
        "acc": (lambda a, b: float(((b >= 0.5).astype(int) == a).mean()), [y, p]),
        "bal_acc": (lambda a, b: balanced_accuracy(a, (b >= 0.5).astype(int)), [y, p]),
        "auc": (auc, [y, p]),
        "brier": (lambda a, b: float(np.mean((b - a) ** 2)), [y, p]),
    }
    if ret is not None:
        fns["rank_ic"] = (lambda b, r: rank_ic(b, r), [p, np.asarray(ret, dtype=float)])
    for name, (fn, arrs) in fns.items():
        lo, hi = bootstrap_ci(fn, arrs, n_boot=n_boot, seed=seed)
        out[f"{name}_lo"], out[f"{name}_hi"] = lo, hi
    return out
