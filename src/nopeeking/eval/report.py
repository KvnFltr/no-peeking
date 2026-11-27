"""Everything a human reads is generated here from ``results/`` -- never typed by hand.

Outputs: ``results/<name>/summary.md``, ``results/<name>/macros.tex`` (for the paper),
``results/<name>/figures/hero.png`` and the README block between the RESULTS markers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from nopeeking.config import DataConfig

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
SCALING_FILE = "scaling.json"

# Reference categorical palette (validated: CVD ΔE 24.7, contrast >= 3:1 on #fcfcfb).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

SPLIT_LABELS = {
    "train": "train (2018-22)",
    "test_pre": "pre-cutoff test (2023)",
    "val": "val (2024)",
    "test_post": "post-cutoff test (2025-07+)",
}


def _f(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def _ci(block: dict, key: str, nd: int = 3) -> str:
    return f"{_f(block.get(key), nd)} [{_f(block.get(f'{key}_lo'), nd)}, {_f(block.get(f'{key}_hi'), nd)}]"


def _load(cfg: DataConfig) -> tuple[dict, dict, dict[str, dict], list[dict]]:
    root = cfg.run_results_dir
    stats = json.loads((cfg.dataset_dir / "stats.json").read_text(encoding="utf-8"))
    flog = json.loads((cfg.dataset_dir / "filter_log.json").read_text(encoding="utf-8"))
    base_path = root / "baselines.json"
    baselines = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    probes = {
        p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "probe").glob("*.json"))
    }
    trained = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "methods").glob("*.json"))]
    if trained:  # post-trained adapters (nopeeking eval-adapter) are listed after the baselines
        baselines = {
            **baselines,
            "methods": {
                **baselines.get("methods", {}),
                **{f"{t['run']} (post-trained)": t["metrics"] for t in trained},
            },
        }
    return stats, baselines, probes, flog


def _method_rows(baselines: dict, probes: dict[str, dict]) -> list[tuple[str, dict]]:
    rows = [(name, m) for name, m in baselines.get("methods", {}).items()]
    rows += [(f"{p['model']} (zero-shot)", p["metrics"]["original"]) for p in probes.values()]
    return rows


def _model_meta() -> dict[str, dict]:
    path = Path("configs/models.yaml")
    if not path.exists():
        return {}
    import yaml

    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("models", {})


def leaderboard_md(baselines: dict, probes: dict[str, dict]) -> list[str]:
    """Models ranked by post-cutoff AUC (the number that cannot be inflated by memory), with their DiD."""
    meta = _model_meta()
    rows = []
    for p in probes.values():
        post = p["metrics"]["original"].get("test_post", {})
        if not post:
            continue
        m = meta.get(p["model"], {})
        gap = p.get("gap", {}).get("auc", {})
        rows.append((post.get("auc", float("nan")), p["model"], m, post, gap))
    tfidf = baselines.get("methods", {}).get("tfidf_lr", {}).get("test_post")
    if tfidf:
        rows.append(
            (
                tfidf["auc"],
                "TF-IDF + LR (trained on 2018-22)",
                {},
                tfidf,
                baselines.get("tfidf_lr_gap", {}).get("auc", {}),
            )
        )
    if not rows:
        return []
    rows.sort(key=lambda r: -r[0] if r[0] == r[0] else 0)
    out = [
        "### Leaderboard: post-cutoff test",
        "",
        "Ranked by AUC on filings published after every model's release (unseen by construction). "
        "DiD > 0 means the model does better on pre-cutoff filings only when it can see who and when: recall.",
        "",
        "| # | model | released | params | AUC post-cutoff [95% CI] | Brier | DiD (memorization) [95% CI] |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for i, (_, name, m, post, gap) in enumerate(rows, 1):
        params = f"{m['params_b']}B" if "params_b" in m else ""
        out.append(
            f"| {i} | {name} | {m.get('released', '')} | {params} | {_ci(post, 'auc')} | {_f(post.get('brier'))} "
            f"| {_ci(gap, 'did') if gap else '–'} |"
        )
    return [*out, ""]


def markdown_summary(cfg: DataConfig) -> str:
    stats, baselines, probes, flog = _load(cfg)
    lines = leaderboard_md(baselines, probes)
    lines += [
        f"Dataset `{cfg.name}`: {stats['n']} filings, {stats['first_filing']} to {stats['last_filing']}. "
        f"Split sizes: " + ", ".join(f"{k} {v}" for k, v in stats["splits"].items()) + ".",
        "",
    ]
    lines += ["**Filter log**", "", "| step | before | after | note |", "|---|---:|---:|---|"]
    lines += [
        f"| {s['step']} | {'' if s['n_before'] is None else s['n_before']} | {s['n_after']} | {s['note']} |"
        for s in flog
    ]
    lines += [
        "",
        "**Share of filings followed by a 3-day outperformance (y_up = 1)**: "
        + ", ".join(f"{k} {v:.3f}" for k, v in stats["up_rate_by_split"].items()),
        "",
    ]

    for split in ("test_post", "test_pre", "val"):
        rows = [(n, m[split]) for n, m in _method_rows(baselines, probes) if split in m]
        if not rows:
            continue
        lines += [
            f"**{SPLIT_LABELS[split]}**: metric [95% bootstrap CI]",
            "",
            "| method | n | AUC | balanced acc. | acc. | Brier | ECE | rank IC |",
            "|---|---:|---|---|---|---|---:|---|",
        ]
        lines += [
            f"| {n} | {m['n']} | {_ci(m, 'auc')} | {_ci(m, 'bal_acc')} | {_ci(m, 'acc')} | "
            f"{_ci(m, 'brier')} | {_f(m['ece'])} | {_ci(m, 'rank_ic')} |"
            for n, m in rows
        ]
        lines.append("")

    gaps = [("TF-IDF + LR (control, trained <= 2022)", baselines.get("tfidf_lr_gap", {}))]
    gaps += [(p["model"], p.get("gap", {})) for p in probes.values()]
    gaps = [(n, g) for n, g in gaps if g]
    if gaps:
        lines += [
            "**Look-ahead probe (AUC)**: DiD = (pre orig - pre anon) - (post orig - post anon); "
            "> 0 suggests recall of pre-cutoff outcomes.",
            "",
            "| model | pre orig | pre anon | post orig | post anon | pre - post (orig) | DiD |",
            "|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {n} | {_ci(g['auc'], 'pre_orig')} | {_ci(g['auc'], 'pre_anon')} | {_ci(g['auc'], 'post_orig')} | "
            f"{_ci(g['auc'], 'post_anon')} | {_ci(g['auc'], 'pre_minus_post_orig')} | {_ci(g['auc'], 'did')} |"
            for n, g in gaps
        ]
        lines.append("")
    lines += scaling_md(cfg)

    ports = [(n, m["portfolio"]) for n, m in _method_rows(baselines, probes) if "portfolio" in m]
    if ports:
        lines += [
            "**Long-short on post-cutoff test** (top vs bottom tercile of score, cut-offs from val, "
            f"{ports[0][1]['cost_bps_one_way']:.0f} bps per side, market-hedged)",
            "",
            "| method | positions | active days | ann. return | ann. vol | Sharpe [95% CI] |",
            "|---|---:|---:|---:|---:|---|",
        ]
        lines += [
            f"| {n} | {p['n_positions']} | {p.get('n_active_days', 0)} | {_f(p.get('ann_return'))} | "
            f"{_f(p.get('ann_vol'))} | {_ci(p, 'sharpe', 2)} |"
            for n, p in ports
        ]
        lines.append("")
    return "\n".join(lines)


def _scaling(cfg: DataConfig) -> dict | None:
    path = cfg.run_results_dir / SCALING_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def scaling_md(cfg: DataConfig) -> list[str]:
    sc = _scaling(cfg)
    if not sc:
        return []
    out = [
        f"**Memorization vs. model size** (exploratory, not pre-registered): one joint bootstrap over the "
        f"{sc['n_models']} zero-shot models, which all score the same {sc['n_pre']} + {sc['n_post']} filings.",
        "",
        "| quantity | estimate [95% CI] |",
        "|---|---|",
        f"| mean DiD over models | {_ci(sc, 'mean_did')} |",
    ]
    if "slope_per_decade" in sc:
        out += [
            f"| slope of DiD per 10x parameters | {_ci(sc, 'slope_per_decade')} |",
            f"| ... from the masking effect before the cutoff | {_ci(sc, 'slope_anon_effect_pre')} |",
            f"| ... minus the masking effect after the cutoff | {_ci(sc, 'slope_anon_effect_post')} |",
        ]
    return [*out, ""]


def summary_macros(baselines: dict, probes: dict[str, dict], scaling: dict | None) -> dict[str, str]:
    """Aggregates the paper's text refers to, so that its sentences stay true when models are added."""
    from nopeeking.eval.latex import num, tex

    zero_shot = [p for p in probes.values() if not p.get("adapter") and p.get("gap")]
    out = {"Probe N Models": str(len(zero_shot))}
    tfidf = baselines.get("methods", {}).get("tfidf_lr", {}).get("test_post", {})
    posts = [(p["model"], p["metrics"]["original"]["test_post"]) for p in zero_shot]
    if posts:
        name, best = max(posts, key=lambda r: r[1]["auc"])
        out |= {
            "Best LLM": tex(name.split("/")[-1]),
            "Best LLM test post auc": num(best["auc"]),
            "Best LLM test post auc lo": num(best["auc_lo"]),
            "Best LLM test post auc hi": num(best["auc_hi"]),
            "N LLM did above zero": str(sum(p["gap"]["auc"]["did_lo"] > 0 for p in zero_shot)),
        }
        if tfidf:
            out["N LLM above tfidf"] = str(sum(m["auc"] > tfidf["auc"] for _, m in posts))
    for k, v in (scaling or {}).items():
        if isinstance(v, float):
            out[f"Scaling {k}"] = num(v)
        elif k == "n_models":
            out["Scaling N Models"] = str(v)
    return out


def scaling_figure(cfg: DataConfig, path: Path) -> Path | None:
    """DiD against parameter count (log scale), one colour per family, with the TF-IDF control as a band."""
    sc = _scaling(cfg)
    if not sc:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _, baselines, _, _ = _load(cfg)
    families = sorted({m["family"] for m in sc["models"]})
    palette = [BLUE, ORANGE, "#1f9d6b", "#8a5cd1", "#c0398b", INK2]
    color = {f: palette[i % len(palette)] for i, f in enumerate(families)}
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ctrl = baselines.get("tfidf_lr_gap", {}).get("auc")
    if ctrl:
        ax.axhspan(ctrl["did_lo"], ctrl["did_hi"], color=GRID, lw=0, label="TF-IDF control (95% CI)")
    ax.axhline(0, color=AXIS, lw=1, ls="--")
    for k, fam in enumerate(families):
        pts = sorted((m for m in sc["models"] if m["family"] == fam), key=lambda m: m["params_b"])
        dodge = 1 + 0.035 * (k - (len(families) - 1) / 2)  # families of equal size side by side, not on top
        xs = np.array([m["params_b"] for m in pts]) * dodge
        ys = np.array([m["did"] for m in pts])
        err = np.array([[m["did"] - m["did_lo"] for m in pts], [m["did_hi"] - m["did"] for m in pts]])
        ax.errorbar(
            xs, ys, yerr=err, fmt="o-", color=color[fam], ms=7, lw=1.5, capsize=0, mec=SURFACE, label=fam
        )
    ax.set_xscale("log")
    sizes = sorted({m["params_b"] for m in sc["models"]})
    ax.set_xticks(sizes, [f"{s:g}B" for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Parameters (log scale)", color=INK2)
    ax.set_ylabel("DiD of AUC (> 0: recall)", color=INK2)
    if "slope_per_decade" in sc:
        ax.set_title(
            f"Slope per 10x parameters: {sc['slope_per_decade']:+.3f} "
            f"[{sc['slope_per_decade_lo']:+.3f}, {sc['slope_per_decade_hi']:+.3f}]",
            loc="left",
            color=INK,
            fontsize=10,
        )
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def hero_figure(cfg: DataConfig, path: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, baselines, probes, _ = _load(cfg)
    entries = []
    if baselines.get("tfidf_lr_gap"):
        entries.append(("TF-IDF + LR\n(control)", baselines["tfidf_lr_gap"]["auc"]))
    entries += [(p["model"].split("/")[-1], p["gap"]["auc"]) for p in probes.values() if p.get("gap")]
    if not entries:
        return None
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 0.7 * len(entries) + 1.8), sharey=True, sharex=True, facecolor=SURFACE
    )
    for ax, variant, title in ((axes[0], "orig", "Original text"), (axes[1], "anon", "Anonymized text")):
        ax.set_facecolor(SURFACE)
        for i, (_, g) in enumerate(entries):
            for dy, key, color, label in (
                (-0.15, "pre", BLUE, "Before cutoff (2023)"),
                (0.15, "post", ORANGE, "After cutoff (2025-07+)"),
            ):
                k = f"{key}_{variant}"
                v, lo, hi = g[k], g[f"{k}_lo"], g[f"{k}_hi"]
                ax.plot([lo, hi], [i + dy, i + dy], color=color, lw=2, solid_capstyle="round")
                ax.plot(
                    v, i + dy, "o", ms=8, color=color, mec=SURFACE, mew=2, label=label if i == 0 else None
                )
        ax.axvline(0.5, color=AXIS, lw=1, ls="--")
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("AUC, 95% bootstrap CI (0.5 = chance)")
        ax.grid(axis="x", color=GRID, lw=0.6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(entries)), [e[0] for e in entries], color=INK2)
    axes[0].set_ylim(len(entries) - 0.5, -0.5)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        f"Does skill vanish after the training cutoff? ({cfg.name} set)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    try:  # vector copy for the paper; the PNG is enough if the PDF backend cannot load
        fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    except Exception as err:
        import logging

        logging.getLogger(__name__).warning("hero.pdf not written (%r); the paper uses hero.png", err)
    plt.close(fig)
    return path


def build_report(cfg: DataConfig, readme: Path | None = None) -> list[Path]:
    from nopeeking.eval.scaling import scaling_analysis

    root = cfg.run_results_dir
    root.mkdir(parents=True, exist_ok=True)
    _, _, probes0, _ = _load(cfg)
    scaling = scaling_analysis(cfg, probes0, _model_meta(), seed=cfg.seed)
    if scaling:
        (root / SCALING_FILE).write_text(json.dumps(scaling, indent=2), encoding="utf-8")
    summary = markdown_summary(cfg)
    written = []
    (root / "summary.md").write_text(summary, encoding="utf-8")
    written.append(root / "summary.md")
    from nopeeking.eval.latex import write_latex

    stats, baselines, probes, flog = _load(cfg)
    gaps = (
        [("TF-IDF + LR (control)", baselines["tfidf_lr_gap"]["auc"])] if baselines.get("tfidf_lr_gap") else []
    )
    gaps += [(p["model"], p["gap"]["auc"]) for p in probes.values() if p.get("gap")]
    trained = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "methods").glob("*.json"))]
    written += write_latex(
        cfg,
        _method_rows(baselines, probes),
        gaps,
        trained,
        stats,
        flog,
        summary_macros(baselines, probes, scaling),
    )
    for fig in (
        hero_figure(cfg, root / "figures" / "hero.png"),
        scaling_figure(cfg, root / "figures" / "scaling.png"),
    ):
        if fig:
            written.append(fig)
    if readme and readme.exists():
        text = readme.read_text(encoding="utf-8")
        if START in text and END in text:
            head, rest = text.split(START, 1)
            _, tail = rest.split(END, 1)
            block = f"{START}\n<!-- generated by `nopeeking report --config ...`; do not edit -->\n\n{summary}\n{END}"
            readme.write_text(head + block + tail, encoding="utf-8")
            written.append(readme)
    return written
