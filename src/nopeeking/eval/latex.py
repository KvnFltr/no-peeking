"""LaTeX tables and macros for the paper, generated from ``results/`` (never edited by hand)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from nopeeking.config import SPLIT_ORDER, DataConfig

DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
SPLIT_TEX = {
    "train": "Train (2018--22)",
    "test_pre": "Pre-cutoff test (2023)",
    "val": "Validation (2024)",
    "test_post": "Post-cutoff test (2025-07+)",
}
METHOD_TEX = {
    "majority": "Majority class",
    "event_type_prior": "Event-type prior",
    "tfidf_lr": "TF-IDF + LR",
    "tfidf_lr_anon": "TF-IDF + LR (anonymized input)",
    "finbert_zero_shot": "FinBERT (zero-shot)",
}


def tex(s: str) -> str:
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        s = s.replace(a, b)
    return s


def macro_name(*parts: str) -> str:
    text = " ".join(parts)
    text = re.sub(r"\d", lambda m: " " + DIGITS[int(m.group(0))] + " ", text)
    words = re.split(r"[^A-Za-z]+", text)
    return "".join(w[:1].upper() + w[1:] for w in words if w)


def num(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def ci(block: dict, key: str, nd: int = 3) -> str:
    v = block.get(key)
    if v is None:
        return "--"
    return f"{num(v, nd)} {{\\scriptsize [{num(block.get(f'{key}_lo'), nd)}, {num(block.get(f'{key}_hi'), nd)}]}}"


def dataset_table(df: pd.DataFrame) -> str:
    rows = []
    for split in SPLIT_ORDER:
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        shares = sub["event_type"].value_counts(normalize=True)
        rows.append(
            f"{SPLIT_TEX[split]} & {len(sub):,} & {sub['cik'].nunique():,} & "
            + " & ".join(
                f"{100 * shares.get(e, 0):.0f}" for e in ("earnings", "deal", "exec_change", "other")
            )
            + f" & {sub['y_up'].mean():.3f} \\\\"
        )
    head = (
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\nSplit & Filings & Firms & Earn.\\,\\% & Deal\\,\\% & "
        "Exec.\\,\\% & Other\\,\\% & Up-rate \\\\\n\\midrule\n"
    )
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def main_table(rows: list[tuple[str, dict]]) -> str:
    lines = []
    for name, m in rows:
        t = m.get("test_post")
        if not t:
            continue
        port = m.get("portfolio", {})
        lines.append(
            f"{tex(METHOD_TEX.get(name, name))} & {ci(t, 'auc')} & {ci(t, 'brier')} & {num(t.get('ece'))} & "
            f"{ci(t, 'rank_ic')} & {ci(port, 'sharpe', 2) if port else '--'} \\\\"
        )
    head = (
        "\\begin{tabular}{lccccc}\n\\toprule\nMethod & AUC & Brier & ECE & Rank IC & Sharpe \\\\\n\\midrule\n"
    )
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


def probe_table(gaps: list[tuple[str, dict]]) -> str:
    lines = [
        f"{tex(n)} & {ci(g, 'pre_orig')} & {ci(g, 'pre_anon')} & {ci(g, 'post_orig')} & "
        f"{ci(g, 'post_anon')} & {ci(g, 'did')} \\\\"
        for n, g in gaps
    ]
    head = (
        "\\begin{tabular}{lccccc}\n\\toprule\n& \\multicolumn{2}{c}{Pre-cutoff (2023)} & "
        "\\multicolumn{2}{c}{Post-cutoff (2025-07+)} & \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "Model & Original & Anonymized & Original & Anonymized & DiD \\\\\n\\midrule\n"
    )
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


def posttrain_table(trained: list[dict]) -> str:
    lines = []
    for t in trained:
        m = t["metrics"].get("test_post", {})
        lines.append(
            f"{tex(t['run'])} & {ci(m, 'auc')} & {ci(m, 'brier')} & {num(m.get('ece'))} & "
            f"{num(t.get('parse_failure_rate'), 2)} & {num(t.get('fact_precision_mean'), 2)} & "
            f"{num(t.get('leak_rate'), 2)} \\\\"
        )
    head = (
        "\\begin{tabular}{lcccccc}\n\\toprule\nRun & AUC & Brier & ECE & Parse fail. & Fact prec. & Leak \\\\\n"
        "\\midrule\n"
    )
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


def filterlog_table(flog: list[dict]) -> str:
    def fmt(x: int | None) -> str:
        return "" if x is None else f"{x:,}"

    lines = [f"{tex(s['step'])} & {fmt(s['n_before'])} & {fmt(s['n_after'])} \\\\" for s in flog]
    head = "\\begin{tabular}{lrr}\n\\toprule\nStep & Before & After \\\\\n\\midrule\n"
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


def prompts_tex() -> str:
    from nopeeking.prompts import REASONING_INSTRUCTIONS, SYSTEM, direct_question

    def box(title: str, body: str) -> str:
        # \small monospace fits ~84 characters in the 5.5in NeurIPS text width; wrap below that
        return (
            f"\\paragraph{{{title}}}\n{{\\small\n\\begin{{verbatim}}\n{_wrap(body)}\n\\end{{verbatim}}\n}}\n"
        )

    return (
        box("System prompt", SYSTEM)
        + box("Direct question (probe, direct-label SFT)", direct_question(3))
        + box("Reasoning instructions (rationales, reasoning students)", REASONING_INSTRUCTIONS)
    )


def _wrap(text: str, width: int = 80) -> str:
    import textwrap

    return "\n".join(textwrap.fill(line, width) if line else "" for line in text.split("\n"))


def write_latex(
    cfg: DataConfig,
    method_rows: list[tuple[str, dict]],
    gaps: list[tuple[str, dict]],
    trained: list[dict],
    stats: dict,
    flog: list[dict],
    extra: dict[str, str] | None = None,
) -> list[Path]:
    out = cfg.run_results_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cfg.dataset_path, columns=["split", "cik", "event_type", "y_up"])
    tables = {
        "dataset": dataset_table(df),
        "main": main_table(method_rows),
        "probe": probe_table(gaps),
        "posttrain": posttrain_table(trained),
        "filterlog": filterlog_table(flog),
        "prompts": prompts_tex(),
    }
    written = []
    for name, body in tables.items():
        path = out / f"{name}.tex"
        path.write_text("% Auto-generated by `nopeeking report`. Do not edit.\n" + body, encoding="utf-8")
        written.append(path)

    macros: dict[str, str] = {}

    def add(name: str, value: str) -> None:
        macros.setdefault(macro_name(name), value)

    add("N Filings", f"{stats['n']:,}")
    add("N Firms", f"{df['cik'].nunique():,}")
    for split, n in stats["splits"].items():
        add(f"N {split}", f"{n:,}")
    pool = next((s["n_after"] for s in flog if s["step"].startswith("one filing per firm")), None)
    if pool is not None:
        add("N Pool", f"{pool:,}")
    for name, m in method_rows:
        for split in ("test_post", "test_pre", "val"):
            for metric in ("auc", "auc_lo", "auc_hi", "acc", "brier", "ece", "rank_ic"):
                if split in m and metric in m[split]:
                    add(f"{name} {split} {metric}", num(m[split][metric]))
        if "portfolio" in m and "sharpe" in m["portfolio"]:
            for k in ("sharpe", "sharpe_lo", "sharpe_hi"):
                add(f"{name} {k}", num(m["portfolio"][k], 2))
    for name, g in gaps:
        for k, v in g.items():
            add(f"{name} gap {k}", num(v))
    add("Post Train N Runs", str(len(trained)))
    for t in trained:
        for k in ("parse_failure_rate", "fact_precision_mean", "leak_rate"):
            if t.get(k) is not None:
                add(f"{t['run']} {k}", num(t[k]))
    for name, value in (extra or {}).items():
        add(name, value)
    path = cfg.run_results_dir / "macros.tex"
    body = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items())
    path.write_text(
        "% Auto-generated by `nopeeking report`. Do not edit by hand.\n" + body + "\n", encoding="utf-8"
    )
    written.append(path)
    return written


def load_trained(cfg: DataConfig) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((cfg.run_results_dir / "methods").glob("*.json"))
    ]
