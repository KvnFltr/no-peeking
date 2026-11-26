"""Download the prediction artifacts of one or more `Probe (CPU)` workflow runs, merge them, and score them.

    python scripts/merge_probe_run.py <run-id> [<rerun-id> ...] --model Qwen/Qwen2.5-0.5B-Instruct         --limit 1000 --max-input-tokens 512

Several run ids merge a run with the reruns of its failed shards (workflow input `only`).

Scoring reuses `nopeeking probe` with every prediction cached, so no model is loaded locally.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nopeeking.config import load_config  # noqa: E402
from nopeeking.eval.probe import model_slug, run_probe  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_ids", nargs="+")
    p.add_argument("--model", required=True)
    p.add_argument("--config", default="configs/v1.yaml")
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--max-input-tokens", type=int, required=True)
    p.add_argument("--repo", default="KvnFltr/no-peeking")
    args = p.parse_args()
    cfg = load_config(args.config)
    name = f"probe__{model_slug(args.model)}__{args.max_input_tokens}.parquet"
    parts = []
    for run_id in args.run_ids:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["gh", "run", "download", run_id, "-R", args.repo, "-D", tmp], check=True)
            found = [pd.read_parquet(f) for f in Path(tmp).rglob(name)]
        if not found:
            raise SystemExit(f"no {name} in run {run_id}")
        parts += found
    merged = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["id", "variant"])
    out = cfg.run_results_dir / "predictions" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        merged = pd.concat([pd.read_parquet(out), merged]).drop_duplicates(subset=["id", "variant"])
    merged.to_parquet(out, index=False)
    print(f"{len(merged)} predictions -> {out}")
    print(
        run_probe(cfg, args.model, limit=args.limit, max_input_tokens=args.max_input_tokens, cached_only=True)
    )


if __name__ == "__main__":
    main()
