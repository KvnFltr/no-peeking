"""Publish NoPeek-8K to the Hugging Face Hub (after `hf auth login`).

    python scripts/publish_hf.py --config configs/v1.yaml --repo <hf-user>/NoPeek-8K

Uploads one parquet file per split (train / test_pre / validation / test_post), the dataset card, the filter log
and the build stats. Raw stock and index returns are dropped: only market-adjusted returns and targets are shipped.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nopeeking.config import load_config  # noqa: E402

HF_SPLITS = {"train": "train", "test_pre": "test_pre", "val": "validation", "test_post": "test_post"}
DROP = ["ret_1d", "ret_3d"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/v1.yaml")
    p.add_argument("--repo", required=True, help="e.g. your-hf-user/NoPeek-8K")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    from huggingface_hub import HfApi

    cfg = load_config(args.config)
    df = pd.read_parquet(cfg.dataset_path).drop(columns=DROP, errors="ignore")
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "data").mkdir()
        for split, hf_name in HF_SPLITS.items():
            sub = df[df["split"] == split]
            if len(sub):
                sub.to_parquet(d / "data" / f"{hf_name}.parquet", index=False, compression="zstd")
        card = (ROOT / "data" / "DATASET_CARD.md").read_text(encoding="utf-8")
        configs = "configs:\n- config_name: default\n  data_files:\n" + "".join(
            f"  - split: {n}\n    path: data/{n}.parquet\n" for n in HF_SPLITS.values()
        )
        card = card.replace("pretty_name: NoPeek-8K\n", f"pretty_name: NoPeek-8K\n{configs}", 1)
        (d / "README.md").write_text(card, encoding="utf-8")
        for name in ("filter_log.json", "stats.json"):
            (d / name).write_bytes((cfg.dataset_dir / name).read_bytes())
        api.upload_folder(
            folder_path=str(d), repo_id=args.repo, repo_type="dataset", commit_message=f"NoPeek-8K {cfg.name}"
        )
    print(f"https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
