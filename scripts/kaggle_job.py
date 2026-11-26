"""Run GPU jobs on Kaggle (free 2x T4 / T4 tier) from a laptop, through the official `kaggle` CLI.

    python scripts/kaggle_job.py push-data --config configs/v1.yaml
    python scripts/kaggle_job.py push-job probe-qwen25 --config configs/v1.yaml \
        --cmd "nopeeking probe Qwen/Qwen2.5-0.5B-Instruct --config configs/v1.yaml --limit 2000"
    python scripts/kaggle_job.py status probe-qwen25
    python scripts/kaggle_job.py pull probe-qwen25 --config configs/v1.yaml

Datasets and notebooks are created private. Credentials come from ~/.kaggle/kaggle.json (never read here).
The job installs `nopeeking` from GitHub at the current commit, so push your commits before launching a job.
Only predictions come back; metrics, portfolios and reports are recomputed locally from them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nopeeking.config import load_config  # noqa: E402

REPO = "https://github.com/KvnFltr/no-peeking.git"
KAGGLE = str(Path(sys.executable).with_name("kaggle"))


def _kaggle(*args: str, check: bool = True) -> str:
    out = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if check and out.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out.stdout}\n{out.stderr}")
    return out.stdout


def _username() -> str:
    for line in _kaggle("config", "view").splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("Kaggle username not found: is ~/.kaggle/kaggle.json in place?")


def _commit() -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    pushed = subprocess.run(
        ["git", "branch", "-r", "--contains", sha], capture_output=True, text=True, cwd=ROOT
    )
    if "origin/" not in pushed.stdout:
        raise SystemExit(f"commit {sha[:8]} is not on GitHub yet: push it before launching a Kaggle job")
    return sha


def dataset_slug(cfg_name: str) -> str:
    return f"nopeek8k-{cfg_name}"


def push_data(config: str) -> None:
    cfg = load_config(config)
    user = _username()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for name in ("nopeek8k.parquet", "stats.json", "filter_log.json"):
            shutil.copy(cfg.dataset_dir / name, d / name)
        meta = {
            "title": f"NoPeek-8K {cfg.name}",
            "id": f"{user}/{dataset_slug(cfg.name)}",
            "licenses": [{"name": "other"}],
        }
        (d / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        exists = _kaggle("datasets", "status", f"{user}/{dataset_slug(cfg.name)}", check=False)
        if "ready" in exists.lower():
            print(_kaggle("datasets", "version", "-p", str(d), "-m", "update", "-q"))
        else:
            print(_kaggle("datasets", "create", "-p", str(d), "-q"))


RUNNER = """import os, subprocess, shutil, sys, glob
def sh(cmd):
    print('$', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
sh('git clone -q {repo} repo && cd repo && git checkout -q {sha}')
os.chdir('repo')
sh('pip -q install -e ".[train]" bitsandbytes')
src = glob.glob('/kaggle/input/{slug}/nopeek8k.parquet') + glob.glob('/kaggle/input/*/{slug}/nopeek8k.parquet') \\
    + glob.glob('/kaggle/input/**/nopeek8k.parquet', recursive=True)
os.makedirs('data/processed/{name}', exist_ok=True)
for f in ('nopeek8k.parquet', 'stats.json', 'filter_log.json'):
    shutil.copy(os.path.join(os.path.dirname(src[0]), f), 'data/processed/{name}/' + f)
for cmd in {cmds!r}:
    try:
        sh(cmd)
    except subprocess.CalledProcessError as err:
        print('FAILED:', cmd, err, flush=True)
shutil.copytree('results', '/kaggle/working/results', dirs_exist_ok=True)
for extra in ('outputs', 'logs'):
    if os.path.isdir(extra):
        shutil.copytree(extra, '/kaggle/working/' + extra, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('*.safetensors.tmp', 'checkpoint-*'))
"""


def push_job(job: str, config: str, cmds: list[str], gpu: str) -> None:
    cfg = load_config(config)
    user = _username()
    sha = _commit()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        script = RUNNER.format(repo=REPO, sha=sha, slug=dataset_slug(cfg.name), name=cfg.name, cmds=cmds)
        (d / "run.py").write_text(script, encoding="utf-8")
        meta = {
            "id": f"{user}/nopeeking-{job}",
            "title": f"nopeeking-{job}",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": gpu != "none",
            "enable_internet": True,
            "machine_shape": gpu if gpu != "none" else "",
            "dataset_sources": [f"{user}/{dataset_slug(cfg.name)}"],
        }
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(_kaggle("kernels", "push", "-p", str(d)))


def status(job: str) -> None:
    print(_kaggle("kernels", "status", f"{_username()}/nopeeking-{job}"))


def pull(job: str, config: str) -> None:
    cfg = load_config(config)
    out = ROOT / "outputs" / "kaggle" / job
    out.mkdir(parents=True, exist_ok=True)
    print(_kaggle("kernels", "output", f"{_username()}/nopeeking-{job}", "-p", str(out), "-o"))
    src = out / "results" / cfg.name / "predictions"
    if src.is_dir():
        dst = cfg.run_results_dir / "predictions"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.parquet"):
            shutil.copy(f, dst / f.name)
            print("predictions:", dst / f.name)
    for sub in ("rationales",):
        r = out / "results" / cfg.name / sub
        if r.is_dir():
            shutil.copytree(r, cfg.run_results_dir / sub, dirs_exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("push-data")
    a.add_argument("--config", required=True)
    b = sub.add_parser("push-job")
    b.add_argument("job")
    b.add_argument("--config", required=True)
    b.add_argument("--cmd", dest="cmds", action="append", required=True, help="shell command, repeatable")
    b.add_argument("--gpu", default="NvidiaTeslaT4", help="Kaggle machine shape, or 'none' for CPU")
    c = sub.add_parser("status")
    c.add_argument("job")
    e = sub.add_parser("pull")
    e.add_argument("job")
    e.add_argument("--config", required=True)
    args = p.parse_args()
    if args.action == "push-data":
        push_data(args.config)
    elif args.action == "push-job":
        push_job(args.job, args.config, args.cmds, args.gpu)
    elif args.action == "status":
        status(args.job)
    else:
        pull(args.job, args.config)


if __name__ == "__main__":
    main()
