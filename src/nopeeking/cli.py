"""nopeeking: data build | baselines | probe <model> | report | rationales | train sft|dpo|grpo | eval-adapter"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(name: str, verbose: bool) -> None:
    Path("logs").mkdir(exist_ok=True)
    handlers: list[logging.Handler] = [logging.FileHandler(f"logs/{name}.log", encoding="utf-8")]
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    handlers.append(console)
    logging.basicConfig(
        level=logging.INFO, handlers=handlers, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("urllib3", "yfinance", "peewee", "filelock", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _cmd_data_build(args: argparse.Namespace) -> None:
    from nopeeking.config import load_config
    from nopeeking.data.build import build_dataset

    cfg = load_config(args.config)
    if args.end:
        cfg.end = args.end
    df = build_dataset(cfg)
    print(f"dataset: {cfg.dataset_path} ({len(df)} rows)")
    print(df.groupby("split")["y_up"].agg(["size", "mean"]).round(3).to_string())


def _cmd_baselines(args: argparse.Namespace) -> None:
    from nopeeking.config import load_config
    from nopeeking.eval.baselines import run_baselines

    cfg = load_config(args.config)
    out = run_baselines(cfg, finbert=args.finbert)
    print(f"wrote {out}")


def _cmd_probe(args: argparse.Namespace) -> None:
    from nopeeking.config import load_config
    from nopeeking.eval.probe import run_probe

    cfg = load_config(args.config)
    out = run_probe(
        cfg,
        args.model,
        splits=args.splits,
        limit=args.limit,
        device=args.device,
        max_input_tokens=args.max_input_tokens,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    print(f"wrote {out}")


def _cmd_report(args: argparse.Namespace) -> None:
    from nopeeking.config import load_config
    from nopeeking.eval.report import build_report

    cfg = load_config(args.config)
    paths = build_report(cfg, readme=Path(args.readme) if args.readme else None)
    for p in paths:
        print(f"wrote {p}")


def _cmd_rationales(args: argparse.Namespace) -> None:
    from nopeeking.train.rationales import generate_rationales

    print(f"wrote {generate_rationales(args.config)}")


def _cmd_train(args: argparse.Namespace) -> None:
    if args.method == "sft":
        from nopeeking.train.sft import train
    elif args.method == "dpo":
        from nopeeking.train.dpo import train
    else:
        from nopeeking.train.grpo import train
    print(f"saved adapter to {train(args.config)}")


def _cmd_eval_adapter(args: argparse.Namespace) -> None:
    from nopeeking.train.evaluate import evaluate_adapter

    print(f"wrote {evaluate_adapter(args.config, splits=tuple(args.splits))}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nopeeking", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true", help="log progress to the console too")
    sub = p.add_subparsers(dest="cmd", required=True)

    data = sub.add_parser("data", help="dataset commands")
    data_sub = data.add_subparsers(dest="data_cmd", required=True)
    build = data_sub.add_parser("build", help="build (or refresh) the dataset from EDGAR + prices")
    build.add_argument("--config", default="configs/pilot.yaml")
    build.add_argument("--end", default=None, help="override the config end date (YYYY-MM-DD)")
    build.set_defaults(func=_cmd_data_build, log_name="data_build")

    base = sub.add_parser("baselines", help="majority, TF-IDF+LR, (FinBERT) on all splits")
    base.add_argument("--config", default="configs/pilot.yaml")
    base.add_argument("--finbert", action="store_true", help="also run ProsusAI/finbert (needs [llm] extra)")
    base.set_defaults(func=_cmd_baselines, log_name="baselines")

    probe = sub.add_parser("probe", help="look-ahead-bias probe of any Hugging Face causal LM")
    probe.add_argument("model", help="e.g. Qwen/Qwen2.5-0.5B-Instruct")
    probe.add_argument("--config", default="configs/pilot.yaml")
    probe.add_argument("--splits", nargs="+", default=["test_pre", "test_post", "val"])
    probe.add_argument("--limit", type=int, default=None, help="max filings per split (seeded sample)")
    probe.add_argument("--device", default=None, help="cpu, cuda, auto (default: cuda if available)")
    probe.add_argument("--dtype", default="auto", help="auto, float32, bfloat16, float16, 4bit")
    probe.add_argument("--max-input-tokens", type=int, default=768)
    probe.add_argument("--batch-size", type=int, default=4)
    probe.set_defaults(func=_cmd_probe, log_name="probe")

    rep = sub.add_parser("report", help="tables, hero figure, LaTeX macros, README results block")
    rep.add_argument("--config", default="configs/pilot.yaml")
    rep.add_argument("--readme", default="README.md")
    rep.set_defaults(func=_cmd_report, log_name="report")

    rat = sub.add_parser("rationales", help="teacher writes hindsight (or control) rationales")
    rat.add_argument("--config", default="configs/train/rationales.yaml")
    rat.set_defaults(func=_cmd_rationales, log_name="rationales")

    tr = sub.add_parser("train", help="post-train a student with LoRA")
    tr.add_argument("method", choices=["sft", "dpo", "grpo"])
    tr.add_argument("--config", required=True)
    tr.set_defaults(func=_cmd_train, log_name="train")

    ev = sub.add_parser("eval-adapter", help="evaluate a trained adapter on val + post-cutoff test")
    ev.add_argument("--config", required=True)
    ev.add_argument("--splits", nargs="+", default=["val", "test_post"])
    ev.set_defaults(func=_cmd_eval_adapter, log_name="eval_adapter")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _setup_logging(args.log_name, args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
