"""Shared helpers for the training scripts: YAML run configs, LoRA, precision, TRL-version-tolerant configs."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from nopeeking.config import DataConfig, load_config

log = logging.getLogger(__name__)


def load_run(path: str | Path) -> tuple[dict, DataConfig]:
    run = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    run["_path"] = str(path)
    return run, load_config(run["dataset_config"])


def filter_kwargs(config_cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only arguments the installed TRL version knows (its configs change between releases)."""
    names = {f.name for f in dataclasses.fields(config_cls)}
    dropped = sorted(set(kwargs) - names)
    if dropped:
        log.warning("%s: ignoring unsupported arguments %s", config_cls.__name__, dropped)
    return {k: v for k, v in kwargs.items() if k in names}


def precision_kwargs() -> dict[str, bool]:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {"bf16": True} if torch.cuda.is_bf16_supported() else {"fp16": True}


def lora_config(run: dict):
    from peft import LoraConfig

    lc = run.get("lora", {})
    return LoraConfig(
        r=lc.get("r", 16),
        lora_alpha=lc.get("alpha", 32),
        lora_dropout=lc.get("dropout", 0.05),
        target_modules=lc.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )


def trainer_args(run: dict) -> dict[str, Any]:
    t = run.get("train", {})
    return {
        "output_dir": run["output_dir"],
        "learning_rate": t.get("learning_rate", 1e-4),
        "per_device_train_batch_size": t.get("batch_size", 2),
        "gradient_accumulation_steps": t.get("grad_accum", 8),
        "num_train_epochs": t.get("epochs", 1),
        "max_steps": t.get("max_steps", -1),
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "logging_steps": 5,
        "save_strategy": "no",
        "report_to": "none",
        "seed": t.get("seed", 0),
        "gradient_checkpointing": t.get("gradient_checkpointing", True),
        **precision_kwargs(),
    }


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy(run: dict):
    """Base model, optionally with a previous LoRA adapter merged in (e.g. GRPO on top of SFT)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(run["base_model"], dtype="auto")
    if run.get("init_adapter"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, run["init_adapter"]).merge_and_unload()
    return model


def train_rows(cfg: DataConfig, splits: tuple[str, ...] = ("train",)) -> pd.DataFrame:
    df = pd.read_parquet(cfg.dataset_path)
    return df[df["split"].isin(splits)].reset_index(drop=True)


def save_run_card(run: dict, out_dir: str | Path, extra: dict | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    card = {k: v for k, v in run.items() if not k.startswith("_")} | (extra or {})
    (out / "run_config.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
