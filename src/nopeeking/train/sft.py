"""Supervised fine-tuning with LoRA. ``data``: ``direct_label`` (answer Yes/No) or ``rationales`` (a teacher variant)."""

from __future__ import annotations

import logging

from nopeeking.prompts import direct_messages, reasoning_messages
from nopeeking.train.common import (
    filter_kwargs,
    load_policy,
    load_run,
    load_tokenizer,
    lora_config,
    save_run_card,
    train_rows,
    trainer_args,
)

log = logging.getLogger(__name__)


def build_examples(run: dict, cfg) -> list[dict]:
    rows = train_rows(cfg).set_index("id", drop=False)
    max_chars = run.get("max_chars", 6000)
    if run["data"] == "direct_label":
        return [
            {
                "prompt": direct_messages(r, max_chars=max_chars, horizon=cfg.horizon),
                "completion": [{"role": "assistant", "content": "Yes" if r["y_up"] else "No"}],
            }
            for _, r in rows.iterrows()
        ]
    from nopeeking.train.rationales import load_kept

    kept = load_kept(cfg, run["rationale_variant"])
    kept = kept[kept["id"].isin(rows.index)]
    return [
        {
            "prompt": reasoning_messages(rows.loc[k["id"]], max_chars=max_chars, horizon=cfg.horizon),
            "completion": [{"role": "assistant", "content": k["completion"]}],
        }
        for _, k in kept.iterrows()
    ]


def train(config_path: str) -> str:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    run, cfg = load_run(config_path)
    examples = build_examples(run, cfg)
    log.info("SFT %s: %d examples", run["output_dir"], len(examples))
    args = SFTConfig(
        **filter_kwargs(
            SFTConfig,
            {**trainer_args(run), "max_length": run.get("max_length", 1536), "completion_only_loss": True},
        )
    )
    trainer = SFTTrainer(
        model=load_policy(run),
        args=args,
        train_dataset=Dataset.from_list(examples),
        processing_class=load_tokenizer(run["base_model"]),
        peft_config=lora_config(run),
    )
    result = trainer.train()
    trainer.save_model(run["output_dir"])
    save_run_card(run, run["output_dir"], {"n_examples": len(examples), "train_metrics": result.metrics})
    return run["output_dir"]
