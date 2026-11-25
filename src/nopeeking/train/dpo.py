"""DPO on (correct vs wrong) rationale pairs: chosen = hindsight rationale for the realized outcome,
rejected = rationale written for the opposite outcome. Both passed the same filters."""

from __future__ import annotations

import logging

from nopeeking.prompts import reasoning_messages
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


def build_pairs(run: dict, cfg) -> list[dict]:
    from nopeeking.train.rationales import load_kept

    rows = train_rows(cfg).set_index("id", drop=False)
    chosen = load_kept(cfg, "hindsight").set_index("id")["completion"]
    rejected = load_kept(cfg, "flipped").set_index("id")["completion"]
    ids = [i for i in chosen.index.intersection(rejected.index) if i in rows.index]
    return [
        {
            "prompt": reasoning_messages(
                rows.loc[i], max_chars=run.get("max_chars", 6000), horizon=cfg.horizon
            ),
            "chosen": [{"role": "assistant", "content": chosen[i]}],
            "rejected": [{"role": "assistant", "content": rejected[i]}],
        }
        for i in ids
    ]


def train(config_path: str) -> str:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    run, cfg = load_run(config_path)
    pairs = build_pairs(run, cfg)
    log.info("DPO %s: %d pairs", run["output_dir"], len(pairs))
    args = DPOConfig(
        **filter_kwargs(
            DPOConfig,
            {
                **trainer_args(run),
                "beta": run.get("beta", 0.1),
                "max_length": run.get("max_length", 1536),
                "max_prompt_length": run.get("max_prompt_length", 1200),
            },
        )
    )
    trainer = DPOTrainer(
        model=load_policy(run),
        args=args,
        train_dataset=Dataset.from_list(pairs),
        processing_class=load_tokenizer(run["base_model"]),
        peft_config=lora_config(run),
    )
    result = trainer.train()
    trainer.save_model(run["output_dir"])
    save_run_card(run, run["output_dir"], {"n_pairs": len(pairs), "train_metrics": result.metrics})
    return run["output_dir"]
