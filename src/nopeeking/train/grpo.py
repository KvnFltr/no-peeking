"""GRPO with market-grounded rewards: (a) outcome, (b) outcome + grounding, (c) + format (see rewards.py).

Typically started from the SFT adapter (``init_adapter``), which is merged before a fresh LoRA is trained.
"""

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
from nopeeking.train.rewards import REWARDS

log = logging.getLogger(__name__)


def build_prompts(run: dict, cfg) -> list[dict]:
    rows = train_rows(cfg)
    if run.get("max_prompts") and len(rows) > run["max_prompts"]:
        rows = rows.sample(n=run["max_prompts"], random_state=cfg.seed)
    return [
        {
            "prompt": reasoning_messages(r, max_chars=run.get("max_chars", 4000), horizon=cfg.horizon),
            "y_up": int(r["y_up"]),
            "text": str(r["text"]),
        }
        for _, r in rows.iterrows()
    ]


def train(config_path: str) -> str:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    run, cfg = load_run(config_path)
    rewards = run.get("rewards", {"outcome": 1.0})
    prompts = build_prompts(run, cfg)
    log.info("GRPO %s: %d prompts, rewards %s", run["output_dir"], len(prompts), rewards)
    g = run.get("grpo", {})
    args = GRPOConfig(
        **filter_kwargs(
            GRPOConfig,
            {
                **trainer_args(run),
                "num_generations": g.get("num_generations", 4),
                "max_completion_length": g.get("max_completion_length", 256),
                "max_prompt_length": g.get("max_prompt_length", 1200),
                "temperature": g.get("temperature", 0.9),
                "beta": g.get("beta", 0.04),
                "reward_weights": list(rewards.values()),
                "log_completions": False,
            },
        )
    )
    trainer = GRPOTrainer(
        model=load_policy(run),
        reward_funcs=[REWARDS[k] for k in rewards],
        args=args,
        train_dataset=Dataset.from_list(prompts),
        processing_class=load_tokenizer(run["base_model"]),
        peft_config=lora_config(run),
    )
    result = trainer.train()
    trainer.save_model(run["output_dir"])
    save_run_card(run, run["output_dir"], {"n_prompts": len(prompts), "train_metrics": result.metrics})
    return run["output_dir"]
