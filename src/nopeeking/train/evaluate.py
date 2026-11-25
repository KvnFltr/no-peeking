"""Evaluate a trained adapter on val + post-cutoff test.

Direct-label adapters are scored with one forward pass (P(Yes)); reasoning adapters generate an answer that is
parsed. Unparsable answers get p = 0.5 and are counted, so format failures cost accuracy instead of vanishing.
Generated reasoning is also scored for fact precision and leakage (faithfulness).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nopeeking.train.common import load_run

log = logging.getLogger(__name__)


def evaluate_adapter(
    config_path: str, splits: tuple[str, ...] = ("val", "test_post"), batch_size: int = 8
) -> str:
    import torch

    from nopeeking.eval.evaluate import evaluate, load_market
    from nopeeking.eval.llm import YesNoScorer, load_model
    from nopeeking.prompts import direct_messages, parse_answer, reasoning_messages
    from nopeeking.train.rewards import fact_precision, leak_flags

    run, cfg = load_run(config_path)
    df = pd.read_parquet(cfg.dataset_path)
    df = df[df["split"].isin(splits)].reset_index(drop=True)
    adapter = run["output_dir"]
    extra: dict = {}
    if run.get("data") == "direct_label":
        scorer = YesNoScorer(
            run["base_model"], adapter=adapter, max_input_tokens=run.get("max_input_tokens", 1024)
        )
        ids = [
            scorer.encode(lambda c, r=r: direct_messages(r, max_chars=c, horizon=cfg.horizon))
            for _, r in df.iterrows()
        ]
        p, _ = scorer.score(ids, batch_size=batch_size)
        preds = pd.DataFrame({"id": df["id"], "p_up": p})
    else:
        tok, model, _ = load_model(run["base_model"], adapter=adapter)
        tok.padding_side = "left"
        rows_out = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            prompts = [
                tok.apply_chat_template(
                    reasoning_messages(r, max_chars=run.get("max_chars", 4000), horizon=cfg.horizon),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for _, r in batch.iterrows()
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(
                model.device
            )
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=320, do_sample=False, pad_token_id=tok.pad_token_id
                )
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for (_, r), text in zip(batch.iterrows(), texts, strict=True):
                parsed = parse_answer(text)
                rows_out.append(
                    {
                        "id": r["id"],
                        "p_up": parsed["p_up"],
                        "completion": text,
                        "fact_precision": fact_precision(str(parsed["reasoning"]), str(r["text"])),
                        "leak": bool(leak_flags(text)),
                    }
                )
        gen = pd.DataFrame(rows_out)
        extra = {
            "parse_failure_rate": float(gen["p_up"].isna().mean()),
            "fact_precision_mean": float(gen["fact_precision"].dropna().mean())
            if gen["fact_precision"].notna().any()
            else None,
            "cites_numbers_rate": float(gen["fact_precision"].notna().mean()),
            "leak_rate": float(gen["leak"].mean()),
        }
        gen["p_up"] = gen["p_up"].fillna(0.5)
        preds = gen[["id", "p_up"]]
        out_dir = cfg.run_results_dir / "generations"
        out_dir.mkdir(parents=True, exist_ok=True)
        gen.to_parquet(out_dir / f"{Path(adapter).name}.parquet", index=False)
    result = {
        "run": Path(adapter).name,
        "config": run["_path"],
        **extra,
        "metrics": evaluate(preds, df, cfg, load_market(cfg, df)),
        "p_up_std": float(np.std(preds["p_up"])),
    }
    out = cfg.run_results_dir / "methods"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(adapter).name}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(path)
