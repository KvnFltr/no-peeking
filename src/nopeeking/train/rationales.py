"""Teacher rationales.

* ``hindsight``: the teacher sees the filing AND the realized outcome, and writes the reasoning an analyst
  could have written before knowing it (RQ2).
* ``control``: same prompt without the outcome (does knowing the answer matter?).
* ``flipped``: hindsight with the opposite outcome -> the "rejected" side of DPO pairs.

Filters (all logged): answer format, direction matches the conditioning outcome, no outcome leakage,
fact precision of cited numbers >= threshold.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from nopeeking.prompts import reasoning_messages
from nopeeking.train.common import load_run, train_rows
from nopeeking.train.rewards import fact_precision, leak_flags

log = logging.getLogger(__name__)


def hindsight_note(ar: float, up: bool, horizon: int) -> str:
    verb = "outperformed" if up else "underperformed"
    return (
        f"\n\nPrivate note for writing training data (never mention it): over the next {horizon} trading days "
        f"the stock {verb} the S&P 500 by {abs(ar) * 100:.1f} percentage points. Write the reasoning a careful "
        f"analyst could have written BEFORE knowing this, using only facts stated in the filing. Do not mention "
        f"or allude to the outcome, the later stock move, or any event after the filing. Your Prediction must be "
        f"{'Up' if up else 'Down'}, and the probability must reflect how strongly the filing alone supports it."
    )


def build_messages(row: pd.Series, variant: str, max_chars: int, horizon: int) -> list[dict[str, str]]:
    msgs = reasoning_messages(row, anonymized=False, max_chars=max_chars, horizon=horizon)
    if variant in ("hindsight", "flipped"):
        up = bool(row["y_up"]) if variant == "hindsight" else not bool(row["y_up"])
        ar = float(row[f"ar_{horizon}d"])
        msgs[-1]["content"] += hindsight_note(ar if variant == "hindsight" else -ar, up, horizon)
    return msgs


def check(completion: str, row: pd.Series, variant: str, min_precision: float) -> dict:
    from nopeeking.prompts import parse_answer

    parsed = parse_answer(completion)
    fp = fact_precision(str(parsed["reasoning"]), str(row["text"]))
    leaks = leak_flags(completion)
    want = None
    if variant == "hindsight":
        want = "Up" if row["y_up"] else "Down"
    elif variant == "flipped":
        want = "Down" if row["y_up"] else "Up"
    reasons = []
    if not (parsed["consistent"] and parsed["has_reasoning"]):
        reasons.append("format")
    if want is not None and parsed["direction"] != want:
        reasons.append("direction")
    if leaks:
        reasons.append("leak")
    if fp is not None and fp < min_precision:
        reasons.append("fact_precision")
    return {
        "p_up": parsed["p_up"],
        "direction": parsed["direction"],
        "fact_precision": fp,
        "n_leaks": len(leaks),
        "kept": not reasons,
        "reject_reasons": ",".join(reasons),
    }


def generate_rationales(config_path: str) -> str:
    import torch

    from nopeeking.eval.llm import chat_text, load_model

    run, cfg = load_run(config_path)
    rows = train_rows(cfg, tuple(run.get("splits", ["train"])))
    if run.get("max_filings") and len(rows) > run["max_filings"]:
        rows = rows.sample(n=run["max_filings"], random_state=cfg.seed).reset_index(drop=True)
    tok, model, _ = load_model(run["teacher"], dtype=run.get("teacher_dtype", "4bit"))
    tok.padding_side = "left"
    gen = run.get("generation", {})
    out_dir = cfg.dataset_dir / "rationales"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for variant in run.get("variants", ["hindsight", "control", "flipped"]):
        path = out_dir / f"{variant}.parquet"
        done = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        seen = set(done["id"]) if len(done) else set()
        todo = rows[~rows["id"].isin(seen)]
        records = done.to_dict("records") if len(done) else []
        bs = gen.get("batch_size", 8)
        for i in range(0, len(todo), bs):
            batch = todo.iloc[i : i + bs]
            prompts = [
                chat_text(tok, build_messages(r, variant, run.get("max_chars", 6000), cfg.horizon))
                for _, r in batch.iterrows()
            ]
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=run.get("max_prompt_tokens", 1536),
            ).to(model.device)
            with torch.no_grad():
                outs = model.generate(
                    **enc,
                    max_new_tokens=gen.get("max_new_tokens", 320),
                    do_sample=True,
                    temperature=gen.get("temperature", 0.7),
                    top_p=gen.get("top_p", 0.9),
                    pad_token_id=tok.pad_token_id,
                )
            texts = tok.batch_decode(outs[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for (_, r), text in zip(batch.iterrows(), texts, strict=True):
                records.append(
                    {
                        "id": r["id"],
                        "split": r["split"],
                        "variant": variant,
                        "completion": text.strip(),
                        **check(text, r, variant, run.get("min_fact_precision", 0.8)),
                    }
                )
            pd.DataFrame(records).to_parquet(path, index=False)
            log.info("%s: %d/%d", variant, len(records), len(rows))
        df = pd.DataFrame(records)
        summary = {
            "variant": variant,
            "n": len(df),
            "kept": int(df["kept"].sum()),
            "reject_reasons": df["reject_reasons"].str.split(",").explode().value_counts().to_dict(),
        }
        (out_dir / f"{variant}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        written.append(str(path))
    return ", ".join(written)


def load_kept(cfg, variant: str) -> pd.DataFrame:
    path = Path(cfg.dataset_dir) / "rationales" / f"{variant}.parquet"
    df = pd.read_parquet(path)
    return df[df["kept"]]
