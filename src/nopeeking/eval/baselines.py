"""Non-LLM baselines: majority class, event-type prior, TF-IDF + logistic regression, optional FinBERT.

TF-IDF is also scored on anonymized text: a model trained only on <= 2022 data cannot remember 2023,
so its anonymization effect measures pure information loss -- the control for the LLM probe.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from nopeeking.config import DataConfig
from nopeeking.eval.evaluate import evaluate, load_market
from nopeeking.eval.metrics import auc
from nopeeking.eval.probe import gap_estimates
from nopeeking.prompts import filing_block

log = logging.getLogger(__name__)


def _tfidf_lr(train: pd.DataFrame, val: pd.DataFrame, seed: int):
    best, best_auc = None, -1.0
    for c in (0.1, 0.3, 1.0, 3.0, 10.0):
        model = make_pipeline(
            TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(C=c, max_iter=2000, random_state=seed),
        )
        model.fit(train["input"], train["y_up"])
        score = auc(val["y_up"].to_numpy(), model.predict_proba(val["input"])[:, 1])
        log.info("tfidf_lr C=%s val AUC=%.4f", c, score)
        if score > best_auc:
            best, best_auc = model, score
    return best, best_auc


def _finbert(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            enc = tok(texts[i : i + 16], truncation=True, max_length=512, padding=True, return_tensors="pt")
            probs = model(**enc).logits.softmax(-1)
            pos, neg = probs[:, labels["positive"]], probs[:, labels["negative"]]
            out.append((pos / (pos + neg)).numpy())
    return np.concatenate(out)


def run_baselines(cfg: DataConfig, finbert: bool = False, max_chars: int = 6000) -> str:
    df = pd.read_parquet(cfg.dataset_path)
    df["input"] = [filing_block(r, anonymized=False, max_chars=max_chars) for _, r in df.iterrows()]
    df["input_anon"] = [filing_block(r, anonymized=True, max_chars=max_chars) for _, r in df.iterrows()]
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    market = load_market(cfg, df)

    preds: dict[str, np.ndarray] = {}
    preds["majority"] = np.full(
        len(df), float(train["y_up"].mean())
    )  # always the majority class, p = base rate
    prior = train.groupby("event_type")["y_up"].mean()
    preds["event_type_prior"] = df["event_type"].map(prior).fillna(train["y_up"].mean()).to_numpy()
    model, val_auc = _tfidf_lr(train, val, cfg.seed)
    preds["tfidf_lr"] = model.predict_proba(df["input"])[:, 1]
    preds["tfidf_lr_anon"] = model.predict_proba(df["input_anon"])[:, 1]
    if finbert:
        preds["finbert_zero_shot"] = _finbert(df["text"].str[:4000].tolist())

    results: dict = {"dataset": cfg.name, "n": len(df), "methods": {}}
    pred_rows = []
    for name, p in preds.items():
        table = pd.DataFrame({"id": df["id"], "p_up": p})
        results["methods"][name] = evaluate(table, df, cfg, market=market)
        pred_rows.append(table.assign(method=name, split=df["split"].to_numpy()))
    results["methods"]["tfidf_lr"]["selected_on_val_auc"] = val_auc

    both = df[["id", "split", "y_up"]].assign(
        p_original=preds["tfidf_lr"], p_anonymized=preds["tfidf_lr_anon"]
    )
    results["tfidf_lr_gap"] = gap_estimates(both, seed=cfg.seed)

    out_dir = cfg.run_results_dir
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    pd.concat(pred_rows).to_parquet(out_dir / "predictions" / "baselines.parquet", index=False)
    path = out_dir / "baselines.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return str(path)
