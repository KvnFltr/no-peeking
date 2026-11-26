import json

import numpy as np
import pandas as pd

from nopeeking.prompts import direct_messages, filing_block, parse_answer, reasoning_messages

ROW = {
    "text": "Acme Widgets Inc. (NYSE: ACME) reported Q3 2024 revenue of $1.2 billion, up 8%.",
    "aliases": json.dumps(["Acme Widgets Inc.", "ACME WIDGETS INC"]),
    "tickers_all": json.dumps(["ACME"]),
    "company": "Acme Widgets Inc.",
    "ticker": "ACME",
    "filing_date": pd.Timestamp("2024-10-30"),
    "items": "2.02,9.01",
    "event_type": "earnings",
}


def test_filing_block_original_vs_anonymized():
    orig = filing_block(ROW, anonymized=False, max_chars=1000)
    anon = filing_block(ROW, anonymized=True, max_chars=1000)
    assert "Acme" in orig and "2024-10-30" in orig
    assert "Acme" not in anon and "ACME" not in anon and "2024" not in anon
    assert "$1.2 billion" in anon and "8%" in anon


def test_messages_shapes():
    for msgs in (direct_messages(ROW), reasoning_messages(ROW, anonymized=True)):
        assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Yes or No" in direct_messages(ROW)[1]["content"]
    assert "Probability of outperforming" in reasoning_messages(ROW)[1]["content"]


def test_parse_answer():
    ok = parse_answer(
        "Reasoning: Revenue grew 8% and guidance was raised, which the market should like.\n"
        "Prediction: Up\nProbability of outperforming: 0.64"
    )
    assert ok["direction"] == "Up" and ok["p_up"] == 0.64 and ok["consistent"] and ok["has_reasoning"]
    bad = parse_answer("Prediction: Down\nProbability of outperforming: 0.7")
    assert bad["direction"] == "Down" and not bad["consistent"] and not bad["has_reasoning"]
    assert parse_answer("no idea")["p_up"] is None


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _NewModel:
    """Forward that knows `logits_to_keep` and returns one position when asked."""

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, attention_mask, position_ids, logits_to_keep=0):
        self.calls.append(logits_to_keep)
        n = logits_to_keep or input_ids.shape[1]
        return _Out(np.arange(input_ids.shape[0] * n * 3, dtype=float).reshape(input_ids.shape[0], n, 3))


class _OldModel:
    """Forward without any logits-trimming argument."""

    def __call__(self, input_ids, attention_mask, position_ids):
        return _Out(
            np.arange(input_ids.shape[0] * input_ids.shape[1] * 3, dtype=float).reshape(
                -1, input_ids.shape[1], 3
            )
        )


def test_last_logits_keeps_only_the_last_position_when_supported():
    from nopeeking.eval.llm import YesNoScorer

    ids = np.zeros((2, 5), dtype=int)
    new = object.__new__(YesNoScorer)
    new.model = _NewModel()
    assert new._last_logits(ids, ids, ids).shape == (2, 3)
    assert new._keep_key == "logits_to_keep" and new.model.calls == [1]
    old = object.__new__(YesNoScorer)
    old.model = _OldModel()
    full = old._last_logits(ids, ids, ids)
    assert old._keep_key is None
    assert full.tolist() == _OldModel()(ids, ids, ids).logits[:, -1, :].tolist()
