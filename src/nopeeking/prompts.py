"""Prompts shared by zero-shot scoring, rationale generation and post-training.

The answer format for reasoning models is fixed so that it can be parsed and rewarded:

    Reasoning: <analysis that cites facts stated in the filing>
    Prediction: Up | Down
    Probability of outperforming: 0.xx
"""

from __future__ import annotations

import json
import re

import pandas as pd

from nopeeking.data.text import anonymize

SYSTEM = (
    "You are an equity analyst. You judge a company filing using only the text you are given. "
    "You do not know, and must not use, anything that happened after the filing was published."
)

EVENT_NAMES = {
    "earnings": "results of operations (earnings)",
    "deal": "material agreement or acquisition/disposition",
    "exec_change": "executive or director change",
    "other": "other material event",
}


def _list(value: str | list[str]) -> list[str]:
    return json.loads(value) if isinstance(value, str) else list(value)


def filing_block(row: pd.Series | dict, anonymized: bool, max_chars: int) -> str:
    text = str(row["text"])[:max_chars]
    if anonymized:
        text = anonymize(text, [*_list(row["aliases"]), row["company"]], _list(row["tickers_all"]))
        header = "Company: [COMPANY] (ticker [TICKER])\nFiling date: [DATE]"
    else:
        date = pd.Timestamp(row["filing_date"]).strftime("%Y-%m-%d")
        header = f"Company: {row['company']} (ticker {row['ticker']})\nFiling date: {date}"
    event = EVENT_NAMES.get(row["event_type"], row["event_type"])
    return f"{header}\nForm 8-K items: {row['items']} ({event})\n\n{text}"


def direct_question(horizon: int = 3) -> str:
    return (
        f"Question: over the next {horizon} trading days, starting with the first session after this filing, "
        "will this company's stock outperform the S&P 500 (SPY)? Answer with a single word: Yes or No."
    )


def direct_messages(
    row: pd.Series | dict, anonymized: bool = False, max_chars: int = 6000, horizon: int = 3
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": filing_block(row, anonymized, max_chars) + "\n\n" + direct_question(horizon),
        },
    ]


REASONING_INSTRUCTIONS = (
    "Think like an analyst reading this filing the moment it was released. Identify the 2-4 facts in the "
    "filing that matter most for the stock relative to the market (for example results versus prior year, "
    "guidance, margins, one-off items, the nature of the deal or departure), quoting numbers exactly as they "
    "appear in the filing. Then answer in exactly this format:\n"
    "Reasoning: <3-6 sentences>\n"
    "Prediction: Up or Down\n"
    "Probability of outperforming: <a number between 0 and 1>"
)


def reasoning_messages(
    row: pd.Series | dict, anonymized: bool = False, max_chars: int = 6000, horizon: int = 3
) -> list[dict[str, str]]:
    question = (
        f"Over the next {horizon} trading days, will this company's stock outperform the S&P 500 (SPY)?\n\n"
        + REASONING_INSTRUCTIONS
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": filing_block(row, anonymized, max_chars) + "\n\n" + question},
    ]


_PRED_RE = re.compile(r"Prediction:\s*(Up|Down)\b", re.I)
_PROB_RE = re.compile(r"Probability of outperforming:\s*([01](?:\.\d+)?|\.\d+)", re.I)


def parse_answer(completion: str) -> dict[str, float | str | bool | None]:
    """Parse the fixed answer format; ``consistent`` means Up <=> probability >= 0.5."""
    pred = _PRED_RE.search(completion)
    prob = _PROB_RE.search(completion)
    direction = pred.group(1).capitalize() if pred else None
    p = float(prob.group(1)) if prob else None
    if p is not None and not 0.0 <= p <= 1.0:
        p = None
    consistent = direction is not None and p is not None and ((p >= 0.5) == (direction == "Up"))
    reasoning = completion.split("Prediction:")[0].replace("Reasoning:", "").strip()
    return {
        "direction": direction,
        "p_up": p,
        "consistent": consistent,
        "reasoning": reasoning,
        "has_reasoning": len(reasoning) >= 40,
    }
