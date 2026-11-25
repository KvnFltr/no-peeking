"""Rationale filters and GRPO rewards (pure Python, unit-tested).

* outcome reward: Brier-based proper scoring rule, 1 - 2 (p - y)^2 in [-1, 1]. A proper rule rewards
  calibrated probabilities, which limits the "always be confident" reward hacking that noisy market labels
  invite with a 0/1 reward.
* grounding reward: share of the numbers cited in the reasoning that appear in the filing (fact precision),
  minus 1 if the reasoning leaks the outcome or later events.
* format reward: the fixed answer format is present and self-consistent.

TRL passes dataset columns to reward functions as keyword lists: ``y_up`` and ``text`` must be columns.
"""

from __future__ import annotations

import re
from typing import Any

from nopeeking.prompts import parse_answer

LEAK_PATTERNS = [
    r"\b(?:shares?|stock|price)\s+(?:subsequently\s+)?(?:rose|fell|jumped|dropped|surged|plunged|climbed|"
    r"declined|gained|lost|rallied|slid|tumbled|soared|sank)\b",
    r"\b(?:subsequently|in hindsight|turned out|ended up|as it happened)\b",
    r"\bthe market (?:reacted|responded|rewarded|punished)\b",
    r"\b(?:realized|actual|abnormal) return\b",
    r"\b(?:outperformed|underperformed)\b",
    r"\b(?:after|following) the (?:announcement|filing|release),? (?:the )?(?:shares|stock)\b",
    r"\bwe (?:now )?know\b",
]
_LEAK_RE = [re.compile(p, re.I) for p in LEAK_PATTERNS]
_NUM_RE = re.compile(r"(?<![\w.])[-+]?\$?\(?(\d[\d,]*(?:\.\d+)?)\)?\s?%?")


def leak_flags(text: str) -> list[str]:
    return [p.pattern for p in _LEAK_RE if p.search(text)]


def _norm_number(raw: str) -> str | None:
    s = raw.replace(",", "")
    try:
        x = float(s)
    except ValueError:
        return None
    if "." not in s and (x < 10 or 1990 <= x <= 2035):  # small counts and years carry no checkable fact
        return None
    return f"{x:.6g}"


def numbers(text: str) -> set[str]:
    out = set()
    for m in _NUM_RE.finditer(text):
        n = _norm_number(m.group(1))
        if n is not None:
            out.add(n)
    return out


def fact_precision(reasoning: str, filing_text: str) -> float | None:
    """Share of checkable numbers in ``reasoning`` that also appear in the filing; None if it cites none."""
    cited = numbers(reasoning)
    if not cited:
        return None
    return len(cited & numbers(filing_text)) / len(cited)


def _completion_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return str(c[-1].get("content", ""))
    return str(c)


def outcome_reward(completions: list, y_up: list[int], **_: Any) -> list[float]:
    out = []
    for c, y in zip(completions, y_up, strict=True):
        parsed = parse_answer(_completion_text(c))
        p = parsed["p_up"]
        out.append(-1.0 if p is None else 1.0 - 2.0 * (float(p) - int(y)) ** 2)
    return out


def grounding_reward(completions: list, text: list[str], **_: Any) -> list[float]:
    out = []
    for c, filing in zip(completions, text, strict=True):
        completion = _completion_text(c)
        reasoning = parse_answer(completion)["reasoning"]
        fp = fact_precision(str(reasoning), filing)
        score = 0.0 if fp is None else fp
        if leak_flags(completion):
            score -= 1.0
        out.append(score)
    return out


def format_reward(completions: list, **_: Any) -> list[float]:
    out = []
    for c in completions:
        parsed = parse_answer(_completion_text(c))
        out.append(1.0 if parsed["consistent"] and parsed["has_reasoning"] else 0.0)
    return out


REWARDS = {"outcome": outcome_reward, "grounding": grounding_reward, "format": format_reward}
