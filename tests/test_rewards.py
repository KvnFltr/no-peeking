import pandas as pd
import pytest

from nopeeking.train.rationales import check, hindsight_note
from nopeeking.train.rewards import (
    fact_precision,
    format_reward,
    grounding_reward,
    leak_flags,
    numbers,
    outcome_reward,
)

FILING = "Revenue was $94.9 billion, up 6 percent. EPS of $0.97; gross margin 46.2%. Shares: 15,204 million."
GOOD = (
    "Reasoning: Revenue reached $94.9 billion and gross margin was 46.2%, both solid; EPS of $0.97 includes "
    "a one-off charge.\nPrediction: Up\nProbability of outperforming: 0.6"
)
MADE_UP = (
    "Reasoning: Revenue reached $97.1 billion and margin hit 48.0%, a clear beat versus consensus.\n"
    "Prediction: Up\nProbability of outperforming: 0.9"
)
LEAKY = (
    "Reasoning: Revenue reached $94.9 billion. The stock rose after the announcement.\n"
    "Prediction: Up\nProbability of outperforming: 0.7"
)


def test_numbers_normalization_skips_years_and_small_counts():
    assert numbers("In 2024 we had 3 segments and $1,234.50 of sales, 46.2% margin") == {"1234.5", "46.2"}


def test_fact_precision():
    reasoning = GOOD.split("Prediction:")[0]
    assert fact_precision(reasoning, FILING) == pytest.approx(1.0)
    assert fact_precision(MADE_UP.split("Prediction:")[0], FILING) == pytest.approx(0.0)
    assert fact_precision(GOOD, FILING) == pytest.approx(0.75)  # the probability line is not a filing fact
    assert fact_precision("Results looked strong overall.", FILING) is None


def test_leak_flags():
    assert leak_flags(LEAKY)
    assert not leak_flags(GOOD)


def test_rewards_shapes_and_values():
    comps = [GOOD, MADE_UP, "garbage", [{"role": "assistant", "content": GOOD}]]
    out = outcome_reward(comps, y_up=[1, 0, 1, 0])
    assert out[0] == pytest.approx(1 - 2 * 0.4**2)
    assert out[1] == pytest.approx(1 - 2 * 0.9**2)
    assert out[2] == -1.0
    assert out[3] == pytest.approx(1 - 2 * 0.6**2)
    g = grounding_reward([GOOD, MADE_UP, LEAKY], text=[FILING] * 3)
    assert g[0] == pytest.approx(1.0) and g[1] == pytest.approx(0.0) and g[2] == pytest.approx(0.0)
    assert format_reward([GOOD, "Prediction: Down\nProbability of outperforming: 0.8"]) == [1.0, 0.0]


def test_rationale_filter():
    row = pd.Series({"y_up": 1, "text": FILING, "ar_3d": 0.031})
    assert check(GOOD, row, "hindsight", 0.8)["kept"]
    assert check(GOOD, row, "flipped", 0.8)["reject_reasons"] == "direction"
    assert "fact_precision" in check(MADE_UP, row, "control", 0.8)["reject_reasons"]
    assert "leak" in check(LEAKY, row, "hindsight", 0.8)["reject_reasons"]
    assert "3.1 percentage points" in hindsight_note(0.031, True, 3)
