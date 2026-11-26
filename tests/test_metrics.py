import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from nopeeking.eval.metrics import auc, bootstrap_ci, ece, metrics_with_ci
from nopeeking.eval.portfolio import long_short_returns, sharpe
from nopeeking.eval.probe import gap_estimates


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.round(rng.random(300), 1)
    assert auc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert np.isnan(auc(np.ones(5), np.arange(5)))


def test_ece_zero_when_calibrated_bins():
    y = np.array([0, 1] * 50)
    p = np.full(100, 0.5)
    assert ece(y, p) == pytest.approx(0.0)
    assert ece(np.ones(10), np.zeros(10)) == pytest.approx(1.0)


def test_bootstrap_is_reproducible_and_brackets_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    a = bootstrap_ci(auc, [y, p], n_boot=300, seed=3)
    b = bootstrap_ci(auc, [y, p], n_boot=300, seed=3)
    assert a == b and a[0] < auc(y, p) < a[1]
    m = metrics_with_ci(y, p, ret=p - 0.5, n_boot=200)
    assert m["auc_lo"] <= m["auc"] <= m["auc_hi"] and m["rank_ic"] > 0


def test_long_short_portfolio_signs_costs_and_tradable_timing():
    cal = pd.bdate_range("2025-01-01", periods=10)
    bench = pd.Series(100.0, index=cal)
    close = pd.DataFrame({"UP": 100 * 1.01 ** np.arange(10), "DN": 100 * 0.99 ** np.arange(10)}, index=cal)
    events = pd.DataFrame({"ticker": ["UP", "DN"], "event_day": [cal[2], cal[2]], "score": [0.9, 0.1]})
    daily = long_short_returns(events, 0.3, 0.7, close, bench, horizon=3, cost_bps=0.0)
    assert list(daily.index) == [cal[3], cal[4]]  # day t (cal[2]) is not tradable: entry at close(t)
    assert (daily > 0.019).all()
    with_cost = long_short_returns(events, 0.3, 0.7, close, bench, horizon=3, cost_bps=10.0)
    assert with_cost.iloc[0] == pytest.approx(daily.iloc[0] - 2 * 2 * 10 / 1e4)
    jump = close.copy()
    jump.loc[cal[2] :, "UP"] *= 1.5  # a 50% reaction on day t must not be earned
    assert long_short_returns(events.iloc[:1], 0.3, 0.7, jump, bench, 3, 0.0).max() < 0.02
    assert np.isnan(sharpe(np.zeros(5)))


def test_gap_estimates_detects_memorization():
    rng = np.random.default_rng(0)
    n = 200
    y_pre, y_post = rng.integers(0, 2, n), rng.integers(0, 2, n)
    noise = rng.random(n)
    both = pd.DataFrame(
        {
            "id": range(2 * n),
            "split": ["test_pre"] * n + ["test_post"] * n,
            "y_up": np.r_[y_pre, y_post],
            "p_original": np.r_[0.8 * y_pre + 0.2 * noise, rng.random(n)],  # "remembers" pre-cutoff outcomes
            "p_anonymized": np.r_[rng.random(n), rng.random(n)],
        }
    )
    g = gap_estimates(both, n_boot=300)
    assert g["auc"]["did"] > 0.3 and g["auc"]["did_lo"] > 0
