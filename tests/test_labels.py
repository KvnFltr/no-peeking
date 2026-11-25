import numpy as np
import pandas as pd
import pytest

from nopeeking.labels.events import add_targets, event_days, market_adjusted_returns

CAL = pd.DatetimeIndex(pd.bdate_range("2024-10-28", "2024-11-08"))  # Mon 28 Oct .. Fri 8 Nov


def _utc(ts_et: str) -> pd.Timestamp:
    return pd.Timestamp(ts_et, tz="America/New_York").tz_convert("UTC")


def test_event_day_before_and_after_close_and_weekend():
    acc = pd.Series(
        [
            _utc("2024-10-31 10:00"),  # during session -> same day
            _utc("2024-10-31 16:30"),  # after close -> next session (Fri 1 Nov)
            _utc("2024-11-02 09:00"),  # Saturday -> Monday 4 Nov
            _utc("2024-11-01 15:59"),  # just before close -> same day
            _utc("2024-11-08 16:01"),  # after the last close in the calendar -> NaT
        ]
    )
    days = event_days(acc, CAL)
    assert days.iloc[0] == pd.Timestamp("2024-10-31")
    assert days.iloc[1] == pd.Timestamp("2024-11-01")
    assert days.iloc[2] == pd.Timestamp("2024-11-04")
    assert days.iloc[3] == pd.Timestamp("2024-11-01")
    assert pd.isna(days.iloc[4])


def test_market_adjusted_returns_window():
    close = pd.DataFrame({"AAA": np.arange(100.0, 100.0 + len(CAL))}, index=CAL)
    bench = pd.Series(np.full(len(CAL), 50.0), index=CAL)
    bench.iloc[5:] = 55.0  # +10% on 4 Nov
    out = market_adjusted_returns(
        pd.Series(["AAA", "AAA", "ZZZ"]), pd.Series([CAL[4], CAL[-1], CAL[4]]), close, bench, horizon=3
    )
    # t = CAL[4] (Fri 1 Nov): close(t-1)=103, close(t)=104, close(t+2)=106
    assert out.loc[0, "ret_1d"] == pytest.approx(104 / 103 - 1)
    assert out.loc[0, "ret_3d"] == pytest.approx(106 / 103 - 1)
    assert out.loc[0, "ar_3d"] == pytest.approx(106 / 103 - 1 - 0.10)
    assert np.isnan(out.loc[1, "ar_3d"])  # window runs past the data
    assert np.isnan(out.loc[2, "ar_1d"])  # unknown ticker


def test_targets_use_train_cutoffs_only():
    df = pd.DataFrame(
        {
            "ar_3d": [-0.05, -0.01, 0.0, 0.02, 0.04, 0.5, -0.5],
            "ar_1d": [0.0] * 7,
            "split": ["train"] * 5 + ["test_post"] * 2,
        }
    )
    cuts = add_targets(df, 3, (0.0, 1.0), df["split"] == "train")
    assert cuts["winsor_hi"] == pytest.approx(0.04)
    assert df["ar_3d_w"].iloc[5] == pytest.approx(0.04)  # test extreme clipped at train max
    assert df["y_up"].tolist() == [0, 0, 0, 1, 1, 1, 0]
    assert set(df["y_tercile"]) <= {0, 1, 2}
