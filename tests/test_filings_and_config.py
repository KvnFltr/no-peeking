import pandas as pd

from nopeeking.config import UNUSED, assign_split
from nopeeking.data.filings import classify_event, combine_text, parse_filing_index

INDEX_HTML = """<html><body><table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/ix?doc=/Archives/edgar/data/1/000000000124000001/a8-k.htm">a8-k.htm</a></td><td>8-K</td><td>30</td></tr>
<tr><td>2</td><td>PRESS RELEASE</td><td><a href="/Archives/edgar/data/1/000000000124000001/ex991.htm">ex991.htm</a></td><td>EX-99.1</td><td>90</td></tr>
<tr><td>3</td><td></td><td><a href="/Archives/edgar/data/1/000000000124000001/logo.jpg">logo.jpg</a></td><td>GRAPHIC</td><td>9</td></tr>
</table></body></html>"""


def test_classify_event_priority():
    assert classify_event("2.02,9.01") == "earnings"
    assert classify_event("5.02,2.02") == "earnings"
    assert classify_event("1.01,5.02") == "deal"
    assert classify_event("5.02") == "exec_change"
    assert classify_event("8.01,9.01") == "other"
    assert classify_event("5.07") is None
    assert classify_event("") is None and classify_event(None) is None


def test_parse_filing_index_strips_ixviewer_prefix():
    rows = parse_filing_index(INDEX_HTML)
    assert [r["type"] for r in rows] == ["8-K", "EX-99.1", "GRAPHIC"]
    assert rows[0]["href"] == "/Archives/edgar/data/1/000000000124000001/a8-k.htm"


def test_combine_text_truncates():
    assert combine_text("body", "release", 1000) == "[FORM 8-K]\nbody\n\n[EXHIBIT 99.1]\nrelease"
    assert len(combine_text("a" * 50, "b" * 50, 40)) == 40


def test_assign_split_windows_and_gap():
    splits = {
        "train": ["2018-01-01", "2022-12-31"],
        "test_pre": ["2023-01-01", "2023-12-31"],
        "val": ["2024-01-01", "2024-12-31"],
        "test_post": ["2025-07-01", None],
    }
    dates = pd.Series(
        pd.to_datetime(
            [
                "2017-12-31",
                "2018-01-01",
                "2022-12-31",
                "2023-06-01",
                "2024-12-31",
                "2025-03-01",
                "2025-07-01",
                "2026-11-01",
            ]
        )
    )
    assert assign_split(dates, splits).tolist() == [
        UNUSED,
        "train",
        "train",
        "test_pre",
        "val",
        UNUSED,
        "test_post",
        "test_post",
    ]
