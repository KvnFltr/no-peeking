import pandas as pd

from nopeeking.eval.latex import ci, dataset_table, filterlog_table, macro_name, num, probe_table, tex
from nopeeking.eval.report import leaderboard_md


def test_macro_names_are_valid_latex_and_distinct():
    a = macro_name("Qwen/Qwen2.5-0.5B-Instruct (zero-shot) test_post auc")
    b = macro_name("Qwen/Qwen2.5-1.5B-Instruct (zero-shot) test_post auc")
    assert a.isalpha() and b.isalpha() and a != b
    assert macro_name("v1 N Filings") == "VOneNFilings"


def test_tex_escaping_and_numbers():
    assert tex("tfidf_lr & 5%") == r"tfidf\_lr \& 5\%"
    assert num(float("nan")) == "--" and num(0.12345) == "0.123"
    assert ci({"auc": 0.5, "auc_lo": 0.4, "auc_hi": 0.6}, "auc").startswith("0.500 ")


def test_tables_render():
    df = pd.DataFrame(
        {
            "split": ["train", "train", "test_post"],
            "cik": [1, 2, 1],
            "event_type": ["earnings", "deal", "other"],
            "y_up": [1, 0, 1],
        }
    )
    t = dataset_table(df)
    assert "\\toprule" in t and "Post-cutoff" in t and "1.000" in t
    assert "1,234" in filterlog_table([{"step": "x_y", "n_before": None, "n_after": 1234, "note": ""}])
    g = {k: 0.5 for k in ("pre_orig", "pre_anon", "post_orig", "post_anon", "did")}
    assert "DiD" in probe_table([("m", g)])


def test_leaderboard_ranks_by_post_cutoff_auc():
    def probe(model, auc):
        return {
            "model": model,
            "metrics": {
                "original": {
                    "test_post": {"auc": auc, "auc_lo": auc - 0.05, "auc_hi": auc + 0.05, "brier": 0.25}
                }
            },
            "gap": {"auc": {"did": 0.01, "did_lo": -0.02, "did_hi": 0.04}},
        }

    lines = leaderboard_md({}, {"a": probe("small", 0.51), "b": probe("big", 0.56)})
    rows = [line for line in lines if line.startswith("| 1 ") or line.startswith("| 2 ")]
    assert "big" in rows[0] and "small" in rows[1]
