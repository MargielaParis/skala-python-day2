"""load 모듈 — 로딩 비교와 정제 단계 검증"""

import pandas as pd
import pytest

from src import load


def test_load_compare_reads_both_engines(raw_csv):
    df, compare = load.load_compare(raw_csv)

    assert list(df.columns) == load.COLS
    assert compare["rows"] == 3 and compare["cols"] == 15
    assert compare["pandas_sec"] > 0 and compare["polars_sec"] > 0


def test_load_compare_strips_test_label_dot(raw_csv):
    df, _ = load.load_compare(raw_csv)

    assert set(df["income"]) == {"<=50K", ">50K"}


def test_load_compare_treats_question_mark_as_na(raw_csv):
    df, _ = load.load_compare(raw_csv)

    assert df["workclass"].isna().sum() == 1
    assert df["occupation"].isna().sum() == 1


def test_load_compare_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        load.load_compare(tmp_path / "nope.data")


def test_clean_drops_na_then_duplicates():
    df = pd.DataFrame(
        {
            "age": [39, 39, 50, 31],
            "workclass": ["Private", "Private", "Private", None],
            "income": ["<=50K", "<=50K", ">50K", "<=50K"],
        }
    )

    cleaned, info = load.clean(df)

    assert info == {
        "raw": 4,
        "after_na": 3,
        "clean": 2,
        "na_cols": {"workclass": 1},
    }
    assert len(cleaned) == 2
    assert list(cleaned.index) == [0, 1]
