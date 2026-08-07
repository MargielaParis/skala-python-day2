"""stats_test 모듈 — 기술통계·상관·Welch t-test 검증"""

import pytest

from src import stats_test


def test_describe_numeric_uses_only_numeric_columns(sample_df):
    describe, corr = stats_test.describe_numeric(sample_df)

    assert "workclass" not in describe.columns
    assert "hours-per-week" in describe.columns
    assert corr.loc["education-num", "hours-per-week"] == pytest.approx(
        corr.loc["hours-per-week", "education-num"]
    )


def test_ttest_detects_hours_gap(sample_df):
    result = stats_test.ttest_hours_by_income(sample_df)

    assert result["mean_high"] > result["mean_low"]
    assert result["p"] < stats_test.ALPHA
    assert bool(result["significant"]) is True


def test_ttest_exits_when_group_empty(sample_df):
    only_low = sample_df[sample_df["income"] == "<=50K"]

    with pytest.raises(SystemExit):
        stats_test.ttest_hours_by_income(only_low)
