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


def test_format_p_marks_underflow_instead_of_zero():
    assert stats_test.format_p(0.0) == stats_test.P_FLOOR_TEXT
    assert "0.000000" not in stats_test.format_p(0.0)


def test_format_p_uses_scientific_notation_for_tiny_values():
    assert stats_test.format_p(1.23e-9) == "1.230e-09"
    assert stats_test.format_p(0.0321) == "0.032100"


def test_ttest_reports_p_text_and_underflow_flag(sample_df):
    result = stats_test.ttest_hours_by_income(sample_df)

    assert result["p_underflow"] is False
    assert result["p_text"] == stats_test.format_p(result["p"])
