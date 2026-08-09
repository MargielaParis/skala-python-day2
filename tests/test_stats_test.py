"""stats_test 모듈 — 기술통계·상관·Welch t-test·효과크기·카이제곱 검증"""

import pandas as pd
import pytest

from src import stats_test
from src.errors import PipelineError


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


def test_ttest_raises_when_group_empty(sample_df):
    only_low = sample_df[sample_df["income"] == "<=50K"]

    with pytest.raises(PipelineError):
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


def test_ttest_reports_effect_size_and_ci(sample_df):
    # p-value만으로는 차이의 크기를 알 수 없으므로 효과크기·신뢰구간을 함께 낸다
    result = stats_test.ttest_hours_by_income(sample_df)

    assert result["diff"] == pytest.approx(result["mean_high"] - result["mean_low"])
    assert result["ci_low"] < result["diff"] < result["ci_high"]
    assert result["cohens_d"] > 0
    assert result["effect"] in {"매우 작음", "작음", "중간", "큼"}


def test_cohens_d_is_zero_for_identical_groups():
    same = pd.Series([1.0, 2.0, 3.0, 4.0])

    assert stats_test.cohens_d(same, same) == pytest.approx(0.0)


def test_effect_label_follows_cohen_thresholds():
    assert stats_test.effect_label(0.1) == "매우 작음"
    assert stats_test.effect_label(-0.3) == "작음"
    assert stats_test.effect_label(0.6) == "중간"
    assert stats_test.effect_label(1.2) == "큼"


def test_chisq_sex_income_detects_association():
    # 성별에 따라 소득 분포가 뚜렷하게 다른 표본
    df = pd.DataFrame(
        {
            "sex": ["Male"] * 100 + ["Female"] * 100,
            "income": [">50K"] * 80 + ["<=50K"] * 20 + [">50K"] * 10 + ["<=50K"] * 90,
        }
    )

    result = stats_test.chisq_sex_income(df)

    assert result["dof"] == 1
    assert result["n"] == 200
    assert result["significant"] is True
    assert 0 < result["cramers_v"] <= 1
    assert result["rate_by_sex"]["Male"] > result["rate_by_sex"]["Female"]
    assert result["p_text"] == stats_test.format_p(result["p"])


def test_chisq_sex_income_finds_no_association_when_balanced():
    # 두 성별 모두 고소득 비율이 정확히 50%라 연관이 없다
    df = pd.DataFrame(
        {
            "sex": ["Male"] * 100 + ["Female"] * 100,
            "income": ([">50K"] * 50 + ["<=50K"] * 50) * 2,
        }
    )

    result = stats_test.chisq_sex_income(df)

    assert result["significant"] is False
    assert result["cramers_v"] == pytest.approx(0.0)


def test_chisq_raises_when_table_is_degenerate(sample_df):
    one_sex = sample_df[sample_df["sex"] == "Male"].assign(income="<=50K")

    with pytest.raises(PipelineError):
        stats_test.chisq_sex_income(one_sex)


def test_chisq_does_not_apply_yates_correction(sample_df):
    # 기대빈도가 충분히 크면 연속성 보정은 chi2를 낮게 만든다. 보정 없이 계산하고 그 사실을 남긴다
    from scipy import stats as sp

    result = stats_test.chisq_sex_income(sample_df)
    table = pd.crosstab(sample_df["sex"], sample_df["income"])
    uncorrected = sp.chi2_contingency(table, correction=False)[0]

    assert result["correction"] is False
    assert result["chi2"] == pytest.approx(uncorrected)
    assert result["expected_min"] > 0


def test_data_caveats_reports_overlap_and_topcoding(sample_df):
    caveats = stats_test.data_caveats(sample_df, sample_df)

    # 같은 프레임을 넘겼으므로 모든 행이 교차 중복이다
    assert caveats["overlap_rows"] == len(sample_df)
    assert caveats["capital_gain_cap"] == stats_test.CAPITAL_GAIN_CAP
    assert 0.0 <= caveats["capital_gain_zero_share"] <= 1.0
    for share in caveats["role_purity"].values():
        assert 0.5 <= share <= 1.0


def test_data_caveats_measures_sex_role_overlap():
    # Husband가 전부 Male이면 성별 편중도는 1.0이어야 한다
    df = pd.DataFrame(
        {
            "sex": ["Male"] * 10 + ["Female"] * 10,
            "relationship": ["Husband"] * 10 + ["Wife"] * 10,
            "capital-gain": [0] * 20,
            "income": ["<=50K"] * 20,
        }
    )

    caveats = stats_test.data_caveats(df, df.iloc[:0])

    assert caveats["role_purity"] == {"Husband": 1.0, "Wife": 1.0}
    assert caveats["overlap_rows"] == 0
