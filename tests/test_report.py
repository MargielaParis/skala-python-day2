"""report 모듈 — report.md 자동 생성과 '입력이 바뀌면 결론도 바뀐다' 회귀 검증"""

import copy

import pytest

from src import ml, report, stats_test


@pytest.fixture
def report_inputs(sample_df, tmp_path):
    # main.py가 넘기는 것과 같은 모양의 단계별 결과를 구성
    loading = {
        "rows": len(sample_df),
        "cols": 15,
        "pandas_sec": 0.01,
        "polars_sec": 0.02,
        "equality": {
            "columns_match": True,
            "shape_match": True,
            "dtype_mismatch": [],
            "null_mismatch": 0,
            "value_mismatch": 0,
            "identical": True,
        },
    }
    cleaning = {
        "strategy": "drop",
        "raw": 70,
        "after_na": 65,
        "clean": 60,
        "na_cols": {"workclass": 5},
        "drop_rate_by_income": {"<=50K": 0.1, ">50K": 0.05},
        "removed_by_income": {"<=50K": 4, ">50K": 1},
        "total_by_income": {"<=50K": 40, ">50K": 20},
        "pos_rate_raw": 0.30,
        "pos_rate_kept": 0.32,
    }
    describe, corr = stats_test.describe_numeric(sample_df)
    ttest = stats_test.ttest_hours_by_income(sample_df)
    chisq = stats_test.chisq_sex_income(sample_df)
    metrics = ml.train_and_evaluate(sample_df, sample_df, tmp_path / "pipeline.pkl")
    ab = ml.compare_strategies({"drop": (sample_df, sample_df)})
    return {
        "loading": loading,
        "cleaning": cleaning,
        "loading_te": loading,
        "cleaning_te": cleaning,
        "describe": describe,
        "corr": corr,
        "ttest": ttest,
        "chisq": chisq,
        "ml": metrics,
        "ab": ab,
    }


def _render(inputs, tmp_path, name="report.md"):
    path = tmp_path / name
    report.write_report(path, **inputs)
    return path.read_text(encoding="utf-8")


def test_write_report_creates_all_sections(report_inputs, tmp_path):
    md = _render(report_inputs, tmp_path)

    for heading in ("## 1. 데이터 준비", "## 3. 통계 분석", "## 4. ML Pipeline", "## 5. 결론"):
        assert heading in md
    assert f"{report_inputs['ml']['accuracy']:.4f}" in md
    assert "sex_" in md


def test_report_documents_reference_category_and_scale_unit(report_inputs, tmp_path):
    md = _render(report_inputs, tmp_path)
    reference = report_inputs["ml"]["reference"]

    assert f"sex={reference['sex']}" in md
    assert "1 표준편차 증가" in md
    assert "기준 범주 대비" in md or "기준 범주 —" in md


def test_report_includes_confusion_matrix_and_imbalance_metrics(report_inputs, tmp_path):
    md = _render(report_inputs, tmp_path)
    cm = report_inputs["ml"]["confusion"]

    assert "실제 \\ 예측" in md
    assert f"{cm['fn']:,}" in md
    assert "ROC-AUC" in md
    assert "정밀도 (precision)" in md and "재현율 (recall)" in md


def test_report_states_equality_only_from_actual_comparison(report_inputs, tmp_path):
    md_ok = _render(report_inputs, tmp_path, "ok.md")
    assert "불일치 0" in md_ok

    broken = copy.deepcopy(report_inputs)
    broken["loading"]["equality"] |= {"identical": False, "value_mismatch": 451592}
    md_bad = _render(broken, tmp_path, "bad.md")

    assert "불일치 있음" in md_bad
    assert "451592" in md_bad or "451,592" in md_bad


def test_conclusion_follows_ttest_significance(report_inputs, tmp_path):
    # ttest.significant를 뒤집으면 3절과 5절 결론이 함께 바뀌어야 한다
    md_sig = _render(report_inputs, tmp_path, "sig.md")

    flipped = copy.deepcopy(report_inputs)
    flipped["ttest"] |= {"significant": False, "p": 0.42, "p_text": "0.420000"}
    md_ns = _render(flipped, tmp_path, "ns.md")

    assert "통계적으로 유의미한 차이 있음 (H0 기각)" in md_sig
    assert "통계적으로 유의미한 차이가 있다" in md_sig
    assert "차이 없음 (우연일 수 있음)" in md_ns
    assert "유의미한 차이가 없다" in md_ns
    assert "통계적으로 유의미한 차이가 있다" not in md_ns


def test_conclusion_follows_chisq_significance(report_inputs, tmp_path):
    flipped = copy.deepcopy(report_inputs)
    flipped["chisq"] |= {"significant": False, "p": 0.8, "p_text": "0.800000"}

    md = _render(flipped, tmp_path, "chi.md")

    assert "독립 가설을 기각하지 못했다" in md
    assert "독립 가설을 기각했다" not in md


def test_conclusion_odds_ratio_follows_coefficient(report_inputs, tmp_path):
    # 고정된 "오즈비 0.37" 같은 문장이 남아 있지 않고, 계수를 바꾸면 결론 숫자도 바뀐다
    md_before = _render(report_inputs, tmp_path, "before.md")
    coef = report_inputs["ml"]["coef"]
    sex_row = next(name for name in coef.index if name.startswith("sex_"))
    original = f"오즈 {coef.loc[sex_row, 'odds_ratio']:.3f}배"
    assert original in md_before

    changed = copy.deepcopy(report_inputs)
    changed["ml"]["coef"].loc[sex_row, "odds_ratio"] = 0.123
    md_after = _render(changed, tmp_path, "after.md")

    assert "오즈 0.123배" in md_after
    assert original not in md_after


def test_report_has_no_hardcoded_conclusion_numbers(report_inputs, tmp_path):
    md = _render(report_inputs, tmp_path)

    assert "오즈비 0.37" not in md
    assert "가장 강한 신호" not in md


def test_report_compares_missing_value_strategies(report_inputs, tmp_path):
    md = _render(report_inputs, tmp_path)

    assert "결측 처리 방식 A/B 비교" in md
    assert report.STRATEGY_LABEL["drop"] in md


def test_write_report_exits_on_bad_path(report_inputs, tmp_path):
    bad = tmp_path / "missing-dir" / "report.md"

    with pytest.raises(SystemExit):
        report.write_report(bad, **report_inputs)
