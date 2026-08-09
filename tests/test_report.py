"""report 모듈 — report.md 자동 생성과 '입력이 바뀌면 결론도 바뀐다' 회귀 검증"""

import copy

import pytest

from src import ml, report, stats_test
from src.errors import PipelineError


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
    ab = ml.compare_strategies({"drop": sample_df}, sample_df)
    caveats = stats_test.data_caveats(sample_df, sample_df)
    sensitivity = ml.sensitivity_without(sample_df, sample_df, "relationship")
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
        "caveats": caveats,
        "sensitivity": sensitivity,
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
    original = f"대비 {coef.loc[sex_row, 'odds_ratio']:.3f}배"
    assert original in md_before

    changed = copy.deepcopy(report_inputs)
    changed["ml"]["coef"].loc[sex_row, "odds_ratio"] = 0.123
    md_after = _render(changed, tmp_path, "after.md")

    assert "대비 0.123배" in md_after
    assert original not in md_after


def test_report_has_no_hardcoded_conclusion_numbers(report_inputs, tmp_path):
    md = _render(report_inputs, tmp_path)

    assert "오즈비 0.37" not in md
    assert "가장 강한 신호" not in md


def test_report_shows_sample_size_behind_each_contrast(report_inputs, tmp_path):
    # 표본 18행짜리 범주의 오즈비가 27,000행짜리와 똑같이 단정적으로 보이면 안 된다
    md = _render(report_inputs, tmp_path)
    counts = report_inputs["ml"]["category_counts"]
    reference = report_inputs["ml"]["reference"]

    assert "학습 표본" in md
    assert f"기준 {counts['sex'][reference['sex']]:,}" in md


def test_report_discloses_data_caveats(report_inputs, tmp_path):
    md = _render(report_inputs, tmp_path)
    caveats = report_inputs["caveats"]

    assert "데이터 한계" in md
    assert f"{caveats['overlap_rows']:,}건" in md  # train/test 교차 중복
    assert "capital-gain" in md and "상한" in md  # top-coding
    assert "relationship" in md  # sex와의 겹침
    assert "성별 정체성" in md  # 1994년 행정 분류 표기


def test_report_uses_association_wording_not_causal(report_inputs, tmp_path):
    md = _render(report_inputs, tmp_path)

    assert "고소득 오즈와 양의 연관이 큰 항목" in md
    assert "고소득 확률을 올리는 요인" not in md


def test_report_does_not_claim_directly_comparable_metrics(report_inputs, tmp_path):
    # 평가셋에도 정제를 적용했으므로 "그대로 비교 가능"이라고 쓰면 1-4절과 모순된다
    md = _render(report_inputs, tmp_path)

    assert "그대로 비교할 수 있는" not in md
    assert "정제 절차부터 맞춰야" in md


def test_strategy_verdict_declines_to_rank_a_tie(report_inputs, tmp_path):
    # F1 격차가 0.001 수준이면 "가장 높았다"고 쓰면 안 된다
    tied = copy.deepcopy(report_inputs)
    base = dict(tied["ab"]["drop"])
    tied["ab"] = {"drop": base, "unknown": {**base, "f1": base["f1"] - 0.0008}}

    md = _render(tied, tmp_path, "tie.md")

    assert "우열을 가릴 수 없다" in md
    assert "가장 높았다" not in md


def test_strategy_verdict_names_a_winner_when_gap_is_real(report_inputs, tmp_path):
    clear = copy.deepcopy(report_inputs)
    base = dict(clear["ab"]["drop"])
    clear["ab"] = {"drop": base, "unknown": {**base, "f1": base["f1"] - 0.2}}

    md = _render(clear, tmp_path, "clear.md")

    assert "가장 높았다" in md


def test_report_compares_missing_value_strategies(report_inputs, tmp_path):
    md = _render(report_inputs, tmp_path)

    assert "결측 처리 방식 A/B 비교" in md
    assert report.STRATEGY_LABEL["drop"] in md


def test_write_report_raises_on_bad_path(report_inputs, tmp_path):
    bad = tmp_path / "missing-dir" / "report.md"

    with pytest.raises(PipelineError, match="리포트 저장 실패"):
        report.write_report(bad, **report_inputs)


def test_report_shows_confidence_intervals(report_inputs, tmp_path):
    # 점추정만 보여주면 18행짜리 범주의 오즈비를 27,000행짜리와 같은 확신으로 읽게 된다
    md = _render(report_inputs, tmp_path)

    assert "오즈비 95% 구간" in md
    assert "라플라스 근사" in md
    assert "구간이 1을 포함하면" in md


def test_report_includes_threshold_analysis(report_inputs, tmp_path):
    md = _render(report_inputs, tmp_path)
    tuned = report_inputs["ml"]["thresholds"]["tuned"]

    assert "## 4. ML Pipeline" in md and "4-2. 분류 임계값" in md
    assert "PR-AUC" in md
    assert f"{tuned['threshold']:.3f}" in md
    assert "out-of-fold" in md  # 임계값 탐색 방법을 밝힌다


def test_report_states_the_ab_test_set_is_shared(report_inputs, tmp_path):
    # 전략별 평가셋이 다르면 비교가 성립하지 않으므로, 고정했다는 사실을 밝혀야 한다
    md = _render(report_inputs, tmp_path)

    assert "평가셋은 두 전략 모두" in md
    assert "나란히" in md


def test_report_reports_unseen_category_rows(report_inputs, tmp_path):
    md = _render(report_inputs, tmp_path)
    caveats = report_inputs["caveats"]

    assert "학습에 없던 범주가 평가셋에 있으면" in md
    assert f"{caveats['unseen_category_rows']:,}행" in md


def test_report_includes_sensitivity_section(report_inputs, tmp_path):
    md = _render(report_inputs, tmp_path)

    assert "민감도 분석" in md
    assert report_inputs["sensitivity"]["excluded"] in md


def test_conclusion_follows_sensitivity_result(report_inputs, tmp_path):
    # 민감도 결과도 고정 문장이 아니라 입력에서 파생돼야 한다
    changed = copy.deepcopy(report_inputs)
    changed["sensitivity"]["coef"]["odds_ratio"] = 0.777

    md = _render(changed, tmp_path, "sens.md")

    assert "0.777" in md
