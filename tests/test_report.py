"""report 모듈 — report.md 자동 생성 검증"""

import pytest

from src import ml, report, stats_test


@pytest.fixture
def report_inputs(sample_df, tmp_path):
    # main.py가 넘기는 것과 같은 모양의 단계별 결과 dict를 구성
    loading = {"rows": len(sample_df), "cols": 15, "pandas_sec": 0.01, "polars_sec": 0.02}
    cleaning = {"raw": 70, "after_na": 65, "clean": 60, "na_cols": {"workclass": 5}}
    describe, corr = stats_test.describe_numeric(sample_df)
    ttest = stats_test.ttest_hours_by_income(sample_df)
    metrics = ml.train_and_evaluate(sample_df, sample_df, tmp_path / "pipeline.pkl")
    return loading, cleaning, describe, corr, ttest, metrics


def test_write_report_creates_all_sections(report_inputs, tmp_path):
    loading, cleaning, describe, corr, ttest, metrics = report_inputs
    path = tmp_path / "report.md"

    report.write_report(path, loading, cleaning, loading, cleaning, describe, corr, ttest, metrics)

    md = path.read_text(encoding="utf-8")
    for heading in ("## 1. 데이터 준비", "## 3. 통계 분석", "## 4. ML Pipeline", "## 5. 결론"):
        assert heading in md
    assert f"{metrics['accuracy']:.4f}" in md
    assert "sex_" in md


def test_write_report_exits_on_bad_path(report_inputs, tmp_path):
    loading, cleaning, describe, corr, ttest, metrics = report_inputs
    bad = tmp_path / "missing-dir" / "report.md"

    with pytest.raises(SystemExit):
        report.write_report(
            bad, loading, cleaning, loading, cleaning, describe, corr, ttest, metrics
        )
