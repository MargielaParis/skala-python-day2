"""
[Day 2 종합실습] End2End 데이터 분석 프로젝트 — Adult Census Income

프로그램 설명
data/raw/adult.data(학습) + data/raw/adult.test(평가) — UCI 공식 분할본을 대상으로
아래 5단계를 수행한다.
1) 데이터 준비: train·test 각각 Pandas·Polars 로딩·동등성 검증, 결측치·중복 처리, 기본 EDA
2) 시각화: Seaborn 정적 차트(PNG) 3장(핵심 4패널 + 전체 변수 부록 2장),
   Plotly 인터랙티브 차트(HTML)
3) 통계 분석: 성별x소득 카이제곱, 소득 그룹 간 근로시간 t-test(효과크기 포함), 기술통계·상관
4) ML Pipeline: 전처리+로지스틱 회귀를 Pipeline으로 학습(train), 평가(test), joblib 저장
   계수 신뢰구간·임계값 분석·민감도 분석 포함
5) 자동화: 분석 결과를 output/report.md로 자동 생성

실행 방법: python main.py  (세부 로직은 src/ 모듈 참조)

변경내역
2026-08-07 최초 작성
2026-08-07 실습 3·4 형식 반영 (구분선·단계별 행 수 출력, Welch t-test)
2026-08-07 train/test 8:2 랜덤 분할 -> UCI 공식 adult.data/adult.test 분할로 변경
2026-08-07 차트 재설계, 전체 변수 플롯 추가, 회귀계수(오즈비) 출력 추가
2026-08-09 Issue #5·#6·#7 반영 — Pandas·Polars 값 동등성 검증, 원핫 기준 범주 명시,
           성별x소득 카이제곱, t-test 효과크기, 혼동행렬·정밀도·재현율·ROC-AUC,
           결측 처리 A/B 비교, 리포트 고정 결론 제거
2026-08-09 계수 신뢰구간(라플라스 근사), PR-AUC·임계값 분석, capital-gain 상한 지시변수,
           relationship 제외 민감도 분석 추가. SystemExit 대신 PipelineError 사용

작성자: 박기연 (판교 7반)
"""

from pathlib import Path
from typing import Any

import pandas as pd

from src import load, ml, report, stats_test, viz
from src.errors import PipelineError

# =========================================================
# 1. 환경 설정 및 파라미터
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "data" / "raw" / "adult.data"
TEST_FILE = BASE_DIR / "data" / "raw" / "adult.test"
OUTPUT_DIR = BASE_DIR / "output"
CHART_FILE = OUTPUT_DIR / "eda_charts.png"
NUM_CHART_FILE = OUTPUT_DIR / "eda_numeric_all.png"
CAT_CHART_FILE = OUTPUT_DIR / "eda_categorical_all.png"
HTML_FILE = OUTPUT_DIR / "income_by_education.html"
MODEL_FILE = OUTPUT_DIR / "income_pipeline.pkl"
REPORT_FILE = OUTPUT_DIR / "report.md"
# 성별과 거의 겹치는 변수 — 이 변수를 뺀 모델로 성별 계수의 통제 의존도를 확인한다
SENSITIVITY_EXCLUDE = "relationship"


def prepare(
    file_path: Path, label: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    # 파일 하나를 로딩·정제하고 단계별 행 수를 출력해 (원본 df, 정제 df, loading, cleaning) 반환
    df_raw, loading = load.load_compare(file_path)
    print(
        f"[{label}] 로딩 {loading['rows']:,}행: "
        f"Pandas {loading['pandas_sec']:.3f}초 / Polars {loading['polars_sec']:.3f}초 "
        f"(단일 참고 측정)"
    )
    eq = loading["equality"]
    print(
        f"[{label}] Pandas·Polars 동등성: dtype 불일치 {len(eq['dtype_mismatch'])}컬럼 / "
        f"null 불일치 {eq['null_mismatch']}셀 / 값 불일치 {eq['value_mismatch']}셀 "
        f"-> {'동일' if eq['identical'] else '불일치'}"
    )
    df, cleaning = load.clean(df_raw)
    print(f"[{label}] 결측 컬럼: {cleaning['na_cols']}")
    print(
        f"[{label}] 결측치 제거: {cleaning['raw']:,}행 -> {cleaning['after_na']:,}행 "
        f"(제거 {cleaning['raw'] - cleaning['after_na']:,}행)"
    )
    print(
        f"[{label}] 소득 그룹별 제거율: "
        + ", ".join(f"{k} {v:.2%}" for k, v in cleaning["drop_rate_by_income"].items())
    )
    print(
        f"[{label}] 중복 제거: {cleaning['after_na']:,}행 -> {cleaning['clean']:,}행 "
        f"(제거 {cleaning['after_na'] - cleaning['clean']:,}행)"
    )
    return df_raw, df, loading, cleaning


def run() -> None:
    # 전체 파이프라인 5단계를 순서대로 수행한다
    print(f"Using data: {TRAIN_FILE.name} (train) / {TEST_FILE.name} (test)")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("\n1) 데이터 준비 — Pandas·Polars 로딩·동등성 검증 + 정제 (train/test 각각)")
    raw_train, df_train, loading, cleaning = prepare(TRAIN_FILE, "train")
    print()
    raw_test, df_test, loading_te, cleaning_te = prepare(TEST_FILE, "test")

    print("\n1) 기본 EDA (train 기준)")
    print(df_train.describe(include="all").iloc[:4, :6])

    print("\n" + "=" * 80)
    print("\n2) 시각화 — Seaborn PNG + Plotly HTML (train 기준)")
    viz.save_seaborn_charts(df_train, CHART_FILE)
    viz.save_numeric_panels(df_train, NUM_CHART_FILE)
    viz.save_categorical_panels(df_train, CAT_CHART_FILE)
    viz.save_plotly_chart(df_train, HTML_FILE)
    print(
        f"저장 완료 -> {CHART_FILE.name}, {NUM_CHART_FILE.name}, "
        f"{CAT_CHART_FILE.name}, {HTML_FILE.name}"
    )

    print("\n" + "=" * 80)
    print("\n3) 통계 분석 — 카이제곱·t-test·기술통계 (train 기준)")
    chisq = stats_test.chisq_sex_income(df_train)
    print("성별 고소득 비율: " + ", ".join(f"{k} {v:.1%}" for k, v in chisq["rate_by_sex"].items()))
    print(
        f"카이제곱: chi2={chisq['chi2']:.3f}, dof={chisq['dof']}, p={chisq['p_text']}, "
        f"Cramer's V={chisq['cramers_v']:.3f} (연속성 보정 없음, 최소 기대빈도 "
        f"{chisq['expected_min']:,.0f})"
    )
    print(
        "-> 성별과 소득은 독립이 아님 (H0 기각)"
        if chisq["significant"]
        else "-> 독립 가설을 기각하지 못함"
    )

    describe, corr = stats_test.describe_numeric(df_train)
    print(describe.round(1).iloc[:3])
    ttest = stats_test.ttest_hours_by_income(df_train)
    print(f"\n주당 근로시간: >50K {ttest['mean_high']:.1f} vs <=50K {ttest['mean_low']:.1f}")
    print(f"t={ttest['t']:.3f}, p={ttest['p_text']}")
    print(
        f"차이 {ttest['diff']:+.2f}시간, 95% CI [{ttest['ci_low']:.2f}, {ttest['ci_high']:.2f}], "
        f"Cohen's d={ttest['cohens_d']:.3f} ({ttest['effect']})"
    )
    print(
        "-> 통계적으로 유의미한 차이 있음"
        if ttest["significant"]
        else "-> 차이 없음 (우연일 수 있음)"
    )

    print("\n" + "=" * 80)
    print("\n4) ML Pipeline — adult.data 학습 / adult.test 평가·저장")
    metrics = ml.train_and_evaluate(df_train, df_test, MODEL_FILE)
    print(
        f"피처: 수치 {len(metrics['num_cols'])}개 + 범주 {len(metrics['cat_cols'])}개"
        f"(sex·race 포함) -> 원핫(기준 범주 제외) 후 {metrics['n_features']}개, "
        f"제외 {metrics['dropped']}"
    )
    print(
        f"학습 {metrics['train']:,}건 (>50K 비율 {metrics['train_pos_rate']:.3f}) / "
        f"평가 {metrics['test']:,}건 (>50K 비율 {metrics['test_pos_rate']:.3f})"
    )
    print(
        f"정확도 {metrics['accuracy']:.4f} / 정밀도 {metrics['precision']:.4f} / "
        f"재현율 {metrics['recall']:.4f} / F1 {metrics['f1']:.4f} / "
        f"ROC-AUC {metrics['roc_auc']:.4f} / PR-AUC {metrics['pr_auc']:.4f}"
    )
    cm = metrics["confusion"]
    print(
        f"혼동행렬: TN {cm['tn']:,} / FP {cm['fp']:,} / FN {cm['fn']:,} / TP {cm['tp']:,}"
        f"  (실제 고소득 {cm['tp'] + cm['fn']:,}명 중 {cm['fn']:,}명을 저소득으로 놓침)"
    )
    print(f"[PASS] 모델 저장·재로딩 검증 -> {MODEL_FILE.name}")

    tuned = metrics["thresholds"]["tuned"]
    print(
        f"\n4) 임계값 — 기본 0.500 재현율 {metrics['recall']:.4f}(FN {cm['fn']:,}) vs "
        f"{metrics['thresholds']['cv']}-폴드 교차검증 F1 최적 {tuned['threshold']:.3f} "
        f"재현율 {tuned['recall']:.4f}(FN {tuned['confusion']['fn']:,})"
    )

    print("\n4) 결측 처리 A/B — dropna vs 범주형 Unknown 보존 (평가셋 고정)")
    unknown_train, _ = load.clean(raw_train, strategy=load.UNKNOWN)
    # 학습셋만 바꾸고 평가셋은 df_test로 고정한다. 전략별 평가셋을 따로 쓰면 비교가 성립하지 않는다.
    # drop 전략은 위에서 이미 같은 평가셋으로 학습·평가했으므로 그 지표를 재사용한다.
    ab = {load.DROP: metrics} | ml.compare_strategies({load.UNKNOWN: unknown_train}, df_test)
    for name, m in ab.items():
        print(
            f"  {name:<8} 학습 {m['train']:,}건 -> 정확도 {m['accuracy']:.4f} / "
            f"정밀도 {m['precision']:.4f} / 재현율 {m['recall']:.4f} / F1 {m['f1']:.4f} / "
            f"PR-AUC {m['pr_auc']:.4f}"
        )

    print("\n4) 회귀계수 — 고소득 오즈와 연관이 큰 상위 5개 (기준 범주 대비 오즈비)")
    coef = metrics["coef"]
    print(f"  기준 선정: {metrics['reference_rule']}")
    print(f"  기준 범주: {metrics['reference']}")
    for name, row in coef.tail(5)[::-1].iterrows():
        print(
            f"  + {name:<32} {row['coef']:+.3f}  odds x{row['odds_ratio']:.2f} "
            f"[{row['or_low']:.2f}, {row['or_high']:.2f}]"
        )
    for name, row in coef.head(5).iterrows():
        print(
            f"  - {name:<32} {row['coef']:+.3f}  odds x{row['odds_ratio']:.2f} "
            f"[{row['or_low']:.2f}, {row['or_high']:.2f}]"
        )
    print("\n4) sex·race 계수 (범주형은 기준 범주 대비, 대괄호는 95% 근사 구간)")
    for name, row in coef.loc[coef.index.str.startswith(("sex_", "race_"))].iterrows():
        print(
            f"    {name:<32} {row['coef']:+.3f}  odds x{row['odds_ratio']:.2f} "
            f"[{row['or_low']:.2f}, {row['or_high']:.2f}]"
        )

    print(f"\n4) 민감도 — {SENSITIVITY_EXCLUDE} 제외 재학습")
    sensitivity = ml.sensitivity_without(df_train, df_test, SENSITIVITY_EXCLUDE)
    for name, row in sensitivity["coef"].iterrows():
        print(
            f"    {name:<32} odds x{row['odds_ratio']:.2f} "
            f"[{row['or_low']:.2f}, {row['or_high']:.2f}] "
            f"(기준 {sensitivity['focus']}={sensitivity['reference']})"
        )

    print("\n" + "=" * 80)
    print("\n5) 자동화 — report.md 생성")
    caveats = stats_test.data_caveats(df_train, df_test)
    print(
        f"데이터 한계: train/test 동일 행 {caveats['overlap_rows']}건, "
        f"미지 범주 행 {caveats['unseen_category_rows']}건, "
        f"capital-gain 상한값 {caveats['capital_gain_capped']}행, "
        f"relationship 성별 편중 "
        + ", ".join(f"{k} {v:.1%}" for k, v in caveats["role_purity"].items())
    )
    report.write_report(
        REPORT_FILE,
        loading=loading,
        cleaning=cleaning,
        loading_te=loading_te,
        cleaning_te=cleaning_te,
        describe=describe,
        corr=corr,
        ttest=ttest,
        chisq=chisq,
        ml=metrics,
        ab=ab,
        caveats=caveats,
        sensitivity=sensitivity,
    )
    print(f"저장 완료 -> {REPORT_FILE.name}")

    print(
        f"\nDone! 산출물: {CHART_FILE.name}, {NUM_CHART_FILE.name}, {CAT_CHART_FILE.name}, "
        f"{HTML_FILE.name}, {MODEL_FILE.name}, {REPORT_FILE.name}"
    )


# =========================================================
# 2. 파이프라인 실행
# =========================================================
if __name__ == "__main__":
    # src/ 모듈은 프로세스를 직접 끝내지 않는다. 종료 코드는 여기서만 정한다
    try:
        run()
    except PipelineError as error:
        raise SystemExit(f"[오류] {error}") from error
