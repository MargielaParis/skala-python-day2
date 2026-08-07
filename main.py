"""
[Day 2 종합실습] End2End 데이터 분석 프로젝트 — Adult Census Income

프로그램 설명
data/raw/adult.data(학습) + data/raw/adult.test(평가) — UCI 공식 분할본을 대상으로
아래 5단계를 수행한다.
1) 데이터 준비: train·test 각각 Pandas·Polars 로딩 비교, 결측치·중복 처리, 기본 EDA
2) 시각화: Seaborn 정적 차트(PNG) 3장(핵심 4패널 + 전체 변수 부록 2장),
   Plotly 인터랙티브 차트(HTML)
3) 통계 분석: 기술통계·상관계수 산출, 소득 그룹 간 근로시간 t-test 및 p-value 해석
4) ML Pipeline: 전처리+로지스틱 회귀를 Pipeline으로 학습(train), 평가(test), joblib 저장
5) 자동화: 분석 결과를 output/report.md로 자동 생성

실행 방법: python main.py  (세부 로직은 src/ 모듈 참조)

변경내역
2026-08-07 최초 작성
2026-08-07 실습 3·4 형식 반영 (구분선·단계별 행 수 출력, Welch t-test)
2026-08-07 train/test 8:2 랜덤 분할 -> UCI 공식 adult.data/adult.test 분할로 변경
2026-08-07 차트 재설계, 전체 변수 플롯 추가, 회귀계수(오즈비) 출력 추가

작성자: 박기연 (판교 7반)
"""

from pathlib import Path

from src import load, ml, report, stats_test, viz

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

print(f"Using data: {TRAIN_FILE.name} (train) / {TEST_FILE.name} (test)")


def prepare(file_path, label):
    # 파일 하나를 로딩·정제하고 단계별 행 수를 출력해 (df, loading, cleaning) 반환
    df_raw, loading = load.load_compare(file_path)
    print(
        f"[{label}] 로딩 {loading['rows']:,}행: "
        f"Pandas {loading['pandas_sec']:.3f}초 / Polars {loading['polars_sec']:.3f}초"
    )
    df, cleaning = load.clean(df_raw)
    print(f"[{label}] 결측 컬럼: {cleaning['na_cols']}")
    print(
        f"[{label}] 결측치 제거: {cleaning['raw']:,}행 -> {cleaning['after_na']:,}행 "
        f"(제거 {cleaning['raw'] - cleaning['after_na']:,}행)"
    )
    print(
        f"[{label}] 중복 제거: {cleaning['after_na']:,}행 -> {cleaning['clean']:,}행 "
        f"(제거 {cleaning['after_na'] - cleaning['clean']:,}행)"
    )
    return df, loading, cleaning


# =========================================================
# 2. 파이프라인 실행
# =========================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("\n1) 데이터 준비 — Pandas·Polars 로딩 비교 + 정제 (train/test 각각)")
    df_train, loading, cleaning = prepare(TRAIN_FILE, "train")
    print()
    df_test, loading_te, cleaning_te = prepare(TEST_FILE, "test")

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
    print("\n3) 통계 분석 — 기술통계·상관·t-test (train 기준)")
    describe, corr = stats_test.describe_numeric(df_train)
    print(describe.round(1).iloc[:3])
    ttest = stats_test.ttest_hours_by_income(df_train)
    print(f"주당 근로시간: >50K {ttest['mean_high']:.1f} vs <=50K {ttest['mean_low']:.1f}")
    print(f"t={ttest['t']:.3f}, p={ttest['p_text']}")
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
        f"(sex·race 포함) -> 원핫 후 {metrics['n_features']}개, 제외 {metrics['dropped']}"
    )
    print(
        f"학습 {metrics['train']:,}건 (>50K 비율 {metrics['train_pos_rate']:.3f}) / "
        f"평가 {metrics['test']:,}건 (>50K 비율 {metrics['test_pos_rate']:.3f})"
    )
    print(f"정확도 {metrics['accuracy']:.4f} / F1 {metrics['f1']:.4f}")
    print(f"[PASS] 모델 저장·재로딩 검증 -> {MODEL_FILE.name}")

    print("\n4) 회귀계수 — 고소득 확률을 올리는/내리는 상위 5개 (오즈비)")
    coef = metrics["coef"]
    for name, row in coef.tail(5)[::-1].iterrows():
        print(f"  + {name:<32} {row['coef']:+.3f}  odds x{row['odds_ratio']:.2f}")
    for name, row in coef.head(5).iterrows():
        print(f"  - {name:<32} {row['coef']:+.3f}  odds x{row['odds_ratio']:.2f}")
    print("\n4) sex·race 계수")
    for name, row in coef.loc[coef.index.str.startswith(("sex_", "race_"))].iterrows():
        print(f"    {name:<32} {row['coef']:+.3f}  odds x{row['odds_ratio']:.2f}")

    print("\n" + "=" * 80)
    print("\n5) 자동화 — report.md 생성")
    report.write_report(
        REPORT_FILE, loading, cleaning, loading_te, cleaning_te, describe, corr, ttest, metrics
    )
    print(f"저장 완료 -> {REPORT_FILE.name}")

    print(
        f"\nDone! 산출물: {CHART_FILE.name}, {NUM_CHART_FILE.name}, {CAT_CHART_FILE.name}, "
        f"{HTML_FILE.name}, {MODEL_FILE.name}, {REPORT_FILE.name}"
    )
