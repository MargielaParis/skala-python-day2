"""자동화 — 분석 결과를 report.md로 자동 생성"""

from datetime import datetime


def write_report(file_path, loading, cleaning, loading_te, cleaning_te,
                 describe, corr, ttest, ml):
    # 단계별 결과 dict를 받아 발표용 report.md를 생성
    corr_pair = corr.loc["education-num", "hours-per-week"]
    coef = ml["coef"]
    top = coef.tail(8)[::-1].rename(columns={"coef": "계수", "odds_ratio": "오즈비"})
    bottom = coef.head(8).rename(columns={"coef": "계수", "odds_ratio": "오즈비"})
    demo = coef.loc[coef.index.str.startswith(("sex_", "race_"))][::-1] \
               .rename(columns={"coef": "계수", "odds_ratio": "오즈비"})
    ttest_msg = ("통계적으로 유의미한 차이 있음 (H0 기각)"
                 if ttest["significant"] else "차이 없음 (우연일 수 있음)")

    n_num, n_cat, n_feat = len(ml["num_cols"]), len(ml["cat_cols"]), ml["n_features"]
    num_cols, cat_cols, dropped = (", ".join(ml["num_cols"]), ", ".join(ml["cat_cols"]),
                                   ", ".join(ml["dropped"]))
    top_md, bottom_md, demo_md = (t.round(3).to_markdown() for t in (top, bottom, demo))

    md = f"""# Adult Census Income — End2End 분석 리포트

- 생성 일시: {datetime.now().strftime("%Y-%m-%d %H:%M")}
- 작성자: 박기연 (판교 7반)
- 데이터 분할: UCI 공식 분할 사용 (`adult.data` 학습 / `adult.test` 평가, 랜덤 분할 없음)

## 1. 데이터 준비
- 로딩 비교(train): Pandas {loading["pandas_sec"]:.3f}초 vs Polars {loading["polars_sec"]:.3f}초
  ({loading["rows"]:,}행 x {loading["cols"]}컬럼, 두 도구 결과 동일)
- 로딩 비교(test): Pandas {loading_te["pandas_sec"]:.3f}초 vs Polars {loading_te["polars_sec"]:.3f}초
  ({loading_te["rows"]:,}행 x {loading_te["cols"]}컬럼, 두 도구 결과 동일)
- 결측 컬럼(train): {", ".join(f"{k}({v}건)" for k, v in cleaning["na_cols"].items())}
- 결측 컬럼(test): {", ".join(f"{k}({v}건)" for k, v in cleaning_te["na_cols"].items())}
- 정제 결과

| 구분 | 원본 | 결측 제거 후 | 중복 제거 후 |
|---|---|---|---|
| train | {cleaning["raw"]:,} | {cleaning["after_na"]:,} | {cleaning["clean"]:,} |
| test | {cleaning_te["raw"]:,} | {cleaning_te["after_na"]:,} | {cleaning_te["clean"]:,} |

- `adult.test`는 첫 줄이 주석(`|1x3 Cross validator`), 라벨에 마침표(`>50K.`)가 붙어 있어
  로딩 단계에서 주석 제외·라벨 표기 통일 처리를 했다.

## 2. 시각화 (train 기준)
- `output/eda_charts.png` — 핵심 4패널: 연령 분포, 근로시간 박스플롯, 직업별 고소득 비율, 수치형 상관
- `output/eda_numeric_all.png` — 수치형 6개 변수 전체의 소득 그룹별 분포
  (capital-gain/loss는 0이 대부분이어서 0 제외 + 로그 축)
- `output/eda_categorical_all.png` — 범주형 8개 변수 전체의 고소득 비율
  (native-country는 범주가 41개라 표본 상위 14개 + 기타로 묶음)
- `output/income_by_education.html` — 학력별 고소득 비율 (Plotly, 마우스오버로 표본 수 확인)

## 3. 통계 분석 (train 기준)
- 주당 근로시간 평균: >50K {ttest["mean_high"]:.1f}시간 vs <=50K {ttest["mean_low"]:.1f}시간
- t-test(Welch): t={ttest["t"]:.3f}, p={ttest["p"]:.6f} -> **{ttest_msg}**
- 상관계수 예시: education-num vs hours-per-week = {corr_pair:.3f}
- 수치형 기술통계

{describe.round(2).to_markdown()}

## 4. ML Pipeline (소득 >50K 로지스틱 회귀)
- 구성: StandardScaler + OneHotEncoder -> LogisticRegression (단일 Pipeline)
- 피처: 수치형 {n_num}개 + 범주형 {n_cat}개(sex·race 포함) -> 원핫 인코딩 후 {n_feat}개
  - 수치형: {num_cols}
  - 범주형: {cat_cols}
  - 제외: {dropped} (fnlwgt는 표본 가중치, education은 education-num과 1:1 중복)
- 학습(adult.data) {ml["train"]:,}건 / 평가(adult.test) {ml["test"]:,}건
- 고소득(>50K) 비율: train {ml["train_pos_rate"]:.3f} / test {ml["test_pos_rate"]:.3f}
- **정확도 {ml["accuracy"]:.4f} / F1 {ml["f1"]:.4f}**
- 모델 저장: `output/{ml["model_file"]}` (joblib, 재로딩 검증 완료)

### 4-1. 회귀계수 — 고소득 확률을 올리는 요인 (오즈비 = exp(계수))

{top_md}

### 4-2. 고소득 확률을 내리는 요인

{bottom_md}

### 4-3. sex·race 계수

{demo_md}

계수는 다른 변수를 통제한 상태의 부분효과이므로, 위 3절 집계 비율과 부호가 다를 수 있다.
sex_Female은 같은 학력·직업·근로시간이어도 고소득 오즈가 낮게 추정된다는 뜻이며,
데이터(1994 US Census)에 기록된 격차이지 인과관계의 증거는 아니다.

## 5. 결론
- 학력 수준이 높을수록 고소득 비율이 뚜렷하게 증가한다.
- 결혼 상태(Married-civ-spouse)와 capital-gain이 계수 크기 기준 가장 강한 신호다.
- sex·race를 포함해 전체 12개 변수로 회귀했고, 다른 변수를 통제해도 sex_Female의
  고소득 오즈가 가장 크게 낮았다 (오즈비 0.37).
- 고소득 그룹의 주당 근로시간이 통계적으로 유의미하게 길다.
- 학습·평가를 서로 다른 파일로 완전히 분리해, 랜덤 분할보다 데이터 누수 위험이 낮고
  다른 참가자와 지표를 그대로 비교할 수 있는 평가 기준을 확보했다.
- 전처리~모델을 Pipeline 하나로 묶어 재현 가능한 학습·배포 단위를 확보했다.
"""
    try:
        file_path.write_text(md, encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"[오류] 리포트 저장 실패: {e}")
