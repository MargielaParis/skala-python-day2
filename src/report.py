"""자동화 — 분석 결과를 report.md로 자동 생성

리포트의 수치·결론 문장은 모두 인자로 받은 결과에서 파생시킨다.
입력이 바뀌면 결론도 함께 바뀌어야 하므로 고정 문장을 두지 않는다.
"""

from datetime import datetime

STRATEGY_LABEL = {"drop": "결측 행 삭제(dropna)", "unknown": "범주형 결측을 Unknown으로 보존"}


def _equality_text(loading):
    # "두 도구 결과 동일"을 실제 비교 결과에서만 만든다
    eq = loading["equality"]
    if eq.get("identical"):
        return "정규화 후 컬럼 순서·shape·null mask·의미상 dtype·셀 값 불일치 0"
    parts = [
        f"dtype {len(eq['dtype_mismatch'])}컬럼",
        f"null {eq['null_mismatch']}셀",
        f"값 {eq['value_mismatch']}셀",
    ]
    return "불일치 있음 — " + ", ".join(parts)


def _reference_note(reference):
    # 원핫에서 빠진 기준 범주 목록 — 모든 범주형 계수는 이 기준 대비 값이다
    return ", ".join(f"{col}={ref}" for col, ref in reference.items() if ref is not None)


def _annotate(coef_slice, reference, counts, n_train):
    # 각 계수가 무엇과 비교된 값인지(기준 범주 / 1 표준편차)와 그 대비를 만든 표본 수를 함께 적는다.
    # 표본 수가 없으면 18행짜리 범주의 오즈비가 27,000행짜리 범주와 똑같이 단정적으로 보인다.
    basis, sizes = [], []
    for name in coef_slice.index:
        feature = next((c for c in reference if name.startswith(f"{c}_")), None)
        if feature is None:
            basis.append("1 표준편차 증가")
            sizes.append(f"{n_train:,}")
            continue
        level = name[len(feature) + 1 :]
        basis.append(f"기준 {feature}={reference[feature]}")
        ref_n = counts.get(feature, {}).get(reference[feature], 0)
        sizes.append(f"{counts.get(feature, {}).get(level, 0):,} / 기준 {ref_n:,}")
    table = coef_slice.assign(**{"비교 기준": basis, "학습 표본": sizes})
    return table.rename(columns={"coef": "계수", "odds_ratio": "오즈비"}).round(3).to_markdown()


def _sex_contrast(coef, reference):
    # 성별 계수를 "기준 대비 오즈비" 문장으로 파생 (고정 숫자를 쓰지 않기 위함)
    ref = reference.get("sex")
    rows = coef.loc[coef.index.str.startswith("sex_")]
    if ref is None or rows.empty:
        return "sex 계수가 모델에 없어 성별 대비를 계산하지 않았다."
    parts = [
        f"{name.split('_', 1)[1]}는 기준({ref}) 대비 오즈 {row['odds_ratio']:.3f}배"
        for name, row in rows.iterrows()
    ]
    return (
        "학력·직업·근로시간과 가구 내 역할(relationship)까지 고정했을 때 "
        + ", ".join(parts)
        + "로 추정됐다. relationship은 성별과 거의 겹치므로 이 값은 성별 전체 격차가 아니다."
    )


def _confusion_md(cm, positive=">50K"):
    return "\n".join(
        [
            f"| 실제 \\ 예측 | <=50K | {positive} |",
            "|---|---:|---:|",
            f"| <=50K | {cm['tn']:,} | {cm['fp']:,} |",
            f"| {positive} | {cm['fn']:,} | {cm['tp']:,} |",
        ]
    )


def _strategy_md(ab):
    header = "| 결측 전략 | 학습 행 수 | 평가 행 수 | 정확도 | 정밀도 | 재현율 | F1 | ROC-AUC |"
    rows = [header, "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in ab.items():
        rows.append(
            f"| {STRATEGY_LABEL.get(name, name)} | {m['train']:,} | {m['test']:,} | "
            f"{m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['roc_auc']:.4f} |"
        )
    return "\n".join(rows)


def _na_impact_md(cleaning, cleaning_te):
    header = "| 구분 | 소득 그룹 | 원본 행 | 결측 행 | 제거율 |"
    rows = [header, "|---|---|---:|---:|---:|"]
    for label, info in (("train", cleaning), ("test", cleaning_te)):
        for group, total in info["total_by_income"].items():
            removed = info["removed_by_income"][group]
            rows.append(
                f"| {label} | {group} | {total:,} | {removed:,} | "
                f"{info['drop_rate_by_income'][group]:.2%} |"
            )
    return "\n".join(rows)


def _conclusions(ttest, chisq, ml, ab):
    # 결론은 전부 입력에서 파생한다 — 입력이 바뀌면 이 문장들도 함께 바뀐다
    coef, reference = ml["coef"], ml["reference"]
    strongest = coef["coef"].abs().idxmax()
    strongest_row = coef.loc[strongest]
    rates = ", ".join(f"{k} {v:.1%}" for k, v in chisq["rate_by_sex"].items())
    best = max(ab, key=lambda k: ab[k]["f1"]) if ab else None
    cm = ml["confusion"]

    lines = [
        f"- 성별과 소득은 카이제곱 검정에서 "
        f"{'독립 가설을 기각했다' if chisq['significant'] else '독립 가설을 기각하지 못했다'} "
        f"(chi2={chisq['chi2']:.1f}, dof={chisq['dof']}, p={chisq['p_text']}, "
        f"Cramer's V={chisq['cramers_v']:.3f}). 성별 고소득 비율: {rates}.",
        f"- 고소득 그룹의 주당 근로시간은 저소득 그룹보다 {ttest['diff']:+.1f}시간이고 "
        f"95% 신뢰구간은 [{ttest['ci_low']:.2f}, {ttest['ci_high']:.2f}], "
        f"Cohen's d={ttest['cohens_d']:.3f}({ttest['effect']})로 "
        f"{'통계적으로 유의미한 차이가 있다' if ttest['significant'] else '유의미한 차이가 없다'}.",
        f"- 계수 절대값이 가장 큰 항은 `{strongest}`이며 "
        f"계수 {strongest_row['coef']:+.3f}, 오즈비 {strongest_row['odds_ratio']:.3f}다. "
        f"범주형은 기준 범주 대비, 수치형은 1 표준편차 증가 기준이며, "
        f"계수 크기만으로 중요도를 단정할 수는 없다(표준오차 미산출).",
        f"- {_sex_contrast(coef, reference)}",
        f"- 평가 성능은 정확도 {ml['accuracy']:.4f}, 정밀도 {ml['precision']:.4f}, "
        f"재현율 {ml['recall']:.4f}, F1 {ml['f1']:.4f}, ROC-AUC {ml['roc_auc']:.4f}다. "
        f"실제 고소득 {cm['tp'] + cm['fn']:,}명 중 {cm['fn']:,}명을 저소득으로 놓쳤다.",
    ]
    if best:
        lines.append(
            f"- 결측 처리 방식은 F1 기준 '{STRATEGY_LABEL.get(best, best)}'가 "
            f"{ab[best]['f1']:.4f}로 가장 높았다."
        )
    return "\n".join(lines)


def write_report(
    file_path,
    *,
    loading,
    cleaning,
    loading_te,
    cleaning_te,
    describe,
    corr,
    ttest,
    chisq,
    ml,
    ab,
    caveats,
):
    # 단계별 결과를 받아 발표용 report.md를 생성
    corr_pair = corr.loc["education-num", "hours-per-week"]
    coef = ml["coef"]
    reference = ml["reference"]
    counts, n_train = ml["category_counts"], ml["train"]
    top_md = _annotate(coef.tail(8)[::-1], reference, counts, n_train)
    bottom_md = _annotate(coef.head(8), reference, counts, n_train)
    demo_md = _annotate(
        coef.loc[coef.index.str.startswith(("sex_", "race_"))][::-1], reference, counts, n_train
    )

    ttest_msg = (
        "통계적으로 유의미한 차이 있음 (H0 기각)"
        if ttest["significant"]
        else "차이 없음 (우연일 수 있음)"
    )
    chisq_msg = (
        "성별과 소득은 독립이 아니다 (H0 기각)"
        if chisq["significant"]
        else "독립 가설을 기각하지 못함"
    )

    n_num, n_cat, n_feat = len(ml["num_cols"]), len(ml["cat_cols"]), ml["n_features"]
    num_cols, cat_cols, dropped = (
        ", ".join(ml["num_cols"]),
        ", ".join(ml["cat_cols"]),
        ", ".join(ml["dropped"]),
    )
    scales = ", ".join(f"{c}={s:.2f}" for c, s in ml["numeric_scales"].items())
    na_cols_tr = ", ".join(f"{k}({v}건)" for k, v in cleaning["na_cols"].items())
    na_cols_te = ", ".join(f"{k}({v}건)" for k, v in cleaning_te["na_cols"].items())
    strategy_label = STRATEGY_LABEL.get(cleaning["strategy"], cleaning["strategy"])
    sex_rates = ", ".join(f"{k} {v:.1%}" for k, v in chisq["rate_by_sex"].items())
    chisq_stats = (
        f"chi2={chisq['chi2']:.3f}, 자유도={chisq['dof']}, "
        f"n={chisq['n']:,}, p={chisq['p_text']}"
    )
    role_purity = ", ".join(f"{role} {share:.1%}" for role, share in caveats["role_purity"].items())
    cg_cap = (
        f"{caveats['capital_gain_capped']:,}행이 상한값 {caveats['capital_gain_cap']:,}이고 "
        f"그중 고소득 비율은 {caveats['capital_gain_capped_pos_rate']:.1%}"
    )

    md = f"""# Adult Census Income — End2End 분석 리포트

- 생성 일시: {datetime.now().strftime("%Y-%m-%d %H:%M")}
- 작성자: 박기연 (판교 7반)
- 데이터 분할: UCI 공식 분할 사용 (`adult.data` 학습 / `adult.test` 평가, 랜덤 분할 없음)
- 분석 범위: 1994년 US Census에서 추출한 비가중 표본. 아래 결과는 이 표본 안의 조건부 연관이며
  모집단 인과효과나 개인 평가의 근거가 아니다.
- 변수 표기: `sex`·`race`는 1994년 당시 행정 분류 체계로 기록된 값이다. `sex`는 두 값만 존재해
  성별 정체성을 나타내지 않고, `race` 범주도 조사 시점의 분류이지 자기 식별과 일치하지 않을 수 있다.
  이 리포트에서 두 변수는 "기록된 범주 간 차이"를 보는 용도로만 쓴다.

## 1. 데이터 준비

### 1-1. Pandas·Polars 로딩 비교
- 소요 시간(단일 참고 측정, 벤치마크 아님)
  - train: Pandas {loading["pandas_sec"]:.3f}초 vs Polars {loading["polars_sec"]:.3f}초
    ({loading["rows"]:,}행 x {loading["cols"]}컬럼)
  - test: Pandas {loading_te["pandas_sec"]:.3f}초 vs Polars {loading_te["polars_sec"]:.3f}초
    ({loading_te["rows"]:,}행 x {loading_te["cols"]}컬럼)
- 동등성 검증(train): {_equality_text(loading)}
- 동등성 검증(test): {_equality_text(loading_te)}

Polars에는 Pandas의 `skipinitialspace`에 해당하는 옵션이 없어, 그대로 읽으면 구분자 뒤 공백이 남고
(`" State-gov"` vs `"State-gov"`) 수치 컬럼까지 문자열로 추론된다. `src/load.py`는 Polars를 전 컬럼
문자열로 읽은 뒤 **공백 제거 -> `?` 결측 처리 -> 라벨 마침표 제거 -> 수치형 캐스팅**이라는 같은 계약을
적용하고, 그 뒤에 컬럼 순서·shape·null mask·의미상 dtype·셀 값을 실제로 비교한다.

### 1-2. 결측·중복 처리
- 결측 컬럼(train): {na_cols_tr}
- 결측 컬럼(test): {na_cols_te}
- 적용 전략: {strategy_label}

| 구분 | 원본 | 결측 처리 후 | 중복 제거 후 |
|---|---:|---:|---:|
| train | {cleaning["raw"]:,} | {cleaning["after_na"]:,} | {cleaning["clean"]:,} |
| test | {cleaning_te["raw"]:,} | {cleaning_te["after_na"]:,} | {cleaning_te["clean"]:,} |

### 1-3. 결측 행 삭제가 소득 그룹에 준 영향
결측 행 삭제는 소득 그룹에 균일하게 적용되지 않는다. 그룹별 제거율은 아래와 같다.

{_na_impact_md(cleaning, cleaning_te)}

- 결측 제거 직후 고소득(>50K) 비율(중복 제거 전 기준):
  train {cleaning["pos_rate_raw"]:.4f} -> {cleaning["pos_rate_kept"]:.4f} /
  test {cleaning_te["pos_rate_raw"]:.4f} -> {cleaning_te["pos_rate_kept"]:.4f}

- `adult.test`는 첫 줄이 주석(`|1x3 Cross validator`), 라벨에 마침표(`>50K.`)가 붙어 있어
  로딩 단계에서 주석 제외·라벨 표기 통일 처리를 했다.

### 1-4. 결과를 읽을 때의 데이터 한계
- **평가셋에도 중복 제거를 적용했다.** test는 {cleaning_te["after_na"]:,}행에서
  {cleaning_te["clean"]:,}행이 됐다. 공식 평가셋을 그대로 쓰지 않았으므로 아래 지표를 다른 참가자와
  비교할 때는 같은 정제 절차를 썼는지 확인해야 한다.
- **train과 test에 완전히 같은 행이 {caveats["overlap_rows"]:,}건 있다.** 공식 분할이지만 이만큼은
  평가가 낙관적으로 치우칠 수 있다. 이번 파이프라인은 이 행들을 제거하지 않았다.
- **`capital-gain`은 상한 처리된 값이다.** 학습 데이터에서 0인 비율이
  {caveats["capital_gain_zero_share"]:.1%}이고, {cg_cap}다. 상한값은 실제 금액이 아니라
  "그 이상"을 뜻하는 표기이므로, 이 변수의 큰 계수를 금액 효과로 읽으면 안 된다.
- **`relationship`은 `sex`와 거의 겹친다.** 성별 편중도 — {role_purity}.
  `relationship`을 함께 통제한 상태의 성별 계수는 "가구 내 역할을 고정했을 때의 차이"이므로,
  성별 전체 격차는 3-1절의 교차표와 카이제곱으로 읽어야 한다.

## 2. 시각화 (train 기준)
- `output/eda_charts.png` — 핵심 4패널: 연령 분포, 근로시간 박스플롯, 직업별 고소득 비율, 수치형 상관
- `output/eda_numeric_all.png` — 수치형 6개 변수 전체의 소득 그룹별 분포
  (capital-gain/loss는 0이 대부분이어서 0 제외 + 로그 축)
- `output/eda_categorical_all.png` — 범주형 8개 변수 전체의 고소득 비율
  (native-country는 범주가 41개라 표본 상위 14개 + 기타로 묶음)
- `output/income_by_education.html` — 학력별 고소득 비율 (Plotly, 마우스오버로 표본 수 확인)

## 3. 통계 분석 (train 기준)

### 3-1. 성별 x 소득 — 카이제곱 독립성 검정
실습 주제가 성별과 소득의 관계이므로, 두 범주형 변수의 연관을 직접 검정한다.

{chisq["table"].to_markdown()}

- {chisq_stats} -> **{chisq_msg}**
- Cramer's V={chisq["cramers_v"]:.3f} (표본 크기와 무관한 연관 강도)
- 성별 고소득 비율: {sex_rates}
- 최소 기대빈도가 {chisq["expected_min"]:,.0f}이라 Yates 연속성 보정을 쓰지 않았다
  (보정은 기대빈도가 작은 2x2를 위한 보수적 장치이고, 여기서 쓰면 chi2와 Cramer's V가 낮게 잡힌다).
- 이 검정은 다른 변수를 통제하지 않은 **전체 연관**이다. 4절의 성별 계수는 학력·직업·근로시간·
  가구 내 역할을 고정한 뒤의 값이라 크기가 다르며, 둘은 서로 다른 질문에 답한다.

### 3-2. 소득 그룹 간 주당 근로시간 — Welch t-test
- 평균: >50K {ttest["mean_high"]:.1f}시간 vs <=50K {ttest["mean_low"]:.1f}시간
  (차이 {ttest["diff"]:+.2f}시간, 95% CI [{ttest["ci_low"]:.2f}, {ttest["ci_high"]:.2f}])
- t={ttest["t"]:.3f}, p={ttest["p_text"]} -> **{ttest_msg}**
- Cohen's d={ttest["cohens_d"]:.3f} ({ttest["effect"]}) — 표본이 크면 작은 차이도 유의해지므로
  유의성과 함께 효과크기를 본다.

### 3-3. 기술통계·상관
- 상관계수 예시: education-num vs hours-per-week = {corr_pair:.3f}

{describe.round(2).to_markdown()}

## 4. ML Pipeline (소득 >50K 로지스틱 회귀)
- 구성: StandardScaler + OneHotEncoder(drop="first") -> LogisticRegression (단일 Pipeline)
- 피처: 수치형 {n_num}개 + 범주형 {n_cat}개(sex·race 포함) -> 원핫 인코딩 후 {n_feat}개
  - 수치형: {num_cols}
  - 범주형: {cat_cols}
  - 제외: {dropped} (fnlwgt는 표본 가중치, education은 education-num과 1:1 중복)
- 학습(adult.data) {ml["train"]:,}건 / 평가(adult.test) {ml["test"]:,}건
- 고소득(>50K) 비율: train {ml["train_pos_rate"]:.3f} / test {ml["test_pos_rate"]:.3f}
- 모델 저장: `output/{ml["model_file"]}` (joblib, 재로딩 검증 완료)

### 4-1. 평가 지표

| 지표 | 값 |
|---|---:|
| 정확도 (accuracy) | {ml["accuracy"]:.4f} |
| 정밀도 (precision) | {ml["precision"]:.4f} |
| 재현율 (recall) | {ml["recall"]:.4f} |
| F1 | {ml["f1"]:.4f} |
| ROC-AUC | {ml["roc_auc"]:.4f} |

고소득 표본이 적어 정확도만으로는 실제 고소득자를 얼마나 놓쳤는지 알 수 없다. 혼동행렬로 확인한다.

{_confusion_md(ml["confusion"])}

- 실제 고소득 {ml["confusion"]["tp"] + ml["confusion"]["fn"]:,}명 중
  {ml["confusion"]["fn"]:,}명을 저소득으로 잘못 예측했다 (재현율 {ml["recall"]:.4f}).
- 실제 저소득 {ml["confusion"]["tn"] + ml["confusion"]["fp"]:,}명 중
  {ml["confusion"]["fp"]:,}명을 고소득으로 잘못 예측했다.

### 4-2. 결측 처리 방식 A/B 비교
같은 Pipeline으로 결측 처리만 바꿔 학습·평가한 결과다.

{_strategy_md(ab)}

### 4-3. 계수 해석의 전제
- 범주형: 각 변수에서 기준 범주 하나를 빼고 원핫했다. 남은 계수는 **그 기준 범주 대비**
  조건부 오즈비 `exp(beta)`다.
  - 기준 선정 규칙 — {ml["reference_rule"]}
  - 기준 범주 — {_reference_note(reference)}
  - 사전순 첫 범주(`drop="first"`)를 쓰면 `native-country`의 기준이 학습 18행짜리 범주가 되어
    나머지 40개 대비가 모두 불안정해진다. 그래서 표본 최다 범주를 기준으로 삼았다.
  - `LogisticRegression`은 기본이 L2 정규화라 기준 범주를 바꾸면 계수뿐 아니라 예측도 미세하게
    달라진다(정규화가 "계수 0"을 기준 범주 대비 0으로 해석하기 때문). 지표를 비교할 때는
    기준 범주 규칙이 같은지 확인해야 한다.
- 수치형: StandardScaler 적용 후 계수이므로 **1 표준편차 증가 기준**이다.
  1 표준편차 크기 — {scales}
- `handle_unknown="ignore"`이므로 학습에 없던 범주는 전부 0으로 인코딩되어 기준 범주와 구분되지 않는다.
- 계수는 다른 변수를 통제한 상태의 부분효과라 3절의 단순 집계 비율과 부호가 다를 수 있다.
- 아래 표의 `학습 표본`은 해당 범주와 기준 범주의 학습 행 수다. 표본이 적은 범주의 오즈비는
  신뢰구간이 넓어 순위를 그대로 읽으면 안 된다. 이 리포트는 계수의 표준오차를 계산하지 않는다.

### 4-4. 고소득 오즈와 양의 연관이 큰 항목 (상위 8개)

{top_md}

### 4-5. 고소득 오즈와 음의 연관이 큰 항목 (하위 8개)

{bottom_md}

### 4-6. sex·race 계수

{demo_md}

sex·race 계수는 1994년 표본에 기록된 조건부 연관이다. 인과관계의 증거가 아니며 개인 평가의 근거로
사용할 수 없다. 관측되지 않은 교란 변수와 표본 선택의 영향을 통제하지 않았다.
성별 계수는 `relationship`(Husband/Wife)을 함께 통제한 값인데 이 변수는 성별과 거의 겹치므로
(1-4절 참고), 성별 전체 격차가 아니라 "가구 내 역할까지 고정했을 때 남는 차이"로 읽어야 한다.
`native-country` 계수도 마찬가지로 국가별 순위가 아니다. 표본이 수십 행인 범주가 많고
이민 시기·직종 구성·표본 추출이 통제되지 않았다.

## 5. 결론 (모두 위 결과에서 계산)

{_conclusions(ttest, chisq, ml, ab)}

### 5-1. 분석자 해석 (자동 계산 아님)
- 학습·평가를 서로 다른 파일로 분리해 랜덤 분할보다 누수 위험이 낮다. 다만 1-4절대로 평가셋에도
  결측·중복 제거를 적용했고 두 파일에 동일 행이 남아 있으므로, 다른 참가자와 지표를 비교하려면
  정제 절차부터 맞춰야 한다.
- 전처리~모델을 Pipeline 하나로 묶어 재현 가능한 학습·배포 단위를 확보했다.
- 결측 행 삭제는 소득 그룹별 제거율이 달라 표본 구성을 바꾼다. 위 1-3의 제거율 차이를 감안해
  결과를 읽어야 한다.
- 이 리포트는 계수의 표준오차·신뢰구간과 PR-AUC, 임계값 조정을 다루지 않는다. 순위와 크기를
  단정적으로 읽지 않도록 주의해야 한다.
"""
    try:
        file_path.write_text(md, encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"[오류] 리포트 저장 실패: {e}") from e
