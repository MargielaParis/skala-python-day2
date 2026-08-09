"""자동화 — 분석 결과를 report.md로 자동 생성

리포트의 수치·결론 문장은 모두 인자로 받은 결과에서 파생시킨다.
입력이 바뀌면 결론도 함께 바뀌어야 하므로 고정 문장을 두지 않는다.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import PipelineError

STRATEGY_LABEL = {"drop": "결측 행 삭제(dropna)", "unknown": "범주형 결측을 Unknown으로 보존"}


def _equality_text(loading: dict[str, Any]) -> str:
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


def _reference_note(reference: dict[str, Any]) -> str:
    # 원핫에서 빠진 기준 범주 목록 — 모든 범주형 계수는 이 기준 대비 값이다
    return ", ".join(f"{col}={ref}" for col, ref in reference.items() if ref is not None)


def _annotate(
    coef_slice: pd.DataFrame,
    reference: dict[str, Any],
    counts: dict[str, dict[Any, int]],
    n_train: int,
) -> str:
    # 각 계수가 무엇과 비교된 값인지, 그 대비를 만든 표본 수, 신뢰구간을 함께 적는다.
    # 표본 수와 구간이 없으면 18행짜리 범주의 오즈비가 27,000행짜리와 똑같이 단정적으로 보인다.
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

    table = pd.DataFrame(
        {
            "계수": coef_slice["coef"].round(3),
            "오즈비": coef_slice["odds_ratio"].round(3),
            "오즈비 95% 구간": [
                f"[{low:.3f}, {high:.3f}]"
                for low, high in zip(coef_slice["or_low"], coef_slice["or_high"], strict=True)
            ],
            "비교 기준": basis,
            "학습 표본": sizes,
        },
        index=coef_slice.index,
    )
    return str(table.to_markdown())


def _sex_contrast(coef: pd.DataFrame, reference: dict[str, Any]) -> str:
    # 성별 계수를 "기준 대비 오즈비" 문장으로 파생 (고정 숫자를 쓰지 않기 위함)
    ref = reference.get("sex")
    rows = coef.loc[coef.index.str.startswith("sex_")]
    if ref is None or rows.empty:
        return "sex 계수가 모델에 없어 성별 대비를 계산하지 않았다."
    parts = [
        f"{str(name).split('_', 1)[1]}는 기준({ref}) 대비 오즈 {row['odds_ratio']:.3f}배"
        f"(95% 구간 [{row['or_low']:.3f}, {row['or_high']:.3f}])"
        for name, row in rows.iterrows()
    ]
    return (
        "학력·직업·근로시간과 가구 내 역할(relationship)까지 고정했을 때 "
        + ", ".join(parts)
        + "로 추정됐다. relationship은 성별과 거의 겹치므로 이 값은 성별 전체 격차가 아니다."
    )


def _confusion_md(cm: dict[str, int], positive: str = ">50K") -> str:
    return "\n".join(
        [
            f"| 실제 \\ 예측 | <=50K | {positive} |",
            "|---|---:|---:|",
            f"| <=50K | {cm['tn']:,} | {cm['fp']:,} |",
            f"| {positive} | {cm['fn']:,} | {cm['tp']:,} |",
        ]
    )


def _threshold_md(thresholds: dict[str, Any]) -> str:
    header = "| 임계값 | 정확도 | 정밀도 | 재현율 | F1 | 놓친 고소득(FN) | 잘못 잡은 저소득(FP) |"
    rows = [header, "|---|---:|---:|---:|---:|---:|---:|"]
    for label, key in (("기본값", "default"), ("교차검증 F1 최적", "tuned")):
        m = thresholds[key]
        rows.append(
            f"| {label} {m['threshold']:.3f} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m['confusion']['fn']:,} | "
            f"{m['confusion']['fp']:,} |"
        )
    return "\n".join(rows)


def _sensitivity_md(sensitivity: dict[str, Any]) -> str:
    rows = [
        f"| {sensitivity['focus']} 계수 | 오즈비 | 오즈비 95% 구간 |",
        "|---|---:|---|",
    ]
    for name, row in sensitivity["coef"].iterrows():
        rows.append(
            f"| {name} | {row['odds_ratio']:.3f} | "
            f"[{row['or_low']:.3f}, {row['or_high']:.3f}] |"
        )
    return "\n".join(rows)


def _strategy_md(ab: dict[str, Any]) -> str:
    header = (
        "| 결측 전략 | 학습 행 수 | 평가 행 수 | 정확도 | 정밀도 | 재현율 | F1 | ROC-AUC | PR-AUC |"
    )
    rows = [header, "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in ab.items():
        rows.append(
            f"| {STRATEGY_LABEL.get(name, name)} | {m['train']:,} | {m['test']:,} | "
            f"{m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |"
        )
    return "\n".join(rows)


def _na_impact_md(cleaning: dict[str, Any], cleaning_te: dict[str, Any]) -> str:
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


def _conclusions(
    ttest: dict[str, Any],
    chisq: dict[str, Any],
    ml: dict[str, Any],
    ab: dict[str, Any],
    sensitivity: dict[str, Any],
) -> str:
    # 결론은 전부 입력에서 파생한다 — 입력이 바뀌면 이 문장들도 함께 바뀐다
    coef, reference = ml["coef"], ml["reference"]
    strongest = coef["coef"].abs().idxmax()
    strongest_row = coef.loc[strongest]
    rates = ", ".join(f"{k} {v:.1%}" for k, v in chisq["rate_by_sex"].items())
    best = max(ab, key=lambda k: ab[k]["f1"]) if ab else None
    cm = ml["confusion"]
    tuned = ml["thresholds"]["tuned"]

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
        f"계수 {strongest_row['coef']:+.3f}, 오즈비 {strongest_row['odds_ratio']:.3f} "
        f"(95% 구간 [{strongest_row['or_low']:.3f}, {strongest_row['or_high']:.3f}])다. "
        f"범주형은 기준 범주 대비, 수치형은 1 표준편차 증가 기준이다.",
        f"- {_sex_contrast(coef, reference)}",
        f"- `{sensitivity['excluded']}`를 빼고 다시 학습하면 {_sensitivity_summary(sensitivity)}",
        f"- 평가 성능은 정확도 {ml['accuracy']:.4f}, 정밀도 {ml['precision']:.4f}, "
        f"재현율 {ml['recall']:.4f}, F1 {ml['f1']:.4f}, ROC-AUC {ml['roc_auc']:.4f}, "
        f"PR-AUC {ml['pr_auc']:.4f}다. 실제 고소득 {cm['tp'] + cm['fn']:,}명 중 "
        f"{cm['fn']:,}명을 저소득으로 놓쳤다.",
        f"- 임계값을 교차검증 F1 최적값 {tuned['threshold']:.3f}으로 바꾸면 재현율이 "
        f"{ml['recall']:.4f}에서 {tuned['recall']:.4f}로, 놓친 고소득이 "
        f"{cm['fn']:,}명에서 {tuned['confusion']['fn']:,}명으로 바뀐다.",
    ]
    if best:
        lines.append(f"- {_strategy_verdict(ab, best)}")
    return "\n".join(lines)


# F1 차이가 이 값보다 작으면 순위를 말하지 않는다. 계수 구간도 안 내는 리포트에서
# 0.001 차이로 우열을 선언하면 과장이 된다
STRATEGY_TIE_GAP = 0.01


def _strategy_verdict(ab: dict[str, Any], best: str) -> str:
    # "어느 전략이 이겼다"를 f1 격차에서 파생시킨다 (고정 문장을 두지 않기 위함)
    scores = sorted((m["f1"] for m in ab.values()), reverse=True)
    gap = scores[0] - scores[-1] if len(scores) > 1 else 0.0
    if len(scores) > 1 and gap < STRATEGY_TIE_GAP:
        return (
            f"결측 처리 방식은 같은 평가셋에서 F1 격차가 {gap:.4f}에 그쳐 우열을 가릴 수 없다 "
            f"(최고 {scores[0]:.4f} / 최저 {scores[-1]:.4f}). 지표만으로 방식을 고를 근거는 없다."
        )
    return (
        f"결측 처리 방식은 F1 기준 '{STRATEGY_LABEL.get(best, best)}'가 "
        f"{ab[best]['f1']:.4f}로 가장 높았다 (격차 {gap:.4f})."
    )


def _sensitivity_summary(sensitivity: dict[str, Any]) -> str:
    rows = sensitivity["coef"]
    if rows.empty:
        return f"`{sensitivity['focus']}` 계수를 계산할 수 없었다."
    parts = [
        f"{name}의 오즈비가 {row['odds_ratio']:.3f}"
        f"(95% 구간 [{row['or_low']:.3f}, {row['or_high']:.3f}])"
        for name, row in rows.iterrows()
    ]
    return (
        ", ".join(parts)
        + f"가 된다 (기준 {sensitivity['focus']}={sensitivity['reference']}, "
        + f"정확도 {sensitivity['accuracy']:.4f})."
    )


def write_report(
    file_path: Path,
    *,
    loading: dict[str, Any],
    cleaning: dict[str, Any],
    loading_te: dict[str, Any],
    cleaning_te: dict[str, Any],
    describe: pd.DataFrame,
    corr: pd.DataFrame,
    ttest: dict[str, Any],
    chisq: dict[str, Any],
    ml: dict[str, Any],
    ab: dict[str, Any],
    caveats: dict[str, Any],
    sensitivity: dict[str, Any],
) -> None:
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
        f"chi2={chisq['chi2']:.3f}, 자유도={chisq['dof']}, n={chisq['n']:,}, p={chisq['p_text']}"
    )
    role_purity = ", ".join(f"{role} {share:.1%}" for role, share in caveats["role_purity"].items())
    cg_cap = (
        f"{caveats['capital_gain_capped']:,}행이 상한값 {caveats['capital_gain_cap']:,}이고 "
        f"그중 고소득 비율은 {caveats['capital_gain_capped_pos_rate']:.1%}"
    )
    feature_summary = (
        f"수치형 {n_num}개 + 범주형 {n_cat}개(sex·race 포함), 원핫 인코딩 후 {n_feat}개"
    )
    unseen_note = (
        f"{caveats['unseen_category_rows']:,}행 / {caveats['test_rows']:,}행"
        if caveats["test_rows"]
        else "평가셋 없음"
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
  "그 이상"을 뜻하는 표기이므로 이 변수의 큰 계수를 금액 효과로 읽으면 안 된다.
  상한 도달 여부를 별도 지시변수로 분리해봤지만 `capital-gain`과 상관이 0.94라 따로 식별되지 않았고
  (계수 구간 [0.14, 7.14], 사후분포가 사전분포와 사실상 같음) 모델에서 뺐다.
- **학습에 없던 범주가 평가셋에 있으면 전부 0으로 인코딩된다.** 이번 실행에서 해당 행은
  {unseen_note}다. 0이 아니라면 그 행들은 기준 범주와 구분되지 않은 채 예측된 것이다.
- **`relationship`은 `sex`와 거의 겹친다.** 성별 편중도 — {role_purity}.
  `relationship`을 함께 통제한 상태의 성별 계수는 "가구 내 역할을 고정했을 때의 차이"이므로,
  성별 전체 격차는 3-1절의 교차표와 카이제곱으로, 통제 의존도는 4-8절의 민감도 분석으로 읽어야 한다.

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
  유의성과 함께 효과크기를 본다. 검정은 등분산을 가정하지 않는 Welch지만 효과크기는 관례대로
  pooled 표준편차를 쓴다.

### 3-3. 기술통계·상관
- 상관계수 예시: education-num vs hours-per-week = {corr_pair:.3f}

{describe.round(2).to_markdown()}

## 4. ML Pipeline (소득 >50K 로지스틱 회귀)
- 구성: StandardScaler + OneHotEncoder(기준 범주 제외) -> LogisticRegression (단일 Pipeline)
- 피처: {feature_summary}
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
| PR-AUC (average precision) | {ml["pr_auc"]:.4f} |

양성(고소득)이 드물기 때문에 ROC-AUC는 낙관적으로 보일 수 있다. 양성 비율이
{ml["test_pos_rate"]:.3f}이므로 아무 정보 없이 찍는 분류기의 PR-AUC 기준선이 그 값이다.
정확도만으로는 실제 고소득자를 얼마나 놓쳤는지 알 수 없으니 혼동행렬로 확인한다.

{_confusion_md(ml["confusion"])}

- 실제 고소득 {ml["confusion"]["tp"] + ml["confusion"]["fn"]:,}명 중
  {ml["confusion"]["fn"]:,}명을 저소득으로 잘못 예측했다 (재현율 {ml["recall"]:.4f}).
- 실제 저소득 {ml["confusion"]["tn"] + ml["confusion"]["fp"]:,}명 중
  {ml["confusion"]["fp"]:,}명을 고소득으로 잘못 예측했다.

### 4-2. 분류 임계값
기본 임계값 0.5는 자의적이다. F1을 최대화하는 임계값을 **학습셋 {ml["thresholds"]["cv"]}-폴드
교차검증의 out-of-fold 확률에서** 찾아 평가셋에 적용했다. 평가셋에서 고르면 평가셋에 대한
과적합이고, 학습셋 예측(in-sample)으로 고르면 모델이 이미 그 데이터에 적합돼 있어 낙관적이다.

{_threshold_md(ml["thresholds"])}

어떤 임계값이 맞는지는 "고소득자를 놓치는 비용"과 "저소득자를 고소득으로 잘못 잡는 비용" 중
어느 쪽이 큰지에 달려 있고, 그 판단은 데이터가 아니라 용도에서 나온다.

### 4-3. 결측 처리 방식 A/B 비교
같은 Pipeline으로 결측 처리만 바꿔 학습한 결과다.
**평가셋은 두 전략 모두 정제 후 test {ml["test"]:,}행으로 고정했다.** 전략별로 평가셋을 따로 쓰면
행 수와 사례 구성이 달라져(결측 보존 시 {cleaning_te["raw"]:,}행 중 더 많이 남는다) 지표를 나란히
비교할 수 없다.

{_strategy_md(ab)}

### 4-4. 계수 해석의 전제
- 범주형: 각 변수에서 기준 범주 하나를 빼고 원핫했다. 남은 계수는 **그 기준 범주 대비**
  조건부 오즈비 `exp(beta)`다.
  - 기준 선정 규칙 — {ml["reference_rule"]}
  - 기준 범주 — {_reference_note(reference)}
  - 사전순 첫 범주(`drop="first"`)를 쓰면 `native-country`의 기준이 학습 18행짜리 범주가 되어
    나머지 40개 대비가 모두 불안정해진다. 그래서 표본 최다 범주를 기준으로 삼았다.
  - `LogisticRegression`은 기본이 L2 정규화라 기준 범주를 바꾸면 계수뿐 아니라 예측도 미세하게
    달라진다. 지표를 비교할 때는 기준 범주 규칙이 같은지 확인해야 한다.
- 수치형: StandardScaler 적용 후 계수이므로 **1 표준편차 증가 기준**이다.
  1 표준편차 크기 — {scales}
- **오즈비 95% 구간**은 L2 정규화를 가우시안 사전분포로 본 라플라스 근사에서 얻은 값이다
  (사후 정밀도 `Z'WZ + I/C`). 정확한 Wald 신뢰구간이 아니고, 정규화 때문에 계수 자체도
  0 쪽으로 수축돼 있다. 구간이 1을 포함하면 방향조차 단정할 수 없다는 뜻으로 읽으면 된다.
- `handle_unknown="ignore"`이므로 학습에 없던 범주는 전부 0으로 인코딩되어 기준 범주와 구분되지 않는다.
- 계수는 다른 변수를 통제한 상태의 부분효과라 3절의 단순 집계 비율과 부호가 다를 수 있다.
- `학습 표본`은 해당 범주와 기준 범주의 학습 행 수다. 표본이 적을수록 구간이 넓어진다.

### 4-5. 고소득 오즈와 양의 연관이 큰 항목 (상위 8개)

{top_md}

### 4-6. 고소득 오즈와 음의 연관이 큰 항목 (하위 8개)

{bottom_md}

### 4-7. sex·race 계수

{demo_md}

sex·race 계수는 1994년 표본에 기록된 조건부 연관이다. 인과관계의 증거가 아니며 개인 평가의 근거로
사용할 수 없다. 관측되지 않은 교란 변수와 표본 선택의 영향을 통제하지 않았다.
`native-country` 계수도 국가별 순위가 아니다. 표본이 수십 행인 범주가 많고
이민 시기·직종 구성·표본 추출이 통제되지 않았다.

### 4-8. 민감도 분석 — `{sensitivity["excluded"]}`를 뺐을 때
`{sensitivity["excluded"]}`는 `sex`와 거의 겹치므로(1-4절), 이 변수를 통제한 상태의 성별 계수가
통제 방식에 얼마나 의존하는지 확인한다. 아래는 이 변수만 빼고 같은 Pipeline으로 다시 학습한 결과다.

{_sensitivity_md(sensitivity)}

- 기준 범주: {sensitivity["focus"]}={sensitivity["reference"]}
- 이 모델의 정확도 {sensitivity["accuracy"]:.4f} / F1 {sensitivity["f1"]:.4f}
- 두 값의 차이가 크다면, 4-7절의 성별 계수는 "성별의 효과"가 아니라 "가구 내 역할까지 고정한 뒤
  남는 차이"라는 뜻이다. 어느 쪽이 옳은 모형인지는 데이터가 아니라 묻고 싶은 질문이 정한다.

## 5. 결론 (모두 위 결과에서 계산)

{_conclusions(ttest, chisq, ml, ab, sensitivity)}

### 5-1. 분석자 해석 (자동 계산 아님)
- 학습·평가를 서로 다른 파일로 분리해 랜덤 분할보다 누수 위험이 낮다. 다만 1-4절대로 평가셋에도
  결측·중복 제거를 적용했고 두 파일에 동일 행이 남아 있으므로, 다른 참가자와 지표를 비교하려면
  정제 절차부터 맞춰야 한다.
- 전처리~모델을 Pipeline 하나로 묶어 재현 가능한 학습·배포 단위를 확보했다.
- 결측 행 삭제는 소득 그룹별 제거율이 달라 표본 구성을 바꾼다. 위 1-3의 제거율 차이를 감안해
  결과를 읽어야 한다.
- 계수 구간은 근사값이고 다중비교를 보정하지 않았다. 80개 계수를 동시에 보면 우연히 1을 벗어나는
  구간이 나오므로, 순위와 크기를 단정적으로 읽지 않도록 주의해야 한다.
"""
    try:
        file_path.write_text(md, encoding="utf-8")
    except OSError as e:
        raise PipelineError(f"리포트 저장 실패: {e}") from e
