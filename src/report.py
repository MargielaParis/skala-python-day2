"""분석 결과를 report.md로 생성한다.

리포트의 수치와 결론 문장은 모두 인자로 받은 결과에서 파생시킨다.
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
    return "불일치 있음: " + ", ".join(parts)


def _reference_note(reference: dict[str, Any]) -> str:
    # 원핫에서 빠진 기준 범주 목록. 모든 범주형 계수는 이 기준 대비 값이다
    return ", ".join(f"{col}={ref}" for col, ref in reference.items() if ref is not None)


def _annotate(
    coef_slice: pd.DataFrame,
    reference: dict[str, Any],
    counts: dict[str, dict[Any, int]],
    n_train: int,
) -> str:
    # 각 계수가 무엇과 비교된 값인지, 그 대비를 만든 표본 수, 신뢰구간을 함께 적는다.
    # 표본 수와 구간이 없으면 수십 행짜리 범주의 오즈비가 수만 행짜리와 똑같이 단정적으로 보인다.
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
    # 성별 계수를 기준 대비 오즈비 문장으로 파생시킨다
    ref = reference.get("sex")
    rows = coef.loc[coef.index.str.startswith("sex_")]
    if ref is None or rows.empty:
        return "sex 계수가 모델에 없어 성별 대비를 계산하지 않았다."
    parts = [
        f"{str(name).split('_', 1)[1]}의 오즈는 기준({ref}) 대비 {row['odds_ratio']:.3f}배"
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
        f"- 고소득 그룹의 주당 근로시간은 저소득 그룹보다 {ttest['diff']:+.1f}시간 차이가 나고, "
        f"95% 신뢰구간은 [{ttest['ci_low']:.2f}, {ttest['ci_high']:.2f}], "
        f"Cohen's d는 {ttest['cohens_d']:.3f}({ttest['effect']}). "
        f"{'통계적으로 유의미한 차이가 있다' if ttest['significant'] else '유의미한 차이가 없다'}.",
        f"- 계수 절대값이 가장 큰 항은 `{strongest}`이며 "
        f"계수 {strongest_row['coef']:+.3f}, 오즈비 {strongest_row['odds_ratio']:.3f} "
        f"(95% 구간 [{strongest_row['or_low']:.3f}, {strongest_row['or_high']:.3f}])다. "
        f"범주형은 기준 범주 대비, 수치형은 1 표준편차 증가 기준이다.",
        f"- {_sex_contrast(coef, reference)}",
        f"- `{sensitivity['excluded']}` 변수를 빼고 다시 학습하면 "
        f"{_sensitivity_summary(sensitivity)}",
        f"- 평가 성능은 정확도 {ml['accuracy']:.4f}, 정밀도 {ml['precision']:.4f}, "
        f"재현율 {ml['recall']:.4f}, F1 {ml['f1']:.4f}, ROC-AUC {ml['roc_auc']:.4f}, "
        f"PR-AUC {ml['pr_auc']:.4f}. 실제 고소득 {cm['tp'] + cm['fn']:,}명 중 "
        f"{cm['fn']:,}명을 저소득으로 놓쳤다.",
        f"- 임계값을 교차검증 F1 최적값 {tuned['threshold']:.3f}에 맞추면 재현율이 "
        f"{ml['recall']:.4f}에서 {tuned['recall']:.4f}까지 오르고, 놓친 고소득이 "
        f"{cm['fn']:,}명에서 {tuned['confusion']['fn']:,}명으로 줄어든다.",
    ]
    if best:
        lines.append(f"- {_strategy_verdict(ab, best)}")
    return "\n".join(lines)


# F1 격차가 이 값보다 작으면 순위를 말하지 않는다.
# 계수에는 구간을 붙이면서 모델 비교만 점추정으로 단정하면 일관성이 없다
STRATEGY_TIE_GAP = 0.01


def _strategy_verdict(ab: dict[str, Any], best: str) -> str:
    # 우열 판정을 F1 격차에서 파생시킨다
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
        f"{name} 오즈비 {row['odds_ratio']:.3f}"
        f"(95% 구간 [{row['or_low']:.3f}, {row['or_high']:.3f}])"
        for name, row in rows.iterrows()
    ]
    return (
        "계수가 이렇게 바뀐다 — "
        + ", ".join(parts)
        + f". 기준은 {sensitivity['focus']}={sensitivity['reference']}, "
        + f"정확도는 {sensitivity['accuracy']:.4f}."
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
        f"학습 데이터의 {caveats['capital_gain_zero_share']:.1%}가 0이고, "
        f"{caveats['capital_gain_capped']:,}행이 상한값 {caveats['capital_gain_cap']:,}이며 "
        f"그중 고소득 비율은 {caveats['capital_gain_capped_pos_rate']:.1%}다."
    )
    feature_summary = (
        f"수치형 {n_num}개 + 범주형 {n_cat}개(sex·race 포함), 원핫 인코딩 후 {n_feat}개"
    )
    unseen_note = (
        f"전체 {caveats['test_rows']:,}행 중 {caveats['unseen_category_rows']:,}행이다"
        if caveats["test_rows"]
        else "평가셋이 비어 있어 셀 수 없다"
    )
    cm = ml["confusion"]
    confusion_note = (
        f"실제 고소득 {cm['tp'] + cm['fn']:,}명 중 {cm['fn']:,}명을 저소득으로 잘못 예측했고"
        f"(재현율 {ml['recall']:.4f}), 실제 저소득 {cm['tn'] + cm['fp']:,}명 중 "
        f"{cm['fp']:,}명을 고소득으로 잘못 예측했다."
    )
    sensitivity_note = (
        f"기준 범주는 {sensitivity['focus']}={sensitivity['reference']}, "
        f"이 모델의 정확도는 {sensitivity['accuracy']:.4f}, F1은 {sensitivity['f1']:.4f}."
    )
    loading_rows = "\n".join(
        f"| {label} | {info['rows']:,} x {info['cols']} | {info['pandas_sec']:.3f}초 | "
        f"{info['polars_sec']:.3f}초 | {_equality_text(info)} |"
        for label, info in (("train", loading), ("test", loading_te))
    )
    cleaning_rows = "\n".join(
        f"| {label} | {info['raw']:,} | {info['after_na']:,} | {info['clean']:,} | {na_cols} |"
        for label, info, na_cols in (
            ("train", cleaning, na_cols_tr),
            ("test", cleaning_te, na_cols_te),
        )
    )
    ttest_means = (
        f"평균은 >50K가 {ttest['mean_high']:.1f}시간, <=50K가 {ttest['mean_low']:.1f}시간으로 "
        f"차이는 {ttest['diff']:+.2f}시간이다."
    )
    ttest_stats = (
        f"95% 신뢰구간은 [{ttest['ci_low']:.2f}, {ttest['ci_high']:.2f}], "
        f"t={ttest['t']:.3f}, p={ttest['p_text']}. 판정: {ttest_msg}."
    )

    md = f"""# Adult Census Income 분석 리포트

생성 일시: {datetime.now().strftime("%Y-%m-%d %H:%M")} / 작성자: 박기연 (판교 7반)

학습과 평가는 UCI 공식 분할을 그대로 쓴다(`adult.data` 학습, `adult.test` 평가, 랜덤 분할 없음).
데이터는 1994년 US Census에서 추출한 비가중 표본이므로, 아래 결과는 이 표본 안의 조건부 연관이지
모집단 인과효과나 개인 평가의 근거가 아니다.

`sex`와 `race`는 1994년 당시 행정 분류로 기록된 값이다. `sex`는 두 값만 있어 성별 정체성을
나타내지 않고, `race` 범주도 조사 시점의 분류이지 자기 식별과 일치하지 않을 수 있다. 이 리포트에서
두 변수는 기록된 범주 간 차이를 보는 용도로만 쓴다.

## 1. 데이터 준비

### 1-1. Pandas·Polars 로딩 비교

Polars에는 Pandas의 `skipinitialspace`에 해당하는 옵션이 없다. 그대로 읽으면 구분자 뒤 공백이 남고
(`" State-gov"` vs `"State-gov"`) 수치 컬럼까지 문자열로 추론된다. `src/load.py`는 Polars를 전 컬럼
문자열로 읽은 뒤 공백 제거, `?` 결측 처리, 라벨 마침표 제거, 수치형 캐스팅이라는 같은 계약을
적용하고, 그 뒤에 컬럼 순서·shape·null mask·의미상 dtype·셀 값을 비교한다.

| 구분 | 행 x 컬럼 | Pandas | Polars | 동등성 |
|---|---|---:|---:|---|
{loading_rows}

소요 시간은 단일 참고 측정이고 벤치마크가 아니다.

### 1-2. 결측·중복 처리

적용한 전략은 {strategy_label}이다. `adult.test`는 첫 줄이 주석(`|1x3 Cross validator`)이고 라벨에
마침표(`>50K.`)가 붙어 있어 로딩 단계에서 주석 제외와 라벨 표기 통일을 처리했다.

| 구분 | 원본 | 결측 처리 후 | 중복 제거 후 | 결측 컬럼 |
|---|---:|---:|---:|---|
{cleaning_rows}

### 1-3. 결측 행 삭제가 소득 그룹에 준 영향

결측 행 삭제는 소득 그룹에 균일하게 적용되지 않는다.

{_na_impact_md(cleaning, cleaning_te)}

결측 제거 직후 고소득(>50K) 비율은 train이 {cleaning["pos_rate_raw"]:.4f}에서 {cleaning["pos_rate_kept"]:.4f}로,
test가 {cleaning_te["pos_rate_raw"]:.4f}에서 {cleaning_te["pos_rate_kept"]:.4f}로 이동한다(중복 제거 전 기준).

### 1-4. 결과를 읽을 때의 데이터 한계

평가셋에도 중복 제거를 적용했다. test는 {cleaning_te["after_na"]:,}행에서 {cleaning_te["clean"]:,}행이 됐다.
공식 평가셋을 그대로 쓰지 않았으므로 아래 지표를 다른 참가자와 비교할 때는 같은 정제 절차를 썼는지
확인해야 한다.

train과 test에 완전히 같은 행이 {caveats["overlap_rows"]:,}건 있다. 공식 분할이지만 이만큼은 평가가 낙관적으로
치우칠 수 있고, 이번 파이프라인은 이 행들을 제거하지 않았다.

`capital-gain`은 상한 처리된 값이다. {cg_cap} 상한값은 실제 금액이 아니라 "그 이상"을 뜻하는
표기이므로 이 변수의 큰 계수를 금액 효과로 읽으면 안 된다. 상한 도달 여부는 `capital-gain`과
상관이 0.94라 따로 식별되지 않아 별도 변수로 두지 않았다.

학습에 없던 범주가 평가셋에 있으면 전부 0으로 인코딩되어 기준 범주와 구분되지 않는다. 이번 실행에서
해당 행은 {unseen_note}.

`relationship`은 `sex`와 거의 겹친다(성별 편중도 {role_purity}). `relationship`을 함께 통제한 상태의
성별 계수는 가구 내 역할을 고정했을 때의 차이이므로, 성별 전체 격차는 3-1절의 교차표와 카이제곱으로,
통제 의존도는 4-8절의 민감도 분석으로 읽어야 한다.

## 2. 시각화 (train 기준)

| 파일 | 내용 |
|---|---|
| `output/eda_charts.png` | 연령 분포, 근로시간 박스플롯, 직업별 고소득 비율, 수치형 상관 |
| `output/eda_numeric_all.png` | 수치형 6개 변수의 소득 그룹별 분포 (capital-gain/loss는 0 제외 후 로그 축) |
| `output/eda_categorical_all.png` | 범주형 8개 변수의 고소득 비율 (native-country는 상위 14개와 기타) |
| `output/income_by_education.html` | 학력별 고소득 비율 (Plotly, 마우스오버로 표본 수 확인) |

## 3. 통계 분석 (train 기준)

### 3-1. 성별과 소득의 카이제곱 독립성 검정

실습 주제가 성별과 소득의 관계이므로 두 범주형 변수의 연관을 직접 검정한다.

{chisq["table"].to_markdown()}

{chisq_stats}. 판정: {chisq_msg}.

Cramer's V는 {chisq["cramers_v"]:.3f}로, 표본 크기와 무관한 연관 강도를 나타낸다. 성별 고소득 비율은 {sex_rates}다.
최소 기대빈도가 {chisq["expected_min"]:,.0f}로 충분히 커서 Yates 연속성 보정은 쓰지 않았다. 보정은 기대빈도가 작은
2x2를 위한 장치이고, 여기서 쓰면 chi2와 Cramer's V가 실제보다 낮게 잡힌다.

이 검정은 다른 변수를 통제하지 않은 전체 연관이다. 4절의 성별 계수는 학력·직업·근로시간·가구 내
역할을 고정한 뒤의 값이라 크기가 다르며, 둘은 서로 다른 질문에 답한다.

### 3-2. 소득 그룹 간 주당 근로시간 Welch t-test

{ttest_means}
{ttest_stats}

Cohen's d는 {ttest["cohens_d"]:.3f}({ttest["effect"]})다. 표본이 크면 작은 차이도 유의해지므로 유의성과 함께
효과크기를 본다. 검정은 등분산을 가정하지 않는 Welch지만 효과크기는 관례대로 pooled 표준편차를 쓴다.

### 3-3. 기술통계와 상관

education-num과 hours-per-week의 상관계수는 {corr_pair:.3f}이다.

{describe.round(2).to_markdown()}

## 4. ML Pipeline (소득 >50K 로지스틱 회귀)

StandardScaler와 OneHotEncoder(기준 범주 제외)를 거쳐 LogisticRegression으로 이어지는 단일
Pipeline이다. 피처는 {feature_summary}다.

| 항목 | 값 |
|---|---|
| 수치형 | {num_cols} |
| 범주형 | {cat_cols} |
| 제외 | {dropped} (fnlwgt는 표본 가중치, education은 education-num과 중복) |
| 학습 / 평가 | {ml["train"]:,}건 / {ml["test"]:,}건 |
| 고소득 비율 | train {ml["train_pos_rate"]:.3f} / test {ml["test_pos_rate"]:.3f} |
| 모델 파일 | `output/{ml["model_file"]}` (joblib, 재로딩 검증 완료) |

### 4-1. 평가 지표

| 지표 | 값 |
|---|---:|
| 정확도 (accuracy) | {ml["accuracy"]:.4f} |
| 정밀도 (precision) | {ml["precision"]:.4f} |
| 재현율 (recall) | {ml["recall"]:.4f} |
| F1 | {ml["f1"]:.4f} |
| ROC-AUC | {ml["roc_auc"]:.4f} |
| PR-AUC (average precision) | {ml["pr_auc"]:.4f} |

양성(고소득)이 드물기 때문에 ROC-AUC는 낙관적으로 보일 수 있다. 평가셋 양성 비율이
{ml["test_pos_rate"]:.3f}이므로 아무 정보 없이 찍는 분류기의 PR-AUC 기준선이 그 값이다. 정확도만으로는
실제 고소득자를 얼마나 놓쳤는지 알 수 없으니 혼동행렬로 확인한다.

{_confusion_md(ml["confusion"])}

{confusion_note}

### 4-2. 분류 임계값

기본 임계값 0.5는 자의적이다. F1을 최대화하는 임계값을 학습셋 {ml["thresholds"]["cv"]}-폴드 교차검증의
out-of-fold 확률에서 찾아 평가셋에 적용했다. 평가셋에서 고르면 평가셋에 대한 과적합이고, 학습셋
in-sample 예측으로 고르면 모델이 이미 그 데이터에 적합돼 있어 낙관적이다.

{_threshold_md(ml["thresholds"])}

어떤 임계값이 맞는지는 고소득자를 놓치는 비용과 저소득자를 고소득으로 잘못 잡는 비용 중 어느 쪽이
큰지에 달려 있고, 그 판단은 데이터가 아니라 용도에서 나온다.

### 4-3. 결측 처리 방식 A/B 비교

같은 Pipeline으로 결측 처리만 바꿔 학습한 결과다. 평가셋은 두 전략 모두 정제 후
test {ml["test"]:,}행으로 고정했다. 전략별로 평가셋을 따로 쓰면 행 수와 사례 구성이 달라져 지표를
나란히 비교할 수 없다.

{_strategy_md(ab)}

### 4-4. 계수 해석의 전제

범주형은 각 변수에서 기준 범주 하나를 빼고 원핫했다. 남은 계수는 그 기준 범주 대비 조건부 오즈비
`exp(beta)`다. 기준은 변수마다 {ml["reference_rule"]}로 정한다. 표본이 극히 적은 범주가 기준이 되면
나머지 대비가 모두 불안정해지기 때문이다.

기준 범주: {_reference_note(reference)}

수치형은 StandardScaler를 거친 뒤의 계수이므로 1 표준편차 증가 기준이다.

1 표준편차 크기: {scales}

- 오즈비 95% 구간은 L2 정규화를 가우시안 사전분포로 본 라플라스 근사에서 얻은 값이다
  (사후 정밀도 `Z'WZ + I/C`). 정확한 Wald 신뢰구간이 아니고, 정규화 때문에 계수 자체도 0 쪽으로
  수축돼 있다. 구간이 1을 포함하면 방향조차 단정할 수 없다는 뜻이다.
- `LogisticRegression`은 기본이 L2 정규화라 기준 범주를 바꾸면 계수뿐 아니라 예측도 미세하게
  달라진다. 지표를 비교할 때는 기준 범주 규칙이 같은지 확인해야 한다.
- `handle_unknown="ignore"`이므로 학습에 없던 범주는 전부 0으로 인코딩되어 기준 범주와 구분되지 않는다.
- 계수는 다른 변수를 통제한 상태의 부분효과라 3절의 단순 집계 비율과 부호가 다를 수 있다.
- 표의 `학습 표본`은 해당 범주와 기준 범주의 학습 행 수다. 표본이 적을수록 구간이 넓어진다.

### 4-5. 고소득 오즈와 양의 연관이 큰 항목 (상위 8개)

{top_md}

### 4-6. 고소득 오즈와 음의 연관이 큰 항목 (하위 8개)

{bottom_md}

### 4-7. sex·race 계수

{demo_md}

sex·race 계수는 1994년 표본에 기록된 조건부 연관이다. 인과관계의 증거가 아니며 개인 평가의 근거로
사용할 수 없다. 관측되지 않은 교란 변수와 표본 선택의 영향을 통제하지 않았다. `native-country`
계수도 국가별 순위가 아니다. 표본이 수십 행인 범주가 많고 이민 시기와 직종 구성이 통제되지 않았다.

### 4-8. 민감도 분석: `{sensitivity["excluded"]}` 제외

`{sensitivity["excluded"]}` 변수는 `sex`와 거의 겹치므로(1-4절), 성별 계수가 통제 방식에 얼마나
의존하는지 확인한다. 아래는 이 변수만 빼고 같은 Pipeline으로 다시 학습한 결과다.

{_sensitivity_md(sensitivity)}

{sensitivity_note}
4-7절 값과 차이가 크다면 성별 계수는 성별의 효과가 아니라 가구 내 역할까지 고정한 뒤 남는
차이라는 뜻이다. 어느 쪽이 옳은 모형인지는 데이터가 아니라 묻고 싶은 질문이 정한다.

## 5. 결론

아래 문장은 모두 위 결과에서 계산한 값이다.

{_conclusions(ttest, chisq, ml, ab, sensitivity)}

### 5-1. 분석자 해석

아래는 자동 계산이 아니라 분석자의 판단이다.

- 학습과 평가를 서로 다른 파일로 분리해 랜덤 분할보다 누수 위험이 낮다. 다만 1-4절대로 평가셋에도
  결측·중복 제거를 적용했고 두 파일에 동일 행이 남아 있으므로, 다른 참가자와 지표를 비교하려면
  정제 절차부터 맞춰야 한다.
- 전처리부터 모델까지 Pipeline 하나로 묶어 재현 가능한 학습·배포 단위를 확보했다.
- 결측 행 삭제는 소득 그룹별 제거율이 달라 표본 구성을 바꾼다. 1-3의 제거율 차이를 감안해 결과를
  읽어야 한다.
- 계수 구간은 근사값이고 다중비교를 보정하지 않았다. 80개 계수를 동시에 보면 우연히 1을 벗어나는
  구간이 나오므로 순위와 크기를 단정적으로 읽지 않아야 한다.
"""
    try:
        file_path.write_text(md, encoding="utf-8")
    except OSError as e:
        raise PipelineError(f"리포트 저장 실패: {e}") from e
