"""통계 분석 — 기술통계, 상관계수, t-test(p-value·효과크기), 성별×소득 카이제곱"""

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05  # 유의수준
# float64가 표현할 수 있는 최소 양수는 약 5e-324라, 그보다 작은 p는 정확히 0.0으로 언더플로된다
P_FLOOR_TEXT = "1e-308 미만 (float64 표현 한계)"
POSITIVE = ">50K"


def describe_numeric(df):
    # 수치형 기술통계(평균·표준편차·분위수)와 상관행렬 반환
    numeric = df.select_dtypes("number")
    return numeric.describe(), numeric.corr()


def format_p(p):
    # p=0은 "차이 없음"이 아니라 언더플로이므로 0.000000으로 찍지 않고 한계값으로 표기
    if p == 0.0:
        return P_FLOOR_TEXT
    if p < 1e-4:
        return f"{p:.3e}"
    return f"{p:.6f}"


def cohens_d(a, b):
    # pooled 표준편차 기준 표준화 평균차 — 표본이 크면 작은 차이도 유의해지므로 크기를 함께 본다
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else 0.0


def effect_label(d):
    # Cohen(1988) 관례 구간 — 절대값 기준
    size = abs(d)
    if size < 0.2:
        return "매우 작음"
    if size < 0.5:
        return "작음"
    if size < 0.8:
        return "중간"
    return "큼"


def ttest_hours_by_income(df):
    # 소득 그룹(>50K vs <=50K) 간 주당 근로시간 평균 차이를 독립표본 t-test로 검정
    high = df.loc[df["income"] == POSITIVE, "hours-per-week"]
    low = df.loc[df["income"] == "<=50K", "hours-per-week"]
    if high.empty or low.empty:
        raise SystemExit("[오류] t-test 그룹이 비어 있습니다. income 값을 확인하세요.")

    result = stats.ttest_ind(high, low, equal_var=False)  # Welch: 등분산 가정 없음
    t, p = float(result.statistic), float(result.pvalue)
    ci = result.confidence_interval(confidence_level=1 - ALPHA)
    d = cohens_d(high, low)
    return {
        "mean_high": high.mean(),
        "mean_low": low.mean(),
        "diff": float(high.mean() - low.mean()),
        "ci_low": float(ci.low),
        "ci_high": float(ci.high),
        "cohens_d": d,
        "effect": effect_label(d),
        "t": t,
        "p": p,
        "p_text": format_p(p),
        "p_underflow": bool(p == 0.0),
        "significant": p < ALPHA,
    }


def chisq_sex_income(df):
    # 실습 주제(성별과 소득의 관계)를 직접 검정 — 교차표 기반 카이제곱 독립성 검정
    table = pd.crosstab(df["sex"], df["income"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise SystemExit("[오류] 카이제곱 교차표가 2x2 미만입니다. sex/income 값을 확인하세요.")

    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    # Cramer's V: 표본 크기와 무관한 연관 강도 (2x2에서는 phi 계수와 같다)
    cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    rate_by_sex = df.assign(high=df["income"].eq(POSITIVE)).groupby("sex")["high"].mean()
    return {
        "table": table,
        "chi2": float(chi2),
        "p": float(p),
        "p_text": format_p(float(p)),
        "dof": int(dof),
        "n": n,
        "cramers_v": cramers_v,
        "rate_by_sex": rate_by_sex.to_dict(),
        "significant": bool(p < ALPHA),
    }
