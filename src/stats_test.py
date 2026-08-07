"""통계 분석 — 기술통계, 상관계수, t-test(p-value 해석)"""

from scipy import stats

ALPHA = 0.05  # 유의수준
# float64가 표현할 수 있는 최소 양수는 약 5e-324라, 그보다 작은 p는 정확히 0.0으로 언더플로된다
P_FLOOR_TEXT = "1e-308 미만 (float64 표현 한계)"


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


def ttest_hours_by_income(df):
    # 소득 그룹(>50K vs <=50K) 간 주당 근로시간 평균 차이를 독립표본 t-test로 검정
    high = df.loc[df["income"] == ">50K", "hours-per-week"]
    low = df.loc[df["income"] == "<=50K", "hours-per-week"]
    if high.empty or low.empty:
        raise SystemExit("[오류] t-test 그룹이 비어 있습니다. income 값을 확인하세요.")

    t, p = stats.ttest_ind(high, low, equal_var=False)  # Welch: 등분산 가정 없음
    return {
        "mean_high": high.mean(),
        "mean_low": low.mean(),
        "t": t,
        "p": p,
        "p_text": format_p(p),
        "p_underflow": bool(p == 0.0),
        "significant": p < ALPHA,
    }
