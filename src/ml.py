"""전처리와 로지스틱 회귀를 묶은 Pipeline. 평가 지표, 계수 불확실성, joblib 저장·재로딩."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .errors import PipelineError

# fnlwgt(표본 가중치)·education(education-num과 중복)은 피처에서 제외
NUM_COLS = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT_COLS = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
TARGET = "income"
POSITIVE = ">50K"
# 제외 컬럼: fnlwgt(표본 가중치, 개인 속성이 아님), education(education-num과 1:1 중복)
DROPPED = ["fnlwgt", "education"]
# 기준 범주 선정 규칙 — 리포트에 그대로 싣는다
REFERENCE_RULE = "학습 표본이 가장 많은 범주(동수면 사전순 앞)"
ALPHA = 0.05  # 계수 신뢰구간의 유의수준
THRESHOLD_CV = 5  # 임계값 탐색용 교차검증 분할 수


def reference_values(df: pd.DataFrame, cat_cols: list[str] | None = None) -> np.ndarray:
    # 각 범주형 변수에서 표본이 가장 많은 범주를 기준으로 고른다.
    # 표본이 극히 적은 범주가 기준이 되면 나머지 대비가 모두 불안정해진다.
    # 동수일 때는 사전순으로 확정해 실행마다 같은 기준이 나오게 한다.
    refs = []
    for col in CAT_COLS if cat_cols is None else cat_cols:
        counts = df[col].value_counts()
        refs.append(counts[counts == counts.max()].index.min())
    return np.array(refs, dtype=object)


def category_counts(df: pd.DataFrame) -> dict[str, dict[Any, int]]:
    # 계수 표에 "이 대비가 몇 개 표본에서 나왔는지"를 함께 싣기 위한 범주별 학습 표본 수
    return {col: df[col].value_counts().to_dict() for col in CAT_COLS if col in df.columns}


def build_pipeline(X_train: pd.DataFrame, cat_cols: list[str] | None = None) -> Pipeline:
    # 수치 스케일링 + 범주 원핫(기준 범주 제외) + 로지스틱 회귀를 하나의 Pipeline으로 결합
    cats = CAT_COLS if cat_cols is None else cat_cols
    preproc = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop=reference_values(X_train, cats)),
                cats,
            ),
        ]
    )
    return Pipeline([("prep", preproc), ("clf", LogisticRegression(max_iter=1000))])


def feature_names(model: Pipeline) -> list[str]:
    return [n.split("__", 1)[1] for n in model.named_steps["prep"].get_feature_names_out()]


def _design_matrix(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    matrix = model.named_steps["prep"].transform(X)
    return np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=float)


def coefficient_stats(model: Pipeline, X_train: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    # 계수와 오즈비에 표준오차·신뢰구간을 붙인다.
    #
    # LogisticRegression은 기본이 L2 정규화라 순수 최대우도 추정이 아니다. 정규화를 분산 C인
    # 가우시안 사전분포로 보고 사후분포를 최빈값 주변에서 정규근사(라플라스 근사)한 표준오차를 쓴다.
    # 사후 정밀도는 Z'WZ + (1/C)I이고 절편은 정규화하지 않는다. 따라서 아래 구간은 정확한 Wald
    # 신뢰구간이 아니고, 정규화 때문에 계수 자체도 0 쪽으로 수축돼 있다.
    clf = model.named_steps["clf"]
    Z = _design_matrix(model, X_train)
    design = np.hstack([np.ones((Z.shape[0], 1)), Z])  # 절편 포함

    # 원본 컬럼을 받는 것은 Pipeline이므로 clf가 아니라 model에 넣는다
    prob = model.predict_proba(X_train)[:, 1]
    weights = prob * (1.0 - prob)
    penalty = np.eye(design.shape[1]) / clf.C
    penalty[0, 0] = 0.0  # 절편은 정규화 대상이 아니다

    precision = design.T @ (design * weights[:, None]) + penalty
    try:
        covariance = np.linalg.inv(precision)
    except np.linalg.LinAlgError as e:
        raise PipelineError(f"계수 공분산 계산 실패(설계행렬 특이): {e}") from e
    se = np.sqrt(np.clip(np.diag(covariance)[1:], 0.0, None))  # 절편 제외

    z = float(norm.ppf(1.0 - alpha / 2.0))
    coef = pd.Series(clf.coef_[0], index=feature_names(model))
    se_series = pd.Series(se, index=coef.index)
    table = pd.DataFrame(
        {
            "coef": coef,
            "se": se_series,
            "odds_ratio": np.exp(coef),
            "or_low": np.exp(coef - z * se_series),
            "or_high": np.exp(coef + z * se_series),
        }
    )
    return table.sort_values("coef")


def reference_categories(model: Pipeline, cat_cols: list[str] | None = None) -> dict[str, Any]:
    # drop으로 빠진 범주 = 오즈비 해석의 기준. 남은 계수는 모두 이 기준 대비 값이다
    cats = CAT_COLS if cat_cols is None else cat_cols
    enc = model.named_steps["prep"].named_transformers_["cat"]
    dropped = enc.drop_idx_ if enc.drop_idx_ is not None else [None] * len(cats)
    return {
        col: (levels[int(idx)] if idx is not None else None)
        for col, levels, idx in zip(cats, enc.categories_, dropped, strict=True)
    }


def numeric_scales(model: Pipeline) -> dict[str, float]:
    # 수치형 계수는 StandardScaler 적용 후 값이므로 "1 표준편차"가 무엇인지 함께 노출한다
    scaler = model.named_steps["prep"].named_transformers_["num"]
    return dict(zip(NUM_COLS, scaler.scale_, strict=True))


def split_xy(df: pd.DataFrame, cat_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    # 피처 행렬과 이진 타깃(>50K=1) 분리
    cats = CAT_COLS if cat_cols is None else cat_cols
    return df[NUM_COLS + cats], df[TARGET].eq(POSITIVE).astype(int)


def _score(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        # PR-AUC: 양성이 드물 때 ROC-AUC보다 성능 차이를 민감하게 드러낸다
        "pr_auc": average_precision_score(y_true, proba),
        # tp = 고소득을 고소득으로, fn = 실제 고소득을 저소득으로 잘못 예측한 건수
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _best_f1_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    if not len(thresholds):
        return 0.5
    denom = precisions + recalls
    f1_grid = np.divide(2 * precisions * recalls, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(f1_grid[:-1]))])


def threshold_analysis(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int = THRESHOLD_CV,
) -> dict[str, Any]:
    # 기본 임계값 0.5는 자의적이라 F1을 최대화하는 값을 따로 찾는다.
    # 평가셋에서 고르면 평가셋에 대한 과적합이고, 학습셋 예측으로 고르면 모델이 이미 그 데이터에
    # 적합돼 있어 낙관적이다. 교차검증 out-of-fold 확률은 각 행을 보지 못한 모델에서 나오므로
    # 처음 보는 데이터에 가장 가깝다.
    try:
        oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    except ValueError as e:
        raise PipelineError(f"임계값 탐색용 교차검증 실패: {e}") from e
    best = _best_f1_threshold(y_train, oof)

    proba_test = model.predict_proba(X_test)[:, 1]
    return {
        "cv": cv,
        "default": {
            "threshold": 0.5,
            **_score(y_test, (proba_test >= 0.5).astype(int), proba_test),
        },
        "tuned": {
            "threshold": best,
            **_score(y_test, (proba_test >= best).astype(int), proba_test),
        },
    }


def fit_score(
    df_train: pd.DataFrame, df_test: pd.DataFrame, cat_cols: list[str] | None = None
) -> tuple[Pipeline, pd.DataFrame, np.ndarray, dict[str, Any]]:
    # 학습 후 평가 지표만 계산해 (model, X_test, pred, 지표 dict) 반환 — 저장은 하지 않는다
    X_train, y_train = split_xy(df_train, cat_cols)
    X_test, y_test = split_xy(df_test, cat_cols)

    model = build_pipeline(X_train, cat_cols)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train": len(X_train),
        "test": len(X_test),
        "train_pos_rate": y_train.mean(),
        "test_pos_rate": y_test.mean(),
        **_score(y_test, pred, proba),
    }
    return model, X_test, pred, metrics


def compare_strategies(
    train_sets: dict[str, pd.DataFrame], df_test: pd.DataFrame
) -> dict[str, Any]:
    # 결측 처리 방식 A/B 비교. 전략마다 학습셋은 다르지만 평가셋은 하나로 고정한다.
    # 전략별 평가셋을 따로 쓰면 행 수와 사례 구성이 달라져 지표를 나란히 놓고 비교할 수 없다.
    return {name: fit_score(tr, df_test)[3] for name, tr in train_sets.items()}


def sensitivity_without(
    df_train: pd.DataFrame, df_test: pd.DataFrame, excluded: str, focus: str = "sex"
) -> dict[str, Any]:
    # 특정 범주형 변수를 빼고 다시 학습해 관심 변수의 계수가 얼마나 달라지는지 본다.
    # relationship(Husband/Wife)은 sex와 거의 겹치므로, 통제 의존도를 확인하지 않으면
    # 성별 계수를 성별 격차로 오독하기 쉽다.
    cat_cols = [c for c in CAT_COLS if c != excluded]
    model, _, _, metrics = fit_score(df_train, df_test, cat_cols)
    coef = coefficient_stats(model, split_xy(df_train, cat_cols)[0])
    rows = coef.loc[coef.index.str.startswith(f"{focus}_")]
    return {
        "excluded": excluded,
        "focus": focus,
        "reference": reference_categories(model, cat_cols).get(focus),
        "coef": rows,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
    }


def train_and_evaluate(
    df_train: pd.DataFrame, df_test: pd.DataFrame, model_path: Path
) -> dict[str, Any]:
    # adult.data로 학습하고 adult.test로 평가, joblib 저장·재로딩 일치까지 검증
    model, X_test, pred, metrics = fit_score(df_train, df_test)

    try:
        joblib.dump(model, model_path)
        reloaded = joblib.load(model_path)
    except OSError as e:
        raise PipelineError(f"모델 저장/로딩 실패: {e}") from e

    if (reloaded.predict(X_test) != pred).any():
        raise PipelineError("재로딩 모델의 예측이 원본과 다릅니다.")

    X_train, y_train = split_xy(df_train)
    coef = coefficient_stats(model, X_train)
    metrics.update(
        {
            "model_file": model_path.name,
            "n_features": len(coef),
            "coef": coef,
            "reference": reference_categories(model),
            "reference_rule": REFERENCE_RULE,
            "category_counts": category_counts(df_train),
            "numeric_scales": numeric_scales(model),
            "thresholds": threshold_analysis(model, X_train, y_train, X_test, split_xy(df_test)[1]),
            "num_cols": NUM_COLS,
            "cat_cols": CAT_COLS,
            "dropped": DROPPED,
            "alpha": ALPHA,
        }
    )
    return metrics
