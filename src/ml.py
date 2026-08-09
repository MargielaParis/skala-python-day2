"""ML Pipeline — 전처리 + 로지스틱 회귀, 평가 지표, joblib 저장·재로딩"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
REFERENCE_RULE = "각 범주형 변수에서 학습 표본이 가장 많은 범주 (동수면 사전순 앞)"


def reference_values(df):
    # 기준 범주를 표본 최다 범주로 고른다.
    # drop="first"(사전순)는 native-country=Cambodia(학습 18행)처럼 극소 표본을 기준으로 잡아
    # 나머지 40개 범주의 대비를 전부 불안정하게 만든다. 동수일 때는 사전순으로 확정한다.
    refs = []
    for col in CAT_COLS:
        counts = df[col].value_counts()
        refs.append(counts[counts == counts.max()].index.min())
    return np.array(refs, dtype=object)


def category_counts(df):
    # 계수 표에 "이 대비가 몇 개 표본에서 나왔는지"를 함께 싣기 위한 범주별 학습 표본 수
    return {col: df[col].value_counts().to_dict() for col in CAT_COLS}


def build_pipeline(df_train):
    # 수치 스케일링 + 범주 원핫(기준 범주 제외) + 로지스틱 회귀를 하나의 Pipeline으로 결합
    preproc = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop=reference_values(df_train)),
                CAT_COLS,
            ),
        ]
    )
    return Pipeline([("prep", preproc), ("clf", LogisticRegression(max_iter=1000))])


def coefficients(model):
    # 원핫 이름을 그대로 붙인 로지스틱 계수와 오즈비를 크기 순으로 반환
    names = [n.split("__", 1)[1] for n in model.named_steps["prep"].get_feature_names_out()]
    coef = pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values()
    return pd.DataFrame({"coef": coef, "odds_ratio": np.exp(coef)})


def reference_categories(model):
    # drop으로 빠진 범주 = 오즈비 해석의 기준. 남은 계수는 모두 이 기준 대비 값이다
    enc = model.named_steps["prep"].named_transformers_["cat"]
    dropped = enc.drop_idx_ if enc.drop_idx_ is not None else [None] * len(CAT_COLS)
    return {
        col: (cats[int(idx)] if idx is not None else None)
        for col, cats, idx in zip(CAT_COLS, enc.categories_, dropped, strict=True)
    }


def numeric_scales(model):
    # 수치형 계수는 StandardScaler 적용 후 값이므로 "1 표준편차"가 무엇인지 함께 노출한다
    scaler = model.named_steps["prep"].named_transformers_["num"]
    return dict(zip(NUM_COLS, scaler.scale_, strict=True))


def split_xy(df):
    # 피처 행렬과 이진 타깃(>50K=1) 분리
    return df[NUM_COLS + CAT_COLS], df[TARGET].eq(POSITIVE).astype(int)


def fit_score(df_train, df_test):
    # 학습 후 평가 지표만 계산해 (model, X_test, pred, 지표 dict) 반환 — 저장은 하지 않는다
    X_train, y_train = split_xy(df_train)
    X_test, y_test = split_xy(df_test)

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    metrics = {
        "train": len(X_train),
        "test": len(X_test),
        "train_pos_rate": y_train.mean(),
        "test_pos_rate": y_test.mean(),
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        # tp = 고소득을 고소득으로, fn = 실제 고소득을 저소득으로 잘못 예측한 건수
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return model, X_test, pred, metrics


def compare_strategies(datasets):
    # {결측 전략: (train_df, test_df)} -> {전략: 지표 dict}. 결측 처리 방식 A/B 비교용
    return {name: fit_score(tr, te)[3] for name, (tr, te) in datasets.items()}


def train_and_evaluate(df_train, df_test, model_path):
    # adult.data로 학습하고 adult.test로 평가, joblib 저장·재로딩 일치까지 검증
    model, X_test, pred, metrics = fit_score(df_train, df_test)

    try:
        joblib.dump(model, model_path)
        reloaded = joblib.load(model_path)
    except OSError as e:
        raise SystemExit(f"[오류] 모델 저장/로딩 실패: {e}") from e

    if (reloaded.predict(X_test) != pred).any():
        raise SystemExit("[오류] 재로딩 모델의 예측이 원본과 다릅니다.")

    coef = coefficients(model)
    metrics.update(
        {
            "model_file": model_path.name,
            "n_features": len(coef),
            "coef": coef,
            "reference": reference_categories(model),
            "reference_rule": REFERENCE_RULE,
            "category_counts": category_counts(df_train),
            "numeric_scales": numeric_scales(model),
            "num_cols": NUM_COLS,
            "cat_cols": CAT_COLS,
            "dropped": DROPPED,
        }
    )
    return metrics
