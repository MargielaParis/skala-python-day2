"""ML Pipeline — 전처리 + 로지스틱 회귀, 평가 지표, joblib 저장·재로딩"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
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


def coefficients(model):
    # 원핫 이름을 그대로 붙인 로지스틱 계수와 오즈비를 크기 순으로 반환
    names = [n.split("__", 1)[1] for n in model.named_steps["prep"].get_feature_names_out()]
    coef = pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values()
    return pd.DataFrame({"coef": coef, "odds_ratio": np.exp(coef)})


def build_pipeline():
    # 수치 스케일링 + 범주 원핫 + 로지스틱 회귀를 하나의 Pipeline으로 결합
    preproc = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )
    return Pipeline([("prep", preproc), ("clf", LogisticRegression(max_iter=1000))])


def split_xy(df):
    # 피처 행렬과 이진 타깃(>50K=1) 분리
    return df[NUM_COLS + CAT_COLS], df[TARGET].eq(POSITIVE).astype(int)


def train_and_evaluate(df_train, df_test, model_path):
    # adult.data로 학습하고 adult.test로 평가, joblib 저장·재로딩 일치까지 검증
    X_train, y_train = split_xy(df_train)
    X_test, y_test = split_xy(df_test)

    model = build_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred)

    try:
        joblib.dump(model, model_path)
        reloaded = joblib.load(model_path)
    except OSError as e:
        raise SystemExit(f"[오류] 모델 저장/로딩 실패: {e}") from e

    if (reloaded.predict(X_test) != pred).any():
        raise SystemExit("[오류] 재로딩 모델의 예측이 원본과 다릅니다.")

    coef = coefficients(model)
    return {
        "train": len(X_train),
        "test": len(X_test),
        "train_pos_rate": y_train.mean(),
        "test_pos_rate": y_test.mean(),
        "accuracy": acc,
        "f1": f1,
        "model_file": model_path.name,
        "n_features": len(coef),
        "coef": coef,
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "dropped": DROPPED,
    }
