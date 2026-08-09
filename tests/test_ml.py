"""ml 모듈 — Pipeline 학습·평가·저장, 기준 범주와 지표 검증"""

import numpy as np
import pandas as pd

from src import ml


def test_split_xy_excludes_dropped_columns(sample_df):
    X, y = ml.split_xy(sample_df)

    assert not set(ml.DROPPED) & set(X.columns)
    assert list(X.columns) == ml.NUM_COLS + ml.CAT_COLS
    assert set(y.unique()) == {0, 1}


def test_train_and_evaluate_returns_metrics_and_saves_model(sample_df, tmp_path):
    model_path = tmp_path / "pipeline.pkl"

    metrics = ml.train_and_evaluate(sample_df, sample_df, model_path)

    assert model_path.exists()
    assert metrics["model_file"] == "pipeline.pkl"
    assert metrics["train"] == metrics["test"] == len(sample_df)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["n_features"] == len(metrics["coef"])


def test_metrics_include_imbalance_aware_scores(sample_df, tmp_path):
    metrics = ml.train_and_evaluate(sample_df, sample_df, tmp_path / "pipeline.pkl")

    for key in ("precision", "recall", "roc_auc"):
        assert 0.0 <= metrics[key] <= 1.0
    cm = metrics["confusion"]
    assert set(cm) == {"tn", "fp", "fn", "tp"}
    assert sum(cm.values()) == len(sample_df)
    # 재현율은 실제 고소득 중 맞춘 비율과 같아야 한다
    assert metrics["recall"] == (cm["tp"] / (cm["tp"] + cm["fn"]))


def test_coefficients_include_onehot_names(sample_df, tmp_path):
    metrics = ml.train_and_evaluate(sample_df, sample_df, tmp_path / "pipeline.pkl")
    coef = metrics["coef"]

    assert list(coef.columns) == ["coef", "se", "odds_ratio", "or_low", "or_high"]
    assert any(name.startswith("sex_") for name in coef.index)
    assert coef["coef"].is_monotonic_increasing  # 계수 오름차순 정렬


def test_onehot_drops_one_reference_category_per_feature(sample_df, tmp_path):
    # drop="first"이므로 범주형 컬럼은 (범주 수 - 1)개만 남고, 빠진 값이 기준 범주가 된다
    metrics = ml.train_and_evaluate(sample_df, sample_df, tmp_path / "pipeline.pkl")
    coef, reference = metrics["coef"], metrics["reference"]

    for col in ml.CAT_COLS:
        n_levels = sample_df[col].nunique()
        n_kept = sum(name.startswith(f"{col}_") for name in coef.index)
        assert n_kept == n_levels - 1
        assert reference[col] not in {
            name.split("_", 1)[1] for name in coef.index if name.startswith(f"{col}_")
        }


def test_reference_category_is_the_most_frequent_level(sample_df, tmp_path):
    # 사전순 첫 범주를 쓰면 극소 표본이 기준이 될 수 있으므로 표본 최다 범주를 기준으로 삼는다
    metrics = ml.train_and_evaluate(sample_df, sample_df, tmp_path / "pipeline.pkl")

    for col in ml.CAT_COLS:
        counts = sample_df[col].value_counts()
        assert counts[metrics["reference"][col]] == counts.max()


def test_reference_prefers_large_category_over_alphabetical():
    # native-country=Cambodia(소수) 대신 United-States(다수)가 기준이 되는지 확인
    df = pd.DataFrame(
        {col: ["United-States"] * 50 + ["Cambodia"] * 2 for col in ml.CAT_COLS},
    )

    refs = dict(zip(ml.CAT_COLS, ml.reference_values(df), strict=True))

    assert set(refs.values()) == {"United-States"}


def test_reference_breaks_ties_alphabetically():
    df = pd.DataFrame({col: ["Male", "Female"] * 10 for col in ml.CAT_COLS})

    refs = dict(zip(ml.CAT_COLS, ml.reference_values(df), strict=True))

    assert set(refs.values()) == {"Female"}


def test_category_counts_match_training_frequencies(sample_df, tmp_path):
    metrics = ml.train_and_evaluate(sample_df, sample_df, tmp_path / "pipeline.pkl")

    counts = metrics["category_counts"]
    assert set(counts) == set(ml.CAT_COLS)
    assert counts["sex"] == sample_df["sex"].value_counts().to_dict()


def test_numeric_scales_match_training_std(sample_df, tmp_path):
    # 수치형 계수는 1 표준편차 단위이므로 그 표준편차를 함께 노출해야 한다
    metrics = ml.train_and_evaluate(sample_df, sample_df, tmp_path / "pipeline.pkl")

    assert set(metrics["numeric_scales"]) == set(ml.NUM_COLS)
    assert metrics["numeric_scales"]["age"] == sample_df["age"].std(ddof=0)


def test_compare_strategies_uses_one_shared_test_set(sample_df):
    # 전략마다 평가셋이 다르면 행 수와 사례 구성이 달라 지표를 나란히 비교할 수 없다
    bigger = pd.concat([sample_df, sample_df.head(10)], ignore_index=True)

    scores = ml.compare_strategies({"a": sample_df, "b": bigger}, sample_df)

    assert set(scores) == {"a", "b"}
    assert scores["a"]["test"] == scores["b"]["test"] == len(sample_df)
    assert scores["a"]["train"] != scores["b"]["train"]


def test_coefficient_stats_report_uncertainty(sample_df, tmp_path):
    metrics = ml.train_and_evaluate(sample_df, sample_df, tmp_path / "pipeline.pkl")
    coef = metrics["coef"]

    assert list(coef.columns) == ["coef", "se", "odds_ratio", "or_low", "or_high"]
    assert (coef["se"] > 0).all()
    assert (coef["or_low"] < coef["odds_ratio"]).all()
    assert (coef["odds_ratio"] < coef["or_high"]).all()


def test_wider_interval_for_rarer_category(sample_df, tmp_path):
    # 표본이 적은 범주일수록 구간이 넓어야 한다
    df = sample_df.copy()
    df.loc[df.index[:2], "workclass"] = "Rare-class"
    metrics = ml.train_and_evaluate(df, df, tmp_path / "pipeline.pkl")
    coef = metrics["coef"]

    rare = coef.loc["workclass_Rare-class", "se"]
    common = coef.loc[[i for i in coef.index if i.startswith("occupation_")][0], "se"]
    assert rare > common


def test_metrics_include_pr_auc(sample_df, tmp_path):
    metrics = ml.train_and_evaluate(sample_df, sample_df, tmp_path / "pipeline.pkl")

    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_threshold_is_tuned_on_out_of_fold_probabilities(sample_df, tmp_path):
    # 평가셋에서 고르면 과적합, 학습셋 in-sample 예측으로 고르면 낙관적이다
    metrics = ml.train_and_evaluate(sample_df, sample_df, tmp_path / "pipeline.pkl")
    thresholds = metrics["thresholds"]

    assert thresholds["default"]["threshold"] == 0.5
    assert thresholds["cv"] == ml.THRESHOLD_CV
    assert 0.0 <= thresholds["tuned"]["threshold"] <= 1.0


def test_best_f1_threshold_picks_the_separating_cut():
    y = pd.Series([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])

    assert 0.2 < ml._best_f1_threshold(y, proba) <= 0.8


def test_sensitivity_without_refits_without_the_column(sample_df, tmp_path):
    result = ml.sensitivity_without(sample_df, sample_df, "relationship")

    assert result["excluded"] == "relationship"
    assert result["focus"] == "sex"
    assert not any(name.startswith("relationship_") for name in result["coef"].index)
    assert all(name.startswith("sex_") for name in result["coef"].index)
    assert result["reference"] in set(sample_df["sex"])
