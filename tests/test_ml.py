"""ml 모듈 — Pipeline 학습·평가·저장, 기준 범주와 지표 검증"""

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

    assert list(coef.columns) == ["coef", "odds_ratio"]
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


def test_compare_strategies_scores_each_dataset(sample_df):
    scores = ml.compare_strategies({"a": (sample_df, sample_df), "b": (sample_df, sample_df)})

    assert set(scores) == {"a", "b"}
    assert scores["a"]["f1"] == scores["b"]["f1"]
