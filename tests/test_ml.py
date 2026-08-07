"""ml 모듈 — Pipeline 학습·평가·저장 검증"""

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


def test_coefficients_include_onehot_names(sample_df, tmp_path):
    metrics = ml.train_and_evaluate(sample_df, sample_df, tmp_path / "pipeline.pkl")
    coef = metrics["coef"]

    assert list(coef.columns) == ["coef", "odds_ratio"]
    assert any(name.startswith("sex_") for name in coef.index)
    assert coef["coef"].is_monotonic_increasing  # 계수 오름차순 정렬
