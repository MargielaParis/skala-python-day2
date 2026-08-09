"""load 모듈 — 로딩 비교, Pandas·Polars 동등성, 정제 단계 검증"""

import pandas as pd
import polars as pl
import pytest

from src import load
from src.errors import PipelineError


def test_load_compare_reads_both_engines(raw_csv):
    df, compare = load.load_compare(raw_csv)

    assert list(df.columns) == load.COLS
    assert compare["rows"] == 3 and compare["cols"] == 15
    assert compare["pandas_sec"] > 0 and compare["polars_sec"] > 0


def test_load_compare_strips_test_label_dot(raw_csv):
    df, _ = load.load_compare(raw_csv)

    assert set(df["income"]) == {"<=50K", ">50K"}


def test_load_compare_treats_question_mark_as_na(raw_csv):
    df, _ = load.load_compare(raw_csv)

    assert df["workclass"].isna().sum() == 1
    assert df["occupation"].isna().sum() == 1


def test_load_compare_missing_file_raises_pipeline_error(tmp_path):
    # src/ 모듈은 프로세스를 죽이지 않고 예외를 올린다. 종료 코드는 main.py가 정한다
    with pytest.raises(PipelineError, match="찾을 수 없습니다"):
        load.load_compare(tmp_path / "nope.data")


def test_pandas_polars_are_identical_after_normalization(raw_csv):
    # shape만이 아니라 컬럼 순서·null mask·의미상 dtype·셀 값까지 불일치 0이어야 한다
    _, compare = load.load_compare(raw_csv)
    equality = compare["equality"]

    assert equality["columns_match"] is True
    assert equality["shape_match"] is True
    assert equality["dtype_mismatch"] == []
    assert equality["null_mismatch"] == 0
    assert equality["value_mismatch"] == 0
    assert equality["identical"] is True


def test_polars_reader_strips_leading_whitespace(raw_csv):
    # Polars에는 skipinitialspace가 없어 정규화 전에는 " State-gov"로 읽힌다
    df_pl = load._read_polars(raw_csv)

    assert df_pl["workclass"].to_list()[0] == "State-gov"
    assert df_pl["age"].dtype == pl.Int64


def test_compare_frames_counts_value_mismatch(raw_csv):
    df_pd, _ = load.load_compare(raw_csv)
    df_pl = load._read_polars(raw_csv)
    broken = df_pl.with_columns(pl.col("workclass").str.to_uppercase())

    result = load.compare_frames(df_pd, broken)

    assert result["identical"] is False
    assert result["value_mismatch"] > 0


def test_clean_drops_na_then_duplicates():
    df = pd.DataFrame(
        {
            "age": [39, 39, 50, 31],
            "workclass": ["Private", "Private", "Private", None],
            "income": ["<=50K", "<=50K", ">50K", "<=50K"],
        }
    )

    cleaned, info = load.clean(df)

    assert info["strategy"] == load.DROP
    assert (info["raw"], info["after_na"], info["clean"]) == (4, 3, 2)
    assert info["na_cols"] == {"workclass": 1}
    assert len(cleaned) == 2
    assert list(cleaned.index) == [0, 1]


def test_clean_reports_drop_rate_by_income():
    df = pd.DataFrame(
        {
            "age": [39, 40, 50, 31],
            "workclass": [None, "Private", "Private", None],
            "income": ["<=50K", "<=50K", ">50K", "<=50K"],
        }
    )

    _, info = load.clean(df)

    assert info["drop_rate_by_income"] == {"<=50K": pytest.approx(2 / 3), ">50K": 0.0}
    assert info["removed_by_income"] == {"<=50K": 2, ">50K": 0}
    assert info["pos_rate_raw"] == pytest.approx(0.25)
    assert info["pos_rate_kept"] == pytest.approx(0.5)


def test_clean_unknown_strategy_keeps_rows(raw_csv):
    df, _ = load.load_compare(raw_csv)

    cleaned, info = load.clean(df, strategy=load.UNKNOWN)

    assert info["strategy"] == load.UNKNOWN
    assert info["after_na"] == info["raw"] == len(df)  # 결측 행을 버리지 않는다
    assert load.UNKNOWN_LABEL in set(cleaned["workclass"])
    assert cleaned["workclass"].isna().sum() == 0
