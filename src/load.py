"""데이터 준비 — Pandas·Polars 로딩 비교·동등성 검증, 결측치·중복 처리, 기본 EDA"""

import time
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from .errors import PipelineError

# Adult Census Income 컬럼 정의 (원본 파일에 헤더 없음)
COLS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT_COLS = [c for c in COLS if c not in NUM_COLS and c != "income"]
NA_TOKEN = "?"  # 원본에서 결측을 뜻하는 문자
COMMENT = "|"  # adult.test 첫 줄 "|1x3 Cross validator" 주석
TARGET = "income"
POSITIVE = ">50K"
# 원본에서 capital-gain은 99999로 상한 처리(top-coding)되어 있다. 실제 금액이 아니다
CAPITAL_GAIN_CAP = 99999

# 결측 처리 전략 — DROP은 결측 행 삭제, UNKNOWN은 범주형 결측을 별도 범주로 보존
DROP, UNKNOWN = "drop", "unknown"
UNKNOWN_LABEL = "Unknown"


def _read_pandas(file_path: Path) -> pd.DataFrame:
    # skipinitialspace로 구분자 뒤 공백을 없애고, "?"를 결측으로, "|" 줄을 주석으로 처리
    df = pd.read_csv(
        file_path,
        header=None,
        names=COLS,
        na_values=NA_TOKEN,
        skipinitialspace=True,
        comment=COMMENT,
    )
    # adult.test의 라벨은 마침표가 붙어 있어(">50K.") train과 표기를 통일
    df[TARGET] = df[TARGET].str.rstrip(".")
    return df


def _read_polars(file_path: Path) -> pl.DataFrame:
    # Polars에는 skipinitialspace가 없어 앞 공백이 남고 수치 컬럼까지 문자열로 추론된다.
    # 전 컬럼을 문자열로 읽은 뒤 Pandas와 같은 계약(공백 제거 -> 결측 토큰 -> 수치 캐스팅)을 직접 적용한다.
    df = pl.read_csv(
        file_path,
        has_header=False,
        new_columns=COLS,
        comment_prefix=COMMENT,
        infer_schema_length=0,
    )
    # Polars는 파일 끝 빈 줄을 전체 null 행으로 읽으므로 제거해 Pandas와 맞춤
    df = df.filter(~pl.all_horizontal(pl.all().is_null()))
    null_str = pl.lit(None, dtype=pl.String)
    df = df.with_columns(pl.col(c).str.strip_chars() for c in COLS)
    df = df.with_columns(
        pl.when(pl.col(c) == NA_TOKEN).then(null_str).otherwise(pl.col(c)).alias(c) for c in COLS
    )
    df = df.with_columns(pl.col(TARGET).str.strip_chars_end("."))
    return df.with_columns(pl.col(c).cast(pl.Int64) for c in NUM_COLS)


def _semantic_dtype(dtype: object) -> str:
    # 엔진마다 이름이 다른 dtype을 의미 단위(int/float/string)로 환산해 비교한다
    name = str(dtype).lower()
    if "int" in name:
        return "int"
    if "float" in name:
        return "float"
    return "string"


def compare_frames(df_pd: pd.DataFrame, df_pl: pl.DataFrame) -> dict[str, Any]:
    # 같은 계약으로 읽은 두 DataFrame을 컬럼·shape·null mask·의미상 dtype·셀 값까지 비교
    cols_pd, cols_pl = list(df_pd.columns), list(df_pl.columns)
    result: dict[str, Any] = {
        "columns_match": cols_pd == cols_pl,
        "shape_match": df_pd.shape == df_pl.shape,
        "dtype_mismatch": [],
        "null_mismatch": None,
        "value_mismatch": None,
    }
    if not (result["columns_match"] and result["shape_match"]):
        result["identical"] = False
        return result

    result["dtype_mismatch"] = [
        (c, str(df_pd[c].dtype), str(df_pl.schema[c]))
        for c in cols_pd
        if _semantic_dtype(df_pd[c].dtype) != _semantic_dtype(df_pl.schema[c])
    ]

    # 셀 값은 Python 값으로 내려 비교한다 (polars.to_pandas는 pyarrow를 요구하므로 쓰지 않는다)
    null_mismatch = value_mismatch = 0
    for c in cols_pd:
        left, right = df_pd[c].tolist(), df_pl[c].to_list()
        left_na, right_na = df_pd[c].isna().tolist(), [v is None for v in right]
        for i in range(len(left)):
            if left_na[i] != right_na[i]:
                null_mismatch += 1
            elif not left_na[i] and left[i] != right[i]:
                value_mismatch += 1
    result["null_mismatch"], result["value_mismatch"] = null_mismatch, value_mismatch

    result["identical"] = (
        not result["dtype_mismatch"]
        and result["null_mismatch"] == 0
        and result["value_mismatch"] == 0
    )
    return result


def load_compare(file_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    # 같은 파일을 Pandas와 Polars로 각각 로딩해 (pandas_df, 비교 결과 dict) 반환
    try:
        t0 = time.perf_counter()
        df_pd = _read_pandas(file_path)
        pd_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        df_pl = _read_polars(file_path)
        pl_sec = time.perf_counter() - t0
    except FileNotFoundError:
        raise PipelineError(f"데이터 파일을 찾을 수 없습니다: {file_path}") from None
    except (pd.errors.ParserError, pl.exceptions.ComputeError) as e:
        raise PipelineError(f"데이터 로딩 실패: {e}") from e

    equality = compare_frames(df_pd, df_pl)
    if not (equality["columns_match"] and equality["shape_match"]):
        raise PipelineError(f"두 도구의 로딩 결과가 다릅니다: {df_pd.shape} vs {df_pl.shape}")

    compare = {
        "rows": len(df_pd),
        "cols": df_pd.shape[1],
        "pandas_sec": pd_sec,
        "polars_sec": pl_sec,
        "equality": equality,
    }
    return df_pd, compare


def _na_impact(df: pd.DataFrame) -> dict[str, Any]:
    # 결측 행 삭제가 소득 그룹별로 얼마나 치우쳤는지 (제거율, 제거 전후 고소득 비율) 계산
    has_na = df.isnull().any(axis=1)
    total = df.groupby(TARGET).size()
    removed = df.loc[has_na].groupby(TARGET).size().reindex(total.index, fill_value=0)
    kept = df.loc[~has_na]
    return {
        "drop_rate_by_income": (removed / total).to_dict(),
        "removed_by_income": removed.to_dict(),
        "total_by_income": total.to_dict(),
        "pos_rate_raw": float(df[TARGET].eq(POSITIVE).mean()),
        "pos_rate_kept": float(kept[TARGET].eq(POSITIVE).mean()) if len(kept) else float("nan"),
    }


def clean(df: pd.DataFrame, strategy: str = DROP) -> tuple[pd.DataFrame, dict[str, Any]]:
    # 결측 처리(전략 선택)와 중복 제거를 단계별로 수행해 (정제 df, 처리 내역 dict) 반환
    n_raw = len(df)
    na_counts = df.isnull().sum()
    impact = _na_impact(df)

    if strategy == UNKNOWN:
        # 범주형 결측만 별도 범주로 채워 행을 보존한다 (수치형에는 원본에 결측이 없다)
        df = df.fillna({c: UNKNOWN_LABEL for c in CAT_COLS}).dropna()
    else:
        df = df.dropna()

    n_after_na = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, {
        "strategy": strategy,
        "raw": n_raw,
        "after_na": n_after_na,
        "clean": len(df),
        "na_cols": na_counts[na_counts > 0].to_dict(),
        **impact,
    }
