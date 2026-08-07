"""데이터 준비 — Pandas·Polars 로딩 비교, 결측치·중복 처리, 기본 EDA"""

import time

import pandas as pd
import polars as pl

# Adult Census Income 컬럼 정의 (원본 파일에 헤더 없음)
COLS = ["age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
NA_TOKEN = "?"     # 원본에서 결측을 뜻하는 문자
COMMENT = "|"      # adult.test 첫 줄 "|1x3 Cross validator" 주석


def load_compare(file_path):
    # 같은 파일을 Pandas와 Polars로 각각 로딩해 (pandas_df, 비교 결과 dict) 반환
    try:
        t0 = time.perf_counter()
        df_pd = pd.read_csv(file_path, header=None, names=COLS,
                            na_values=NA_TOKEN, skipinitialspace=True,
                            comment=COMMENT)
        pd_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        df_pl = pl.read_csv(file_path, has_header=False, new_columns=COLS,
                            null_values=[NA_TOKEN, f" {NA_TOKEN}"],
                            comment_prefix=COMMENT)
        # Polars는 파일 끝 빈 줄을 전체 null 행으로 읽으므로 제거해 Pandas와 맞춤
        df_pl = df_pl.filter(~pl.all_horizontal(pl.all().is_null()))
        pl_sec = time.perf_counter() - t0
    except FileNotFoundError:
        raise SystemExit(f"[오류] 데이터 파일을 찾을 수 없습니다: {file_path}")
    except (pd.errors.ParserError, pl.exceptions.ComputeError) as e:
        raise SystemExit(f"[오류] 데이터 로딩 실패: {e}")

    if df_pd.shape != df_pl.shape:
        raise SystemExit(f"[오류] 두 도구의 로딩 결과가 다릅니다: {df_pd.shape} vs {df_pl.shape}")

    # adult.test의 라벨은 마침표가 붙어 있어(">50K.") train과 표기를 통일
    df_pd["income"] = df_pd["income"].str.rstrip(".")

    compare = {"rows": len(df_pd), "cols": df_pd.shape[1],
               "pandas_sec": pd_sec, "polars_sec": pl_sec}
    return df_pd, compare


def clean(df):
    # 결측 제거와 중복 제거를 단계별로 수행해 (정제 df, 처리 내역 dict) 반환
    n_raw = len(df)
    na_counts = df.isnull().sum()
    df = df.dropna()
    n_after_na = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, {"raw": n_raw, "after_na": n_after_na, "clean": len(df),
                "na_cols": na_counts[na_counts > 0].to_dict()}
