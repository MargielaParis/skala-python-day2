"""테스트 공용 픽스처 — 원본 데이터 없이도 돌아가는 소형 합성 데이터셋"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_rows(n, high_income):
    # 고소득/저소득 그룹을 근로시간·학력이 뚜렷이 다르게 생성해 검정 결과를 결정적으로 만든다
    label = ">50K" if high_income else "<=50K"
    return [
        {
            "age": 40 + (i % 5) if high_income else 30 + (i % 5),
            "workclass": "Private" if i % 2 else "Self-emp-not-inc",
            "fnlwgt": 100000 + i,
            "education": "Bachelors" if high_income else "HS-grad",
            "education-num": 13 + (i % 3) if high_income else 9 + (i % 3),
            "marital-status": "Married-civ-spouse" if high_income else "Never-married",
            "occupation": "Exec-managerial" if high_income else "Other-service",
            "relationship": "Husband" if high_income else "Not-in-family",
            "race": "White" if i % 3 else "Black",
            "sex": "Male" if i % 2 else "Female",
            "capital-gain": 500 * (i % 3) if high_income else 0,
            "capital-loss": 0,
            "hours-per-week": 50 + (i % 5) if high_income else 35 + (i % 5),
            "native-country": "United-States",
            "income": label,
        }
        for i in range(n)
    ]


@pytest.fixture
def sample_df():
    # 고소득 30건 + 저소득 30건, 결측·중복 없는 정제 완료 상태
    return pd.DataFrame(_make_rows(30, True) + _make_rows(30, False))


@pytest.fixture
def raw_csv(tmp_path):
    # adult.test 형식(주석 첫 줄, 라벨 마침표, "?" 결측)을 흉내 낸 원본 파일
    path = tmp_path / "adult.sample"
    lines = [
        "|1x3 Cross validator",
        "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical,"
        " Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K.",
        "52, Self-emp-inc, 287927, HS-grad, 9, Married-civ-spouse, Exec-managerial,"
        " Wife, White, Female, 15024, 0, 40, United-States, >50K.",
        "31, ?, 84154, Some-college, 10, Married-civ-spouse, ?,"
        " Husband, White, Male, 0, 0, 38, ?, <=50K.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
