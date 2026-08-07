# Day 2 종합실습 — End2End 데이터 분석 프로젝트

Adult Census Income(UCI) 데이터로 데이터 준비 → 시각화 → 통계 분석 → ML Pipeline →
리포트 자동 생성까지 한 번에 수행하는 분석 파이프라인.
학습/평가는 UCI가 제공하는 공식 분할(`adult.data` / `adult.test`)을 그대로 사용한다.

작성자: 박기연 (판교 7반)

## 프로젝트 구조

```
판교_7반_박기연_day2종합실습/
├── data/raw/adult.data     # 학습 원본 (수정 금지, git 미포함)
├── data/raw/adult.test     # 평가 원본 (수정 금지, git 미포함)
├── src/
│   ├── load.py             # Pandas·Polars 로딩 비교, 결측·중복 처리
│   ├── viz.py              # Seaborn 정적 / Plotly 인터랙티브 차트
│   ├── stats_test.py       # 기술통계·상관계수·t-test
│   ├── ml.py               # sklearn Pipeline 학습·평가·저장
│   └── report.py           # report.md 자동 생성
├── tests/                  # pytest 테스트 (원본 데이터 없이 실행 가능)
├── main.py                 # 전체 파이프라인 실행 진입점
├── output/                 # 차트·모델·리포트 산출물
├── .pre-commit-config.yaml # 커밋 전 자동 검사 설정
├── pyproject.toml          # black·ruff·mypy·pytest 설정
├── .flake8                 # flake8 설정
├── requirements.txt
└── requirements-dev.txt    # 검사 도구
```

## 개발 환경 설정

```bash
python3.11 -m pip install -r requirements.txt
python3.11 -m pip install -r requirements-dev.txt
pre-commit install          # 커밋 전 자동 검사 활성화
```

## 코드 품질 검사

커밋할 때 `pre-commit`이 아래 순서로 자동 실행되고, 하나라도 실패하면 커밋이 중단된다.

| 도구 | 역할 |
|---|---|
| `no-commit-to-branch` | `main` 직접 커밋 차단 (브랜치 -> PR 흐름 강제) |
| `black` | 코드 포매팅 (line-length 100) |
| `ruff` | 린트 + import 정렬 (E/W/F/I/UP/B/SIM) |
| `flake8` | 추가 스타일 검사 |
| `mypy` | 정적 타입 검사 |
| `pytest` | 테스트 전체 실행 |

전체 파일을 한 번에 검사하려면:

```bash
pre-commit run --all-files
pytest                      # 테스트만 실행
```

## 데이터 준비

데이터는 git에 올리지 않는다. 아래 명령으로 원본을 내려받는다.

```bash
curl -o data/raw/adult.data \
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -o data/raw/adult.test \
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

`adult.test`는 첫 줄이 주석(`|1x3 Cross validator`)이고 라벨에 마침표(`>50K.`)가 붙어 있어,
`src/load.py`에서 주석 제외·라벨 표기 통일을 처리한다.

## 실행 방법

```bash
python3 main.py
```

## 산출물 (output/)

| 파일 | 내용 |
|---|---|
| `eda_charts.png` | 핵심 4패널 — 연령 분포, 근로시간 박스플롯, 직업별 고소득 비율, 수치형 상관 (Seaborn) |
| `eda_numeric_all.png` | 수치형 6개 변수 전체의 소득 그룹별 분포 |
| `eda_categorical_all.png` | 범주형 8개 변수 전체의 고소득 비율 |
| `income_by_education.html` | 학력별 고소득 비율 (Plotly, 브라우저에서 열기) |
| `income_pipeline.pkl` | 전처리+모델 Pipeline (joblib) |
| `report.md` | 분석 결과 자동 생성 리포트 (발표용, 회귀계수·오즈비 포함) |

## 분석 요약

- 소득 >50K 그룹의 주당 근로시간이 유의미하게 길다 (t-test p < 0.05)
- 학력 수준이 높을수록 고소득 비율 증가
- LogisticRegression Pipeline: 정확도 0.8477 / F1 0.6610
  (학습 adult.data 30,139건 / 평가 adult.test 15,055건, 정제 후 기준)
- 회귀 피처는 sex·race를 포함한 12개 변수(원핫 후 87개). fnlwgt(표본 가중치)와
  education(education-num과 중복)만 제외했고, 계수·오즈비는 `output/report.md` 4절 참고
