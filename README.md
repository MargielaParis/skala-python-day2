# Day 2 종합실습 — Adult Census Income End2End 분석

UCI Adult Census Income 데이터로 데이터 준비, 시각화, 통계 분석, ML Pipeline, 리포트 생성까지
한 번에 수행하는 분석 파이프라인이다. 학습과 평가는 UCI 공식 분할(`adult.data` / `adult.test`)을
그대로 쓴다.

작성자: 박기연 (판교 7반)

## 프로젝트 구조

```
판교_7반_박기연_day2종합실습/
├── data/raw/adult.data     # 학습 원본 (수정 금지, git 미포함)
├── data/raw/adult.test     # 평가 원본 (수정 금지, git 미포함)
├── src/
│   ├── errors.py           # 공통 예외 (PipelineError)
│   ├── load.py             # Pandas·Polars 로딩·동등성 검증, 결측·중복 처리
│   ├── viz.py              # Seaborn 정적 차트 / Plotly 인터랙티브 차트
│   ├── stats_test.py       # 기술통계·상관·t-test·카이제곱·데이터 한계 산출
│   ├── ml.py               # sklearn Pipeline 학습·평가·계수 구간·임계값·민감도
│   └── report.py           # report.md 생성
├── tests/                  # pytest 테스트 (원본 데이터 없이 실행 가능)
├── docs/verification.md    # 재검증 기록 (커밋 SHA, 입력 해시, 지표)
├── main.py                 # 전체 파이프라인 실행 진입점
├── output/                 # 차트·모델·리포트 산출물
├── .pre-commit-config.yaml # 커밋 전 자동 검사 설정
├── pyproject.toml          # black·ruff·mypy·pytest 설정
├── .flake8                 # flake8 설정
├── requirements.txt
└── requirements-dev.txt    # 검사 도구
```

## 개발 환경 설정

Python 3.12 이상이 필요하다. 가상환경을 만들어 활성화한 뒤 의존성을 설치한다.

macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

이후 모든 명령은 가상환경이 활성화된 셸에서 실행한다. `.pre-commit-config.yaml`의 `pytest` 훅은
`language: system`이라 별도 환경을 만들지 않고 현재 PATH의 `pytest`를 쓴다. 가상환경을 활성화하지
않으면 `Executable 'pytest' not found`로 이 훅만 실패한다.

훅 환경은 `default_language_version: python3`이라 pre-commit을 실행한 인터프리터를 따라간다.
특정 마이너 버전을 고정하면 그 버전이 없는 장비에서 훅 환경 생성 자체가 실패한다.

## 데이터 준비

데이터는 git에 올리지 않는다. 아래 명령으로 원본을 내려받는다. `data/raw` 디렉터리를 먼저
만들어야 curl이 실패하지 않는다.

macOS / Linux

```bash
mkdir -p data/raw
curl -fL -o data/raw/adult.data \
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -fL -o data/raw/adult.test \
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

Windows (PowerShell)

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
curl.exe -fL -o data\raw\adult.data `
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl.exe -fL -o data\raw\adult.test `
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

`adult.test`는 첫 줄이 주석(`|1x3 Cross validator`)이고 라벨에 마침표(`>50K.`)가 붙어 있다.
`src/load.py`에서 주석 제외와 라벨 표기 통일을 처리한다.

## 실행 방법

```bash
python main.py
```

## 코드 품질 검사

커밋할 때 pre-commit이 아래 순서로 자동 실행되고, 하나라도 실패하면 커밋이 중단된다.

| 도구 | 역할 |
|---|---|
| `no-commit-to-branch` | `main` 직접 커밋 차단 (브랜치에서 PR로 올리는 흐름 강제) |
| `black` | 코드 포매팅 (line-length 100) |
| `ruff` | 린트와 import 정렬 (E/W/F/I/UP/B/SIM) |
| `flake8` | 추가 스타일 검사 |
| `mypy` | 정적 타입 검사 |
| `pytest` | 테스트 전체 실행 |

위 검사는 `pre-commit install`로 훅을 설치한 클론에서만 동작한다. 새로 clone한 뒤 설치를
건너뛰면 커밋이 아무 검사 없이 통과하므로, 먼저 훅이 걸려 있는지 확인한다.

```bash
ls .git/hooks/pre-commit    # 없으면 pre-commit install 을 실행한다
```

전체 파일을 한 번에 검사하려면 아래를 실행한다. `no-commit-to-branch` 훅은 `main`에서 의도적으로
실패하므로 작업 브랜치에서 실행해야 전체 통과를 확인할 수 있다.

```bash
pre-commit run --all-files
```

개별 도구를 직접 돌릴 수도 있다.

```bash
pytest
ruff check .
black --check .
flake8
mypy .
```

## 산출물 (output/)

| 파일 | 내용 |
|---|---|
| `eda_charts.png` | 핵심 4패널: 연령 분포, 근로시간 박스플롯, 직업별 고소득 비율, 수치형 상관 |
| `eda_numeric_all.png` | 수치형 6개 변수의 소득 그룹별 분포 |
| `eda_categorical_all.png` | 범주형 8개 변수의 고소득 비율 |
| `income_by_education.html` | 학력별 고소득 비율 (Plotly, 브라우저에서 열기) |
| `income_pipeline.pkl` | 전처리와 모델을 묶은 Pipeline (joblib) |
| `report.md` | 분석 결과 리포트 |

차트의 한글 라벨은 `Helvetica Neue`, `AppleGothic`, `Malgun Gothic`, `Noto Sans CJK KR`,
`DejaVu Sans` 순으로 설치된 폰트를 찾는다. 목록에 있는 한글 폰트가 하나도 없으면 라벨이 네모로
표시된다.

## 분석 요약

전체 수치와 표는 `output/report.md`에 있다.

### 성별과 소득

성별과 소득은 독립이 아니다. 카이제곱 검정 결과 chi2=1414.873, 자유도 1, p=1.232e-309,
Cramer's V=0.217이고 고소득 비율은 Female 11.4%, Male 31.4%다. 최소 기대빈도가 2,434라
Yates 연속성 보정은 쓰지 않았다.

다만 통제 방식에 따라 성별 계수의 크기가 크게 달라진다.

| 기준 | `sex_Female` 오즈비 (기준 Male) |
|---|---|
| 통제 없는 교차표 | Female 11.4% vs Male 31.4% |
| 전체 변수 통제 | 0.43 (95% 근사 구간 [0.37, 0.50]) |
| `relationship` 제외 | 0.83 (95% 근사 구간 [0.75, 0.93]) |

`relationship`은 Husband가 남성 100.0%, Wife가 여성 99.9%로 `sex`와 거의 겹친다. 세 값은 서로
다른 질문에 답하므로 하나만 뽑아 성별 격차라고 부를 수 없다.

### 근로시간

소득 >50K 그룹의 주당 근로시간이 6.36시간 길다. 95% 신뢰구간은 [6.07, 6.64], Cohen's d는
0.545(중간)다.

### 모델

학습 30,139건, 평가 15,055건(정제 후)으로 학습한 LogisticRegression Pipeline이다.

| 지표 | 값 |
|---|---|
| 정확도 | 0.8476 |
| 정밀도 | 0.7288 |
| 재현율 | 0.6049 |
| F1 | 0.6611 |
| ROC-AUC | 0.9027 |
| PR-AUC | 0.7651 |

기본 임계값 0.5에서는 실제 고소득 3,700명 중 1,462명을 저소득으로 놓친다. 학습셋 5-폴드
교차검증의 out-of-fold 확률에서 F1을 최대화하는 임계값 0.304를 쓰면 재현율이 0.6049에서
0.7786으로, 놓친 고소득이 819명으로 바뀐다. 어느 쪽이 맞는지는 두 오류의 비용 비교에서 정해진다.

### 계수 해석

회귀 피처는 sex와 race를 포함한 12개 변수이고, 범주마다 기준 범주를 하나 빼 원핫 인코딩 후
80개다. fnlwgt(표본 가중치)와 education(education-num과 중복)만 제외했다.

- 기준 범주는 각 변수에서 학습 표본이 가장 많은 범주로 정한다. 표본이 극히 적은 범주가 기준이
  되면 나머지 모든 대비가 불안정해진다.
- 범주형 계수는 기준 범주 대비 조건부 오즈비이고, 수치형 계수는 1 표준편차 증가 기준이다.
- 계수 표에 95% 근사 구간과 범주별 학습 표본 수를 함께 싣는다. 예를 들어
  `native-country_Cambodia`는 오즈비 2.46이지만 구간이 [0.88, 6.87]이라 방향조차 단정할 수 없다.

### 결측 처리

평가셋을 15,055행으로 고정하고 학습셋만 바꿔 두 방식을 비교했다. dropna의 F1이 0.6611,
범주형 결측을 Unknown으로 보존한 쪽이 0.6603으로 격차가 0.0008이라 우열을 가릴 수 없다.

다만 결측 행 삭제는 소득 그룹에 균일하게 적용되지 않는다. 제거율이 `<=50K` 8.36%,
`>50K` 4.25%로 두 배 차이라 표본 구성 자체가 바뀐다.

## 결과를 읽을 때의 한계

아래 항목은 `output/report.md` 1-4절에서 실행할 때마다 계산해 싣는다.

- 평가셋에도 결측·중복 제거를 적용했으므로 공식 평가셋 지표와 직접 비교할 수 없다.
- train과 test에 완전히 같은 행이 19건 있고, 제거하지 않았다.
- `capital-gain`은 99,999로 상한 처리된 값이다. 학습 데이터의 91.6%가 0이고 상한값은 148행이며
  그중 100%가 고소득이다. 큰 계수를 금액 효과로 읽으면 안 된다.
- 학습에 없던 범주는 전부 0으로 인코딩되어 기준 범주와 구분되지 않는다. 이번 실행에서 해당
  평가 행은 0건이다.
- 계수 구간은 L2 정규화를 사전분포로 본 라플라스 근사값이지 정확한 Wald 신뢰구간이 아니고,
  다중비교를 보정하지 않았다.
- sex와 race는 1994년 당시 행정 분류로 기록된 값이다. 결과는 이 표본 안의 조건부 연관이며
  인과관계나 개인 평가의 근거가 아니다.

재검증 기록(커밋 SHA, 입력 SHA-256, 실행 환경, 지표)은 [`docs/verification.md`](docs/verification.md)에 있다.
