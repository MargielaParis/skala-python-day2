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
│   ├── load.py             # Pandas·Polars 로딩·동등성 검증, 결측·중복 처리
│   ├── viz.py              # Seaborn 정적 / Plotly 인터랙티브 차트
│   ├── stats_test.py       # 기술통계·상관·t-test·카이제곱
│   ├── ml.py               # sklearn Pipeline 학습·평가·저장
│   └── report.py           # report.md 자동 생성
├── tests/                  # pytest 테스트 (원본 데이터 없이 실행 가능)
├── docs/verification.md    # 재검증 기록 (커밋 SHA·입력 해시·지표)
├── main.py                 # 전체 파이프라인 실행 진입점
├── output/                 # 차트·모델·리포트 산출물
├── .pre-commit-config.yaml # 커밋 전 자동 검사 설정
├── pyproject.toml          # black·ruff·mypy·pytest 설정
├── .flake8                 # flake8 설정
├── requirements.txt
└── requirements-dev.txt    # 검사 도구
```

## 개발 환경 설정

Python **3.11**을 기준으로 한다. 가상환경을 만들고 활성화한 뒤 의존성을 설치한다.

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
pre-commit install          # 커밋 전 자동 검사 활성화
```

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

이후의 모든 명령은 **가상환경이 활성화된 셸**에서 실행한다.
`.pre-commit-config.yaml`의 `pytest` 훅은 `language: system`이라 별도 환경을 만들지 않고
현재 PATH의 `pytest`를 그대로 쓴다. 가상환경을 활성화하지 않으면 시스템 인터프리터의
`pytest`가 실행되어 의존성을 찾지 못한다.

## 데이터 준비

데이터는 git에 올리지 않는다. 아래 명령으로 원본을 내려받는다.
`data/raw` 디렉터리를 **먼저 만들어야** curl이 실패하지 않는다.

**macOS / Linux**

```bash
mkdir -p data/raw
curl -fL -o data/raw/adult.data \
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -fL -o data/raw/adult.test \
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

**Windows (PowerShell)**

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
curl.exe -fL -o data\raw\adult.data `
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl.exe -fL -o data\raw\adult.test `
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

`adult.test`는 첫 줄이 주석(`|1x3 Cross validator`)이고 라벨에 마침표(`>50K.`)가 붙어 있어,
`src/load.py`에서 주석 제외·라벨 표기 통일을 처리한다.

## 실행 방법

```bash
python main.py
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
pre-commit run --all-files   # 반드시 main이 아닌 브랜치에서 실행
pytest                       # 테스트만 실행
```

> **브랜치 조건**: `no-commit-to-branch` 훅은 `main`에서 **의도적으로 실패**한다.
> `main`에 체크아웃한 상태로 `pre-commit run --all-files`를 돌리면 이 훅만 실패하는 것이 정상이다.
> 전체 훅 통과를 확인하려면 작업 브랜치(`git switch -c <branch>`)에서 실행한다.

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
| `eda_charts.png` | 핵심 4패널 — 연령 분포, 근로시간 박스플롯, 직업별 고소득 비율, 수치형 상관 (Seaborn) |
| `eda_numeric_all.png` | 수치형 6개 변수 전체의 소득 그룹별 분포 |
| `eda_categorical_all.png` | 범주형 8개 변수 전체의 고소득 비율 |
| `income_by_education.html` | 학력별 고소득 비율 (Plotly, 브라우저에서 열기) |
| `income_pipeline.pkl` | 전처리+모델 Pipeline (joblib) |
| `report.md` | 분석 결과 자동 생성 리포트 (발표용, 회귀계수·오즈비 포함) |

차트의 한글 라벨은 설치된 폰트를 순서대로 찾는다
(`Helvetica Neue` → `AppleGothic` → `Malgun Gothic` → `Noto Sans CJK KR` → `DejaVu Sans`).
목록에 있는 한글 폰트가 하나도 없으면 라벨이 네모로 표시된다.

## 분석 요약

- 성별과 소득은 독립이 아니다 — 카이제곱 chi2=1413.8, dof=1, p=2.1e-309, Cramer's V=0.217
  (고소득 비율 Female 11.4% vs Male 31.4%)
- 소득 >50K 그룹의 주당 근로시간이 6.36시간 길다 (95% CI [6.07, 6.64], Cohen's d=0.545, 효과크기 중간)
- 학력 수준이 높을수록 고소득 비율 증가
- LogisticRegression Pipeline: 정확도 0.8479 / 정밀도 0.7299 / 재현율 0.6049 / F1 0.6615 / ROC-AUC 0.9029
  (학습 adult.data 30,139건 / 평가 adult.test 15,055건, 정제 후 기준)
  - 실제 고소득 3,700명 중 1,462명을 저소득으로 놓쳤다 (혼동행렬은 `output/report.md` 4-1절)
- 회귀 피처는 sex·race를 포함한 12개 변수. `OneHotEncoder(drop="first")`로 범주마다 기준을 하나 빼
  원핫 후 80개다. fnlwgt(표본 가중치)와 education(education-num과 중복)만 제외했다.
  - **모든 범주형 계수는 기준 범주 대비 조건부 오즈비**이고, 수치형 계수는 **1 표준편차 증가 기준**이다.
    기준 범주 목록과 표준편차 크기는 `output/report.md` 4-3절 참고.
- 결측 처리 방식 A/B: dropna(F1 0.6615) vs 범주형 Unknown 보존(F1 0.6575). 결측 행 삭제는
  소득 그룹별 제거율이 달라(`<=50K` 8.36% vs `>50K` 4.25%) 표본 구성을 바꾼다.

재검증 기록(커밋 SHA·입력 SHA-256·환경·지표)은 [`docs/verification.md`](docs/verification.md) 참고.
