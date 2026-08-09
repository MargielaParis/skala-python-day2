# 재검증 기록

Issue #5의 완료 조건에 따라, 수정 후 상태를 다시 실행해 확인한 기록이다.

## 1. 대상

| 항목 | 값 |
|---|---|
| 검증 대상 커밋 | `74f88282fc9446061de78af8564c69a241ba8c88` |
| 기준(수정 전) 커밋 | `829deb880af7d5b7fd3c49371fc100a36a42c4b0` |
| 브랜치 | `fix/issues-5-6-7` |
| 검증 일자 | 2026-08-09 |

## 2. 실행 환경

Python **3.12 이상**을 요구한다. 아래 두 버전에서 각각 전체를 실행했고 지표가 완전히 같았다.

| 항목 | 값 |
|---|---|
| OS | Windows 11 Pro 10.0.26200 (x64) |
| Python | 3.12.4 / 3.14.6 |

| 패키지 | 버전 (두 환경 동일) |
|---|---|
| pandas | 3.0.5 |
| polars | 1.43.2 |
| numpy | 2.5.1 |
| scikit-learn | 1.9.0 |
| scipy | 1.18.0 |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| plotly | 6.9.0 |
| joblib | 1.5.3 |
| tabulate | 0.10.0 |
| pytest | 9.1.1 |
| black | 26.5.1 |
| ruff | 0.16.2 |
| flake8 | 7.3.0 |
| mypy | 2.3.0 (pandas-stubs, types-tabulate 포함) |
| pre-commit | 4.6+ |

`pandas-stubs`·`types-tabulate`는 pre-commit의 mypy 훅이 `additional_dependencies`로 설치하는
스텁이다. 로컬 mypy에 없으면 훅보다 느슨하게 검사되므로 `requirements-dev.txt`에도 넣었다.

## 3. 입력 데이터

README의 `curl -fL` 명령으로 내려받은 UCI 원본이다.

| 파일 | 줄 수 | SHA-256 |
|---|---:|---|
| `data/raw/adult.data` | 32,561 | `5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d` |
| `data/raw/adult.test` | 16,282 | `a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05` |

`adult.test`의 첫 줄은 주석(`|1x3 Cross validator`)이라 데이터 행은 16,281개다.

## 4. 실행한 명령과 결과

두 Python 버전 모두 아래와 같다.

| 명령 | 종료 코드 | 결과 |
|---|---:|---|
| `pytest` | 0 | 61 passed |
| `ruff check .` | 0 | All checks passed |
| `black --check .` | 0 | 13 files unchanged |
| `flake8` | 0 | 출력 없음 |
| `mypy .` | 0 | no issues found in 13 source files |
| `pre-commit run --all-files` | 0 | 훅 12개 전부 Passed |
| `python main.py` | 0 | 산출물 6개 생성 |

`pre-commit`은 가상환경을 활성화한(또는 venv의 `Scripts`/`bin`을 PATH 앞에 둔) 셸에서 실행해야 한다.
`pytest` 훅이 `language: system`이라 PATH의 `pytest`를 그대로 쓰기 때문이다.
활성화하지 않으면 `Executable 'pytest' not found`로 이 훅만 실패한다.

### 이전 기록에서 해소된 항목

직전 검증에서 미검증으로 남겼던 `pre-commit run --all-files`가 이제 통과한다.
원인은 `.pre-commit-config.yaml`의 `default_language_version: python3.11` 고정이었다.
실제 작성 환경이 3.14인데 설정만 3.11을 요구해, 3.11이 없는 장비에서는 훅 환경 생성 자체가
실패했다(`failed to find interpreter for ... python_spec='python3.11'`). `python3`로 바꿔 해소했다.

## 5. 핵심 지표

### 로딩 동등성 (Issue #5-3 / #6-3)

정규화(공백 제거 → 결측 토큰 → 라벨 마침표 제거 → 수치 캐스팅) 후 비교 결과다.

| 대상 | 컬럼 순서 | shape | dtype 불일치 | null mask 불일치 | 셀 값 불일치 |
|---|---|---|---:|---:|---:|
| train | 일치 | 일치 | 0 | 0 | 0 |
| test | 일치 | 일치 | 0 | 0 | 0 |

### 정제 (Issue #7-1)

| 대상 | 원본 | 결측 제거 후 | 중복 제거 후 |
|---|---:|---:|---:|
| train | 32,561 | 30,162 | 30,139 |
| test | 16,281 | 15,060 | 15,055 |

소득 그룹별 결측 행 제거율 — train `<=50K` 8.36% / `>50K` 4.25%,
test `<=50K` 8.64% / `>50K` 3.80%.
고소득 비율은 train 0.2408 → 0.2489, test 0.2362 → 0.2457로 이동했다(중복 제거 전 기준).

### 통계 (Issue #6-1 / #7-3)

| 검정 | 결과 |
|---|---|
| 성별 x 소득 카이제곱 | chi2=1414.873, dof=1, n=30,139, p=1.232e-309, Cramer's V=0.217 |
| 연속성 보정 | 사용 안 함 (최소 기대빈도 2,434) |
| 성별 고소득 비율 | Female 11.4% / Male 31.4% |
| 근로시간 Welch t-test | t=43.170, p<1e-308(float64 한계), 차이 +6.36시간 |
| 효과크기 | 95% CI [6.07, 6.64], Cohen's d=0.545 (중간) |

### 모델 (Issue #7-2)

| 지표 | 값 |
|---|---:|
| 정확도 | 0.8474 |
| 정밀도 | 0.7291 |
| 재현율 | 0.6030 |
| F1 | 0.6601 |
| ROC-AUC | 0.9026 |
| PR-AUC | 0.7650 |

혼동행렬 — TN 10,526 / FP 829 / FN 1,469 / TP 2,231.
실제 고소득 3,700명 중 1,469명을 저소득으로 예측했다.

결측 처리 A/B — dropna: F1 0.6601 / 정확도 0.8474 / PR-AUC 0.7650,
Unknown 보존: F1 0.6572 / 정확도 0.8522 / PR-AUC 0.7619.

### 분류 임계값

기본 0.5는 자의적이다. 학습셋에서 F1을 최대화하는 임계값을 찾아 평가셋에 적용했다.

| 임계값 | 재현율 | 놓친 고소득(FN) |
|---|---:|---:|
| 0.500 (기본) | 0.6030 | 1,469 |
| 0.341 (학습셋 F1 최적) | 0.7497 | 926 |

### 민감도 분석

`relationship`(Husband 100.0% 남성 / Wife 99.9% 여성)은 `sex`와 거의 겹친다.
이 변수를 빼고 다시 학습하면 성별 계수가 크게 달라진다.

| 모형 | `sex_Female` 오즈비 (기준 Male) | 95% 근사 구간 |
|---|---:|---|
| 전체 변수 | 0.43 | [0.37, 0.50] |
| `relationship` 제외 | 0.84 | [0.75, 0.93] |

3-1절의 통제 없는 교차표에서는 Female 11.4% vs Male 31.4%다.
세 값은 서로 다른 질문에 답하므로 하나만 뽑아 "성별 격차"라고 부를 수 없다.

### 계수 해석 (Issue #5-1 / #6-2)

범주마다 기준 범주 하나를 빼고 `capital-gain-capped` 지시변수를 더해 원핫 피처는 **81개**다
(기준 커밋은 87개).
기준은 **각 변수의 학습 표본 최다 범주**다(동수는 사전순).

| 변수 | 기준 범주 | 기준 표본 |
|---|---|---:|
| workclass | Private | 22,264 |
| marital-status | Married-civ-spouse | 14,059 |
| occupation | Prof-specialty | 4,034 |
| relationship | Husband | 12,457 |
| race | White | 25,912 |
| sex | Male | 20,366 |
| native-country | United-States | 27,487 |

사전순 첫 범주(`drop="first"`)를 쓰면 `native-country`의 기준이 학습 18행짜리 Cambodia,
`race`가 286행짜리 Amer-Indian-Eskimo가 되어 나머지 대비가 전부 불안정해진다.

교차 확인: `sex_Female`의 오즈비는 **0.43**(기준 Male, 95% 근사 구간 [0.37, 0.50])이다.
Issue #5가 기준 커밋에서 계산한 조건부 대비 `exp(beta_Female - beta_Male)=0.423`과 대응한다.

`LogisticRegression`은 기본이 L2 정규화라 기준 범주를 바꾸면 예측도 미세하게 달라진다.
사전순 기준일 때 정확도 0.8479, 표본 최다 기준일 때 0.8476이었다
(현재 0.8474는 `capital-gain-capped` 지시변수를 더한 뒤 값이다).

수치형 계수의 1 표준편차 크기 — `age=13.13`, `education-num=2.55`, `capital-gain=7408.99`,
`capital-loss=404.44`, `hours-per-week=11.98`

### 계수의 불확실성

계수마다 표준오차와 오즈비 95% 근사 구간을 산출한다. L2 정규화를 분산 `C`인 가우시안 사전분포로
보고 사후분포를 최빈값 주변에서 정규근사한 값이며(사후 정밀도 `Z'WZ + I/C`, 절편은 비정규화),
정확한 Wald 신뢰구간이 아니다. 다중비교도 보정하지 않았다.

표본 크기가 구간 폭에 그대로 드러난다.

| 항목 | 학습 표본 | 오즈비 | 95% 근사 구간 |
|---|---:|---:|---|
| `education-num` (수치형, 1 SD) | 30,139 | 2.06 | [1.97, 2.17] |
| `sex_Female` | 9,773 | 0.43 | [0.37, 0.50] |
| `native-country_Italy` | 68 | 2.12 | [1.15, 3.93] |
| `native-country_Cambodia` | 18 | 2.36 | [0.84, 6.60] |

`native-country_Cambodia`는 구간이 1을 포함하므로 방향조차 단정할 수 없다.
점추정만 실었을 때 이 값이 `education-num`과 같은 확신으로 읽히던 문제를 없앤다.

### 리포트가 자동 산출하는 데이터 한계

| 항목 | 값 |
|---|---|
| train/test 완전 동일 행 | 19건 (제거하지 않음) |
| 평가셋 중복 제거 | 15,060 → 15,055행 (공식 평가셋과 직접 비교 불가) |
| `capital-gain` 상한값(99,999) | 148행, 그중 고소득 비율 100.0% |
| `capital-gain` 0인 비율 | 91.6% |
| `relationship` 성별 편중 | Husband 100.0% 남성 / Wife 99.9% 여성 |

## 6. 산출물

`python main.py` 실행으로 생성된 파일이다.

| 파일 | 크기(B) | 비고 |
|---|---:|---|
| `output/eda_charts.png` | 194,470 | 차트 코드 변경 없음 — 기존 파일 유지 |
| `output/eda_numeric_all.png` | 136,994 | 차트 코드 변경 없음 — 기존 파일 유지 |
| `output/eda_categorical_all.png` | 476,235 | 차트 코드 변경 없음 — 기존 파일 유지 |
| `output/income_by_education.html` | 약 4.9MB | 재생성 |
| `output/income_pipeline.pkl` | 약 6.9KB | 재생성 (기준 범주 변경 반영) |
| `output/report.md` | 약 15KB | 재생성 |

PNG 3장은 `src/viz.py`의 차트 로직이 바뀌지 않아 내용이 동일하다. 검증 장비에는 코드가 우선 지정한
macOS 폰트(`Helvetica Neue`/`AppleGothic`)가 없어 재생성하면 한글이 네모로 나오므로,
원본(macOS 렌더링)을 그대로 두었다. 폰트 목록에는 `Malgun Gothic`·`Noto Sans CJK KR`·
`DejaVu Sans` 폴백을 추가해 다른 환경에서도 렌더링되게 했다.

`income_by_education.html`과 `report.md`는 실행할 때마다 내용이 달라진다
(Plotly의 div id, 리포트의 생성 일시). 따라서 해시를 재현 기준으로 쓰지 않는다.

## 7. 반영 완료 / 남은 후속 항목

3개 관점 재검토에서 "개선 제안"으로 남겼던 6개 항목은 모두 반영했다.

| 제안 | 반영 내용 |
|---|---|
| 계수의 표준오차·신뢰구간 | `ml.coefficient_stats()` — 라플라스 근사, 리포트 4-5~4-8절 표에 구간 표시 |
| PR-AUC·임계값 조정 | `average_precision_score` 추가, `ml.threshold_analysis()`로 학습셋 F1 최적 임계값 산출 |
| `relationship` 제외 민감도 모델 | `ml.sensitivity_without()` — 리포트 4-8절 |
| `capital-gain` 상한값 분리 | `capital-gain-capped` 지시변수(스케일링 없이 passthrough) |
| 타입 힌트 | `src/`·`main.py` 전 함수에 어노테이션, mypy `disallow_untyped_defs` 활성화 |
| `SystemExit` 대신 예외 | `src/errors.py`의 `PipelineError`, 종료 코드는 `main.py`만 결정 |

타입 힌트를 켜자마자 mypy가 실제 오류 5건(`Literal` 축 인자, 누락된 파라미터 어노테이션,
`Hashable.split`, dict 키 타입, `Series`/`DataFrame` 혼동)을 잡았다.

아직 남은 항목이다.

- train/test 교차 중복 19행 제거 정책 결정 (현재는 공개만 하고 제거하지 않음)
- 계수 구간의 다중비교 보정
- target 라벨 집합 검증과 예상 밖 값 실패 처리
- 결과 범위를 1994년 비가중 표본으로 제한하는 UCI DOI·라이선스 표기
- 전체 pipeline 통합 테스트
- 검증한 패키지 버전을 constraints/lock으로 고정하고 입력 SHA-256을 코드에서 검증
- Git SHA·입력 해시·환경 버전·지표·산출물 해시를 한 manifest로 연결
- 실패 뒤 기존 output을 성공 결과로 오인하지 않도록 success marker·run ID 적용
- joblib 로드 대상 신뢰 범위와 학습 당시 의존성 버전 기록
