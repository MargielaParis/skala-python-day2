# 검증 기록

Issue #5의 완료 조건에 따라 최종 상태를 실행해 확인한 기록이다.

## 1. 대상

| 항목 | 값 |
|---|---|
| 검증 대상 커밋 | `59bcdcf4db91399eefbd11af1d3163141277b67f` |
| 기준(수정 전) 커밋 | `829deb880af7d5b7fd3c49371fc100a36a42c4b0` |
| 브랜치 | `fix/issues-5-6-7` |
| 검증 일자 | 2026-08-09 |

## 2. 실행 환경

Python 3.12 이상이 필요하다. 아래 두 버전에서 각각 전체를 실행했고 지표가 완전히 같았다.

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
| mypy | 2.3.0 |
| pre-commit | 4.6+ |

`pandas-stubs`와 `types-tabulate`는 pre-commit의 mypy 훅이 `additional_dependencies`로 설치하는
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
| `pytest` | 0 | 65 passed |
| `ruff check .` | 0 | All checks passed |
| `black --check .` | 0 | 13 files unchanged |
| `flake8` | 0 | 출력 없음 |
| `mypy .` | 0 | no issues found in 13 source files |
| `pre-commit run --all-files` | 0 | 훅 12개 전부 Passed |
| `python main.py` | 0 | 산출물 6개 생성 |

pre-commit은 가상환경을 활성화한(또는 venv의 `Scripts`/`bin`을 PATH 앞에 둔) 셸에서 실행해야 한다.
`pytest` 훅이 `language: system`이라 PATH의 `pytest`를 그대로 쓰기 때문이다. 활성화하지 않으면
`Executable 'pytest' not found`로 이 훅만 실패한다.

## 5. 핵심 지표

### 로딩 동등성 (Issue #5-3 / #6-3)

정규화(공백 제거, 결측 토큰, 라벨 마침표 제거, 수치 캐스팅) 후 비교 결과다.

| 대상 | 컬럼 순서 | shape | dtype 불일치 | null mask 불일치 | 셀 값 불일치 |
|---|---|---|---:|---:|---:|
| train | 일치 | 일치 | 0 | 0 | 0 |
| test | 일치 | 일치 | 0 | 0 | 0 |

### 정제 (Issue #7-1)

| 대상 | 원본 | 결측 제거 후 | 중복 제거 후 |
|---|---:|---:|---:|
| train | 32,561 | 30,162 | 30,139 |
| test | 16,281 | 15,060 | 15,055 |

소득 그룹별 결측 행 제거율은 train이 `<=50K` 8.36% / `>50K` 4.25%, test가 `<=50K` 8.64% /
`>50K` 3.80%다. 고소득 비율은 train 0.2408에서 0.2489로, test 0.2362에서 0.2457로 이동했다
(중복 제거 전 기준).

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
| 정확도 | 0.8476 |
| 정밀도 | 0.7288 |
| 재현율 | 0.6049 |
| F1 | 0.6611 |
| ROC-AUC | 0.9027 |
| PR-AUC | 0.7651 |

혼동행렬은 TN 10,522 / FP 833 / FN 1,462 / TP 2,238이다.
실제 고소득 3,700명 중 1,462명을 저소득으로 예측했다.

### 결측 처리 A/B

전략마다 학습셋만 바꾸고 평가셋은 15,055행으로 고정한다. 전략별 평가셋을 따로 쓰면 행 수와 사례
구성이 달라져 비교가 성립하지 않는다.

| 전략 | 학습 행 | 평가 행 | 정확도 | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| dropna | 30,139 | 15,055 | 0.8476 | 0.6611 | 0.7651 |
| Unknown 보존 | 32,537 | 15,055 | 0.8472 | 0.6603 | 0.7651 |

F1 격차가 0.0008이라 우열을 가릴 수 없다. 리포트도 격차가 0.01 미만이면 순위를 말하지 않는다.

### 분류 임계값

기본 0.5는 자의적이다. 평가셋에서 고르면 과적합이고 학습셋 in-sample 예측으로 고르면 낙관적이라,
학습셋 5-폴드 교차검증 out-of-fold 확률에서 F1 최적값을 찾아 평가셋에 적용한다. 참고로 in-sample
예측으로 고르면 0.341이 나오고 평가셋 재현율은 0.7497(FN 926)이었다.

| 임계값 | 재현율 | 놓친 고소득(FN) | 잘못 잡은 저소득(FP) |
|---|---:|---:|---:|
| 0.500 (기본) | 0.6049 | 1,462 | 833 |
| 0.304 (교차검증 F1 최적) | 0.7786 | 819 | 1,843 |

### 민감도 분석

`relationship`(Husband 100.0% 남성, Wife 99.9% 여성)은 `sex`와 거의 겹친다. 이 변수를 빼고 다시
학습하면 성별 계수가 크게 달라진다.

| 모형 | `sex_Female` 오즈비 (기준 Male) | 95% 근사 구간 |
|---|---:|---|
| 전체 변수 | 0.43 | [0.37, 0.50] |
| `relationship` 제외 | 0.83 | [0.75, 0.93] |

통제 없는 교차표에서는 Female 11.4% vs Male 31.4%다. 세 값은 서로 다른 질문에 답하므로 하나만
뽑아 성별 격차라고 부를 수 없다.

### 계수 해석 (Issue #5-1 / #6-2)

범주마다 기준 범주 하나를 빼 원핫 피처는 80개다(기준 커밋은 87개). 기준은 각 변수의 학습 표본
최다 범주다(동수는 사전순). 사전순 첫 범주를 쓰면 `native-country`의 기준이 학습 18행짜리
Cambodia, `race`가 286행짜리 Amer-Indian-Eskimo가 되어 나머지 대비가 전부 불안정해진다.

| 변수 | 기준 범주 | 기준 표본 |
|---|---|---:|
| workclass | Private | 22,264 |
| marital-status | Married-civ-spouse | 14,059 |
| occupation | Prof-specialty | 4,034 |
| relationship | Husband | 12,457 |
| race | White | 25,912 |
| sex | Male | 20,366 |
| native-country | United-States | 27,487 |

교차 확인으로 `sex_Female`의 오즈비는 0.43(기준 Male, 95% 근사 구간 [0.37, 0.50])이고, Issue #5가
기준 커밋에서 계산한 조건부 대비 `exp(beta_Female - beta_Male)=0.423`과 대응한다.

`LogisticRegression`은 기본이 L2 정규화라 기준 범주를 바꾸면 예측도 미세하게 달라진다. 사전순
기준일 때 정확도가 0.8479, 표본 최다 기준일 때 0.8476이었다.

수치형 계수의 1 표준편차 크기는 `age=13.13`, `education-num=2.55`, `capital-gain=7408.99`,
`capital-loss=404.44`, `hours-per-week=11.98`이다.

### 계수의 불확실성

계수마다 표준오차와 오즈비 95% 근사 구간을 산출한다. L2 정규화를 분산 `C`인 가우시안 사전분포로
보고 사후분포를 최빈값 주변에서 정규근사한 값이며(사후 정밀도 `Z'WZ + I/C`, 절편은 비정규화),
정확한 Wald 신뢰구간이 아니다. 다중비교도 보정하지 않았다.

표본 크기가 구간 폭에 그대로 드러난다.

| 항목 | 학습 표본 | 오즈비 | 95% 근사 구간 |
|---|---:|---:|---|
| `education-num` (수치형, 1 SD) | 30,139 | 2.06 | [1.96, 2.16] |
| `sex_Female` | 9,773 | 0.43 | [0.37, 0.50] |
| `native-country_Italy` | 68 | 2.03 | [1.09, 3.77] |
| `native-country_Cambodia` | 18 | 2.46 | [0.88, 6.87] |

`native-country_Cambodia`는 구간이 1을 포함하므로 방향조차 단정할 수 없다.

이 구간은 변수 선택의 진단에도 쓰인다. `capital-gain` 상한 도달 지시변수를 넣었을 때 계수가
`+0.006`, 표준오차가 `1.000`, 구간이 `[0.14, 7.14]`로 나왔다. 표준오차가 사전분포 표준편차와
정확히 같다는 것은 데이터가 그 계수에 아무 정보도 주지 못했다는 뜻이다(`capital-gain`과 상관 0.94,
학습 148행). 따라서 그 변수는 모델에 넣지 않았다.

### 리포트가 매번 산출하는 데이터 한계

| 항목 | 값 |
|---|---|
| train/test 완전 동일 행 | 19건 (제거하지 않음) |
| 평가셋 중복 제거 | 15,060 → 15,055행 (공식 평가셋과 직접 비교 불가) |
| 학습에 없던 범주가 있는 평가 행 | 0건 / 15,055행 |
| `capital-gain` 상한값(99,999) | 148행, 그중 고소득 비율 100.0% |
| `capital-gain` 0인 비율 | 91.6% |
| `relationship` 성별 편중 | Husband 100.0% 남성 / Wife 99.9% 여성 |

## 6. 산출물

`python main.py` 실행으로 생성된 파일이다.

| 파일 | 크기 | 비고 |
|---|---:|---|
| `output/eda_charts.png` | 194,470 B | macOS 렌더링본 유지 |
| `output/eda_numeric_all.png` | 136,994 B | macOS 렌더링본 유지 |
| `output/eda_categorical_all.png` | 476,235 B | macOS 렌더링본 유지 |
| `output/income_by_education.html` | 약 4.9 MB | 재생성 |
| `output/income_pipeline.pkl` | 약 6.9 KB | 재생성 |
| `output/report.md` | 약 15 KB | 재생성 |

PNG 3장은 `src/viz.py`의 차트 로직이 그대로라 내용이 동일하다. 검증 장비에는 코드가 우선 지정한
macOS 폰트(`Helvetica Neue`, `AppleGothic`)가 없어 재생성하면 한글이 네모로 나오므로 원본을 두었다.
폰트 목록에는 `Malgun Gothic`, `Noto Sans CJK KR`, `DejaVu Sans` 폴백을 넣어 다른 환경에서도
렌더링되게 했다.

`income_by_education.html`과 `report.md`는 실행할 때마다 내용이 달라진다(Plotly의 div id,
리포트의 생성 일시). 따라서 해시를 재현 기준으로 쓰지 않는다.

## 7. 남은 후속 항목

- train/test 교차 중복 19행 제거 정책 결정 (현재는 공개만 하고 제거하지 않음)
- 계수 구간의 다중비교 보정
- target 라벨 집합 검증과 예상 밖 값 실패 처리
- 결과 범위를 1994년 비가중 표본으로 제한하는 UCI DOI·라이선스 표기
- 전체 pipeline 통합 테스트
- 검증한 패키지 버전을 constraints/lock으로 고정하고 입력 SHA-256을 코드에서 검증
- Git SHA, 입력 해시, 환경 버전, 지표, 산출물 해시를 한 manifest로 연결
- 실패 뒤 기존 output을 성공 결과로 오인하지 않도록 success marker나 run ID 적용
- joblib 로드 대상 신뢰 범위와 학습 당시 의존성 버전 기록
