# Adult Census Income — End2End 분석 리포트

- 생성 일시: 2026-08-09 16:46
- 작성자: 박기연 (판교 7반)
- 데이터 분할: UCI 공식 분할 사용 (`adult.data` 학습 / `adult.test` 평가, 랜덤 분할 없음)
- 분석 범위: 1994년 US Census에서 추출한 비가중 표본. 아래 결과는 이 표본 안의 조건부 연관이며
  모집단 인과효과나 개인 평가의 근거가 아니다.

## 1. 데이터 준비

### 1-1. Pandas·Polars 로딩 비교
- 소요 시간(단일 참고 측정, 벤치마크 아님)
  - train: Pandas 0.030초 vs Polars 0.015초
    (32,561행 x 15컬럼)
  - test: Pandas 0.016초 vs Polars 0.011초
    (16,281행 x 15컬럼)
- 동등성 검증(train): 정규화 후 컬럼 순서·shape·null mask·의미상 dtype·셀 값 불일치 0
- 동등성 검증(test): 정규화 후 컬럼 순서·shape·null mask·의미상 dtype·셀 값 불일치 0

Polars에는 Pandas의 `skipinitialspace`에 해당하는 옵션이 없어, 그대로 읽으면 구분자 뒤 공백이 남고
(`" State-gov"` vs `"State-gov"`) 수치 컬럼까지 문자열로 추론된다. `src/load.py`는 Polars를 전 컬럼
문자열로 읽은 뒤 **공백 제거 -> `?` 결측 처리 -> 라벨 마침표 제거 -> 수치형 캐스팅**이라는 같은 계약을
적용하고, 그 뒤에 컬럼 순서·shape·null mask·의미상 dtype·셀 값을 실제로 비교한다.

### 1-2. 결측·중복 처리
- 결측 컬럼(train): workclass(1836건), occupation(1843건), native-country(583건)
- 결측 컬럼(test): workclass(963건), occupation(966건), native-country(274건)
- 적용 전략: 결측 행 삭제(dropna)

| 구분 | 원본 | 결측 처리 후 | 중복 제거 후 |
|---|---:|---:|---:|
| train | 32,561 | 30,162 | 30,139 |
| test | 16,281 | 15,060 | 15,055 |

### 1-3. 결측 행 삭제가 소득 그룹에 준 영향
결측 행 삭제는 소득 그룹에 균일하게 적용되지 않는다. 그룹별 제거율은 아래와 같다.

| 구분 | 소득 그룹 | 원본 행 | 결측 행 | 제거율 |
|---|---|---:|---:|---:|
| train | <=50K | 24,720 | 2,066 | 8.36% |
| train | >50K | 7,841 | 333 | 4.25% |
| test | <=50K | 12,435 | 1,075 | 8.64% |
| test | >50K | 3,846 | 146 | 3.80% |

- 고소득(>50K) 비율 변화: train 0.2408 -> 0.2489 /
  test 0.2362 -> 0.2457

- `adult.test`는 첫 줄이 주석(`|1x3 Cross validator`), 라벨에 마침표(`>50K.`)가 붙어 있어
  로딩 단계에서 주석 제외·라벨 표기 통일 처리를 했다.

## 2. 시각화 (train 기준)
- `output/eda_charts.png` — 핵심 4패널: 연령 분포, 근로시간 박스플롯, 직업별 고소득 비율, 수치형 상관
- `output/eda_numeric_all.png` — 수치형 6개 변수 전체의 소득 그룹별 분포
  (capital-gain/loss는 0이 대부분이어서 0 제외 + 로그 축)
- `output/eda_categorical_all.png` — 범주형 8개 변수 전체의 고소득 비율
  (native-country는 범주가 41개라 표본 상위 14개 + 기타로 묶음)
- `output/income_by_education.html` — 학력별 고소득 비율 (Plotly, 마우스오버로 표본 수 확인)

## 3. 통계 분석 (train 기준)

### 3-1. 성별 x 소득 — 카이제곱 독립성 검정
실습 주제가 성별과 소득의 관계이므로, 두 범주형 변수의 연관을 직접 검정한다.

| sex    |   <=50K |   >50K |
|:-------|--------:|-------:|
| Female |    8661 |   1112 |
| Male   |   13972 |   6394 |

- chi2=1413.803, 자유도=1, n=30,139, p=2.104e-309 -> **성별과 소득은 독립이 아니다 (H0 기각)**
- Cramer's V=0.217 (표본 크기와 무관한 연관 강도)
- 성별 고소득 비율: Female 11.4%, Male 31.4%

### 3-2. 소득 그룹 간 주당 근로시간 — Welch t-test
- 평균: >50K 45.7시간 vs <=50K 39.4시간
  (차이 +6.36시간, 95% CI [6.07, 6.64])
- t=43.170, p=1e-308 미만 (float64 표현 한계) -> **통계적으로 유의미한 차이 있음 (H0 기각)**
- Cohen's d=0.545 (중간) — 표본이 크면 작은 차이도 유의해지므로
  유의성과 함께 효과크기를 본다.

### 3-3. 기술통계·상관
- 상관계수 예시: education-num vs hours-per-week = 0.153

|       |      age |          fnlwgt |   education-num |   capital-gain |   capital-loss |   hours-per-week |
|:------|---------:|----------------:|----------------:|---------------:|---------------:|-----------------:|
| count | 30139    |  30139          |        30139    |       30139    |       30139    |         30139    |
| mean  |    38.44 | 189795          |           10.12 |        1092.84 |          88.44 |            40.93 |
| std   |    13.13 | 105659          |            2.55 |        7409.11 |         404.45 |            11.98 |
| min   |    17    |  13769          |            1    |           0    |           0    |             1    |
| 25%   |    28    | 117628          |            9    |           0    |           0    |            40    |
| 50%   |    37    | 178417          |           10    |           0    |           0    |            40    |
| 75%   |    47    | 237604          |           13    |           0    |           0    |            45    |
| max   |    90    |      1.4847e+06 |           16    |       99999    |        4356    |            99    |

## 4. ML Pipeline (소득 >50K 로지스틱 회귀)
- 구성: StandardScaler + OneHotEncoder(drop="first") -> LogisticRegression (단일 Pipeline)
- 피처: 수치형 5개 + 범주형 7개(sex·race 포함) -> 원핫 인코딩 후 80개
  - 수치형: age, education-num, capital-gain, capital-loss, hours-per-week
  - 범주형: workclass, marital-status, occupation, relationship, race, sex, native-country
  - 제외: fnlwgt, education (fnlwgt는 표본 가중치, education은 education-num과 1:1 중복)
- 학습(adult.data) 30,139건 / 평가(adult.test) 15,055건
- 고소득(>50K) 비율: train 0.249 / test 0.246
- 모델 저장: `output/income_pipeline.pkl` (joblib, 재로딩 검증 완료)

### 4-1. 평가 지표

| 지표 | 값 |
|---|---:|
| 정확도 (accuracy) | 0.8479 |
| 정밀도 (precision) | 0.7299 |
| 재현율 (recall) | 0.6049 |
| F1 | 0.6615 |
| ROC-AUC | 0.9029 |

고소득 표본이 적어 정확도만으로는 실제 고소득자를 얼마나 놓쳤는지 알 수 없다. 혼동행렬로 확인한다.

| 실제 \ 예측 | <=50K | >50K |
|---|---:|---:|
| <=50K | 10,527 | 828 |
| >50K | 1,462 | 2,238 |

- 실제 고소득 3,700명 중
  1,462명을 저소득으로 잘못 예측했다 (재현율 0.6049).
- 실제 저소득 11,355명 중
  828명을 고소득으로 잘못 예측했다.

### 4-2. 결측 처리 방식 A/B 비교
같은 Pipeline으로 결측 처리만 바꿔 학습·평가한 결과다.

| 결측 전략 | 학습 행 수 | 평가 행 수 | 정확도 | 정밀도 | 재현율 | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 결측 행 삭제(dropna) | 30,139 | 15,055 | 0.8479 | 0.7299 | 0.6049 | 0.6615 | 0.9029 |
| 범주형 결측을 Unknown으로 보존 | 32,537 | 16,276 | 0.8525 | 0.7283 | 0.5993 | 0.6575 | 0.9050 |

### 4-3. 계수 해석의 전제
- 범주형: `drop="first"`로 각 변수의 첫 범주를 빼고 원핫했다. 남은 계수는 **그 기준 범주 대비**
  조건부 오즈비 `exp(beta)`다. 기준 범주 — workclass=Federal-gov, marital-status=Divorced, occupation=Adm-clerical, relationship=Husband, race=Amer-Indian-Eskimo, sex=Female, native-country=Cambodia
- 수치형: StandardScaler 적용 후 계수이므로 **1 표준편차 증가 기준**이다.
  1 표준편차 크기 — age=13.13, education-num=2.55, capital-gain=7408.99, capital-loss=404.44, hours-per-week=11.98
- `handle_unknown="ignore"`이므로 학습에 없던 범주는 전부 0으로 인코딩되어 기준 범주와 구분되지 않는다.
- 계수는 다른 변수를 통제한 상태의 부분효과라 3절의 단순 집계 비율과 부호가 다를 수 있다.

### 4-4. 고소득 확률을 올리는 요인 (상위 8개)

|                                   |    계수 |    오즈비 | 비교 기준                      |
|:----------------------------------|------:|-------:|:---------------------------|
| capital-gain                      | 2.339 | 10.374 | 1 표준편차 증가                  |
| marital-status_Married-AF-spouse  | 1.822 |  6.183 | 기준 marital-status=Divorced |
| marital-status_Married-civ-spouse | 1.729 |  5.635 | 기준 marital-status=Divorced |
| relationship_Wife                 | 1.312 |  3.712 | 기준 relationship=Husband    |
| native-country_Italy              | 0.859 |  2.361 | 기준 native-country=Cambodia |
| sex_Male                          | 0.858 |  2.358 | 기준 sex=Female              |
| occupation_Exec-managerial        | 0.799 |  2.224 | 기준 occupation=Adm-clerical |
| race_Asian-Pac-Islander           | 0.735 |  2.086 | 기준 race=Amer-Indian-Eskimo |

### 4-5. 고소득 확률을 내리는 요인 (하위 8개)

|                                   |     계수 |   오즈비 | 비교 기준                      |
|:----------------------------------|-------:|------:|:---------------------------|
| occupation_Priv-house-serv        | -1.556 | 0.211 | 기준 occupation=Adm-clerical |
| native-country_Columbia           | -1.34  | 0.262 | 기준 native-country=Cambodia |
| workclass_Without-pay             | -1.116 | 0.328 | 기준 workclass=Federal-gov   |
| native-country_South              | -1.027 | 0.358 | 기준 native-country=Cambodia |
| relationship_Own-child            | -1.027 | 0.358 | 기준 relationship=Husband    |
| occupation_Farming-fishing        | -0.999 | 0.368 | 기준 occupation=Adm-clerical |
| workclass_Self-emp-not-inc        | -0.95  | 0.387 | 기준 workclass=Federal-gov   |
| native-country_Dominican-Republic | -0.931 | 0.394 | 기준 native-country=Cambodia |

### 4-6. sex·race 계수

|                         |    계수 |   오즈비 | 비교 기준                      |
|:------------------------|------:|------:|:---------------------------|
| sex_Male                | 0.858 | 2.358 | 기준 sex=Female              |
| race_Asian-Pac-Islander | 0.735 | 2.086 | 기준 race=Amer-Indian-Eskimo |
| race_White              | 0.544 | 1.722 | 기준 race=Amer-Indian-Eskimo |
| race_Black              | 0.431 | 1.539 | 기준 race=Amer-Indian-Eskimo |
| race_Other              | 0.086 | 1.09  | 기준 race=Amer-Indian-Eskimo |

sex·race 계수는 1994년 표본에 기록된 조건부 연관이다. 인과관계의 증거가 아니며 개인 평가의 근거로
사용할 수 없다. 관측되지 않은 교란 변수와 표본 선택의 영향을 통제하지 않았다.

## 5. 결론 (모두 위 결과에서 계산)

- 성별과 소득은 카이제곱 검정에서 독립 가설을 기각했다 (chi2=1413.8, dof=1, p=2.104e-309, Cramer's V=0.217). 성별 고소득 비율: Female 11.4%, Male 31.4%.
- 고소득 그룹의 주당 근로시간은 저소득 그룹보다 +6.4시간이고 95% 신뢰구간은 [6.07, 6.64], Cohen's d=0.545(중간)로 통계적으로 유의미한 차이가 있다.
- 계수 절대값이 가장 큰 항은 `capital-gain`이며 계수 +2.339, 오즈비 10.374다. 범주형은 기준 범주 대비, 수치형은 1 표준편차 증가 기준이다.
- 다른 변수를 통제했을 때 Male는 기준(Female) 대비 오즈 2.358배로 추정됐다.
- 평가 성능은 정확도 0.8479, 정밀도 0.7299, 재현율 0.6049, F1 0.6615, ROC-AUC 0.9029다. 실제 고소득 3,700명 중 1,462명을 저소득으로 놓쳤다.
- 결측 처리 방식은 F1 기준 '결측 행 삭제(dropna)'가 0.6615로 가장 높았다.

### 5-1. 분석자 해석 (자동 계산 아님)
- 학습·평가를 서로 다른 파일로 완전히 분리해, 랜덤 분할보다 데이터 누수 위험이 낮고
  다른 참가자와 지표를 그대로 비교할 수 있는 평가 기준을 확보했다.
- 전처리~모델을 Pipeline 하나로 묶어 재현 가능한 학습·배포 단위를 확보했다.
- 결측 행 삭제는 소득 그룹별 제거율이 달라 표본 구성을 바꾼다. 위 1-3의 제거율 차이를 감안해
  결과를 읽어야 한다.
