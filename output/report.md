# Adult Census Income — End2End 분석 리포트

- 생성 일시: 2026-08-07 16:22
- 작성자: 박기연 (판교 7반)
- 데이터 분할: UCI 공식 분할 사용 (`adult.data` 학습 / `adult.test` 평가, 랜덤 분할 없음)

## 1. 데이터 준비
- 로딩 비교(train): Pandas 0.046초 vs Polars 0.041초
  (32,561행 x 15컬럼, 두 도구 결과 동일)
- 로딩 비교(test): Pandas 0.019초 vs Polars 0.003초
  (16,281행 x 15컬럼, 두 도구 결과 동일)
- 결측 컬럼(train): workclass(1836건), occupation(1843건), native-country(583건)
- 결측 컬럼(test): workclass(963건), occupation(966건), native-country(274건)
- 정제 결과

| 구분 | 원본 | 결측 제거 후 | 중복 제거 후 |
|---|---|---|---|
| train | 32,561 | 30,162 | 30,139 |
| test | 16,281 | 15,060 | 15,055 |

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
- 주당 근로시간 평균: >50K 45.7시간 vs <=50K 39.4시간
- t-test(Welch): t=43.170, p=0.000000 -> **통계적으로 유의미한 차이 있음 (H0 기각)**
- 상관계수 예시: education-num vs hours-per-week = 0.153
- 수치형 기술통계

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
- 구성: StandardScaler + OneHotEncoder -> LogisticRegression (단일 Pipeline)
- 피처: 수치형 5개 + 범주형 7개(sex·race 포함) -> 원핫 인코딩 후 87개
  - 수치형: age, education-num, capital-gain, capital-loss, hours-per-week
  - 범주형: workclass, marital-status, occupation, relationship, race, sex, native-country
  - 제외: fnlwgt, education (fnlwgt는 표본 가중치, education은 education-num과 1:1 중복)
- 학습(adult.data) 30,139건 / 평가(adult.test) 15,055건
- 고소득(>50K) 비율: train 0.249 / test 0.246
- **정확도 0.8477 / F1 0.6610**
- 모델 저장: `output/income_pipeline.pkl` (joblib, 재로딩 검증 완료)

### 4-1. 회귀계수 — 고소득 확률을 올리는 요인 (오즈비 = exp(계수))

|                                   |    계수 |    오즈비 |
|:----------------------------------|------:|-------:|
| capital-gain                      | 2.328 | 10.258 |
| marital-status_Married-AF-spouse  | 1.444 |  4.236 |
| marital-status_Married-civ-spouse | 1.182 |  3.262 |
| relationship_Wife                 | 1.099 |  3.001 |
| native-country_Italy              | 1.051 |  2.859 |
| native-country_Cambodia           | 0.908 |  2.479 |
| occupation_Exec-managerial        | 0.842 |  2.322 |
| education-num                     | 0.716 |  2.047 |

### 4-2. 고소득 확률을 내리는 요인

|                              |     계수 |   오즈비 |
|:-----------------------------|-------:|------:|
| occupation_Priv-house-serv   | -1.431 | 0.239 |
| native-country_Columbia      | -1.26  | 0.284 |
| relationship_Own-child       | -1.165 | 0.312 |
| marital-status_Never-married | -1.146 | 0.318 |
| native-country_South         | -1.088 | 0.337 |
| sex_Female                   | -0.997 | 0.369 |
| occupation_Farming-fishing   | -0.954 | 0.385 |
| native-country_Vietnam       | -0.848 | 0.428 |

### 4-3. sex·race 계수

|                         |     계수 |   오즈비 |
|:------------------------|-------:|------:|
| race_Asian-Pac-Islander |  0.149 | 1.161 |
| race_White              | -0.053 | 0.948 |
| sex_Male                | -0.137 | 0.872 |
| race_Black              | -0.154 | 0.857 |
| race_Other              | -0.467 | 0.627 |
| race_Amer-Indian-Eskimo | -0.61  | 0.544 |
| sex_Female              | -0.997 | 0.369 |

계수는 다른 변수를 통제한 상태의 부분효과이므로, 위 3절 집계 비율과 부호가 다를 수 있다.
sex_Female은 같은 학력·직업·근로시간이어도 고소득 오즈가 낮게 추정된다는 뜻이며,
데이터(1994 US Census)에 기록된 격차이지 인과관계의 증거는 아니다.

## 5. 결론
- 학력 수준이 높을수록 고소득 비율이 뚜렷하게 증가한다.
- 결혼 상태(Married-civ-spouse)와 capital-gain이 계수 크기 기준 가장 강한 신호다.
- sex·race를 포함해 전체 12개 변수로 회귀했고, 다른 변수를 통제해도 sex_Female의
  고소득 오즈가 가장 크게 낮았다 (오즈비 0.37).
- 고소득 그룹의 주당 근로시간이 통계적으로 유의미하게 길다.
- 학습·평가를 서로 다른 파일로 완전히 분리해, 랜덤 분할보다 데이터 누수 위험이 낮고
  다른 참가자와 지표를 그대로 비교할 수 있는 평가 기준을 확보했다.
- 전처리~모델을 Pipeline 하나로 묶어 재현 가능한 학습·배포 단위를 확보했다.
