# 재검증 기록

Issue #5의 완료 조건에 따라, 수정 후 상태를 다시 실행해 확인한 기록이다.

## 1. 대상

| 항목 | 값 |
|---|---|
| 검증 대상 커밋 | `2d9d5bf7777e81b405c452603c98a6352504c76b` |
| 기준(수정 전) 커밋 | `829deb880af7d5b7fd3c49371fc100a36a42c4b0` |
| 브랜치 | `fix/issues-5-6-7` |
| 검증 일자 | 2026-08-09 |

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| OS | Windows 11 Pro 10.0.26200 (x64) |
| Python | 3.12.4 |

> **주의**: 프로젝트가 선언한 기준은 Python **3.11**이지만, 검증 장비에 3.11이 없어 **3.12.4**로 실행했다.
> `pyproject.toml`의 `python_version`·`target-version`과 README의 설치 절차는 3.11 그대로 두었다.
> 3.11에서의 재확인은 별도로 필요하다.

| 패키지 | 버전 |
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

## 3. 입력 데이터

README의 `curl -fL` 명령으로 내려받은 UCI 원본이다.

| 파일 | 줄 수 | SHA-256 |
|---|---:|---|
| `data/raw/adult.data` | 32,561 | `5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d` |
| `data/raw/adult.test` | 16,282 | `a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05` |

`adult.test`의 첫 줄은 주석(`|1x3 Cross validator`)이라 데이터 행은 16,281개다.

## 4. 실행한 명령과 결과

| 명령 | 종료 코드 | 결과 |
|---|---:|---|
| `pytest` | 0 | 40 passed |
| `ruff check .` | 0 | All checks passed |
| `black --check .` | 0 | 12 files unchanged |
| `flake8` | 0 | 출력 없음 |
| `mypy .` | 0 | no issues found in 12 source files |
| `python main.py` | 0 | 산출물 6개 생성 |

### 미검증 항목

- **`pre-commit run --all-files`** — `.pre-commit-config.yaml`의 `default_language_version: python3.11`
  때문에 훅 환경 생성 단계에서 실패한다
  (`RuntimeError: failed to find interpreter for Builtin discover of python_spec='python3.11'`, 종료 코드 3).
  검증 장비에 Python 3.11이 없어서 생긴 문제이므로 설정은 바꾸지 않았다.
  위 5개 도구를 직접 실행한 결과로 대신했다. Python 3.11 환경에서의 확인이 별도로 필요하다.
- **Python 3.11에서의 clean clone 전체 재현** — 위와 같은 이유로 3.12에서만 확인했다.

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
고소득 비율은 train 0.2408 → 0.2489, test 0.2362 → 0.2457로 이동했다.

### 통계 (Issue #6-1 / #7-3)

| 검정 | 결과 |
|---|---|
| 성별 x 소득 카이제곱 | chi2=1413.803, dof=1, n=30,139, p=2.104e-309, Cramer's V=0.217 |
| 성별 고소득 비율 | Female 11.4% / Male 31.4% |
| 근로시간 Welch t-test | t=43.170, p<1e-308(float64 한계), 차이 +6.36시간 |
| 효과크기 | 95% CI [6.07, 6.64], Cohen's d=0.545 (중간) |

### 모델 (Issue #7-2)

| 지표 | 값 |
|---|---:|
| 정확도 | 0.8479 |
| 정밀도 | 0.7299 |
| 재현율 | 0.6049 |
| F1 | 0.6615 |
| ROC-AUC | 0.9029 |

혼동행렬 — TN 10,527 / FP 828 / FN 1,462 / TP 2,238.
실제 고소득 3,700명 중 1,462명을 저소득으로 예측했다.

결측 처리 A/B — dropna: F1 0.6615 / 정확도 0.8479, Unknown 보존: F1 0.6575 / 정확도 0.8525.

### 계수 해석 (Issue #5-1 / #6-2)

`OneHotEncoder(drop="first")` 적용으로 원핫 피처는 87개에서 **80개**가 됐고,
각 범주형 변수의 정렬 첫 범주가 기준이 된다.

기준 범주 — `workclass=Federal-gov`, `marital-status=Divorced`, `occupation=Adm-clerical`,
`relationship=Husband`, `race=Amer-Indian-Eskimo`, `sex=Female`, `native-country=Cambodia`

교차 확인: `sex_Male`의 오즈비는 **2.358**이고 역수는 **0.424**다.
Issue #5가 기준 커밋에서 계산한 조건부 대비 `exp(beta_Female - beta_Male)=0.423`과 일치한다.

수치형 계수의 1 표준편차 크기 — `age=13.13`, `education-num=2.55`, `capital-gain=7408.99`,
`capital-loss=404.44`, `hours-per-week=11.98`

## 6. 산출물

`python main.py` 실행으로 생성된 파일이다.

| 파일 | 크기(B) | 비고 |
|---|---:|---|
| `output/eda_charts.png` | 194,470 | 차트 코드 변경 없음 — 기존 파일 유지 |
| `output/eda_numeric_all.png` | 136,994 | 차트 코드 변경 없음 — 기존 파일 유지 |
| `output/eda_categorical_all.png` | 476,235 | 차트 코드 변경 없음 — 기존 파일 유지 |
| `output/income_by_education.html` | 4,864,656 | 재생성 |
| `output/income_pipeline.pkl` | 6,865 | 재생성 (drop="first" 반영) |
| `output/report.md` | 12,331 | 재생성 |

PNG 3장은 `src/viz.py`의 차트 로직이 바뀌지 않아 내용이 동일하다. 검증 장비에는 코드가 우선 지정한
macOS 폰트(`Helvetica Neue`/`AppleGothic`)가 없어 재생성하면 한글이 네모로 나오므로,
원본(macOS 렌더링)을 그대로 두었다. 폰트 목록에는 `Malgun Gothic`·`Noto Sans CJK KR`·
`DejaVu Sans` 폴백을 추가해 다른 환경에서도 렌더링되게 했다.

`income_by_education.html`과 `report.md`는 실행할 때마다 내용이 달라진다
(Plotly의 div id, 리포트의 생성 일시). 따라서 해시를 재현 기준으로 쓰지 않는다.

## 7. 남은 후속 항목

Issue #5가 별도 Issue 후보로 분리한 항목 중 이번 PR에서 다루지 않은 것들이다.

- target 라벨 집합 검증과 예상 밖 값 실패 처리
- 공식 train/test의 교차 동일 행과 중복 제거 정책 명시
- 결과 범위를 1994년 비가중 표본으로 제한하는 UCI DOI·라이선스 표기
- PR-AUC 추가, 전체 pipeline 통합 테스트
- 검증한 패키지 버전을 constraints/lock으로 고정하고 입력 SHA-256을 코드에서 검증
- Git SHA·입력 해시·환경 버전·지표·산출물 해시를 한 manifest로 연결
- 실패 뒤 기존 output을 성공 결과로 오인하지 않도록 success marker·run ID 적용
- joblib 로드 대상 신뢰 범위와 학습 당시 의존성 버전 기록
