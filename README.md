# Capital Rotation Tracker & Strategy Backtester

글로벌 자산군 간 **자금 순환(Capital Rotation)** 을 분석하고, 로테이션 기반 포트폴리오
전략을 **백테스팅·비교**하는 Python CLI 프로젝트입니다.

> 📚 **교육·연구 목적** 입니다. 실제 투자 판단에 사용하지 마세요. 자세한 주의사항은
> [맨 아래](#주의사항)를 참고하세요.

---

## 핵심 특징

- **26개 자산** (미국/한국/신흥/선진국 주식, 크립토, 채권, 원자재, 부동산, FX, VIX)을
  `yfinance`로 수집 — ETF/지수 기반으로 자산군 전체 흐름을 추적
- **6개 분석 엔진**: 모멘텀 · 거래량 · 변동성 · 상관관계(+PCA) · 시장 레짐(Risk-On/Off) · 자금 순환
- **4개 전략 + 3개 벤치마크**를 **4가지 리밸런싱 주기**(주간/월간/분기/신호)로 백테스트
- **Look-ahead bias 구조적 방지** — 모든 지표·시그널은 해당 시점까지의 데이터만 사용
  (전 모듈이 *절단 검사*로 자동 검증됨, `tests/check_module.py`)
- **전 성과 지표 계산**: CAGR, Sharpe, Sortino, MDD(+회복기간), Calmar, 승률, 손익비,
  Alpha/Beta, 회전율, 월별/연도별 수익률
- **설정 중심(config-driven)** — 자산·전략·수수료·기간을 YAML로 관리, 코드 수정 없이 실험
- rich 기반 CLI 테이블 + matplotlib 차트 + CSV/JSON 내보내기

---

## 설치

```bash
pip install -r requirements.txt    # Python >= 3.10
```

주요 의존성: `yfinance, pandas, numpy, scipy, scikit-learn, matplotlib, plotly, rich, click, pyyaml, pyarrow`

---

## 빠른 시작

```bash
# 1) 데이터 수집 (증분 업데이트 — 처음엔 전체 히스토리, 이후 새 날짜만)
python cli.py data update
python cli.py data status                  # 자산별 수집 현황

# 2) 현재 시장 분석 (최신 시점 기준)
python cli.py analyze regime               # Risk-On / Risk-Off 판단
python cli.py analyze momentum --top 12    # 모멘텀 스코어 순위
python cli.py analyze rotation             # 카테고리별 자금 유입/유출
python cli.py analyze correlation          # 평균 상관(분산투자 관점) + PCA

# 3) 백테스트
python cli.py backtest run                 # 전 전략 × 전 주기 비교
python cli.py backtest run --strategy topn --rebalance monthly
python cli.py backtest compare --sort cagr # 비교표 + 전략별 최적 주기 + 다중검정 보정
python cli.py backtest oos                  # 검증창에서 선택→테스트창 확인 (과적합 점검)
python cli.py backtest export --format both # output/ 에 CSV+JSON 저장
python cli.py backtest report              # output/REPORT.md (표+차트+보정+분석 통합 보고서)

# 4) 차트 (output/*.png)
python cli.py chart equity                 # 누적 수익 곡선
python cli.py chart regime                 # 누적수익 + Risk-Off 구간 음영
python cli.py chart drawdown               # 드로다운
python cli.py chart correlation            # 상관관계 히트맵
python cli.py chart rotation               # 자금 순환 막대
```

> Windows 콘솔에서 박스 문자가 깨지면 `set PYTHONIOENCODING=utf-8` 후 실행하세요
> (CLI는 자동으로 UTF-8 출력을 시도합니다).

---

## 전략

| 전략 | 키 | 설명 |
|------|----|------|
| **Top-N 로테이션** | `topn` | 복합 모멘텀 상위 N개 동일비중. 절대모멘텀 음수면 현금 |
| **리스크 패리티** | `risk_parity` | 변동성 역수 비중. `method: erc`로 진짜 위험기여도 균등화(공분산 사용) 가능 |
| **복합 스코어링** | `momentum_score` | 모멘텀·거래량·변동성·상관관계를 z-score로 종합한 상위 N |
| **레짐 기반** | `regime_based` | Risk-On/Off에 따라 자산군 배분 규칙 전환 |
| **최소 분산** | `min_variance` | Ledoit-Wolf 수축 공분산 기반 long-only 최소분산(상관 고려, SLSQP) |
| 60/40 | `bench_6040` | SPY 60% + TLT 40% (벤치마크) |
| 동일비중 | `bench_equal` | 전체 거래가능 자산 1/N (벤치마크) |
| Buy & Hold | `bench_bh` | SPY 100% 보유 (벤치마크) |

리밸런싱 주기: **weekly / monthly / quarterly / signal**(모멘텀·레짐 변화 시,
최소 간격·회전율 임계값 적용). 거래비용은 편도 수수료 0.1% + 슬리피지 0.05% (조정 가능).

---

## 설정 파일 (`config/`)

- **`assets.yaml`** — 자산 목록, 카테고리, `tradable` 플래그, 레짐용 `roles`
- **`strategy.yaml`** — 전략별 파라미터(N, 룩백, 가중치, 필터), 벤치마크 정의
- **`backtest.yaml`** — 초기자본, 기간, 무위험수익률, 거래비용, 리밸런싱 옵션, 데이터 경로

자산 추가/제거는 `assets.yaml` 한 줄로 끝납니다. 각 자산의 가용 기간은 자동 감지됩니다.

---

## 프로젝트 구조

```
capital_rotation/
├── config/            # assets.yaml, strategy.yaml, backtest.yaml
├── src/
│   ├── config.py      # 설정 로더
│   ├── data/          # fetcher(yfinance) · storage(parquet) · preprocess(MarketData)
│   ├── analysis/      # momentum · volume · volatility · correlation · regime · regime_hmm · _hmm · rotation
│   ├── strategy/      # base(+registry) · topn · risk_parity · momentum_score · regime_based · min_variance · regime_budget · benchmark · _riskopt
│   ├── backtest/      # engine · metrics · runner · report · overlay · splits · sharpe_correction
│   └── visualization/ # tables(rich) · charts(matplotlib)
├── tests/             # fixtures(합성데이터) · check_module(수용 테스트 + look-ahead 검사)
├── data/raw/          # 티커별 parquet 저장소
├── output/            # 백테스트 결과 · 차트
└── cli.py             # CLI 엔트리포인트
```

### 데이터 계약 (`MarketData`)

모든 분석/전략은 동일한 영업일 캘린더·티커 정렬을 공유하는 *wide* DataFrame 묶음을 사용합니다:
`prices`(수정종가) · `close` · `open/high/low` · `volume` · `dollar_volume` · `returns`.
**불변식**: 날짜 `t`의 값은 `index <= t` 데이터만 사용 — 이것이 look-ahead 방지의 토대입니다.

---

## Look-ahead Bias 방지

백테스터의 가장 흔한 함정인 미래참조를 **구조적으로** 차단합니다:

1. 모든 분석 함수는 **후행(trailing) 윈도우**만 사용 — 시그널 패널의 행 `t`는 `t` 이하 데이터로만 계산
2. 엔진은 매일 **수익률을 먼저 반영한 뒤** 종가에 리밸런싱 — `t`에 결정된 비중은 `t+1`부터 수익 발생
3. `tests/check_module.py`가 **절단 검사**로 자동 검증: 미래 데이터를 잘라내도 과거 값이 동일해야 통과

```bash
python -m tests.check_module all   # 19개 모듈/기능 전체 수용 테스트 (look-ahead 절단 검사 포함)
```

---

## 검증 엄밀성 (과적합 방어)

여러 전략·주기를 탐색하고 최고 Sharpe만 보고하면 **데이터 스누핑**(승자의 저주)입니다.
이를 보정하는 두 장치를 내장했습니다:

- **다중검정 보정** (`backtest compare` 하단): 
  - *James-Stein 수축* — 모든 시도 Sharpe를 평균으로 수축(순위 강건성 점검). 긴 일간 표본에선 추정이 정밀해 거의 no-op.
  - *Deflated Sharpe Ratio (DSR)* — 탐색한 시도 수 N을 감안한 "진짜 Sharpe>0" 확률. ≥0.9 강건 / ≤0.5 운과 구별 곤란.
- **Out-of-Sample 프로토콜** (`backtest oos`): 시간순 train/valid/test 분할 →
  **검증창에서 최적 설정 선택 → 테스트창에서 확인**. valid·test Sharpe를 나란히 보여
  취약한 설정(valid 우수 → test 붕괴)을 즉시 식별. `splits.walk_forward_windows`로 임바고
  적용 워크포워드도 지원.

> 예: 본 데이터에서 valid 최고였던 60/40은 test Sharpe가 반토막(1.04→0.51)인 반면,
> 모멘텀 전략은 test에서도 강건 — 단일 백테스트가 숨기는 진실을 드러냅니다.

## HMM 레짐 + 레짐 버짓 (선택)

`regime_budget` 전략은 베이스 전략의 **익스포저를 시장 레짐에 따라 조절**합니다
(`final = budget(레짐) × 베이스가중치`, 나머지 현금). 레짐 신호는 두 가지:
- `regime_source: rule` — 규칙 기반(빠름)
- `regime_source: hmm` — **자체 구현 가우시안 HMM**(`src/analysis/_hmm.py`, numpy/scipy; hmmlearn 불필요).
  Baum-Welch 학습 + **Hamilton 전방필터**(look-ahead 안전) + BIC 상태선택 + 확장윈도우 재학습.
  모든 위기 포착(2008·2020·2022). precompute ~90초.

**검증**: HMM 버짓이 rule 버짓을 전 구간 압도(test Sharpe 1.00 vs 0.75). *엣지 베이스*
(필터 끈 Top-N)에 HMM 버짓을 결합하면 최근 OOS에서 plain Top-N과 **Sharpe 동등하나
낙폭이 −22.8% vs −37.2%(Calmar 1.02 vs 0.79)** — 드로다운 관리형 변형. 단 raw 수익은
plain Top-N이 우위. 레시피는 `config/strategy.yaml`의 `regime_budget` 주석 참조.

## 리스크 오버레이 (선택, 기본 off)

`config/backtest.yaml`의 `risk_overlay.enabled: true`로 모든 전략에 적용:
- **포지션 캡**: 단일 종목 상한 + 그로스 익스포저 상한(초과분 현금)
- **상관 급등 디레버리징**: 평균 쌍상관이 임계 초과(위기 동조화) 시 익스포저 축소

---

## 예시 결과 (참고용)

실데이터(2001–2026, 월간 리밸런싱) 백테스트 일부 — *과거 성과이며 미래를 보장하지 않습니다*:

| 전략 | CAGR | Sharpe | MaxDD | Calmar |
|------|------|--------|-------|--------|
| Composite Score | ~20.3% | 0.74 | -56% | 0.35 |
| Top-N Momentum | ~18.0% | 0.73 | -41% | 0.44 |
| Equal Weight | ~11% | 0.51 | -47% | 0.21 |
| Buy & Hold SPY | ~9.6% | 0.46 | -55% | 0.17 |
| Risk Parity | ~3.8% | 0.22 | -32% | 0.12 |

전체 지표·기간·주기 비교는 `python cli.py backtest compare` 와 `output/metrics.csv` 참고.

---

## 주의사항

- 본 프로젝트는 **교육 및 연구 목적**입니다. 실제 투자에는 추가 리스크 관리·검증이 필요합니다.
- **과거 성과가 미래 수익을 보장하지 않습니다.**
- `yfinance`는 공식 데이터 피드가 아니므로 프로덕션 트레이딩에 부적합합니다.
- 백테스트는 **과적합(overfitting)** 위험이 있습니다. in-sample / out-of-sample 분리를 권장합니다.
- 거래비용·슬리피지는 보수적 가정값이며 실제 체결과 다를 수 있습니다.
