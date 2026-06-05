# Capital Rotation Tracker & Strategy Backtester

글로벌 자산군 간 자금 순환(Capital Rotation)을 분석하고, 로테이션 기반 투자 전략을 백테스팅하는 Python CLI 프로젝트.

---

## 프로젝트 목표

1. 전 세계 주요 시장/자산군의 과거 데이터를 수집한다
2. 시장 간 상관관계, 자금 흐름 방향, 순환 패턴을 분석한다
3. 자본 순환을 이용한 여러 포트폴리오 전략을 구현하고 백테스팅한다
4. 전략 간 성과를 비교하여 실제 수익성을 검증한다

---

## 1. 추적 대상 시장/자산군

### 1.1 대표 티커 매핑

모든 데이터는 **yfinance**를 통해 수집한다. 개별 종목이 아닌 **ETF 또는 지수**를 사용하여 자산군 전체의 흐름을 추적한다.

| 카테고리 | 자산 | 티커 (yfinance) | 비고 |
|---------|------|----------------|------|
| **미국 주식** | S&P 500 | `SPY` | 미국 대형주 |
| | NASDAQ 100 | `QQQ` | 미국 기술주 |
| | Russell 2000 | `IWM` | 미국 소형주 |
| **한국 주식** | KOSPI | `EWY` | iShares MSCI South Korea ETF |
| | (대안) | `^KS11` | KOSPI 지수 직접 (거래량 없음) |
| **크립토** | Bitcoin | `BTC-USD` | |
| | Ethereum | `ETH-USD` | |
| **미국 채권** | 미국 장기채 | `TLT` | 20+ Year Treasury |
| | 미국 단기채 | `SHY` | 1-3 Year Treasury |
| | 미국 회사채 | `LQD` | Investment Grade Corporate |
| **원자재** | 금 | `GLD` | Gold ETF |
| | 은 | `SLV` | Silver ETF |
| | 원유 | `USO` | Crude Oil ETF |
| | 원자재 종합 | `DJP` 또는 `GSG` | Commodity Index |
| **외환** | 달러 인덱스 | `DX-Y.NYB` | USD Index |
| | 엔/달러 | `JPY=X` | |
| | 유로/달러 | `EUR=X` | |
| **신흥국** | 신흥국 종합 | `EEM` | MSCI Emerging Markets |
| | 중국 | `FXI` | China Large-Cap |
| | 인도 | `INDA` | MSCI India |
| | 브라질 | `EWZ` | MSCI Brazil |
| **선진국** | 일본 | `EWJ` | MSCI Japan |
| | 독일 | `EWG` | MSCI Germany |
| | 영국 | `EWU` | MSCI United Kingdom |
| | 유럽 종합 | `VGK` | FTSE Europe |
| **부동산** | 미국 REITs | `VNQ` | Vanguard Real Estate |
| **변동성** | VIX | `^VIX` | 공포 지수 (매매 불가, 지표용) |

### 1.2 티커 관리 원칙

- 티커 목록은 `config.yaml` 또는 `config.json`으로 외부 관리한다
- 카테고리별 그룹핑 정보를 포함한다
- 사용자가 티커를 자유롭게 추가/제거할 수 있어야 한다
- 각 티커의 데이터 시작일이 다르므로, 수집 시 자동으로 가용 기간을 감지한다

---

## 2. 데이터 수집

### 2.1 소스

- **주 데이터 소스**: `yfinance` (무료, 별도 API 키 불필요)
- **보조 데이터 소스** (선택적, 추후 확장용):
  - FRED API (연준 금리, 경제 지표) — 무료 API 키 발급
  - CoinGecko API (크립토 보조 데이터) — 무료 티어

### 2.2 수집 사양

- **수집 주기**: Daily OHLCV (Open, High, Low, Close, Volume)
- **분석 주기**: Daily + Weekly (weekly는 daily에서 리샘플링)
- **백테스팅 기간**: 가능한 최대. 대부분 ETF는 2005~2010년 이후 데이터 존재. 각 자산별 가용 시작일에 맞춰 유연하게 처리
- **데이터 저장**: 로컬 파일 시스템 (Parquet 또는 CSV)
- **업데이트 방식**: 증분 업데이트 (이미 받은 데이터 이후 날짜만 추가 수집)

### 2.3 데이터 전처리

- 결측치 처리: forward fill → backward fill → 잔여 NaN 제거
- 수정 종가(Adjusted Close) 기준 수익률 계산
- 거래량: 달러 거래량(Close × Volume)으로 정규화
- 타임존 통일: UTC 기준
- Weekly 데이터: Daily → Weekly 리샘플링 (금요일 종가 기준)

---

## 3. 분석 엔진

### 3.1 핵심 지표

#### A. 가격 모멘텀 (Price Momentum)

```
- 수익률: 1주, 1개월, 3개월, 6개월, 12개월 구간별 수익률
- 모멘텀 스코어: 복수 기간 수익률의 가중 평균 (예: 12개월 수익률 × 0.5 + 6개월 × 0.3 + 3개월 × 0.2)
- 듀얼 모멘텀: 절대 모멘텀(해당 자산 수익률 > 0) + 상대 모멘텀(타 자산 대비 순위)
```

#### B. 거래량 변화 (Volume Dynamics)

```
- 거래량 이동평균 대비 비율: 현재 거래량 / 20일 평균 거래량
- 거래량 추세: 거래량의 N일 이동평균 기울기
- OBV (On-Balance Volume): 가격 방향에 따른 누적 거래량
- 자금 유입/유출 추정: 가격 상승 + 거래량 증가 = 유입, 가격 하락 + 거래량 증가 = 유출
```

#### C. 변동성 (Volatility)

```
- 실현 변동성: N일 수익률의 표준편차 (연율화)
- ATR (Average True Range): 일간 변동 폭
- VIX 레벨 및 변화율: 시장 전체 공포 수준
- 변동성 레짐 판단: 저변동(VIX < 15), 보통(15-25), 고변동(25-35), 극단(35+)
```

#### D. 상관관계 변화 추적 (Correlation Regime)

```
- 롤링 상관관계: N일(60일, 120일, 252일) 윈도우 상관 행렬
- 상관관계 변화 속도: 상관계수의 1차 미분 (급격한 변화 감지)
- 상관관계 레짐: 정상 vs 위기 (위기 시 상관관계 급등 = 동조화)
- PCA (주성분 분석): 시장 움직임의 주요 팩터 추출, 설명력 추적
```

### 3.2 파생 분석

#### 자금 순환 감지 (Rotation Detection)

```
- 섹터/자산군 간 상대 강도 변화 추적
- "자금 유입 중" vs "자금 유출 중" 자산 분류
- 순환 사이클 평균 기간 추정
- 선행/후행 관계 분석: 자산 A의 움직임이 자산 B에 N일 선행하는지 (교차상관)
```

#### 시장 레짐 분류 (Market Regime)

```
- Risk-On vs Risk-Off 판단
  - Risk-On 신호: 주식↑, 채권↓, VIX↓, 원자재↑
  - Risk-Off 신호: 주식↓, 채권↑, VIX↑, 금↑, 달러↑
- 레짐 전환 시점 감지 (Hidden Markov Model 또는 rule-based)
```

---

## 4. 포트폴리오 전략

아래 전략을 **모두 구현**하고, 동일 조건에서 백테스팅 후 비교한다.

### 4.1 전략 목록

#### Strategy 1: Top-N 로테이션

```
- 모멘텀 스코어 상위 N개 자산에 동일 비중 투자
- 파라미터: N (3, 5, 7), 모멘텀 룩백 기간, 캐시 필터(절대 모멘텀 음수면 현금 보유)
```

#### Strategy 2: 리스크 패리티 (Risk Parity)

```
- 각 자산의 변동성 역수로 비중 결정
- 변동성 높은 자산 = 적은 비중, 낮은 자산 = 많은 비중
- 목표: 각 자산이 포트폴리오 리스크에 동일하게 기여
```

#### Strategy 3: 모멘텀 스코어링 (Composite Scoring)

```
- 여러 지표(모멘텀, 거래량, 변동성, 상관관계)를 종합한 점수 산출
- 점수 기반 비중 배분 (점수 비례 or 점수 구간별 등급)
- 파라미터: 각 지표의 가중치
```

#### Strategy 4: 레짐 기반 전략 (Regime-Based)

```
- 시장 레짐(Risk-On/Off)에 따라 자산군 배분 규칙 변경
- Risk-On: 주식/크립토 비중↑
- Risk-Off: 채권/금/현금 비중↑
```

#### Benchmark (비교 기준)

```
- 60/40 포트폴리오 (SPY 60% + TLT 40%)
- 동일 비중 포트폴리오 (전체 자산 1/N)
- Buy & Hold SPY (단순 미국 주식 보유)
```

### 4.2 리밸런싱 옵션

아래 리밸런싱 주기를 **모두 테스트**하여 비교한다:

- **주간 (Weekly)**: 매주 금요일 종가 기준
- **월간 (Monthly)**: 매월 마지막 거래일
- **분기 (Quarterly)**: 3개월마다
- **신호 기반 (Signal-Triggered)**: 모멘텀 스코어 또는 레짐 변화 시에만 리밸런싱. 최소 간격 제한(예: 최소 5거래일) 설정

### 4.3 거래 비용

- 거래 수수료: 편도 0.1% (보수적 가정)
- 슬리피지: 편도 0.05%
- 설정 파일에서 조정 가능하게

---

## 5. 백테스팅 엔진

### 5.1 요구사항

- **Look-ahead bias 방지**: 미래 데이터 절대 참조 금지. 모든 지표는 해당 시점까지의 데이터만으로 계산
- **Survivorship bias 주의**: ETF 기반이므로 크게 해당 없으나, 상장폐지된 자산 처리 로직 포함
- **초기 자본**: 기본값 $100,000 (설정 변경 가능)
- **현금 포지션**: 신호가 없거나 절대 모멘텀 음수 시 현금 보유 가능 (수익률 0% 가정)

### 5.2 성과 지표 (전부 계산)

| 지표 | 설명 |
|-----|------|
| **총 수익률** | 전체 기간 누적 수익률 |
| **CAGR** | 연평균 복리 수익률 |
| **Sharpe Ratio** | (연 수익률 - 무위험 수익률) / 연 변동성 |
| **Sortino Ratio** | 하방 변동성만 사용한 Sharpe 변형 |
| **MDD (Maximum Drawdown)** | 최고점 대비 최대 하락폭 |
| **MDD 기간** | MDD에서 회복까지 걸린 기간 |
| **Calmar Ratio** | CAGR / MDD |
| **승률 (Win Rate)** | 수익 발생 리밸런싱 비율 |
| **손익비 (Profit Factor)** | 총 이익 / 총 손실 |
| **월별 수익률 히트맵** | 연도 × 월 매트릭스 |
| **연도별 수익률** | 벤치마크 대비 연간 비교 |
| **Alpha / Beta** | 벤치마크(SPY) 대비 |
| **거래 횟수 & 회전율** | 리밸런싱당 평균 교체 비율 |

### 5.3 출력

- CLI 테이블: 전략 × 리밸런싱 주기별 성과 비교표
- CSV/JSON 내보내기: 상세 결과 저장
- 차트 (matplotlib/plotly): 누적 수익 곡선, 드로다운 차트, 상관관계 히트맵 등

---

## 6. 프로젝트 구조

```
capital-rotation/
├── config/
│   ├── assets.yaml          # 자산 목록 및 카테고리
│   ├── strategy.yaml        # 전략 파라미터
│   └── backtest.yaml        # 백테스팅 설정 (기간, 수수료, 자본금)
├── src/
│   ├── data/
│   │   ├── fetcher.py       # yfinance 데이터 수집
│   │   ├── storage.py       # 로컬 저장/로드 (Parquet)
│   │   └── preprocess.py    # 전처리 (결측치, 리샘플링)
│   ├── analysis/
│   │   ├── momentum.py      # 모멘텀 지표
│   │   ├── volume.py        # 거래량 분석
│   │   ├── volatility.py    # 변동성 지표
│   │   ├── correlation.py   # 상관관계 분석
│   │   ├── regime.py        # 시장 레짐 판단
│   │   └── rotation.py      # 자금 순환 감지
│   ├── strategy/
│   │   ├── base.py          # 전략 베이스 클래스
│   │   ├── topn.py          # Top-N 로테이션
│   │   ├── risk_parity.py   # 리스크 패리티
│   │   ├── momentum_score.py # 모멘텀 스코어링
│   │   ├── regime_based.py  # 레짐 기반
│   │   └── benchmark.py     # 벤치마크 (60/40, 동일비중, B&H)
│   ├── backtest/
│   │   ├── engine.py        # 백테스팅 엔진
│   │   ├── metrics.py       # 성과 지표 계산
│   │   └── report.py        # 결과 리포트 생성
│   └── visualization/
│       ├── charts.py        # 차트 생성
│       └── tables.py        # CLI 테이블 출력
├── data/                    # 수집된 데이터 저장 디렉토리
├── output/                  # 백테스팅 결과 저장 디렉토리
├── cli.py                   # CLI 엔트리포인트
├── requirements.txt
└── README.md
```

---

## 7. CLI 인터페이스

### 주요 명령어

```bash
# 데이터 수집/업데이트
python cli.py data update              # 전체 자산 데이터 수집/업데이트
python cli.py data update --asset SPY  # 특정 자산만
python cli.py data status              # 수집 현황 (자산별 기간, 결측치)

# 분석
python cli.py analyze correlation       # 상관관계 매트릭스 출력
python cli.py analyze momentum          # 현재 모멘텀 스코어 순위
python cli.py analyze regime            # 현재 시장 레짐 판단
python cli.py analyze rotation          # 자금 순환 현황
python cli.py analyze all               # 전체 분석 리포트

# 백테스팅
python cli.py backtest run              # 모든 전략 × 모든 리밸런싱 주기 백테스팅
python cli.py backtest run --strategy topn --rebalance monthly
python cli.py backtest compare          # 전략 간 성과 비교표
python cli.py backtest export           # 결과 CSV/JSON 내보내기

# 차트
python cli.py chart equity              # 누적 수익 곡선
python cli.py chart drawdown            # 드로다운 차트
python cli.py chart correlation         # 상관관계 히트맵
python cli.py chart rotation            # 자금 순환 시각화
```

---

## 8. 기술 스택 & 의존성

```
python >= 3.10
yfinance             # 데이터 수집
pandas               # 데이터 처리
numpy                # 수치 연산
scipy                # 통계 분석, 최적화
scikit-learn         # PCA, 클러스터링
matplotlib           # 기본 차트
plotly               # 인터랙티브 차트 (선택)
rich                 # CLI 테이블, 프로그레스바
click                # CLI 프레임워크
pyyaml               # 설정 파일 파싱
pyarrow              # Parquet 저장
```

---

## 9. 개발 순서 (권장)

### Phase 1: 데이터 파이프라인
1. `config/assets.yaml` 설계 및 티커 목록 정의
2. `fetcher.py` — yfinance 래퍼, 증분 수집
3. `storage.py` — Parquet 저장/로드
4. `preprocess.py` — 결측치, 리샘플링, 수익률 계산
5. CLI `data` 명령어 연결

### Phase 2: 분석 엔진
6. `momentum.py` — 다기간 모멘텀 스코어
7. `volume.py` — 거래량 분석
8. `volatility.py` — 변동성 계산
9. `correlation.py` — 롤링 상관관계, PCA
10. `regime.py` — 레짐 분류
11. `rotation.py` — 순환 감지 종합
12. CLI `analyze` 명령어 연결

### Phase 3: 전략 & 백테스팅
13. `base.py` — 전략 인터페이스 정의
14. 4개 전략 + 3개 벤치마크 구현
15. `engine.py` — 백테스팅 엔진 (look-ahead bias 방지)
16. `metrics.py` — 성과 지표 전체 구현
17. `report.py` — 비교 리포트
18. CLI `backtest` 명령어 연결

### Phase 4: 시각화 & 마무리
19. `charts.py` — matplotlib/plotly 차트
20. `tables.py` — rich 기반 CLI 출력
21. CLI `chart` 명령어 연결
22. README.md 작성
23. 전체 통합 테스트

---

## 10. 핵심 원칙

- **설정 중심**: 모든 파라미터(자산 목록, 전략 파라미터, 리밸런싱 주기, 수수료)는 YAML 설정 파일로 관리. 코드 수정 없이 실험 가능해야 함
- **모듈화**: 각 분석/전략 모듈이 독립적으로 동작. 새 전략 추가가 쉬워야 함
- **재현성**: 동일 설정이면 동일 결과. 랜덤 시드 고정
- **속도**: 15년치 30개 자산 백테스팅이 합리적 시간 내 완료 (벡터화 연산 활용)
- **확장성**: 추후 웹 대시보드(FastAPI + React)로 전환할 수 있도록 비즈니스 로직과 UI를 분리

---

## 11. 주의사항

- 이 프로젝트는 **교육 및 연구 목적**이다. 실제 투자 결정에 사용할 경우 추가적인 리스크 관리와 검증이 필요하다
- 과거 성과가 미래 수익을 보장하지 않는다
- yfinance 데이터는 공식 데이터 피드가 아니므로, 프로덕션 트레이딩에는 적합하지 않다
- 백테스팅 결과에는 과적합(overfitting) 위험이 있으므로, in-sample / out-of-sample 분리를 권장한다
