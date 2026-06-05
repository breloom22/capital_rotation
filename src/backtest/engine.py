"""Event-driven backtesting engine (vectorised where it matters).

Look-ahead bias prevention -- the core guarantee
------------------------------------------------
Time advances over a daily calendar. For each day ``t`` we FIRST apply day-t
returns to holdings carried in from the previous rebalance, THEN (if ``t`` is a
rebalance day) we rebalance at the close. Consequence: weights chosen on day t
(from signals with index <= t) only earn returns from day t+1 onward. No part
of the simulation can read a price it could not have known.

Cash earns 0% (per spec). Transaction cost = one-way cost x traded notional,
charged on every rebalance and paid out of cash.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data import MarketData
from .metrics import compute_metrics


# --------------------------------------------------------------------------
# rebalance schedule
# --------------------------------------------------------------------------
def generate_rebalance_dates(calendar: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    """Trading days on which a fixed-schedule strategy rebalances.

    Always the LAST available trading day of each period so we never invent a
    date the market was closed.
    """
    if len(calendar) == 0:
        return []
    s = pd.Series(np.arange(len(calendar)), index=calendar)
    if freq == "weekly":
        key = calendar.to_period("W")
    elif freq == "monthly":
        key = calendar.to_period("M")
    elif freq == "quarterly":
        key = calendar.to_period("Q")
    elif freq == "signal":
        # candidates evaluated weekly; engine decides whether to execute
        key = calendar.to_period("W")
    else:
        raise ValueError(f"unknown rebalance freq '{freq}'")
    last_idx = s.groupby(key).last()
    return [calendar[i] for i in last_idx.values]


@dataclass
class BacktestResult:
    name: str
    freq: str
    equity: pd.Series                       # daily portfolio value
    returns: pd.Series                       # daily returns
    weights: pd.DataFrame                    # target weights at each rebalance (date x ticker)
    cash_weight: pd.Series                   # cash fraction at each rebalance
    rebalance_dates: list = field(default_factory=list)
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    period_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: dict = field(default_factory=dict)


class BacktestEngine:
    def __init__(self, data: MarketData, config):
        self.data = data
        self.config = config
        self.cost = config.one_way_cost
        self.initial_capital = float(config.backtest.get("initial_capital", 100000))
        self.periods_per_year = 252 if data.freq == "daily" else 52
        self.rf = config.risk_free_rate
        # daily returns matrix used to evolve holdings (NaN -> 0 for held assets)
        self._ret = data.returns
        # opt-in risk overlay (position caps + correlation-spike de-risking)
        from .overlay import RiskOverlay
        self.overlay = RiskOverlay(data, config)

    # ---- single run --------------------------------------------------------
    def run(self, strategy, freq: str = "monthly",
            start_date: pd.Timestamp | None = None,
            end_date: pd.Timestamp | None = None) -> BacktestResult:
        strategy.ensure_precomputed()
        # NOTE: signals/precompute see the FULL history (trailing => look-ahead
        # safe); only the EVALUATION calendar is windowed. So an out-of-sample
        # test window is evaluated with signals legitimately available at each
        # date (which include pre-window history -- the past relative to the test).
        calendar = self.data.dates
        if start_date is not None:
            calendar = calendar[calendar >= pd.Timestamp(start_date)]
        if end_date is not None:
            calendar = calendar[calendar <= pd.Timestamp(end_date)]
        if len(calendar) < 2:
            raise ValueError("not enough data to backtest in the chosen window")

        schedule = set(generate_rebalance_dates(calendar, freq))
        signal_mode = freq == "signal"
        sig_cfg = self.config.rebalance.get("signal", {})
        min_interval = int(sig_cfg.get("min_interval_days", 5))
        turn_threshold = float(sig_cfg.get("turnover_threshold", 0.0))

        tickers = list(self.data.prices.columns)
        n = len(tickers)
        # positional numpy view of returns aligned to the calendar (fast inner loop)
        ret_values = (self._ret.reindex(index=calendar, columns=tickers)
                      .to_numpy(dtype=float))
        ret_values = np.nan_to_num(ret_values, nan=0.0)

        holdings = np.zeros(n)
        cash = self.initial_capital
        equity_idx, equity_val = [], []
        weights_hist: dict[pd.Timestamp, pd.Series] = {}
        cash_hist: dict[pd.Timestamp, float] = {}
        turnover_hist: dict[pd.Timestamp, float] = {}

        last_rebalance_pos = None
        started = False

        for pos, date in enumerate(calendar):
            # 1) apply today's returns to existing holdings (skip the very first
            #    active day -- holdings are still zero before the first rebalance)
            if started:
                holdings = holdings * (1.0 + ret_values[pos])

            V = holdings.sum() + cash

            # 2) decide whether to rebalance at today's close
            do_rebalance = date in schedule
            if not started:
                # first scheduled date becomes the inception of the portfolio
                do_rebalance = date in schedule
            if do_rebalance:
                target = strategy.target_weights(date)
                target = _to_series(target, tickers)
                target = self.overlay.apply(target, date)  # opt-in caps / corr gate
                desired = (target.reindex(tickers).fillna(0.0).to_numpy() * V)

                # signal-mode gating: skip tiny / too-frequent rebalances
                if started and signal_mode:
                    gross = np.abs(desired - holdings).sum()
                    turn = gross / V if V > 0 else 0.0
                    too_soon = (last_rebalance_pos is not None and
                                (pos - last_rebalance_pos) < min_interval)
                    if turn < turn_threshold or too_soon:
                        do_rebalance = False

            if do_rebalance:
                trades = desired - holdings
                cost = self.cost * np.abs(trades).sum()
                new_cash = V - desired.sum() - cost
                # if fully invested, scale holdings down so the cost is funded.
                # Iterate to a fixed point: scaling can change the traded notional
                # (hence the cost), so a single step may still leave cash < 0.
                for _ in range(8):
                    if new_cash >= -1e-9 or desired.sum() <= 0:
                        break
                    scale = max(V - cost, 0.0) / desired.sum()
                    desired = desired * scale
                    trades = desired - holdings
                    cost = self.cost * np.abs(trades).sum()
                    new_cash = V - desired.sum() - cost
                turnover_hist[date] = float(np.abs(trades).sum() / V) if V > 0 else 0.0
                holdings = desired
                cash = new_cash
                # record fractions of the REALISED (post-cost) value so the
                # weights + cash add to 1 and describe the portfolio that drifts
                # into the next period.
                vp = holdings.sum() + cash
                weights_hist[date] = (pd.Series(holdings, index=tickers) / vp) if vp > 0 else pd.Series(0.0, index=tickers)
                cash_hist[date] = float(cash / vp) if vp > 0 else 1.0
                last_rebalance_pos = pos
                started = True

            if started:
                equity_idx.append(date)
                equity_val.append(holdings.sum() + cash)

        if not equity_idx:
            raise ValueError("strategy never produced a position in the window")

        equity = pd.Series(equity_val, index=pd.DatetimeIndex(equity_idx), name=strategy.display_name())
        returns = equity.pct_change().dropna()
        weights = pd.DataFrame(weights_hist).T.reindex(columns=tickers).fillna(0.0)
        weights.index.name = "Date"
        cash_w = pd.Series(cash_hist).sort_index()
        turnover = pd.Series(turnover_hist).sort_index()

        rb_dates = sorted(weights_hist.keys())
        eq_at_rb = equity.reindex(rb_dates).ffill()
        eq_at_rb = pd.concat([eq_at_rb, equity.iloc[[-1]]])
        eq_at_rb = eq_at_rb[~eq_at_rb.index.duplicated(keep="last")].sort_index()
        period_returns = eq_at_rb.pct_change().dropna()

        bench_returns = self._benchmark_returns(equity.index)
        metrics = compute_metrics(
            equity, periods_per_year=self.periods_per_year, rf=self.rf,
            benchmark_returns=bench_returns, period_returns=period_returns,
            turnover=turnover, n_trades=len(rb_dates),
        )

        return BacktestResult(
            name=strategy.display_name(), freq=freq, equity=equity, returns=returns,
            weights=weights, cash_weight=cash_w, rebalance_dates=rb_dates,
            turnover=turnover, period_returns=period_returns, metrics=metrics,
        )

    def _benchmark_returns(self, index: pd.DatetimeIndex) -> pd.Series | None:
        bt = self.config.benchmark_ticker
        if bt in self.data.returns.columns:
            return self.data.returns[bt].reindex(index)
        return None

    # ---- warmup / common start --------------------------------------------
    def common_start(self, strategies: list, extra: int = 5,
                     min_assets: int | None = None) -> pd.Timestamp:
        """First date at which the suite can start fairly.

        We need (a) enough warmup history for the slowest strategy and (b) a
        minimum number of TRADABLE assets that already have that much history --
        otherwise an early universe dominated by long-history non-tradables
        (e.g. the dollar index from 1971) would leave the portfolio in cash for
        decades and distort CAGR. All strategies share this start so their
        equity curves are directly comparable.
        """
        dates = self.data.dates
        if len(dates) < 4:
            return dates[0]
        warm = max((getattr(s, "warmup", 252) for s in strategies), default=252)
        tradable = self.data.tradable_present()
        if min_assets is None:
            min_assets = max(3, min(8, len(tradable) // 2))

        if tradable:
            avail = self.data.available[tradable]
            age = avail.cumsum()                 # periods of history per asset
            ready = (age >= warm).sum(axis=1)    # # tradable assets warmed up
            qualifying = ready[ready >= min_assets]
            if len(qualifying):
                pos = dates.get_loc(qualifying.index[0])
                return dates[min(pos + extra, len(dates) - 2)]

        # fallback: plain warmup from the start of the calendar
        return dates[min(warm + extra, len(dates) - 2)]


def _to_series(weights, tickers: list[str]) -> pd.Series:
    if isinstance(weights, pd.Series):
        return weights
    if isinstance(weights, dict):
        return pd.Series(weights, dtype=float)
    if weights is None:
        return pd.Series(dtype=float)
    raise TypeError(f"strategy returned unsupported weights type {type(weights)}")
