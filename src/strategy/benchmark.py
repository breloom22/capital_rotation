"""Benchmarks: 60/40, equal-weight (1/N), and Buy & Hold SPY.

These are plain Strategy subclasses with simple, mostly-static target weights so
the engine treats them uniformly (same costs / rebalancing) as the active
strategies.

Registry names:
    bench_6040   -> 60% SPY / 40% TLT   (config benchmarks.sixty_forty)
    bench_equal  -> equal weight over all tradable assets (config benchmarks.equal_weight)
    bench_bh     -> Buy & Hold (config benchmarks.buy_hold, default SPY 100%)

Implementation guidance:
* warmup should be small (e.g. 1) -- benchmarks need no signal history. The
  engine aligns the whole suite to the largest warmup, so keep these minimal.
* target_weights(date): return the configured weights restricted to assets
  available_at(date), renormalised. For equal-weight, 1/k over available tradables.
* Buy & Hold should set its target once and let it drift (return the same target
  each rebalance; drift is handled by the engine). Returning constant targets is
  fine -- turnover will be ~0 after the first allocation.
"""
from __future__ import annotations

import pandas as pd

from .base import Strategy, register


@register("bench_6040")
class SixtyFortyBenchmark(Strategy):
    label = "60/40"

    def __init__(self, data, config, params=None):
        super().__init__(data, config, params)
        self.warmup = 1

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        cfg = self.config.benchmarks_cfg.get("sixty_forty", {}) or {}
        # restrict to tradable tickers present in this panel and available at date
        available = set(self.available_at(date))
        weights = pd.Series(
            {t: float(w) for t, w in cfg.items() if t in available},
            dtype=float,
        )
        return self.normalize(weights)


@register("bench_equal")
class EqualWeightBenchmark(Strategy):
    label = "Equal Weight"

    def __init__(self, data, config, params=None):
        super().__init__(data, config, params)
        self.warmup = 1

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        cfg = self.config.benchmarks_cfg.get("equal_weight", {}) or {}
        available = self.available_at(date)
        if cfg:
            # an explicit (possibly weighted) universe: equal-weight its members
            tickers = [t for t in cfg if t in available]
        else:
            # default: 1/k over all available tradables
            tickers = available
        k = len(tickers)
        if k == 0:
            return pd.Series(dtype=float)
        weights = pd.Series(1.0 / k, index=tickers, dtype=float)
        return self.normalize(weights)


@register("bench_bh")
class BuyHoldBenchmark(Strategy):
    label = "Buy & Hold SPY"

    def __init__(self, data, config, params=None):
        super().__init__(data, config, params)
        self.warmup = 1

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        cfg = self.config.benchmarks_cfg.get("buy_hold", {}) or {"SPY": 1.0}
        available = set(self.available_at(date))
        weights = pd.Series(
            {t: float(w) for t, w in cfg.items() if t in available},
            dtype=float,
        )
        return self.normalize(weights)
