"""Strategy 5 -- Minimum-Variance portfolio.

Holds the long-only weights that minimise portfolio variance using a Ledoit-Wolf
shrinkage covariance over a trailing window. Unlike inverse-vol risk parity, it
uses the FULL covariance (correlations), so it concentrates in assets that are
both low-vol AND good diversifiers.

Params (config/strategy.yaml -> strategies.min_variance):
    cov_window, min_weight, shrinkage ('ledoit_wolf'|'sample'), min_obs,
    absolute_filter, momentum_lookback.

Look-ahead safe: the covariance at each date uses only the trailing return block
ending at that date (see _riskopt.trailing_cov), and the engine applies day-t
weights to day-t+1 returns.
"""
from __future__ import annotations

import pandas as pd

from .base import Strategy, register
from ._riskopt import trailing_cov, solve_min_variance


@register("min_variance")
class MinVarianceStrategy(Strategy):
    label = "Min Variance"

    def precompute(self) -> None:
        self.window = int(self.params.get("cov_window", 120))
        self.min_weight = float(self.params.get("min_weight", 0.0))
        self.shrinkage = self.params.get("shrinkage", "ledoit_wolf")
        self.min_obs = int(self.params.get("min_obs", max(20, self.window // 2)))
        self.absolute_filter = bool(self.params.get("absolute_filter", False))
        self.mom_lookback = int(self.params.get("momentum_lookback", 126))
        self.mom_panel = None
        if self.absolute_filter:
            from ..analysis import momentum
            self.mom_panel = momentum.total_return(self.prices[self.tradable], self.mom_lookback)
        self.warmup = max(self.window, self.mom_lookback if self.absolute_filter else 0)

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        avail = self.available_at(date)
        if not avail:
            return pd.Series(dtype=float)
        if self.absolute_filter and self.mom_panel is not None:
            mom = self.signal_at(self.mom_panel, date)
            if isinstance(mom, pd.Series):
                avail = [t for t in avail if mom.get(t, float("-inf")) > 0]
            if not avail:
                return pd.Series(dtype=float)

        keep, cov = trailing_cov(self.returns, date, avail, self.window,
                                 self.min_obs, self.shrinkage)
        if cov is None:
            return pd.Series(1.0, index=keep) if len(keep) == 1 else pd.Series(dtype=float)
        w = solve_min_variance(cov, self.min_weight)
        return pd.Series(w, index=keep)
