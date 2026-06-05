"""Strategy 1 -- Top-N momentum rotation.

Equal-weight the N assets with the highest composite momentum score. When
``absolute_filter`` is on, any selected asset whose trailing momentum is <= 0 is
replaced by cash (dual-momentum style).

Params (config/strategy.yaml -> strategies.topn):
    n, lookback_weights (or null => global momentum.lookback_weights),
    absolute_filter, min_assets.

Implementation guidance (look-ahead safe):
* In precompute(): build the momentum score panel once via
  ``momentum.momentum_score(self.prices[self.tradable], weights, skip_recent)``
  and (if absolute_filter) ``momentum.absolute_momentum(...)``.
* In target_weights(date): take ``signal_at`` of the score panel, restrict to
  ``available_at(date)``, pick top-N, drop assets failing the absolute filter,
  equal-weight the survivors (remainder = cash). Use ``self.normalize`` if needed.
"""
from __future__ import annotations

import pandas as pd

from .base import Strategy, register
from ..analysis import momentum


@register("topn")
class TopNStrategy(Strategy):
    label = "Top-N Momentum"

    def precompute(self) -> None:
        cfg_mom = self.config.momentum_cfg

        # lookback weights: strategy override or the global momentum block.
        weights = self.params.get("lookback_weights")
        if not weights:
            weights = cfg_mom.get("lookback_weights") or {252: 1.0}
        # normalise key types to int (YAML keys may arrive as int already)
        self._weights = {int(k): float(v) for k, v in weights.items()}

        skip_recent = int(cfg_mom.get("skip_recent_days", 0) or 0)

        self._n = int(self.params.get("n", 5))
        self._absolute_filter = bool(self.params.get("absolute_filter", False))
        self._min_assets = int(self.params.get("min_assets", 1))

        prices = self.prices[self.tradable]

        # composite momentum score panel (look-ahead safe: trailing windows only)
        self._scores = momentum.momentum_score(prices, self._weights, skip_recent)

        # absolute (time-series) momentum over the longest lookback used; an asset
        # passes the filter when its trailing return is strictly positive.
        self._abs_lookback = max(self._weights) if self._weights else 252
        if self._absolute_filter:
            self._abs_mom = momentum.absolute_momentum(prices, self._abs_lookback)
        else:
            self._abs_mom = None

        # need enough history for the longest lookback (plus any skip).
        self.warmup = max(self._weights, default=252) + skip_recent

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        # last score row with index <= date (structural look-ahead guard)
        scores = self.signal_at(self._scores, date)
        if not isinstance(scores, pd.Series) or scores.empty:
            return pd.Series(dtype=float)

        # restrict to assets that have a real, post-inception price at `date`
        available = self.available_at(date)
        scores = scores.reindex(available).dropna()
        if scores.empty:
            return pd.Series(dtype=float)

        # highest composite momentum first; take top-N
        ranked = scores.sort_values(ascending=False)
        selected = ranked.index[: self._n]

        # dual-momentum: drop selected assets whose trailing momentum <= 0
        if self._absolute_filter and self._abs_mom is not None:
            abs_row = self.signal_at(self._abs_mom, date)
            if isinstance(abs_row, pd.Series):
                survivors = [
                    t for t in selected
                    if t in abs_row.index and pd.notna(abs_row[t]) and abs_row[t] > 0
                ]
            else:
                survivors = []
            # if too few pass the filter, hold the rest in cash
            if len(survivors) < self._min_assets:
                survivors = []
            selected = survivors

        if len(selected) == 0:
            return pd.Series(dtype=float)

        # equal-weight survivors; remainder (1 - sum) is held as cash
        w = pd.Series(1.0 / len(selected), index=list(selected), dtype=float)
        return self.normalize(w)
