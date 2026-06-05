"""Strategy 4 -- Regime-based allocation.

Switch the category mix on the market regime from ``analysis.regime``:
* Risk-On  -> equities / crypto / commodities heavy (config risk_on block)
* Risk-Off -> bonds / gold / cash heavy (config risk_off block)

Within each category, the category weight is split equally across that
category's tradable members that are available at the date. A 'cash' key in the
config block is left as cash (not allocated).

Params (strategies.regime_based): risk_on{category:weight}, risk_off{...}.

Implementation guidance (look-ahead safe):
* precompute(): regime panel via ``regime.classify_regime(self.data, self.config)``.
* target_weights(date): regime label via signal_at(regime_df['regime'], date);
  pick the matching config block; expand category weights to per-ticker weights
  over available_at(date) members; normalise (remainder/'cash' => cash).
"""
from __future__ import annotations

import pandas as pd

from .base import Strategy, register
from ..analysis import regime


@register("regime_based")
class RegimeStrategy(Strategy):
    label = "Regime Based"

    def __init__(self, data, config, params=None):
        super().__init__(data, config, params)
        # Largest lookback used: the regime classifier's trailing window.
        self._regime_lookback = 63
        self.warmup = 126
        self._regime_df: pd.DataFrame | None = None

    def precompute(self) -> None:
        # Trailing, look-ahead-safe regime panel (columns: 'regime', 'score').
        self._regime_df = regime.classify_regime(
            self.data, self.config, lookback=self._regime_lookback
        )

    def _block_for_label(self, label: str) -> dict:
        """Pick the config category->weight block for a regime ``label``.

        risk_on / risk_off map directly to their blocks. Anything else
        (e.g. 'neutral', NaN, missing) falls back to the defensive risk_off
        block so the portfolio de-risks when the regime is undecided.
        """
        risk_on = self.params.get("risk_on", {}) or {}
        risk_off = self.params.get("risk_off", {}) or {}
        if label == "risk_on":
            return risk_on
        if label == "risk_off":
            return risk_off
        # neutral / unknown / warmup -> default to the defensive block.
        return risk_off

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        self.ensure_precomputed()

        # Most recent regime label with index <= date (look-ahead safe).
        label = None
        if self._regime_df is not None and "regime" in self._regime_df.columns:
            label = self.signal_at(self._regime_df["regime"], date)
        if not isinstance(label, str):
            label = "neutral"

        block = self._block_for_label(label)

        available = set(self.available_at(date))

        weights: dict[str, float] = {}
        for category, cat_weight in block.items():
            # 'cash' is an explicit hold-cash bucket: allocate nothing.
            if category == "cash":
                continue
            try:
                cw = float(cat_weight)
            except (TypeError, ValueError):
                continue
            if cw <= 0:
                continue
            members = [
                t for t in self.config.tradable_in_category(category)
                if t in self.tradable and t in available
            ]
            if not members:
                # Members not yet available -> that slice falls through to cash.
                continue
            per = cw / len(members)
            for t in members:
                weights[t] = weights.get(t, 0.0) + per

        if not weights:
            return pd.Series(dtype=float)

        w = pd.Series(weights, dtype=float)
        # Renormalise so the total never exceeds 1; remainder is cash.
        return self.normalize(w, cap_total=1.0)
