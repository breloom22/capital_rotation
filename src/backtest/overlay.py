"""Config-driven risk overlay applied to every strategy's target weights.

Two interpretable, strategy-agnostic risk controls ported (as backtest signals,
not live plumbing) from the regime-portfolio project's ``production/risk.py``:

* **Position caps** -- clip each weight to ``max_single_weight`` and scale the
  basket down if gross exposure exceeds ``max_gross`` (excess -> cash). Prevents
  a single momentum-hot asset from dominating.
* **Correlation-spike de-risking** -- when the average pairwise correlation of
  the tradable universe spikes above ``threshold`` (crash synchronisation, when
  diversification vanishes), scale all weights by ``override_budget`` (rest ->
  cash).

OPT-IN: disabled by default (``risk_overlay.enabled: false``) so baseline
results are unchanged. When enabled it applies uniformly to all strategies,
including benchmarks. Look-ahead safe: the correlation level at ``date`` comes
from a trailing-window panel (rolling_avg_correlation), read via the last row
<= date.
"""
from __future__ import annotations

import pandas as pd


class RiskOverlay:
    def __init__(self, data, config):
        cfg = config.backtest_cfg.get("risk_overlay", {}) or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.max_single = float(cfg.get("max_single_weight", 1.0))
        self.max_gross = float(cfg.get("max_gross", 1.0))
        corr = cfg.get("corr_spike", {}) or {}
        self.corr_enabled = bool(corr.get("enabled", False)) and self.enabled
        self.corr_threshold = float(corr.get("threshold", 0.85))
        self.corr_budget = float(corr.get("override_budget", 0.5))
        self.corr_window = int(corr.get("window", 120))
        self._avg_corr = None

        if self.corr_enabled:
            # precompute the daily average pairwise correlation of the tradable
            # universe ONCE (trailing windows -> look-ahead safe).
            from ..analysis import correlation
            tr = data.tradable_present()
            if len(tr) >= 2:
                rac = correlation.rolling_avg_correlation(data.returns[tr], self.corr_window)
                self._avg_corr = rac.mean(axis=1)

    def apply(self, weights: pd.Series, date) -> pd.Series:
        if not self.enabled or weights is None or len(weights) == 0:
            return weights
        w = weights.copy()

        # 1) per-position cap, then gross cap
        if self.max_single < 1.0:
            w = w.clip(upper=self.max_single)
        gross = w.sum()
        if gross > self.max_gross and gross > 0:
            w = w * (self.max_gross / gross)

        # 2) correlation-spike de-risking
        if self.corr_enabled and self._avg_corr is not None:
            sub = self._avg_corr.loc[:date]
            if len(sub):
                ac = sub.iloc[-1]
                if pd.notna(ac) and ac > self.corr_threshold:
                    w = w * self.corr_budget
        return w
