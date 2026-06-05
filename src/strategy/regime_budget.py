"""Strategy 6 -- Regime-conditioned risk budget (ported mechanism).

Adapted from the regime-portfolio project's signature rebalancing idea:

    final_weights = risk_budget(regime) x base_weights ,   rest -> cash

A *base* strategy decides WHICH assets to hold (security selection); a scalar
risk budget driven by the market regime decides HOW MUCH gross exposure to take
(market timing). Dial exposure toward cash in Risk-Off, full in Risk-On. This is
distinct from ``regime_based`` (which switches the category MIX) -- here we keep
the base portfolio and scale its size.

Drives off the existing rule-based ``analysis.regime`` (no HMM dependency). The
source's hard-won lesson -- tune the budget to each regime's MEASURED return,
not its name -- is honoured because our regime is constructed directionally
(risk_on = risk-on sleeve outperforming), and budgets stay tunable + OOS-checked.

Params (config/strategy.yaml -> strategies.regime_budget):
    base            : registry name of the base strategy (e.g. 'topn', 'min_variance')
    budgets         : {risk_on, neutral, risk_off} -> gross exposure in [0,1]
    mode            : 'discrete' (regime label) | 'continuous' (smooth from score)
    min_budget      : floor on the continuous budget
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, register, get_strategy


@register("regime_budget")
class RegimeBudgetStrategy(Strategy):
    label = "Regime-Budget"

    def precompute(self) -> None:
        self.base_name = self.params.get("base", "topn")
        self.budgets = {k: float(v) for k, v in (self.params.get("budgets") or {
            "risk_on": 1.0, "neutral": 0.6, "risk_off": 0.3}).items()}
        self.mode = str(self.params.get("mode", "discrete")).lower()
        self.min_budget = float(self.params.get("min_budget", 0.0))
        self.regime_source = str(self.params.get("regime_source", "rule")).lower()

        # build + precompute the base strategy (shares data/config). base_params
        # override lets us configure an edge base, e.g. Top-N with the cash filter
        # OFF so it stays fully invested and the regime budget adds the timing.
        base_params = {**self.config.strategy_params(self.base_name),
                       **(self.params.get("base_params") or {})}
        self.base = get_strategy(self.base_name)(self.data, self.config, base_params)
        self.base.ensure_precomputed()

        if self.regime_source == "hmm":
            from ..analysis import regime_hmm
            self.regime_df = regime_hmm.classify_regime_hmm(self.data, self.config)
        else:
            from ..analysis import regime
            self.regime_df = regime.classify_regime(self.data, self.config)

        self.warmup = max(getattr(self.base, "warmup", 252), 126)
        tag = "/HMM" if self.regime_source == "hmm" else ""
        self.label = f"Regime-Budget ({self.base.display_name()}{tag})"

    def _budget(self, date: pd.Timestamp) -> float:
        if self.mode == "continuous":
            score = self.signal_at(self.regime_df["score"], date)
            if not np.isfinite(score):
                return self.budgets.get("neutral", 0.6)
            lo = self.budgets.get("risk_off", 0.3)
            hi = self.budgets.get("risk_on", 1.0)
            # smooth map: score ~0 -> midpoint, large +/- -> hi/lo
            b = lo + (hi - lo) * (0.5 + 0.5 * np.tanh(float(score)))
            return float(np.clip(max(b, self.min_budget), 0.0, 1.0))
        # discrete: regime label -> budget
        label = self.signal_at(self.regime_df["regime"], date)
        label = label if isinstance(label, str) else "neutral"
        return float(np.clip(self.budgets.get(label, self.budgets.get("neutral", 0.6)), 0.0, 1.0))

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        base_w = self.base.target_weights(date)
        if not isinstance(base_w, pd.Series) or len(base_w) == 0:
            return pd.Series(dtype=float)
        return base_w * self._budget(date)
