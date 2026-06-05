"""Strategy 2 -- Risk parity (inverse-volatility weighting).

Weight each tradable asset proportional to 1/realised_vol so every holding
contributes a similar share of portfolio risk. Optionally pre-filter assets with
negative trailing momentum (absolute_filter).

Params (strategies.risk_parity): vol_lookback, absolute_filter, momentum_lookback.

Implementation guidance (look-ahead safe):
* precompute(): vol panel via
  ``volatility.realized_vol(self.returns[self.tradable], vol_lookback, ppy)``;
  if absolute_filter, a momentum panel via ``momentum.total_return``.
* target_weights(date): inverse-vol of assets available_at(date) (and passing
  the momentum filter), normalised to sum to 1. ppy = 252 daily / 52 weekly
  (use len-based or self.data.freq).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, register
from ..analysis import volatility, momentum


@register("risk_parity")
class RiskParityStrategy(Strategy):
    label = "Risk Parity"

    def precompute(self) -> None:
        self.vol_lookback = int(self.params.get("vol_lookback", 63))
        self.absolute_filter = bool(self.params.get("absolute_filter", False))
        self.momentum_lookback = int(self.params.get("momentum_lookback", 126))
        # 'inverse_vol' (naive, ignores correlations) or 'erc' (equal risk
        # contribution -- uses the covariance, true risk parity).
        self.method = str(self.params.get("method", "inverse_vol")).lower()
        self.shrinkage = self.params.get("shrinkage", "ledoit_wolf")
        self.cov_min_obs = int(self.params.get("min_obs", max(20, self.vol_lookback // 2)))

        # periods per year for annualisation (matches panel frequency)
        ppy = 52 if getattr(self.data, "freq", "daily") == "weekly" else 252

        cols = self.tradable
        ret = self.returns[cols]

        # trailing realised volatility (look-ahead safe: rolling std of past returns)
        self.vol_panel = volatility.realized_vol(ret, self.vol_lookback, ppy)
        # zero/negative vol (e.g. zero-volume FX/^VIX flat segments) -> not usable
        self.vol_panel = self.vol_panel.replace([np.inf, -np.inf], np.nan)
        self.vol_panel = self.vol_panel.where(self.vol_panel > 0)

        if self.absolute_filter:
            self.mom_panel = momentum.total_return(
                self.prices[cols], self.momentum_lookback
            )
        else:
            self.mom_panel = None

        # largest lookback consumed before producing a meaningful weight
        if self.absolute_filter:
            self.warmup = max(self.vol_lookback, self.momentum_lookback)
        else:
            self.warmup = self.vol_lookback

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        vol_row = self.signal_at(self.vol_panel, date)
        if not isinstance(vol_row, pd.Series) or len(vol_row) == 0:
            return pd.Series(dtype=float)

        eligible = self.available_at(date)
        if not eligible:
            return pd.Series(dtype=float)

        vol = vol_row.reindex(eligible)
        # drop assets without a valid (positive, finite) vol estimate
        vol = vol.replace([np.inf, -np.inf], np.nan).dropna()
        vol = vol[vol > 0]
        if vol.empty:
            return pd.Series(dtype=float)

        if self.absolute_filter and self.mom_panel is not None:
            mom_row = self.signal_at(self.mom_panel, date)
            if isinstance(mom_row, pd.Series) and len(mom_row):
                passing = mom_row.reindex(vol.index)
                vol = vol[passing > 0]
            else:
                return pd.Series(dtype=float)
            if vol.empty:
                return pd.Series(dtype=float)

        inv_vol = 1.0 / vol
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
        if inv_vol.empty:
            return pd.Series(dtype=float)

        if self.method == "erc":
            # true equal-risk-contribution: needs the covariance, seeded by inv-vol
            from ._riskopt import trailing_cov, solve_erc
            keep, cov = trailing_cov(self.returns, date, list(inv_vol.index),
                                     self.vol_lookback, self.cov_min_obs, self.shrinkage)
            if cov is not None and len(keep) >= 2:
                seed = inv_vol.reindex(keep).to_numpy()
                w = solve_erc(cov, inv_vol_seed=seed)
                return pd.Series(w, index=keep)
            # fall through to inverse-vol if covariance unavailable

        return self.normalize(inv_vol)
