"""Strategy 3 -- Composite multi-factor scoring.

Blend several signals into one score, then hold the top-N by score (equal- or
score-proportional weight). Signals (each cross-sectionally z-scored per date so
they are comparable, higher = better):
    momentum    : composite momentum (higher better)
    volume      : volume / dollar-volume ratio (higher better -> participation)
    volatility  : realised vol (LOWER better -> invert)
    correlation : avg pairwise correlation (LOWER better -> invert, diversifier)

Params (strategies.momentum_score): n, weights{momentum,volume,volatility,
correlation}, vol_lookback, corr_lookback, volume_lookback, absolute_filter.

Implementation guidance (look-ahead safe):
* precompute(): build each signal panel from analysis modules, cross-sectionally
  z-score per row, combine with the configured weights into one score panel.
  Invert volatility & correlation before combining.
* target_weights(date): signal_at the composite, restrict to available_at(date),
  apply absolute momentum filter if set, take top-N, weight (equal or by positive
  score). Remainder = cash.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, register
from ..analysis import momentum, volume, volatility, correlation


def _cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row across assets: (x - row mean) / row std.

    Uses only data within the same row (date), so it is look-ahead safe. Where a
    row has zero dispersion (all present values equal) the z-score collapses to 0
    (neutral) rather than inf/NaN. Genuinely missing cells stay NaN.
    """
    mean = panel.mean(axis=1)
    std = panel.std(axis=1, ddof=0)
    # divide only where std is strictly positive; elsewhere the centred value is
    # 0 for any present cell (all equal) -> neutral score of 0.
    safe_std = std.where(std > 0)
    z = panel.sub(mean, axis=0).div(safe_std, axis=0)
    z = z.replace([np.inf, -np.inf], np.nan)
    # present cells in a zero-dispersion row -> 0; truly missing cells stay NaN
    present = panel.notna()
    z = z.where(present, np.nan)
    z = z.mask(present & z.isna(), 0.0)
    return z


@register("momentum_score")
class CompositeScoreStrategy(Strategy):
    label = "Composite Score"

    def precompute(self) -> None:
        params = self.params
        tradable = self.tradable

        # --- signal weights (auto-normalised) ------------------------------
        raw_weights = dict(params.get("weights", {}) or {})
        default_weights = {
            "momentum": 0.40,
            "volume": 0.20,
            "volatility": 0.20,
            "correlation": 0.20,
        }
        weights = {k: float(raw_weights.get(k, default_weights[k])) for k in default_weights}
        wsum = float(sum(abs(v) for v in weights.values()))
        if wsum <= 0:
            weights = dict(default_weights)
            wsum = float(sum(weights.values()))
        self._weights = {k: v / wsum for k, v in weights.items()}

        # --- lookbacks -----------------------------------------------------
        self._n = int(params.get("n", 8))
        vol_lookback = int(params.get("vol_lookback", 63))
        corr_lookback = int(params.get("corr_lookback", 120))
        volume_lookback = int(params.get("volume_lookback", 20))
        self._absolute_filter = bool(params.get("absolute_filter", True))

        # momentum lookbacks inherit the global momentum block
        mom_cfg = self.config.momentum_cfg
        lookback_weights = mom_cfg.get("lookback_weights") or {252: 0.5, 126: 0.3, 63: 0.2}
        # YAML keys may arrive as strings; coerce to int periods
        lookback_weights = {int(k): float(v) for k, v in lookback_weights.items()}
        skip_recent = int(mom_cfg.get("skip_recent_days", 0) or 0)

        # periods per year for annualised vol (daily=252, weekly=52)
        ppy = 52 if getattr(self.data, "freq", "daily") == "weekly" else 252

        # --- restrict to tradable columns ----------------------------------
        prices = self.prices[tradable]
        returns = self.returns[tradable]
        # dollar volume / volume restricted to tradable
        dollar_volume = self.data.dollar_volume[tradable]

        # --- raw signal panels (all trailing-window, look-ahead safe) ------
        mom_panel = momentum.momentum_score(prices, lookback_weights, skip_recent)
        vol_panel = volume.dollar_volume_ratio(dollar_volume, volume_lookback)
        vola_panel = volatility.realized_vol(returns, vol_lookback, ppy)
        corr_panel = correlation.rolling_avg_correlation(returns, corr_lookback)

        # invert "lower is better" signals so higher = better everywhere
        vola_panel = -vola_panel
        corr_panel = -corr_panel

        # --- cross-sectional z-score each panel per row --------------------
        z_mom = _cross_sectional_zscore(mom_panel)
        z_vol = _cross_sectional_zscore(vol_panel)
        z_vola = _cross_sectional_zscore(vola_panel)
        z_corr = _cross_sectional_zscore(corr_panel)

        # --- combine into one composite panel ------------------------------
        composite = (
            self._weights["momentum"] * z_mom
            + self._weights["volume"] * z_vol
            + self._weights["volatility"] * z_vola
            + self._weights["correlation"] * z_corr
        )
        composite = composite.replace([np.inf, -np.inf], np.nan)
        self._composite = composite

        # absolute-momentum filter panel: keep only assets with positive
        # trailing return over the longest momentum lookback
        if self._absolute_filter:
            abs_lookback = max(lookback_weights.keys()) if lookback_weights else 252
            self._abs_mom = momentum.absolute_momentum(prices, int(abs_lookback))
        else:
            self._abs_mom = None

        # warmup = largest lookback used anywhere
        self.warmup = max(
            max(lookback_weights.keys()) if lookback_weights else 252,
            vol_lookback,
            corr_lookback,
            volume_lookback,
            252,
        )

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        self.ensure_precomputed()

        scores = self.signal_at(self._composite, date)
        if not isinstance(scores, pd.Series) or scores.empty:
            return pd.Series(dtype=float)

        # restrict to tradable assets that actually have a price at `date`
        available = self.available_at(date)
        scores = scores.reindex(available).dropna()
        if scores.empty:
            return pd.Series(dtype=float)

        # absolute momentum filter: drop assets whose trailing return <= 0
        if self._abs_mom is not None:
            abs_row = self.signal_at(self._abs_mom, date)
            if isinstance(abs_row, pd.Series):
                ok = abs_row.reindex(scores.index)
                scores = scores[ok > 0]
            if scores.empty:
                return pd.Series(dtype=float)

        # top-N by composite score
        n = max(1, self._n)
        top = scores.sort_values(ascending=False).head(n)

        # weight by positive composite score; fall back to equal-weight when
        # no selected asset has a positive score
        pos = top[top > 0]
        if len(pos) > 0 and pos.sum() > 0:
            weights = pos / pos.sum()
        else:
            weights = pd.Series(1.0 / len(top), index=top.index)

        return self.normalize(weights, cap_total=1.0)
