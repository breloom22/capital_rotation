"""Volatility indicators: realised vol, ATR, VIX-based regime.

LOOK-AHEAD INVARIANT: value at date ``t`` uses data with index <= ``t`` only.

PUBLIC API:
    realized_vol(returns, window=21, periods_per_year=252) -> DataFrame
    atr(high, low, close, window=14)                       -> DataFrame
    atr_pct(high, low, close, window=14)                   -> DataFrame
    vix_regime(vix, bins=(15,25,35))                       -> Series (categorical)
    vol_regime_label(vix_level, bins=(15,25,35))           -> str
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(returns: pd.DataFrame, window: int = 21, periods_per_year: int = 252) -> pd.DataFrame:
    """Annualised trailing realised volatility: rolling std of returns over
    ``window`` periods * sqrt(periods_per_year). Wide [date x ticker]."""
    min_periods = max(5, window // 2)
    std = returns.rolling(window, min_periods=min_periods).std(ddof=1)
    out = std * np.sqrt(periods_per_year)
    return out


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average True Range over ``window`` periods (Wilder/SMA acceptable). Wide."""
    prev_close = close.shift(1)
    hl = (high - low).abs()
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    # element-wise max across the three candidate ranges (NaN-aware: only the
    # very first bar has NaN prev_close, where TR falls back to high-low)
    true_range = pd.concat([hl, hc, lc]).groupby(level=0).max()
    # restore original index order/labels after groupby
    true_range = true_range.reindex(index=high.index, columns=high.columns)
    out = true_range.rolling(window, min_periods=max(1, window // 2)).mean()
    return out


def atr_pct(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """ATR expressed as a fraction of close (ATR / close)."""
    a = atr(high, low, close, window)
    out = a / close
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def vix_regime(vix: pd.Series, bins: tuple = (15, 25, 35)) -> pd.Series:
    """Map a VIX level series to a categorical regime per date:
    'low' (<15), 'normal' (15-25), 'high' (25-35), 'extreme' (35+)."""
    lo, mid, hi = bins
    labels = ["low", "normal", "high", "extreme"]
    # right-closed digitize: value < lo -> low, lo<=v<mid -> normal, etc.
    edges = np.array([lo, mid, hi], dtype=float)
    values = vix.to_numpy(dtype=float)
    idx = np.digitize(values, edges, right=False)
    out = np.array(labels, dtype=object)[idx]
    result = pd.Series(out, index=vix.index)
    # preserve NaN inputs as missing regimes (digitize maps NaN to the top bin)
    result[vix.isna()] = np.nan
    return result


def vol_regime_label(vix_level: float, bins: tuple = (15, 25, 35)) -> str:
    """Scalar version of :func:`vix_regime` for a single VIX value."""
    lo, mid, hi = bins
    if pd.isna(vix_level):
        return "normal"
    if vix_level < lo:
        return "low"
    if vix_level < mid:
        return "normal"
    if vix_level < hi:
        return "high"
    return "extreme"
