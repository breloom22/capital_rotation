"""Volume dynamics: liquidity, OBV and money-flow proxies.

LOOK-AHEAD INVARIANT: value at date ``t`` uses data with index <= ``t`` only.
All wide DataFrames are [date x ticker].

PUBLIC API:
    volume_ratio(volume, window=20)                 -> DataFrame
    dollar_volume_ratio(dollar_volume, window=20)   -> DataFrame
    volume_trend(volume, window=20)                 -> DataFrame
    obv(close, volume)                              -> DataFrame
    money_flow(close, volume, window=20)            -> DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ratio(numer: pd.DataFrame, window: int) -> pd.DataFrame:
    """Helper: value / trailing-``window`` mean, inf-safe."""
    min_p = max(2, window // 2)
    mean = numer.rolling(window, min_periods=min_p).mean()
    out = numer / mean
    # zero-mean (zero-volume) assets create 0/0 -> NaN and x/0 -> inf
    return out.replace([np.inf, -np.inf], np.nan)


def volume_ratio(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Current volume / trailing ``window``-period mean volume (shifted so the
    mean excludes the current bar is NOT required; trailing mean including t is
    fine as it uses only data <= t). Returns a wide ratio panel."""
    return _ratio(volume, window)


def dollar_volume_ratio(dollar_volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Same as :func:`volume_ratio` but on dollar volume (Close*Volume)."""
    return _ratio(dollar_volume, window)


def volume_trend(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Slope of the trailing ``window``-period moving average of volume
    (normalised, e.g. pct change of the MA). Positive => rising participation."""
    min_p = max(2, window // 2)
    ma = volume.rolling(window, min_periods=min_p).mean()
    out = ma.pct_change()
    # zero-volume assets -> ma all 0 -> 0/0 = NaN, x/0 = inf
    return out.replace([np.inf, -np.inf], np.nan)


def obv(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume: cumulative signed volume where the sign follows the
    daily close change. Wide [date x ticker]."""
    # sign of the daily close change; first row (and pre-inception gaps) -> 0
    sign = np.sign(close.diff())
    signed = sign * volume
    # drop NaN contributions (first bar, pre-inception) so cumsum is well-defined
    # and identical whether or not future rows exist (trailing-only).
    out = signed.fillna(0.0).cumsum()
    return out


def money_flow(close: pd.DataFrame, volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Inflow/outflow proxy over a trailing window: sum of signed dollar volume
    (sign = sign of close change) normalised by total dollar volume in the
    window. Range ~[-1, 1]; >0 => net accumulation."""
    min_p = max(2, window // 2)
    dollar = close * volume
    signed = np.sign(close.diff()) * dollar
    num = signed.rolling(window, min_periods=min_p).sum()
    den = dollar.rolling(window, min_periods=min_p).sum()
    out = num / den
    # zero-volume assets -> den == 0 -> inf/NaN; clean it up
    return out.replace([np.inf, -np.inf], np.nan)
