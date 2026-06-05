"""Price-momentum indicators.

LOOK-AHEAD INVARIANT (applies to every function here): a value indexed at date
``t`` may use prices/returns with index <= ``t`` only. Trailing rolling windows
satisfy this automatically; never use centred windows or `.shift(-k)`.

All wide DataFrames are [date x ticker]. Lookbacks are in periods (rows), i.e.
trading days for a daily panel.

PUBLIC API (stable -- strategies import these exact names/signatures):
    total_return(prices, lookback)                         -> DataFrame
    momentum_score(prices, lookback_weights, skip_recent)  -> DataFrame
    momentum_rank(scores, ascending=False)                 -> DataFrame
    absolute_momentum(prices, lookback)                    -> DataFrame (bool-ish)
    dual_momentum(prices, lookback)                        -> DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing total return over ``lookback`` periods ending at each date:
    ``prices_t / prices_{t-lookback} - 1``. NaN until enough history exists."""
    out = prices / prices.shift(lookback) - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def momentum_score(
    prices: pd.DataFrame,
    lookback_weights: dict[int, float] | None = None,
    skip_recent: int = 0,
) -> pd.DataFrame:
    """Composite momentum: weighted blend of trailing total returns over several
    lookbacks. ``lookback_weights`` maps lookback(periods) -> weight (auto-
    normalised). ``skip_recent`` optionally excludes the most recent N periods
    (reversal filter) by measuring return from ``t-lookback`` to ``t-skip_recent``.
    Returns a wide [date x ticker] score panel."""
    if not lookback_weights:
        lookback_weights = {252: 1.0}

    total_weight = float(sum(lookback_weights.values()))
    score = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    valid = pd.DataFrame(False, index=prices.index, columns=prices.columns)

    for lookback, weight in lookback_weights.items():
        if total_weight != 0.0:
            w = weight / total_weight
        else:
            w = 0.0
        # return from t-lookback to t-skip_recent (trailing only: past data only)
        leg = prices.shift(skip_recent) / prices.shift(lookback) - 1.0
        leg = leg.replace([np.inf, -np.inf], np.nan)
        score = score.add(leg * w, fill_value=0.0)
        valid = valid | leg.notna()

    # where no leg produced a value, result must be NaN (pre-inception/no history)
    return score.where(valid)


def momentum_rank(scores: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """Cross-sectional rank of each asset per date (1 = best when ascending=False)."""
    return scores.rank(axis=1, ascending=ascending, method="min")


def absolute_momentum(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Absolute (time-series) momentum: trailing return over ``lookback`` > 0.
    Returns a float panel of the trailing return (callers threshold at 0)."""
    return total_return(prices, lookback)


def dual_momentum(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Dual momentum signal: relative rank combined with the absolute filter.
    Returns a panel where assets failing absolute momentum (<=0) are NaN and the
    rest carry their cross-sectional rank."""
    tr = total_return(prices, lookback)
    rank = momentum_rank(tr, ascending=False)
    # keep rank only where the absolute (trailing) return is strictly positive
    return rank.where(tr > 0)
