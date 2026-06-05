"""Market-regime classification: Risk-On vs Risk-Off.

Uses the asset ROLES defined in ``config/assets.yaml`` (risk_on, risk_off, vix,
dollar). Rule-based and look-ahead safe: the label at date ``t`` is built only
from trailing signals (momentum of risk-on vs risk-off sleeves, VIX level/trend)
available at ``t``.

PUBLIC API (regime_based strategy depends on `classify_regime`):
    classify_regime(data, config, lookback=63, vix_threshold=20) -> DataFrame
        columns at least: ['regime', 'score']  (regime in
        {'risk_on','risk_off','neutral'}), indexed by date.
    regime_at(regime_df, date) -> str
    risk_on_off_score(data, config, lookback=63) -> Series  (continuous, >0 risk-on)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# NOTE: import the data/config types lazily inside functions if needed to avoid
# import cycles. `data` is a src.data.MarketData; `config` is a src.config.Config.


def _present(tickers, columns) -> list:
    """Filter a list of role tickers down to those present in the panel."""
    cols = set(columns)
    return [t for t in (tickers or []) if t in cols]


def _sleeve_mean_return(returns: pd.DataFrame, tickers: list, lookback: int) -> pd.Series:
    """Average (across tickers) of the trailing-mean daily return per date.

    Trailing rolling mean over ``lookback`` of each ticker's daily return, then
    averaged across the sleeve's tickers. Returns a Series indexed by date; all
    operations are trailing so the result is look-ahead safe. If no tickers are
    present, returns a zero Series on the panel index.
    """
    if not tickers:
        return pd.Series(0.0, index=returns.index)
    # min_periods=lookback so a value only appears once a full trailing window
    # exists -> independent of where the panel is truncated (look-ahead safe).
    roll = returns[tickers].rolling(lookback, min_periods=lookback).mean()
    # mean across tickers, skipping NaN (assets not yet incepted / short history)
    return roll.mean(axis=1, skipna=True)


def risk_on_off_score(data, config, lookback: int = 63) -> pd.Series:
    """Continuous risk appetite score per date. Positive => risk-on.

    Construction (look-ahead safe, all trailing):
      + average trailing daily return of role 'risk_on' tickers
      - average trailing daily return of role 'risk_off' tickers
      - VIX deviation above its trailing mean (normalised by trailing std)
    Returns a Series indexed by date."""
    roles = config.roles
    returns = data.returns
    idx = returns.index

    risk_on = _present(roles.get("risk_on", []), returns.columns)
    risk_off = _present(roles.get("risk_off", []), returns.columns)

    on_score = _sleeve_mean_return(returns, risk_on, lookback)
    off_score = _sleeve_mean_return(returns, risk_off, lookback)

    score = on_score.subtract(off_score, fill_value=0.0)

    # VIX deviation: (vix - trailing mean) / trailing std, all trailing windows.
    vix_ticker = roles.get("vix")
    vix_dev = pd.Series(0.0, index=idx)
    if vix_ticker and vix_ticker in data.prices.columns:
        vix = data.prices[vix_ticker]
        vmean = vix.rolling(lookback, min_periods=lookback).mean()
        vstd = vix.rolling(lookback, min_periods=lookback).std()
        # avoid division by zero -> inf; replace zero std with NaN then fill 0.
        vstd = vstd.replace(0.0, np.nan)
        vix_dev = (vix - vmean) / vstd
        vix_dev = vix_dev.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # higher VIX deviation => more risk-off => subtract it from the score.
    score = score.subtract(vix_dev, fill_value=0.0)
    score = score.replace([np.inf, -np.inf], np.nan)
    score.name = "score"
    return score.reindex(idx)


def classify_regime(data, config, lookback: int = 63, vix_threshold: float = 20.0) -> pd.DataFrame:
    """Per-date regime label. Returns a DataFrame indexed by date with columns
    ['regime', 'score'] where regime in {'risk_on','risk_off','neutral'}.

    The strategy reads this via ``signal_at(regime_df['regime'], date)`` so row t
    MUST be computable from data <= t."""
    score = risk_on_off_score(data, config, lookback)
    idx = score.index

    roles = config.roles
    vix_ticker = roles.get("vix")
    if vix_ticker and vix_ticker in data.prices.columns:
        vix = data.prices[vix_ticker].reindex(idx)
    else:
        vix = pd.Series(np.nan, index=idx)

    s = score.to_numpy(dtype=float)
    v = vix.to_numpy(dtype=float)

    have = ~np.isnan(s)
    # An elevated-VIX flag; when VIX is unavailable (all NaN) treat as not elevated
    # so classification still rests on the score sign.
    elevated = np.where(np.isnan(v), False, v > vix_threshold)

    # risk_off: negative score AND elevated volatility (fear confirmed).
    # risk_on : positive score.
    # neutral : everything else (incl. warmup rows with no score yet).
    conditions = [
        have & (s < 0.0) & elevated,
        have & (s > 0.0),
    ]
    choices = ["risk_off", "risk_on"]
    regime = np.select(conditions, choices, default="neutral")

    out = pd.DataFrame({"regime": regime, "score": score.to_numpy()}, index=idx)
    out.index.name = data.prices.index.name
    return out


def regime_at(regime_df: pd.DataFrame, date) -> str:
    """Most recent regime label at or before ``date`` ('neutral' if none yet)."""
    if regime_df is None or "regime" not in regime_df.columns or regime_df.empty:
        return "neutral"
    ts = pd.Timestamp(date)
    prior = regime_df.loc[:ts]
    if prior.empty:
        return "neutral"
    label = prior["regime"].iloc[-1]
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return "neutral"
    return str(label)
