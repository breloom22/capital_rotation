"""Correlation-regime analysis: rolling correlation, change speed, PCA.

LOOK-AHEAD INVARIANT: value at date ``t`` uses returns with index <= ``t`` only.

PUBLIC API:
    correlation_matrix(returns, window=120, as_of=None)      -> DataFrame (NxN)
    rolling_avg_correlation(returns, window=120)             -> DataFrame [date x ticker]
    correlation_change(returns, window=60)                   -> Series (market-level)
    pca_explained(returns, window=252, n_components=3, as_of=None) -> dict
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _nanmean_safe(arr, axis=None):
    """np.nanmean that returns NaN (not a warning) for all-NaN slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, axis=axis)


def correlation_matrix(returns: pd.DataFrame, window: int = 120, as_of=None) -> pd.DataFrame:
    """Pairwise correlation matrix over the trailing ``window`` periods ending at
    ``as_of`` (default: last date). Square DataFrame indexed/columned by ticker."""
    tickers = list(returns.columns)
    if returns.empty:
        return pd.DataFrame(index=tickers, columns=tickers, dtype=float)

    # resolve the trailing-window end date (last valid date by default)
    if as_of is None:
        end = returns.index[-1]
    else:
        end = pd.Timestamp(as_of)

    # use only data with index <= end (look-ahead safe), then take the trailing
    # `window` rows ending at `end`.
    win = returns.loc[:end]
    if window is not None and window > 0:
        win = win.iloc[-window:]

    cm = win.corr()
    # ensure a stable, square layout in the original ticker order
    cm = cm.reindex(index=tickers, columns=tickers)
    return cm


def rolling_avg_correlation(returns: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """For each date and asset, the mean pairwise correlation of that asset to
    all others over the trailing ``window``. Lower => better diversifier.
    Wide [date x ticker]. Computed look-ahead-safe (trailing windows only)."""
    tickers = list(returns.columns)
    out = pd.DataFrame(np.nan, index=returns.index, columns=tickers, dtype=float)
    n = len(returns)
    if n == 0 or window is None or window <= 0:
        return out

    values = returns.to_numpy(dtype=float)

    for i in range(window - 1, n):
        # trailing window of `window` rows ending at row i (uses index <= i only)
        block = values[i - window + 1 : i + 1, :]
        df = pd.DataFrame(block, columns=tickers)
        cm = np.array(df.corr().to_numpy(), dtype=float)  # k x k, writable copy
        # exclude self by masking the diagonal, then average the off-diagonal
        np.fill_diagonal(cm, np.nan)
        row_mean = _nanmean_safe(cm, axis=1)
        out.iloc[i, :] = row_mean

    # NaN where insufficient history or undefined correlation; never inf
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def correlation_change(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Speed of correlation change: per-date average pairwise correlation, then
    its first difference (how fast the market is synchronising). Series by date."""
    n = len(returns)
    avg = pd.Series(np.nan, index=returns.index, dtype=float)
    if n == 0 or window is None or window <= 0:
        return avg.diff()

    values = returns.to_numpy(dtype=float)
    cols = list(returns.columns)

    for i in range(window - 1, n):
        block = values[i - window + 1 : i + 1, :]
        df = pd.DataFrame(block, columns=cols)
        cm = np.array(df.corr().to_numpy(), dtype=float)  # writable copy
        # average of the off-diagonal entries (each pair counted; symmetric)
        np.fill_diagonal(cm, np.nan)
        avg.iloc[i] = _nanmean_safe(cm)

    avg = avg.replace([np.inf, -np.inf], np.nan)
    change = avg.diff()
    return change


def pca_explained(returns: pd.DataFrame, window: int = 252, n_components: int = 3, as_of=None) -> dict:
    """PCA on the trailing-``window`` return covariance ending at ``as_of``.
    Returns {'explained_variance_ratio': np.ndarray, 'n_obs': int, 'as_of': date,
    'tickers': [...]}. Used by the `analyze correlation` command."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if returns.empty:
        return {
            "explained_variance_ratio": np.array([], dtype=float),
            "n_obs": 0,
            "as_of": None,
            "tickers": [],
        }

    if as_of is None:
        end = returns.index[-1]
    else:
        end = pd.Timestamp(as_of)

    win = returns.loc[:end]
    if window is not None and window > 0:
        win = win.iloc[-window:]

    resolved_as_of = win.index[-1] if len(win) else None

    # drop columns that are entirely NaN over the window, then fill remaining
    # NaN with 0 (a neutral, mean-like value after standardisation)
    win = win.dropna(axis=1, how="all")
    tickers = list(win.columns)

    if win.shape[0] < 2 or win.shape[1] < 1:
        return {
            "explained_variance_ratio": np.array([], dtype=float),
            "n_obs": int(win.shape[0]),
            "as_of": resolved_as_of,
            "tickers": tickers,
        }

    X = win.to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)

    # standardise (zero-variance columns -> StandardScaler leaves them at 0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs = np.where(np.isfinite(Xs), Xs, 0.0)

    k = Xs.shape[1]
    n_comp = max(1, min(int(n_components), k, Xs.shape[0]))
    pca = PCA(n_components=n_comp)
    pca.fit(Xs)

    return {
        "explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=float),
        "n_obs": int(Xs.shape[0]),
        "as_of": resolved_as_of,
        "tickers": tickers,
    }
