"""Covariance estimation + portfolio optimisers shared by the risk strategies.

Adapted from the regime-portfolio project's ``backtest/minvar.py``:
* Ledoit-Wolf shrinkage covariance (essential -- a ~20-asset / 120-day window
  has ~200 covariance entries from 120 obs, so the sample covariance is noisy
  and a bare min-variance solver over-bets on spurious low correlations).
* Long-only minimum-variance via SLSQP with a closed-form warm start and a
  fallback ladder (SLSQP -> projection -> inverse-variance -> equal weight).
* Equal-Risk-Contribution (true risk parity) via SLSQP.

LOOK-AHEAD: callers pass a TRAILING return block ending at the decision date, so
every covariance uses only data <= date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.covariance import LedoitWolf
except Exception:  # pragma: no cover
    LedoitWolf = None

try:
    from scipy.optimize import minimize as _minimize
except Exception:  # pragma: no cover
    _minimize = None


def shrunk_cov(block: np.ndarray, shrinkage: str = "ledoit_wolf") -> np.ndarray:
    """Covariance of an (n_obs x m) return block. Ledoit-Wolf if requested and
    available, else sample covariance; a tiny ridge keeps it positive-definite."""
    m = block.shape[1]
    if str(shrinkage) == "ledoit_wolf" and LedoitWolf is not None and block.shape[0] >= 2:
        try:
            cov = LedoitWolf(assume_centered=False).fit(block).covariance_
        except Exception:
            cov = np.cov(block, rowvar=False)
    else:
        cov = np.cov(block, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    return cov + np.eye(m) * 1e-10


def trailing_cov(returns: pd.DataFrame, date, tickers: list[str], window: int,
                 min_obs: int, shrinkage: str = "ledoit_wolf"):
    """Build the trailing covariance ending at ``date`` over ``tickers``.

    Returns ``(kept_tickers, cov)``. Keeps only tickers with >= ``min_obs`` real
    observations in the window; ``cov`` is None if fewer than 2 qualify."""
    block = returns.loc[:date, tickers]
    if window and window > 0:
        block = block.iloc[-window:]
    counts = block.notna().sum()
    keep = [t for t in tickers if counts.get(t, 0) >= min_obs]
    if len(keep) < 2:
        return keep, None
    sub = np.nan_to_num(block[keep].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return keep, shrunk_cov(sub, shrinkage)


def _finalize(w: np.ndarray, lo: float) -> np.ndarray:
    """Project to sum=1, each >= lo (lift sub-floor, shave proportional slack)."""
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    m = w.size
    if m == 0:
        return w
    if m * lo >= 1.0:
        return np.full(m, 1.0 / m)
    s = float(w.sum())
    w = w / s if s > 1e-15 else np.full(m, 1.0 / m)
    for _ in range(m + 1):
        below = w < lo - 1e-6
        if not below.any():
            break
        deficit = float((lo - w[below]).sum())
        w[below] = lo
        above = w > lo + 1e-12
        room = float((w[above] - lo).sum())
        if (not above.any()) or room <= deficit + 1e-15:
            return np.full(m, 1.0 / m)
        w[above] -= deficit * (w[above] - lo) / room
    t = float(w.sum())
    return w / t if t > 1e-15 else np.full(m, 1.0 / m)


def solve_min_variance(cov: np.ndarray, min_weight: float = 0.0) -> np.ndarray:
    """Long-only minimum variance: min w'Cov w  s.t. sum w = 1, w in [lo, 1]."""
    m = cov.shape[0]
    lo = float(min_weight)
    if m * lo > 1.0:
        return np.full(m, 1.0 / m)
    # warm start: analytic (unconstrained) min-var, clipped
    try:
        inv = np.linalg.pinv(cov)
        ones = np.ones(m)
        w0 = inv @ ones
        s = float(ones @ inv @ ones)
        w0 = w0 / s if abs(s) > 1e-15 else np.full(m, 1.0 / m)
        w0 = np.clip(w0, lo, 1.0)
        w0 = w0 / w0.sum()
    except Exception:
        w0 = np.full(m, 1.0 / m)

    if _minimize is None:
        iv = 1.0 / np.clip(np.diag(cov), 1e-12, None)
        return _finalize(iv / iv.sum(), lo)

    cons = ({"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)},)
    bounds = [(lo, 1.0)] * m
    try:
        res = _minimize(lambda x: float(x @ cov @ x), w0, method="SLSQP",
                        jac=lambda x: 2.0 * (cov @ x), bounds=bounds, constraints=cons,
                        options={"maxiter": 200, "ftol": 1e-12})
        if res.success and np.all(np.isfinite(res.x)):
            x = np.clip(res.x, lo, 1.0)
            return x / x.sum()
    except Exception:
        pass
    return _finalize(w0, lo)


def solve_erc(cov: np.ndarray, inv_vol_seed: np.ndarray | None = None) -> np.ndarray:
    """Equal-Risk-Contribution (true risk parity): each asset contributes the
    same share of portfolio variance, RC_i = w_i (Cov w)_i. Minimise the spread
    of risk contributions s.t. sum w = 1, w >= 0."""
    m = cov.shape[0]
    if m == 1:
        return np.array([1.0])
    if inv_vol_seed is not None and np.all(np.isfinite(inv_vol_seed)) and inv_vol_seed.sum() > 0:
        w0 = inv_vol_seed / inv_vol_seed.sum()
    else:
        iv = 1.0 / np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        w0 = iv / iv.sum()

    if _minimize is None:
        return w0

    def obj(w):
        rc = w * (cov @ w)
        return float(np.sum((rc - rc.mean()) ** 2))

    cons = ({"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)},)
    bounds = [(1e-6, 1.0)] * m
    try:
        res = _minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 300, "ftol": 1e-14})
        if res.success and np.all(np.isfinite(res.x)) and res.x.sum() > 0:
            x = np.clip(res.x, 0.0, 1.0)
            return x / x.sum()
    except Exception:
        pass
    return w0
