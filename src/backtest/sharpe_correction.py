"""Multiple-testing corrections for Sharpe ratios.

When you backtest many configurations (here 7 strategies x 4 rebalance freqs = 28
trials) and report the best Sharpe, that maximum is upward-biased -- the winner's
curse / data-snooping. Two complementary corrections:

1. **James-Stein shrinkage** (ported from the regime-portfolio project, after the
   "Post-Selection Estimation of Sharpe Ratios" idea): shrink every trial Sharpe
   toward the grand mean. The robustness question it answers is *does the winner
   stay the winner?* -- a ranking check, not a significance test.

2. **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014): the probability the
   strategy's true Sharpe is positive AFTER accounting for the number of trials
   and the non-normality of returns. The significance answer to "we searched N
   configs and picked the max."

Both are post-hoc statistics on completed equity curves -- no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm as _norm
except Exception:  # pragma: no cover
    _norm = None

_EULER = 0.5772156649015329  # Euler-Mascheroni constant


# --------------------------------------------------------------------------
# James-Stein shrinkage  (ported)
# --------------------------------------------------------------------------
def james_stein_shrinkage(sharpe_ratios: np.ndarray, sigma2: float | np.ndarray = 1.0) -> np.ndarray:
    """Shrink a vector of Sharpe ratios toward their grand mean.

        SR_JS_i = SR_mean + c * (SR_i - SR_mean)
        c = 1 - (k-2) * sigma2 / sum_i (SR_i - SR_mean)^2 ,  clipped to [0, 1]

    Returns the input unchanged if k < 3 or the spread is zero. NaN-safe (stats
    use finite entries only)."""
    sr = np.asarray(sharpe_ratios, dtype=np.float64).ravel()
    if sr.size < 3:
        return sr.copy()
    finite = sr[np.isfinite(sr)]
    if finite.size < 3:
        return sr.copy()
    sr_mean = float(np.mean(finite))
    ss = float(np.sum((finite - sr_mean) ** 2))
    if not np.isfinite(ss) or ss <= 0.0:
        return sr.copy()
    s2 = float(np.mean(np.asarray(sigma2, dtype=np.float64)))
    c = 1.0 - (finite.size - 2) * s2 / ss
    c = min(1.0, max(0.0, c))
    return sr_mean + c * (sr - sr_mean)


def sharpe_sampling_variance(sharpe: float, n_obs: int) -> float:
    """Lo (2002) sampling variance of a Sharpe estimate (iid):
    ``Var(SR_hat) ~= (1 + SR^2 / 2) / n_obs``. NaN if n_obs <= 1."""
    if n_obs is None or n_obs <= 1 or not np.isfinite(sharpe):
        return float("nan")
    return float((1.0 + 0.5 * sharpe ** 2) / n_obs)


# --------------------------------------------------------------------------
# Deflated Sharpe Ratio  (Bailey & Lopez de Prado)
# --------------------------------------------------------------------------
def _phi(x: float) -> float:
    if _norm is not None:
        return float(_norm.cdf(x))
    return float(0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0))))


def _phi_inv(p: float) -> float:
    p = min(1 - 1e-12, max(1e-12, p))
    if _norm is not None:
        return float(_norm.ppf(p))
    # rational approximation fallback (Acklam) -- only if scipy missing
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = np.sqrt(-2*np.log(1-p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def expected_max_sharpe(n_trials: int, var_trial_sharpe: float) -> float:
    """Expected maximum (per-period) Sharpe across ``n_trials`` independent
    strategies whose true Sharpe is zero -- the multiple-testing benchmark.

        E[max SR] ~= sqrt(V) * [ (1-g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e)) ]

    with V = variance of the trial Sharpes, g = Euler-Mascheroni, Z^-1 = inverse
    standard normal CDF."""
    if n_trials is None or n_trials < 2 or not np.isfinite(var_trial_sharpe) or var_trial_sharpe <= 0:
        return 0.0
    sd = np.sqrt(var_trial_sharpe)
    a = _phi_inv(1.0 - 1.0 / n_trials)
    b = _phi_inv(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * ((1.0 - _EULER) * a + _EULER * b))


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n_obs: int,
                               skew: float = 0.0, kurt: float = 3.0) -> float:
    """P(true Sharpe > sr_benchmark) given an observed (per-period) Sharpe ``sr``,
    sample length ``n_obs`` and return skew / (raw) kurtosis. Bailey-Lopez de Prado."""
    if n_obs is None or n_obs < 2 or not np.isfinite(sr):
        return float("nan")
    denom = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr ** 2
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / np.sqrt(denom)
    return _phi(z)


def deflated_sharpe_ratio(sr_period: float, returns: pd.Series, n_trials: int,
                          var_trial_sharpe: float) -> float:
    """Deflated Sharpe Ratio: PSR evaluated against the expected-max-Sharpe
    benchmark for ``n_trials``. ``sr_period`` is the per-period (non-annualised)
    Sharpe; ``returns`` provides the skew/kurtosis and sample length."""
    r = pd.Series(returns).dropna()
    if len(r) < 8 or not np.isfinite(sr_period):
        return float("nan")
    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3.0  # pandas gives excess kurtosis -> raw
    sr_star = expected_max_sharpe(n_trials, var_trial_sharpe)
    return probabilistic_sharpe_ratio(sr_period, sr_star, len(r), skew, kurt)


# --------------------------------------------------------------------------
# suite-level correction table
# --------------------------------------------------------------------------
def corrections_table(suite, periods_per_year: int = 252) -> pd.DataFrame:
    """Per (strategy, freq): raw Sharpe, James-Stein shrunk Sharpe, rank change,
    and Deflated Sharpe Ratio across all trials in the suite. Index matches the
    suite metrics_table (display-label, freq)."""
    rows: dict[tuple, dict] = {}
    for (name, freq), res in suite.results.items():
        label = suite.labels.get(name, name) if hasattr(suite, "labels") else name
        rows[(label, freq)] = {
            "sharpe_raw": float(res.metrics.get("sharpe", np.nan)),
            "returns": res.returns,
            "n": int(res.metrics.get("n_periods", len(res.returns))),
        }
    if not rows:
        return pd.DataFrame()

    keys = list(rows)
    raw = np.array([rows[k]["sharpe_raw"] for k in keys], dtype=np.float64)

    # James-Stein with Lo sampling variance (averaged across trials)
    sig = np.array([sharpe_sampling_variance(rows[k]["sharpe_raw"], rows[k]["n"]) for k in keys])
    sigma2 = float(np.nanmean(sig)) if np.isfinite(sig).any() else 1.0
    js = james_stein_shrinkage(raw, sigma2=sigma2)

    # Deflated Sharpe: trials measured in per-period units
    n_trials = len(keys)
    sr_period = raw / np.sqrt(periods_per_year)
    var_trials = float(np.nanvar(sr_period[np.isfinite(sr_period)])) if np.isfinite(sr_period).any() else 0.0
    dsr = np.array([
        deflated_sharpe_ratio(rows[k]["sharpe_raw"] / np.sqrt(periods_per_year),
                              rows[k]["returns"], n_trials, var_trials)
        for k in keys
    ])

    idx = pd.MultiIndex.from_tuples(keys, names=["strategy", "freq"])
    out = pd.DataFrame({"sharpe_raw": raw, "sharpe_js": js, "dsr": dsr}, index=idx)
    out["rank_raw"] = out["sharpe_raw"].rank(ascending=False, method="min").astype(int)
    out["rank_js"] = out["sharpe_js"].rank(ascending=False, method="min").astype(int)
    out["rank_change"] = out["rank_raw"] - out["rank_js"]
    return out.sort_values("sharpe_raw", ascending=False)
