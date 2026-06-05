"""Minimal Gaussian HMM (full covariance) in numpy/scipy.

Implemented in-house rather than depending on ``hmmlearn`` because:
* the project targets Python 3.14, for which hmmlearn has no prebuilt wheel;
* we need the FORWARD-only filter for look-ahead-safe inference, and rolling our
  own makes that explicit (hmmlearn's predict_proba is forward-BACKWARD smoothed
  and would leak the future).

Training (``fit_hmm``) uses full Baum-Welch (forward-backward) on a TRAIN window
-- using the whole training sequence to estimate parameters is correct; the
look-ahead concern is only about INFERENCE. Inference for the backtest uses
``forward_filter`` which returns P(S_t | y_1..y_t) -- past observations only.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

_LOG2PI = np.log(2.0 * np.pi)


def _lse(a: np.ndarray, axis: int) -> np.ndarray:
    """Fast log-sum-exp along ``axis`` (manual -> avoids scipy per-call overhead
    in the hot forward/backward loops)."""
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis)


def _mvn_logpdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log pdf of a multivariate normal for each row of X. Returns (T,)."""
    d = X.shape[1]
    cov = cov + np.eye(d) * 1e-6
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + np.eye(d) * 1e-3
        L = np.linalg.cholesky(cov)
    diff = X - mean
    sol = np.linalg.solve(L, diff.T).T          # (T, d)
    maha = np.sum(sol ** 2, axis=1)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * _LOG2PI + logdet + maha)


def _log_emission(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """(T, K) log emission probabilities."""
    T, K = X.shape[0], means.shape[0]
    out = np.empty((T, K))
    for k in range(K):
        out[:, k] = _mvn_logpdf(X, means[k], covars[k])
    return out


def _forward(log_emit, log_pi, log_A):
    T, K = log_emit.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_emit[0]
    for t in range(1, T):
        log_alpha[t] = log_emit[t] + _lse(log_alpha[t - 1][:, None] + log_A, axis=0)
    return log_alpha


def _backward(log_emit, log_A):
    T, K = log_emit.shape
    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_beta[t] = _lse(log_A + log_emit[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
    return log_beta


def fit_hmm(X: np.ndarray, n_states: int, n_iter: int = 100, tol: float = 1e-4,
            seed: int = 42) -> dict:
    """Fit a Gaussian HMM by EM. Returns a params dict with keys
    ``log_pi, log_A, means, covars, n_states, loglik``. Deterministic given seed."""
    from sklearn.cluster import KMeans

    X = np.asarray(X, dtype=np.float64)
    T, d = X.shape
    K = int(n_states)

    # --- init via KMeans (stable, deterministic) ---------------------------
    km = KMeans(n_clusters=K, n_init=4, random_state=seed).fit(X)
    labels = km.labels_
    means = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else X.mean(axis=0)
                      for k in range(K)])
    covars = []
    for k in range(K):
        xk = X[labels == k]
        c = np.cov(xk, rowvar=False) if xk.shape[0] > d else np.cov(X, rowvar=False)
        covars.append(np.atleast_2d(c) + np.eye(d) * 1e-4)
    covars = np.array(covars)
    pi = np.full(K, 1.0 / K)
    A = np.full((K, K), 0.1 / max(K - 1, 1))
    np.fill_diagonal(A, 0.9)

    log_pi = np.log(pi + 1e-12)
    log_A = np.log(A + 1e-12)
    prev_ll = -np.inf

    for _ in range(n_iter):
        log_emit = _log_emission(X, means, covars)
        log_alpha = _forward(log_emit, log_pi, log_A)
        log_beta = _backward(log_emit, log_A)
        ll = logsumexp(log_alpha[-1])
        log_gamma = log_alpha + log_beta - ll
        gamma = np.exp(log_gamma)                 # (T, K)

        # transition expected counts
        log_xi = (log_alpha[:-1, :, None] + log_A[None, :, :]
                  + (log_emit[1:] + log_beta[1:])[:, None, :] - ll)
        xi_sum = np.exp(logsumexp(log_xi, axis=0))   # (K, K)

        # M-step
        gamma_sum = gamma.sum(axis=0) + 1e-12
        means = (gamma.T @ X) / gamma_sum[:, None]
        covars = np.empty((K, d, d))
        for k in range(K):
            diff = X - means[k]
            covars[k] = (gamma[:, k][:, None] * diff).T @ diff / gamma_sum[k] + np.eye(d) * 1e-4
        log_pi = np.log(gamma[0] + 1e-12)
        A = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-12)
        log_A = np.log(A + 1e-12)

        if ll - prev_ll < tol and _ > 0:
            break
        prev_ll = ll

    return {"log_pi": log_pi, "log_A": log_A, "means": means, "covars": covars,
            "n_states": K, "loglik": float(prev_ll)}


def forward_filter(X: np.ndarray, params: dict) -> np.ndarray:
    """LOOK-AHEAD-SAFE inference: filtered state probabilities P(S_t | y_1..y_t).
    Forward recursion only -- row t depends on observations <= t. Returns (T, K)."""
    X = np.asarray(X, dtype=np.float64)
    log_emit = _log_emission(X, params["means"], params["covars"])
    log_alpha = _forward(log_emit, params["log_pi"], params["log_A"])
    return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))


def bic(X: np.ndarray, params: dict) -> float:
    """Bayesian Information Criterion (lower is better)."""
    T, d = X.shape
    K = params["n_states"]
    # free params: startprob(K-1) + transmat K(K-1) + means K*d + covars K*d(d+1)/2
    n_params = (K - 1) + K * (K - 1) + K * d + K * d * (d + 1) // 2
    return -2.0 * params["loglik"] + n_params * np.log(max(T, 2))


def mean_return_order(params: dict, col: int = 0) -> np.ndarray:
    """State indices sorted by ascending mean of feature ``col`` (canonical sort:
    worst-return state first, best last)."""
    return np.argsort(params["means"][:, col])
