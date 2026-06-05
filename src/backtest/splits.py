"""Time-series train / validation / test splits for out-of-sample evaluation.

Chronological only -- never shuffled (that would leak the future). Adapted from
the regime-portfolio project's ``utils/splits.py``, with an added walk-forward
generator and an embargo gap (the source had neither).

Why this matters here: the backtester searches 8 strategies x 4 rebalance freqs
(plus parameter choices). Reporting the full-sample maximum Sharpe is data
snooping. The honest protocol: pick the best configuration on the VALIDATION
window, then report its performance on the untouched TEST window.

Our strategies are non-parametric (momentum / vol / correlation are trailing
computations, nothing is *fit*), so the only leakage risk is future data in a
signal -- which the look-ahead invariant already prevents. Selecting on valid
and confirming on test removes the remaining selection bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    """A chronological split. Each field is the DatetimeIndex of that window."""

    train: pd.DatetimeIndex
    valid: pd.DatetimeIndex
    test: pd.DatetimeIndex

    @property
    def train_end(self):
        return self.train[-1] if len(self.train) else None

    @property
    def valid_end(self):
        return self.valid[-1] if len(self.valid) else self.train_end

    def window(self, name: str) -> tuple:
        idx = getattr(self, name)
        if len(idx) == 0:
            return (None, None)
        return (idx[0], idx[-1])

    def summary(self) -> str:
        def _r(idx):
            return f"{idx[0].date()}~{idx[-1].date()} (n={len(idx)})" if len(idx) else "(empty)"
        return f"train {_r(self.train)} | valid {_r(self.valid)} | test {_r(self.test)}"


def chronological_split(index, train_ratio: float = 0.6,
                        valid_ratio: float = 0.2, test_ratio: float = 0.2) -> Split:
    """Split a sorted DatetimeIndex into contiguous train/valid/test blocks.

    Ratios are normalised if they do not sum to 1. Boundaries floor-rounded; the
    remainder is absorbed into test. (Defaults 0.6/0.2/0.2 -- larger OOS fraction
    than a 0.7/0.15/0.15 because this project has ~25y of history.)
    """
    idx = pd.DatetimeIndex(index).sort_values().unique()
    n = len(idx)
    if n == 0:
        empty = pd.DatetimeIndex([])
        return Split(empty, empty, empty)
    total = train_ratio + valid_ratio + test_ratio
    if total <= 0:
        raise ValueError("ratios must sum to > 0")
    n_train = max(1, int(n * train_ratio / total))
    n_valid = max(0, int(n * valid_ratio / total))
    if n_train + n_valid >= n:
        n_valid = max(0, n - n_train - 1)
    return Split(
        pd.DatetimeIndex(idx[:n_train]),
        pd.DatetimeIndex(idx[n_train:n_train + n_valid]),
        pd.DatetimeIndex(idx[n_train + n_valid:]),
    )


def walk_forward_windows(index, n_folds: int = 5, train_frac: float = 0.5,
                         embargo: int = 1) -> list[tuple]:
    """Generate expanding-window walk-forward folds.

    Returns a list of ``(train_end, test_start, test_end)`` Timestamps. The first
    ``train_frac`` of history seeds the initial train window; the remainder is cut
    into ``n_folds`` contiguous test segments, each preceded by all prior data as
    train. An ``embargo`` of N bars is left between train_end and test_start so a
    trailing signal at the test boundary cannot peek across the seam.
    """
    idx = pd.DatetimeIndex(index).sort_values().unique()
    n = len(idx)
    if n < 10 or n_folds < 1:
        return []
    start = max(1, int(n * train_frac))
    seg = (n - start) // n_folds
    if seg <= embargo + 1:
        return []
    folds = []
    for k in range(n_folds):
        test_lo = start + k * seg
        test_hi = (start + (k + 1) * seg - 1) if k < n_folds - 1 else (n - 1)
        train_end_pos = test_lo - 1 - embargo
        if train_end_pos < 1 or test_lo > test_hi:
            continue
        folds.append((idx[train_end_pos], idx[test_lo], idx[test_hi]))
    return folds
