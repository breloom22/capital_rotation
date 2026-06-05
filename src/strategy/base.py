"""Strategy interface + registry.

THE STRATEGY CONTRACT (read before writing any strategy)
--------------------------------------------------------
A strategy turns market data into *target weights* on each rebalance date.

    class MyStrat(Strategy):
        def precompute(self):
            # OPTIONAL. Build vectorised signal panels once (wide date x ticker
            # DataFrames). Every value at row t must use only data <= t.
            ...
        def target_weights(self, date) -> pd.Series:
            # REQUIRED. Return weights indexed by ticker for THIS date.
            # Sum may be <= 1; the remainder (1 - sum) is held in CASH.
            # Use only information available at `date` (rows with index <= date).
            ...

Look-ahead safety is structural: ``self.signal_at(panel, date)`` returns the
last panel row with index <= date, so as long as a panel row at t is built from
data <= t (which the analysis modules guarantee), strategies cannot peek ahead.

Register a strategy so the CLI / engine can find it by name::

    @register("topn")
    class TopNStrategy(Strategy): ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # avoid import cycles at runtime
    from ..config import Config
    from ..data import MarketData


# --- registry --------------------------------------------------------------
STRATEGY_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        STRATEGY_REGISTRY[name] = cls
        return cls
    return deco


def get_strategy(name: str) -> type:
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"unknown strategy '{name}'. known: {sorted(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name]


class Strategy(ABC):
    """Base class. Subclasses implement :meth:`target_weights`."""

    name: str = "base"
    #: human-friendly label (falls back to ``name``)
    label: str = ""

    def __init__(self, data: "MarketData", config: "Config", params: dict | None = None):
        self.data = data
        self.config = config
        self.params = dict(params or {})
        # tradable tickers that actually exist in this panel
        self.tradable: list[str] = data.tradable_present()
        self.prices: pd.DataFrame = data.prices
        self.returns: pd.DataFrame = data.returns
        self._precomputed = False
        # number of leading periods of history the strategy needs before it can
        # produce a meaningful weight; the engine aligns all strategies to the
        # largest warmup so their equity curves are comparable. Override freely.
        self.warmup: int = 252

    # ---- lifecycle ---------------------------------------------------------
    def precompute(self) -> None:
        """One-time vectorised signal preparation. Default: nothing."""

    def ensure_precomputed(self) -> None:
        if not self._precomputed:
            self.precompute()
            self._precomputed = True

    @abstractmethod
    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        """Return target weights (index=ticker) for ``date``; remainder = cash."""
        raise NotImplementedError

    # ---- helpers for subclasses -------------------------------------------
    @staticmethod
    def signal_at(panel: pd.DataFrame | pd.Series, date: pd.Timestamp):
        """Last available row of ``panel`` with index <= ``date``.

        Returns a Series (for a DataFrame) / scalar (for a Series), or an empty
        Series / NaN when nothing is available yet.
        """
        sub = panel.loc[:date]
        if len(sub) == 0:
            if isinstance(panel, pd.DataFrame):
                return pd.Series(dtype=float)
            return np.nan
        return sub.iloc[-1]

    def available_at(self, date: pd.Timestamp) -> list[str]:
        """Tradable tickers that have a real price at ``date``."""
        row = self.signal_at(self.prices, date)
        if not isinstance(row, pd.Series):
            return []
        ok = row.dropna()
        return [t for t in self.tradable if t in ok.index]

    @staticmethod
    def normalize(weights: pd.Series, cap_total: float = 1.0) -> pd.Series:
        """Drop NaN/negative, rescale so the sum is at most ``cap_total``."""
        w = weights.dropna()
        w = w[w > 0]
        total = w.sum()
        if total <= 0:
            return pd.Series(dtype=float)
        if total > cap_total:
            w = w * (cap_total / total)
        return w

    def display_name(self) -> str:
        return self.label or self.name
