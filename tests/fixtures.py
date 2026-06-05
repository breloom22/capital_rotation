"""Synthetic-data fixtures for fast, offline testing of analysis / strategy /
backtest modules. Builds a realistic ``MarketData`` covering the full configured
universe (including ^VIX and FX) with a fixed seed so tests are reproducible.

Usage::

    from tests.fixtures import make_data
    md, cfg = make_data()                 # daily MarketData + Config
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import build_market_data


def make_data(seed: int = 0, start: str = "2015-01-01", end: str = "2021-12-31",
              freq: str = "daily"):
    """Return (MarketData, Config) over the full configured universe."""
    cfg = load_config()
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end, name="Date")
    n = len(idx)
    raw: dict[str, pd.DataFrame] = {}

    for i, t in enumerate(cfg.tickers):
        cat = cfg.categories.get(t, "other")
        # stagger inception dates so `available` masks are exercised
        first = int(rng.integers(0, max(1, n // 6)))
        sub = idx[first:]
        m = len(sub)
        if t == "^VIX":
            base = 20 + 10 * np.sin(np.linspace(0, 12, m)) + rng.normal(0, 2, m)
            px = np.clip(base, 9, 80)
            vol = np.zeros(m)
        elif cat == "fx":
            lvl = {"JPY=X": 110.0, "EUR=X": 1.12, "DX-Y.NYB": 95.0}.get(t, 1.0)
            px = lvl * np.exp(np.cumsum(rng.normal(0, 0.004, m)))
            vol = np.zeros(m)
        else:
            drift = {"crypto": 0.0010, "us_equity": 0.0004, "bonds": 0.0001}.get(cat, 0.0003)
            sd = {"crypto": 0.035, "bonds": 0.004}.get(cat, 0.012)
            px = (50 + 5 * i) * np.exp(np.cumsum(rng.normal(drift, sd, m)))
            vol = rng.integers(1_000_000, 8_000_000, m).astype(float)
        raw[t] = pd.DataFrame(
            {"Open": px, "High": px * 1.005, "Low": px * 0.995,
             "Close": px, "Adj Close": px, "Volume": vol},
            index=sub,
        )

    md = build_market_data(
        raw, categories=cfg.categories, tradable=cfg.tradable_tickers,
        freq=freq, min_obs=cfg.data_cfg.get("min_obs", 100),
    )
    return md, cfg


if __name__ == "__main__":
    md, cfg = make_data()
    print("MarketData:", md.prices.shape, "tradable:", len(md.tradable_present()))
    print("freq:", md.freq, "date range:", md.dates.min().date(), "->", md.dates.max().date())
