"""Capital-rotation detection: relative strength, inflow/outflow, lead-lag.

LOOK-AHEAD INVARIANT: value at date ``t`` uses data with index <= ``t`` only.

PUBLIC API:
    relative_strength(prices, lookback=63, benchmark=None)  -> DataFrame
    inflow_outflow(close, volume, lookback=20)              -> DataFrame  (score in ~[-1,1])
    category_rotation(data, config, lookback=63, as_of=None) -> DataFrame
        per-category summary at `as_of`: columns ['rel_strength','flow','status']
        status in {'inflow','outflow','neutral'}.
    lead_lag(returns, leader, follower, max_lag=10)         -> dict
        {'best_lag': int, 'best_corr': float, 'corr_by_lag': {lag: corr}}
    cycle_length(returns, window=252)                        -> float  (avg rotation period, periods)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# small dead-band around zero used to label a category 'neutral'
_FLOW_EPS = 1e-3


def relative_strength(prices: pd.DataFrame, lookback: int = 63, benchmark: str | None = None) -> pd.DataFrame:
    """Relative strength per asset: trailing return minus a baseline. If
    ``benchmark`` is given, baseline = that column's trailing return; else the
    cross-sectional mean trailing return. Wide [date x ticker]."""
    # trailing total return over `lookback` periods (uses only data <= t)
    tr = prices / prices.shift(lookback) - 1.0
    tr = tr.replace([np.inf, -np.inf], np.nan)

    if benchmark is not None and benchmark in tr.columns:
        baseline = tr[benchmark]
    else:
        # cross-sectional mean trailing return per date (ignores NaN/pre-inception)
        baseline = tr.mean(axis=1)

    rs = tr.sub(baseline, axis=0)
    return rs.replace([np.inf, -np.inf], np.nan)


def inflow_outflow(close: pd.DataFrame, volume: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Money inflow/outflow score: price up + volume up => inflow (>0); price
    down + volume up => outflow (<0). Aggregated over a trailing window. Wide."""
    close = close.astype(float)
    volume = volume.astype(float)

    # signed dollar flow: sign of the daily close change * traded dollar volume
    sign = np.sign(close.diff())
    dollar = close * volume
    signed = sign * dollar

    # trailing-window sums (look-ahead safe: only data <= t)
    num = signed.rolling(lookback, min_periods=1).sum()
    den = dollar.rolling(lookback, min_periods=1).sum()

    score = num.div(den)
    score = score.replace([np.inf, -np.inf], np.nan)
    # where there was no traded volume at all in the window (e.g. ^VIX, FX),
    # the denominator is 0 -> NaN above; leave as NaN (no signal).
    return score


def category_rotation(data, config, lookback: int = 63, as_of=None) -> pd.DataFrame:
    """Category-level rotation snapshot at ``as_of`` (default last date). Index =
    category; columns ['rel_strength','flow','status'] with status in
    {'inflow','outflow','neutral'}. Aggregates member tickers (mean)."""
    prices = data.prices
    close = data.close
    volume = data.volume

    if as_of is None:
        as_of = prices.index[-1]
    as_of = pd.Timestamp(as_of)

    # restrict to data up to and including as_of (look-ahead safe)
    prices = prices.loc[:as_of]
    close = close.loc[:as_of]
    volume = volume.loc[:as_of]

    rs = relative_strength(prices, lookback)
    flow = inflow_outflow(close, volume, lookback)

    # values as of the snapshot date
    rs_row = rs.loc[as_of] if as_of in rs.index else rs.iloc[-1]
    flow_row = flow.loc[as_of] if as_of in flow.index else flow.iloc[-1]

    tradable = set(data.tradable_present())

    rows = {}
    for category, members in config.category_members.items():
        # only tradable members that are present in this panel
        mem = [t for t in members if t in tradable and t in prices.columns]
        if not mem:
            continue
        rs_val = rs_row.reindex(mem).mean(skipna=True)
        flow_val = flow_row.reindex(mem).mean(skipna=True)

        if pd.isna(flow_val):
            status = "neutral"
        elif flow_val > _FLOW_EPS:
            status = "inflow"
        elif flow_val < -_FLOW_EPS:
            status = "outflow"
        else:
            status = "neutral"

        rows[category] = {
            "rel_strength": float(rs_val) if pd.notna(rs_val) else np.nan,
            "flow": float(flow_val) if pd.notna(flow_val) else np.nan,
            "status": status,
        }

    out = pd.DataFrame.from_dict(rows, orient="index",
                                 columns=["rel_strength", "flow", "status"])
    out.index.name = "category"
    return out


def lead_lag(returns: pd.DataFrame, leader: str, follower: str, max_lag: int = 10) -> dict:
    """Cross-correlation lead/lag between two assets over the full sample.
    Returns {'best_lag', 'best_corr', 'corr_by_lag'}; best_lag>0 => `leader`
    leads `follower` by that many periods."""
    lead = returns[leader]
    foll = returns[follower]

    corr_by_lag: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        # corr(leader.shift(lag), follower):
        #   lag>0 shifts leader's past values forward to align with follower's
        #   present -> leader leading the follower.
        c = lead.shift(lag).corr(foll)
        corr_by_lag[lag] = float(c) if pd.notna(c) else np.nan

    # pick the lag with the largest absolute correlation among valid entries
    valid = {k: v for k, v in corr_by_lag.items() if pd.notna(v)}
    if valid:
        best_lag = max(valid, key=lambda k: abs(valid[k]))
        best_corr = valid[best_lag]
    else:
        best_lag = 0
        best_corr = np.nan

    return {
        "best_lag": int(best_lag),
        "best_corr": float(best_corr) if pd.notna(best_corr) else np.nan,
        "corr_by_lag": corr_by_lag,
    }


def cycle_length(returns: pd.DataFrame, window: int = 252) -> float:
    """Rough average rotation-cycle length (in periods) estimated from the sign
    changes of the market-average relative-strength dispersion. Best-effort."""
    if returns.shape[1] == 0 or len(returns) == 0:
        return float("nan")

    # cross-sectional dispersion of returns per date: how spread out assets are.
    disp = returns.std(axis=1, skipna=True)
    disp = disp.replace([np.inf, -np.inf], np.nan).dropna()
    if len(disp) < 3:
        return float("nan")

    # de-mean against a trailing rolling mean (look-ahead safe) so we measure
    # oscillation of dispersion around its own recent average.
    w = max(2, min(window, len(disp)))
    baseline = disp.rolling(w, min_periods=1).mean()
    centered = (disp - baseline).to_numpy()

    signs = np.sign(centered)
    # treat exact zeros as continuation of the previous sign
    nz = signs[signs != 0]
    if len(nz) < 2:
        return float("nan")

    flips = int(np.sum(nz[1:] != nz[:-1]))
    if flips == 0:
        return float("nan")

    # two sign flips make one full cycle; spread the spanned periods over them.
    span = len(centered)
    return float(2.0 * span / flips)
