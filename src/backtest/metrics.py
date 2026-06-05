"""Performance metrics for an equity curve.

All functions take a daily (or weekly) equity Series and/or its period returns.
``periods_per_year`` annualises (252 for daily, 52 for weekly). Everything is
computed from realised data only -- no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
def to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] == 0:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def years_elapsed(equity: pd.Series) -> float:
    idx = equity.dropna().index
    if len(idx) < 2:
        return float("nan")
    days = (idx[-1] - idx[0]).days
    return max(days / 365.25, 1e-9)


def cagr(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    yrs = years_elapsed(equity)
    growth = equity.iloc[-1] / equity.iloc[0]
    if growth <= 0:
        return float("nan")
    return float(growth ** (1.0 / yrs) - 1.0)


def annual_volatility(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int, rf: float = 0.0) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_per = rf / periods_per_year
    excess = r - rf_per
    sd = excess.std(ddof=1)
    # tolerance (not == 0): a nonzero constant series leaves a tiny FP residual
    if not np.isfinite(sd) or sd <= 1e-12:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int, rf: float = 0.0) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_per = rf / periods_per_year
    excess = r - rf_per
    downside = excess.clip(upper=0.0)
    dd = np.sqrt((downside ** 2).mean())
    if not np.isfinite(dd) or dd <= 1e-12:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    equity = equity.dropna()
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    dd = drawdown_series(equity)
    return float(dd.min()) if len(dd) else float("nan")


def drawdown_details(equity: pd.Series) -> dict:
    """Worst drawdown depth, its trough date, and the longest underwater span."""
    equity = equity.dropna()
    if len(equity) < 2:
        return {"mdd": float("nan"), "mdd_duration_days": float("nan"),
                "trough": None, "peak": None, "recovered": False}
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough = dd.idxmin()
    mdd = float(dd.loc[trough])

    # peak preceding the trough
    pk_val = peak.loc[trough]
    pk_date = equity.loc[:trough][equity.loc[:trough] >= pk_val].index
    peak_date = pk_date[-1] if len(pk_date) else equity.index[0]

    # recovery: first date after trough where equity regains the prior peak
    after = equity.loc[trough:]
    rec = after[after >= pk_val]
    recovered = len(rec) > 0
    end = rec.index[0] if recovered else equity.index[-1]

    # longest underwater stretch overall (consecutive days with dd < 0).
    # Loop over runs, not rows: find run boundaries with numpy.
    uw = (dd < -1e-12).to_numpy()
    idx = equity.index
    longest = 0
    if uw.any():
        change = np.diff(uw.astype(np.int8))
        starts = list(np.where(change == 1)[0] + 1)
        ends = list(np.where(change == -1)[0] + 1)
        if uw[0]:
            starts = [0] + starts
        if uw[-1]:
            ends = ends + [len(uw)]
        for s, e in zip(starts, ends):
            longest = max(longest, (idx[e - 1] - idx[s]).days)

    return {
        "mdd": mdd,
        "mdd_duration_days": float(longest),
        "mdd_recovery_days": float((end - peak_date).days),
        "trough": trough,
        "peak": peak_date,
        "recovered": recovered,
    }


def calmar_ratio(equity: pd.Series) -> float:
    c = cagr(equity)
    mdd = max_drawdown(equity)
    if mdd is None or np.isnan(mdd) or mdd == 0:
        return float("nan")
    return float(c / abs(mdd))


def win_rate(period_returns: pd.Series) -> float:
    r = period_returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float((r > 0).sum() / len(r))


def profit_factor(period_returns: pd.Series) -> float:
    r = period_returns.dropna()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def alpha_beta(returns: pd.Series, benchmark_returns: pd.Series,
               periods_per_year: int, rf: float = 0.0) -> tuple[float, float]:
    """Annualised alpha and beta vs a benchmark (CAPM, OLS on excess returns)."""
    df = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(df) < 3:
        return float("nan"), float("nan")
    rf_per = rf / periods_per_year
    y = df.iloc[:, 0] - rf_per
    x = df.iloc[:, 1] - rf_per
    var = x.var(ddof=1)
    if not np.isfinite(var) or var <= 1e-18:
        return float("nan"), float("nan")
    beta = float(x.cov(y) / var)
    alpha_per = float(y.mean() - beta * x.mean())
    # simple (linear) annualisation of the per-period CAPM intercept. Geometric
    # compounding of a per-BAR intercept explodes for larger intercepts and needs
    # an arbitrary branch to stay finite; linear is the standard, stable choice.
    alpha_ann = alpha_per * periods_per_year
    return alpha_ann, beta


# --------------------------------------------------------------------------
# tabular views
# --------------------------------------------------------------------------
def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Year x Month matrix of compounded returns (fraction)."""
    r = returns.dropna()
    if len(r) == 0:
        return pd.DataFrame()
    monthly = (1 + r).resample("ME").prod() - 1
    tbl = monthly.to_frame("ret")
    tbl["year"] = tbl.index.year
    tbl["month"] = tbl.index.month
    pivot = tbl.pivot(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot["Year"] = (1 + pivot.fillna(0)).prod(axis=1) - 1
    return pivot


def annual_returns(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    if len(r) == 0:
        return pd.Series(dtype=float)
    ann = (1 + r).resample("YE").prod() - 1
    ann.index = ann.index.year
    return ann


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------
def compute_metrics(
    equity: pd.Series,
    periods_per_year: int = 252,
    rf: float = 0.0,
    benchmark_returns: pd.Series | None = None,
    period_returns: pd.Series | None = None,
    turnover: pd.Series | float | None = None,
    n_trades: int | None = None,
) -> dict:
    """Compute the full metric suite. Returns a flat dict of scalars.

    ``period_returns`` are the per-rebalance returns used for win-rate / profit
    factor; if omitted, daily returns are used. ``turnover`` is the per-rebalance
    fraction traded (Series or mean).
    """
    equity = equity.dropna().astype(float)
    rets = to_returns(equity)
    dd = drawdown_details(equity)

    if period_returns is None:
        period_returns = rets

    out: dict = {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "ann_volatility": annual_volatility(rets, periods_per_year),
        "sharpe": sharpe_ratio(rets, periods_per_year, rf),
        "sortino": sortino_ratio(rets, periods_per_year, rf),
        "mdd": dd["mdd"],
        "mdd_duration_days": dd["mdd_duration_days"],
        "mdd_recovery_days": dd.get("mdd_recovery_days", float("nan")),
        "calmar": calmar_ratio(equity),
        "win_rate": win_rate(period_returns),
        "profit_factor": profit_factor(period_returns),
        "final_equity": float(equity.iloc[-1]) if len(equity) else float("nan"),
        "n_periods": int(len(rets)),
    }

    if benchmark_returns is not None:
        a, b = alpha_beta(rets, benchmark_returns, periods_per_year, rf)
        out["alpha"] = a
        out["beta"] = b

    if turnover is not None:
        if isinstance(turnover, pd.Series):
            out["avg_turnover"] = float(turnover.mean()) if len(turnover) else 0.0
            out["total_turnover"] = float(turnover.sum())
        else:
            out["avg_turnover"] = float(turnover)
    if n_trades is not None:
        out["n_rebalances"] = int(n_trades)

    return out


METRIC_LABELS = {
    "total_return": "Total Return",
    "cagr": "CAGR",
    "ann_volatility": "Ann.Vol",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "mdd": "MaxDD",
    "mdd_duration_days": "Max Underwater Days",
    "mdd_recovery_days": "Recovery Days",
    "calmar": "Calmar",
    "win_rate": "Win Rate",
    "profit_factor": "Profit Factor",
    "alpha": "Alpha",
    "beta": "Beta",
    "avg_turnover": "Avg Turnover",
    "n_rebalances": "Rebalances",
    "final_equity": "Final Equity",
}
