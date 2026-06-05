"""Matplotlib charts saved to the output directory (headless 'Agg' backend).

PUBLIC API (the CLI imports these exact names). Each returns the saved Path.
Use a non-interactive backend: ``import matplotlib; matplotlib.use('Agg')`` at
import time. Never call plt.show().

    equity_curve(suite, output_dir, freq=None, logy=True)      -> Path
    drawdown_chart(suite, output_dir, freq=None)               -> Path
    correlation_heatmap(corr_df, output_dir, names=None)       -> Path
    rotation_chart(rotation_df, output_dir)                    -> Path
    monthly_heatmap(result, output_dir)                        -> Path

``suite`` is a SuiteResult; pick the given ``freq`` (or each strategy's first
available) and plot equity / drawdown of every strategy on one figure.
``result`` is a single BacktestResult (for the monthly heatmap).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..backtest import metrics as _metrics


def _ensure_dir(output_dir) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _iter_curves(suite, freq: str | None):
    """Yield ``(label, BacktestResult)`` for each strategy, choosing the given
    ``freq`` when present otherwise that strategy's first available freq.

    ``suite.results`` maps ``(strategy_name, freq) -> BacktestResult``. We keep
    config/insertion order and, for each strategy, prefer the requested freq.
    """
    results = getattr(suite, "results", {}) or {}
    labels = getattr(suite, "labels", {}) or {}

    # collect available freqs per strategy, preserving first-seen order
    by_strategy: dict[str, list[str]] = {}
    for key in results:
        name, f = key
        by_strategy.setdefault(name, []).append(f)

    for name, freqs in by_strategy.items():
        chosen = None
        if freq is not None and freq in freqs:
            chosen = freq
        elif freqs:
            chosen = freqs[0]
        if chosen is None:
            continue
        res = results.get((name, chosen))
        if res is None:
            continue
        label = labels.get(name) or getattr(res, "name", None) or str(name)
        yield label, res


def equity_curve(suite, output_dir, freq: str | None = None, logy: bool = True) -> Path:
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(11, 6))

    plotted = False
    for label, res in _iter_curves(suite, freq):
        eq = getattr(res, "equity", None)
        if eq is None:
            continue
        eq = pd.Series(eq).dropna()
        if logy:
            eq = eq[eq > 0]
        if len(eq) < 2:
            continue
        ax.plot(eq.index, eq.to_numpy(), label=str(label), linewidth=1.3)
        plotted = True

    if logy and plotted:
        ax.set_yscale("log")
    ax.set_title("Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value" + (" (log)" if logy else ""))
    ax.grid(True, which="both", alpha=0.3)
    if plotted:
        ax.legend(loc="best", fontsize=8)

    path = out / "equity_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def drawdown_chart(suite, output_dir, freq: str | None = None) -> Path:
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(11, 6))

    plotted = False
    for label, res in _iter_curves(suite, freq):
        eq = getattr(res, "equity", None)
        if eq is None:
            continue
        eq = pd.Series(eq).dropna()
        if len(eq) < 2:
            continue
        peak = eq.cummax()
        dd = (eq / peak - 1.0).replace([np.inf, -np.inf], np.nan)
        dd = dd.dropna()
        if len(dd) < 2:
            continue
        ax.plot(dd.index, dd.to_numpy() * 100.0, label=str(label), linewidth=1.3)
        plotted = True

    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    if plotted:
        ax.legend(loc="best", fontsize=8)

    path = out / "drawdown.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def correlation_heatmap(corr_df: pd.DataFrame, output_dir, names: dict | None = None) -> Path:
    out = _ensure_dir(output_dir)
    corr = pd.DataFrame(corr_df).astype(float)
    corr = corr.replace([np.inf, -np.inf], np.nan)

    tickers = list(corr.columns)
    if names:
        labels = [names.get(t, t) for t in tickers]
    else:
        labels = [str(t) for t in tickers]

    n = len(tickers)
    side = max(6.0, 0.5 * n + 2.0)
    fig, ax = plt.subplots(figsize=(side, side))

    data = np.ma.masked_invalid(corr.to_numpy(dtype=float))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Return Correlation")

    path = out / "correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def rotation_chart(rotation_df: pd.DataFrame, output_dir) -> Path:
    out = _ensure_dir(output_dir)
    df = pd.DataFrame(rotation_df)

    # category flow: prefer the 'flow' column, fall back to 'rel_strength'
    if "flow" in df.columns:
        series = df["flow"]
    elif "rel_strength" in df.columns:
        series = df["rel_strength"]
    else:
        # first numeric column available
        numeric = df.select_dtypes(include="number")
        series = numeric.iloc[:, 0] if numeric.shape[1] else pd.Series(dtype=float)

    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    categories = [str(c) for c in series.index]
    values = series.fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.5 * len(categories) + 1.5)))
    y = np.arange(len(categories))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Flow")
    ax.set_title("Category Rotation Flow")
    ax.grid(True, axis="x", alpha=0.3)

    path = out / "rotation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def monthly_heatmap(result, output_dir) -> Path:
    out = _ensure_dir(output_dir)
    returns = getattr(result, "returns", result)
    table = _metrics.monthly_return_table(pd.Series(returns))

    fig, ax = plt.subplots(figsize=(11, max(3.0, 0.4 * (len(table) + 1) + 1.5)))

    if table.empty:
        ax.set_title("Monthly Returns (no data)")
        ax.set_xticks([])
        ax.set_yticks([])
        path = out / "monthly_heatmap.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    cols = list(table.columns)
    rows = list(table.index)
    data = table.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)

    # symmetric color scale around 0 based on observed magnitudes
    finite = data[np.isfinite(data)]
    lim = float(np.nanmax(np.abs(finite))) if finite.size else 0.01
    lim = lim if lim > 0 else 0.01

    im = ax.imshow(masked, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels([str(c) for c in cols], fontsize=8)
    ax.set_yticklabels([str(r) for r in rows], fontsize=8)

    # annotate cells with percentage values
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center",
                        fontsize=6, color="black")

    label = getattr(result, "name", "") or ""
    ax.set_title(f"Monthly Returns (%){(' - ' + str(label)) if label else ''}")

    path = out / "monthly_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def regime_equity(suite, data, config, output_dir, freq: str | None = None) -> Path:
    """Equity curves with Risk-Off regime periods shaded (from classify_regime)."""
    from ..analysis import regime
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(11, 6))

    plotted = False
    for label, res in _iter_curves(suite, freq):
        eq = pd.Series(getattr(res, "equity", None)).dropna()
        eq = eq[eq > 0]
        if len(eq) < 2:
            continue
        ax.plot(eq.index, eq.to_numpy(), label=str(label), linewidth=1.2)
        plotted = True
    if plotted:
        ax.set_yscale("log")

    # shade contiguous Risk-Off spans
    try:
        rdf = regime.classify_regime(data, config)
        lab = rdf["regime"].reindex(data.dates).ffill()
        off = (lab == "risk_off").to_numpy()
        idx = lab.index
        if off.any():
            change = np.diff(off.astype(np.int8))
            starts = list(np.where(change == 1)[0] + 1)
            ends = list(np.where(change == -1)[0] + 1)
            if off[0]:
                starts = [0] + starts
            if off[-1]:
                ends = ends + [len(off)]
            for s, e in zip(starts, ends):
                ax.axvspan(idx[s], idx[min(e, len(idx) - 1)], color="red", alpha=0.08)
    except Exception:
        pass

    ax.set_title("Equity with Risk-Off regime shading")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (log)")
    ax.grid(True, which="both", alpha=0.3)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    path = out / "regime_equity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
