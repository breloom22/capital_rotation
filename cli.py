#!/usr/bin/env python
"""Capital Rotation Tracker & Strategy Backtester -- CLI entrypoint.

    python cli.py data update [--asset SPY] [--full]
    python cli.py data status
    python cli.py analyze (correlation|momentum|regime|rotation|all) [--window N]
    python cli.py backtest run [--strategy NAME] [--rebalance FREQ]
    python cli.py backtest compare [--sort sharpe]
    python cli.py backtest export [--format csv|json]
    python cli.py chart (equity|drawdown|correlation|rotation) [--rebalance FREQ]
"""
from __future__ import annotations

import sys
from pathlib import Path

# make `import src...` work no matter the cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import click
import pandas as pd
from rich.console import Console

from src.config import load_config
from src.data import build_market_data, storage, fetcher
from src.strategy import ACTIVE_STRATEGIES, BENCHMARK_STRATEGIES, STRATEGY_REGISTRY

# make box-drawing / non-ASCII render on Windows code pages
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# wide enough for the multi-column comparison table even when output is piped
console = Console(width=160)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _load_market_data(cfg, freq: str | None = None):
    """Load stored raw frames and build the MarketData panel."""
    freq = freq or cfg.data_cfg.get("freq", "daily")
    frames = storage.load_many(cfg.raw_dir, cfg.tickers)
    if not frames:
        console.print("[red]No data found.[/red] Run:  python cli.py data update")
        raise SystemExit(1)
    bt = cfg.backtest
    md = build_market_data(
        frames, categories=cfg.categories, tradable=cfg.tradable_tickers,
        freq=freq, start=bt.get("start"), end=bt.get("end"),
        min_obs=cfg.data_cfg.get("min_obs", 100),
    )
    return md


def _build_suite(cfg, md, strategies=None, freqs=None):
    from src.backtest import run_suite
    strategies = strategies or (ACTIVE_STRATEGIES + BENCHMARK_STRATEGIES)
    freqs = freqs or cfg.rebalance.get("frequencies", ["monthly"])
    seen = {"n": 0, "total": len(strategies) * len(freqs)}

    def progress(name, freq, error=None):
        seen["n"] += 1
        tag = "[red]ERR[/red]" if error else "[green]ok[/green]"
        console.print(f"  [{seen['n']}/{seen['total']}] {name} / {freq} {tag}"
                      + (f"  {error}" if error else ""))

    with console.status("[bold]running backtests..."):
        suite = run_suite(md, cfg, strategies, freqs, progress=progress)
    return suite


# ---------------------------------------------------------------------------
# root group
# ---------------------------------------------------------------------------
@click.group()
@click.pass_context
def cli(ctx):
    """Capital Rotation Tracker & Strategy Backtester."""
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config()


# ---- data -----------------------------------------------------------------
@cli.group()
def data():
    """Data collection / status."""


@data.command("update")
@click.option("--asset", default=None, help="Update a single ticker only.")
@click.option("--full", is_flag=True, help="Re-download full history (ignore cache).")
@click.pass_context
def data_update(ctx, asset, full):
    cfg = ctx.obj["cfg"]
    tickers = [asset] if asset else cfg.tickers
    console.print(f"[bold]Updating {len(tickers)} ticker(s)[/bold] -> {cfg.raw_dir}")
    total_added = 0
    with click.progressbar(tickers, label="fetching") as bar:
        for t in bar:
            added, first, last = fetcher.update_ticker(t, cfg.raw_dir, full=full)
            total_added += added
            rng = f"{first.date()}..{last.date()}" if first is not None else "no data"
            console.print(f"  {t:10s} +{added:5d} rows  [{rng}]")
    console.print(f"[green]Done.[/green] {total_added} new rows.")


@data.command("status")
@click.pass_context
def data_status(ctx):
    cfg = ctx.obj["cfg"]
    rows = []
    for a in cfg.assets:
        df = storage.load_raw(cfg.raw_dir, a.ticker)
        if df is None or df.empty:
            rows.append(dict(ticker=a.ticker, name=a.name, start="-", end="-",
                             obs=0, missing="-", tradable=a.tradable))
            continue
        close = df["Close"] if "Close" in df else df.iloc[:, 0]
        miss = float(close.isna().mean())
        rows.append(dict(ticker=a.ticker, name=a.name,
                         start=str(df.index.min().date()), end=str(df.index.max().date()),
                         obs=int(len(df)), missing=f"{miss:.1%}", tradable=a.tradable))
    from src.visualization import tables
    tables.print_data_status(rows, console=console)


# ---- analyze --------------------------------------------------------------
@cli.group()
def analyze():
    """Market analysis snapshots (as of the latest date)."""


@analyze.command("correlation")
@click.option("--window", default=120, help="Rolling window (periods).")
@click.pass_context
def analyze_correlation(ctx, window):
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    import numpy as np
    from rich.table import Table
    from src.analysis import correlation
    from src.visualization import tables
    cm = correlation.correlation_matrix(md.returns, window)

    if cm.shape[0] <= 12:
        tables.print_correlation(cm, names=cfg.names, console=console)
    else:
        # full 26x26 matrix is unreadable in a terminal -> show the per-asset
        # average correlation (diversification ranking); heatmap has the detail.
        off = cm.where(~np.eye(len(cm), dtype=bool))
        avg = off.mean(axis=1).sort_values()
        t = Table(title=f"Avg Correlation (trailing {window}) — low = better diversifier",
                  title_style="bold")
        t.add_column("Ticker", style="cyan")
        t.add_column("Name")
        t.add_column("Avg Corr", justify="right")
        for tk, v in avg.items():
            colour = "green" if v < 0.3 else ("red" if v > 0.6 else "")
            t.add_row(str(tk), cfg.names.get(tk, str(tk)),
                      f"[{colour}]{v:+.2f}[/{colour}]" if colour else f"{v:+.2f}")
        console.print(t)
        console.print("[dim]Full pairwise matrix → python cli.py chart correlation[/dim]")
    pca = correlation.pca_explained(md.returns, max(window, 252), 3)
    evr = pca.get("explained_variance_ratio", [])
    if len(evr):
        pcs = ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(evr))
        console.print(f"[dim]PCA (trailing {max(window,252)}): {pcs}[/dim]")


@analyze.command("momentum")
@click.option("--top", default=None, type=int, help="Show only top-N.")
@click.pass_context
def analyze_momentum(ctx, top):
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    from src.analysis import momentum
    from src.visualization import tables
    scores = momentum.momentum_score(md.prices[md.tradable_present()],
                                     cfg.momentum_cfg.get("lookback_weights"))
    latest = scores.ffill().iloc[-1].dropna().sort_values(ascending=False)
    tables.print_momentum(latest, names=cfg.names, top=top, console=console)


@analyze.command("regime")
@click.pass_context
def analyze_regime(ctx):
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    from src.analysis import regime
    from src.visualization import tables
    rdf = regime.classify_regime(md, cfg)
    label = str(rdf["regime"].iloc[-1])
    score = float(rdf["score"].iloc[-1])
    tables.print_regime(label, score, detail={"as_of": str(md.dates.max().date())}, console=console)


@analyze.command("rotation")
@click.option("--lookback", default=63, help="Lookback (periods).")
@click.pass_context
def analyze_rotation(ctx, lookback):
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    from src.analysis import rotation
    from src.visualization import tables
    rdf = rotation.category_rotation(md, cfg, lookback)
    tables.print_rotation(rdf, console=console)


@analyze.command("all")
@click.pass_context
def analyze_all(ctx):
    for name in ("regime", "momentum", "rotation", "correlation"):
        console.rule(f"[bold]{name}")
        ctx.invoke({"regime": analyze_regime, "momentum": analyze_momentum,
                    "rotation": analyze_rotation, "correlation": analyze_correlation}[name])


# ---- backtest -------------------------------------------------------------
@cli.group()
def backtest():
    """Run and compare backtests."""


def _parse_strategies(strategy):
    if not strategy:
        return ACTIVE_STRATEGIES + BENCHMARK_STRATEGIES
    if strategy not in STRATEGY_REGISTRY:
        raise click.BadParameter(f"unknown strategy '{strategy}'. choices: {sorted(STRATEGY_REGISTRY)}")
    return [strategy]


@backtest.command("run")
@click.option("--strategy", default=None, help="Single strategy (default: all).")
@click.option("--rebalance", default=None, help="Single frequency (default: all configured).")
@click.option("--sort", default="sharpe", help="Sort comparison by this metric.")
@click.pass_context
def backtest_run(ctx, strategy, rebalance, sort):
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    strategies = _parse_strategies(strategy)
    freqs = [rebalance] if rebalance else cfg.rebalance.get("frequencies")
    suite = _build_suite(cfg, md, strategies, freqs)
    from src.visualization import tables
    tables.print_metrics_table(suite, sort_by=sort, console=console)
    ctx.obj["suite"] = suite


@backtest.command("compare")
@click.option("--sort", default="sharpe")
@click.pass_context
def backtest_compare(ctx, sort):
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    suite = _build_suite(cfg, md)
    from src.backtest import report, sharpe_correction
    from src.visualization import tables
    tables.print_metrics_table(suite, sort_by=sort, console=console)
    console.rule("[bold]best rebalance per strategy")
    bp = report.best_per_strategy(suite, sort)
    tables.print_metrics_table(bp, sort_by=sort, console=console)
    # multiple-testing correction over all tested configs
    console.rule("[bold]multiple-testing correction")
    ppy = 252 if md.freq == "daily" else 52
    tables.print_corrections(sharpe_correction.corrections_table(suite, ppy), console=console)


@backtest.command("oos")
@click.option("--select", default="sharpe", help="Metric to select the best config on the validation window.")
@click.option("--train", default=0.6, type=float)
@click.option("--valid", default=0.2, type=float)
@click.option("--test", default=0.2, type=float)
@click.pass_context
def backtest_oos(ctx, select, train, valid, test):
    """Out-of-sample: select the best config on validation, confirm on test."""
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    from src.backtest import run_oos
    from src.visualization import tables
    with console.status("[bold]running out-of-sample valid/test..."):
        oos = run_oos(md, cfg, ratios=(train, valid, test), select_by=select)
    tables.print_oos(oos, console=console)


@backtest.command("report")
@click.pass_context
def backtest_report(ctx):
    """Generate a consolidated Markdown report (output/REPORT.md) + charts."""
    from datetime import datetime
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    suite = _build_suite(cfg, md)
    from src.backtest import report
    from src.analysis import correlation, rotation
    from src.visualization import charts
    out = cfg.output_dir
    with console.status("[bold]building charts + report..."):
        report.export(suite, out, formats=("csv", "json"))
        charts.equity_curve(suite, out, freq="monthly")
        charts.regime_equity(suite, md, cfg, out, freq="monthly")
        charts.drawdown_chart(suite, out, freq="monthly")
        charts.correlation_heatmap(correlation.correlation_matrix(md.returns, 120), out, names=cfg.names)
        charts.rotation_chart(rotation.category_rotation(md, cfg, 63), out)
        if ("momentum_score", "monthly") in suite.results:
            charts.monthly_heatmap(suite.results[("momentum_score", "monthly")], out)
        path = report.markdown_report(suite, md, cfg, out,
                                      generated=datetime.now().strftime("%Y-%m-%d %H:%M"))
    console.print(f"[green]Report written:[/green] {path}")


@backtest.command("export")
@click.option("--format", "fmt", default="csv", type=click.Choice(["csv", "json", "both"]))
@click.pass_context
def backtest_export(ctx, fmt):
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    suite = _build_suite(cfg, md)
    from src.backtest import report
    formats = ("csv", "json") if fmt == "both" else (fmt,)
    files = report.export(suite, cfg.output_dir, formats=formats)
    console.print(f"[green]Exported {len(files)} file(s) to {cfg.output_dir}[/green]")
    for f in files:
        console.print(f"  {f}")


# ---- chart ----------------------------------------------------------------
@cli.group()
def chart():
    """Generate charts into the output directory."""


@chart.command("equity")
@click.option("--rebalance", default=None, help="Frequency to plot (default: first configured).")
@click.pass_context
def chart_equity(ctx, rebalance):
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    suite = _build_suite(cfg, md, freqs=[rebalance] if rebalance else None)
    from src.visualization import charts
    p = charts.equity_curve(suite, cfg.output_dir, freq=rebalance)
    console.print(f"[green]Saved[/green] {p}")


@chart.command("drawdown")
@click.option("--rebalance", default=None)
@click.pass_context
def chart_drawdown(ctx, rebalance):
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    suite = _build_suite(cfg, md, freqs=[rebalance] if rebalance else None)
    from src.visualization import charts
    p = charts.drawdown_chart(suite, cfg.output_dir, freq=rebalance)
    console.print(f"[green]Saved[/green] {p}")


@chart.command("regime")
@click.option("--rebalance", default=None)
@click.pass_context
def chart_regime(ctx, rebalance):
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    suite = _build_suite(cfg, md, freqs=[rebalance] if rebalance else None)
    from src.visualization import charts
    p = charts.regime_equity(suite, md, cfg, cfg.output_dir, freq=rebalance)
    console.print(f"[green]Saved[/green] {p}")


@chart.command("correlation")
@click.option("--window", default=120)
@click.pass_context
def chart_correlation(ctx, window):
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    from src.analysis import correlation
    from src.visualization import charts
    cm = correlation.correlation_matrix(md.returns, window)
    p = charts.correlation_heatmap(cm, cfg.output_dir, names=cfg.names)
    console.print(f"[green]Saved[/green] {p}")


@chart.command("rotation")
@click.option("--lookback", default=63)
@click.pass_context
def chart_rotation(ctx, lookback):
    cfg = ctx.obj["cfg"]
    md = _load_market_data(cfg)
    from src.analysis import rotation
    from src.visualization import charts
    rdf = rotation.category_rotation(md, cfg, lookback)
    p = charts.rotation_chart(rdf, cfg.output_dir)
    console.print(f"[green]Saved[/green] {p}")


if __name__ == "__main__":
    cli(obj={})
