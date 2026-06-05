"""Rich-based CLI tables for analysis output and the backtest comparison.

PUBLIC API (the CLI imports these exact names). Each prints to a rich Console
(create a default one if ``console`` is None) and returns None.

    print_metrics_table(suite, sort_by="sharpe", console=None)
    print_data_status(rows, console=None)
    print_correlation(corr_df, names=None, console=None)
    print_momentum(scores, names=None, top=None, console=None)
    print_regime(label, score, detail=None, console=None)
    print_rotation(rotation_df, console=None)

Formatting conventions:
* Ratios (Sharpe/Sortino/Calmar/beta/profit factor): 2 decimals.
* Percent metrics (returns/CAGR/vol/MDD/win rate/turnover/alpha): show as %.
* Right-align numbers; colour positive green / negative red where it helps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _console(console):
    if console is not None:
        return console
    from rich.console import Console
    return Console()


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------
# Metrics shown as percentages.
_PCT_METRICS = {
    "total_return", "cagr", "ann_volatility", "mdd", "win_rate",
    "avg_turnover", "total_turnover", "alpha",
}
# Metrics shown as 2-decimal ratios.
_RATIO_METRICS = {
    "sharpe", "sortino", "calmar", "beta", "profit_factor",
}
# Return-like fields where positive should read green / negative red.
_SIGNED_METRICS = {
    "total_return", "cagr", "mdd", "alpha", "sharpe", "sortino",
    "calmar", "profit_factor",
}

# Friendly column headers (fall back to the raw key otherwise).
_LABELS = {
    "total_return": "Total Return",
    "cagr": "CAGR",
    "ann_volatility": "Ann.Vol",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "mdd": "MaxDD",
    "mdd_duration_days": "Max Underwater Days",
    "mdd_recovery_days": "Recovery",
    "calmar": "Calmar",
    "win_rate": "Win Rate",
    "profit_factor": "Profit Factor",
    "alpha": "Alpha",
    "beta": "Beta",
    "avg_turnover": "Avg Turnover",
    "total_turnover": "Total Turnover",
    "n_rebalances": "Rebalances",
    "n_periods": "Periods",
    "final_equity": "Final Equity",
}

# Order metrics appear in the table when present.
_METRIC_ORDER = [
    "total_return", "cagr", "ann_volatility", "sharpe", "sortino",
    "mdd", "calmar", "win_rate", "profit_factor", "alpha", "beta",
    "avg_turnover", "n_rebalances", "final_equity",
]

# Curated columns for the on-screen comparison table (export keeps everything).
_KEY_METRICS = [
    "total_return", "cagr", "sharpe", "sortino", "mdd", "calmar",
    "win_rate", "avg_turnover", "final_equity",
]


def _is_number(x) -> bool:
    try:
        return x is not None and not isinstance(x, bool) and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _fmt_value(key: str, value) -> str:
    """Format a single metric value as plain text (no markup)."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        # inf / -inf / nan -> dash
        if _is_number(value):
            pass
        else:
            return "-"
    if not _is_number(value):
        return "-" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)

    v = float(value)
    if key in _PCT_METRICS:
        return f"{v * 100:+.2f}%" if key in _SIGNED_METRICS else f"{v * 100:.2f}%"
    if key in _RATIO_METRICS:
        return f"{v:.2f}"
    if key == "final_equity":
        return f"{v:,.0f}"
    if key in ("n_rebalances", "n_periods", "mdd_duration_days", "mdd_recovery_days"):
        return f"{v:,.0f}"
    # default numeric
    return f"{v:.2f}"


def _colour_for(key: str, value) -> str | None:
    """Return a rich style for a return-like field, else None."""
    if key not in _SIGNED_METRICS or not _is_number(value):
        return None
    v = float(value)
    if v > 0:
        return "green"
    if v < 0:
        return "red"
    return None


# metrics where a LOWER value is better (everything else higher-better; note MDD
# is negative, so a higher/less-negative MDD is better -> not listed here).
_LOWER_BETTER = {"ann_volatility", "avg_turnover", "total_turnover",
                 "mdd_duration_days", "mdd_recovery_days"}


def _cell(key: str, value, highlight: bool = False):
    """Build a rich-friendly (text, style) for a metric cell. ``highlight`` marks
    the best value in its column (bold + reverse)."""
    from rich.text import Text
    txt = _fmt_value(key, value)
    if highlight:
        return Text(txt, style="bold green reverse")
    style = _colour_for(key, value)
    return Text(txt, style=style or "")


def _best_per_column(df, metric_cols):
    """Index of the best value in each metric column (NaN-safe)."""
    best = {}
    for c in metric_cols:
        col = df[c].dropna()
        if col.empty:
            continue
        best[c] = col.idxmin() if c in _LOWER_BETTER else col.idxmax()
    return best


# --------------------------------------------------------------------------
# metrics comparison table
# --------------------------------------------------------------------------
def _suite_to_df(suite) -> pd.DataFrame:
    """Extract the numeric metrics DataFrame from a SuiteResult or DataFrame."""
    if isinstance(suite, pd.DataFrame):
        return suite
    mt = getattr(suite, "metrics_table", None)
    if isinstance(mt, pd.DataFrame):
        return mt
    # last resort: try to coerce
    return pd.DataFrame(suite)


def print_metrics_table(suite, sort_by: str = "sharpe", console=None) -> None:
    """Render the strategy x freq comparison table. ``suite`` is a SuiteResult
    (or its metrics DataFrame)."""
    from rich.table import Table

    con = _console(console)
    df = _suite_to_df(suite)

    table = Table(title="Backtest Comparison", title_style="bold")

    if df is None or df.empty:
        table.add_column("info")
        table.add_row("(no results)")
        con.print(table)
        return

    df = df.copy()

    # Sort by the requested metric if present (descending = best first).
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False, kind="mergesort")

    # Index columns (e.g. strategy, freq) become leading label columns.
    index_names = [n if n is not None else f"idx{i}"
                   for i, n in enumerate(df.index.names)]
    for name in index_names:
        table.add_column(str(name).title(), style="cyan", no_wrap=True)

    # Console shows a curated set of key metrics (the full metric set is kept in
    # the CSV/JSON export). This keeps the table readable on normal terminals.
    metric_cols = [c for c in _KEY_METRICS if c in df.columns]
    if not metric_cols:  # fall back to whatever numeric columns exist
        metric_cols = [c for c in _METRIC_ORDER if c in df.columns]

    for c in metric_cols:
        header = _LABELS.get(c, str(c))
        table.add_column(header, justify="right")

    best = _best_per_column(df, metric_cols)
    for idx, row in df.iterrows():
        idx_vals = idx if isinstance(idx, tuple) else (idx,)
        cells = [str(v) for v in idx_vals]
        # pad in case index has fewer levels than columns
        while len(cells) < len(index_names):
            cells.append("")
        rich_cells = list(cells)
        for c in metric_cols:
            rich_cells.append(_cell(c, row.get(c), highlight=(best.get(c) == idx)))
        table.add_row(*rich_cells)

    con.print(table)


# --------------------------------------------------------------------------
# data status
# --------------------------------------------------------------------------
def print_data_status(rows, console=None) -> None:
    """``rows``: list of dicts with keys ticker,name,start,end,obs,missing,tradable."""
    from rich.table import Table
    from rich.text import Text

    con = _console(console)
    table = Table(title="Data Status", title_style="bold")
    for col in ("Ticker", "Name", "Start", "End"):
        table.add_column(col, no_wrap=True)
    table.add_column("Obs", justify="right")
    table.add_column("Missing", justify="right")
    table.add_column("Tradable", justify="center")

    if not rows:
        con.print(table)
        return

    def _fmt_date(d):
        if d is None or (isinstance(d, float) and np.isnan(d)):
            return "-"
        try:
            return pd.Timestamp(d).date().isoformat()
        except (ValueError, TypeError):
            return str(d)

    def _fmt_int(x):
        return f"{int(x):,}" if _is_number(x) else "-"

    for r in rows:
        tradable = bool(r.get("tradable", False))
        trad_txt = Text("yes", style="green") if tradable else Text("no", style="red")
        table.add_row(
            str(r.get("ticker", "")),
            str(r.get("name", "")),
            _fmt_date(r.get("start")),
            _fmt_date(r.get("end")),
            _fmt_int(r.get("obs")),
            _fmt_int(r.get("missing")),
            trad_txt,
        )
    con.print(table)


# --------------------------------------------------------------------------
# correlation matrix
# --------------------------------------------------------------------------
def print_correlation(corr_df: pd.DataFrame, names: dict | None = None, console=None) -> None:
    from rich.table import Table
    from rich.text import Text

    con = _console(console)
    table = Table(title="Correlation Matrix", title_style="bold")

    if corr_df is None or corr_df.empty:
        table.add_column("info")
        table.add_row("(no data)")
        con.print(table)
        return

    names = names or {}
    cols = list(corr_df.columns)

    def _label(t):
        return str(names.get(t, t))

    table.add_column("", style="cyan", no_wrap=True)
    for c in cols:
        table.add_column(_label(c), justify="right")

    for r in corr_df.index:
        cells = [Text(_label(r), style="cyan")]
        for c in cols:
            v = corr_df.loc[r, c]
            if not _is_number(v):
                cells.append(Text("-"))
                continue
            v = float(v)
            if r == c:
                style = "dim"
            elif v >= 0.7:
                style = "red"
            elif v <= -0.3:
                style = "green"
            else:
                style = ""
            cells.append(Text(f"{v:.2f}", style=style))
        table.add_row(*cells)

    con.print(table)


# --------------------------------------------------------------------------
# momentum ranking
# --------------------------------------------------------------------------
def print_momentum(scores: pd.Series, names: dict | None = None, top: int | None = None, console=None) -> None:
    """``scores``: Series ticker -> momentum score (latest), sorted desc."""
    from rich.table import Table
    from rich.text import Text

    con = _console(console)
    table = Table(title="Momentum Ranking", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("Score", justify="right")

    if scores is None or len(scores) == 0:
        con.print(table)
        return

    names = names or {}
    s = scores.dropna().sort_values(ascending=False)
    if top is not None:
        s = s.head(int(top))

    for rank, (ticker, val) in enumerate(s.items(), start=1):
        style = "green" if _is_number(val) and float(val) > 0 else (
            "red" if _is_number(val) and float(val) < 0 else "")
        score_txt = f"{float(val):+.4f}" if _is_number(val) else "-"
        table.add_row(
            str(rank),
            str(ticker),
            str(names.get(ticker, ticker)),
            Text(score_txt, style=style),
        )
    con.print(table)


# --------------------------------------------------------------------------
# regime
# --------------------------------------------------------------------------
def print_regime(label: str, score: float, detail: dict | None = None, console=None) -> None:
    from rich.table import Table
    from rich.text import Text

    con = _console(console)

    label_str = str(label) if label is not None else "unknown"
    colour = {
        "risk_on": "green",
        "risk_off": "red",
        "neutral": "yellow",
    }.get(label_str, "white")

    table = Table(title="Market Regime", title_style="bold", show_header=False)
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value")

    table.add_row("Regime", Text(label_str.replace("_", " ").title(), style=f"bold {colour}"))

    if _is_number(score):
        sv = float(score)
        sstyle = "green" if sv > 0 else ("red" if sv < 0 else "")
        table.add_row("Score", Text(f"{sv:+.4f}", style=sstyle))
    else:
        table.add_row("Score", Text("-"))

    if detail:
        for k, v in detail.items():
            if _is_number(v):
                txt = f"{float(v):.4f}"
            else:
                txt = "-" if v is None else str(v)
            table.add_row(str(k).replace("_", " ").title(), txt)

    con.print(table)


# --------------------------------------------------------------------------
# category rotation
# --------------------------------------------------------------------------
def print_rotation(rotation_df: pd.DataFrame, console=None) -> None:
    """``rotation_df``: index=category, columns include rel_strength/flow/status."""
    from rich.table import Table
    from rich.text import Text

    con = _console(console)
    table = Table(title="Category Rotation", title_style="bold")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Rel.Strength", justify="right")
    table.add_column("Flow", justify="right")
    table.add_column("Status", justify="center")

    if rotation_df is None or rotation_df.empty:
        con.print(table)
        return

    def _signed_text(v, pct=False):
        if not _is_number(v):
            return Text("-")
        v = float(v)
        style = "green" if v > 0 else ("red" if v < 0 else "")
        txt = f"{v * 100:+.2f}%" if pct else f"{v:+.4f}"
        return Text(txt, style=style)

    status_style = {"inflow": "green", "outflow": "red", "neutral": "yellow"}

    for cat in rotation_df.index:
        row = rotation_df.loc[cat]
        rs = row.get("rel_strength") if hasattr(row, "get") else None
        flow = row.get("flow") if hasattr(row, "get") else None
        status = row.get("status") if hasattr(row, "get") else None
        status = str(status) if status is not None else "-"
        table.add_row(
            str(cat),
            _signed_text(rs, pct=True),
            _signed_text(flow, pct=False),
            Text(status, style=status_style.get(status, "")),
        )
    con.print(table)


# --------------------------------------------------------------------------
# Sharpe multiple-testing corrections
# --------------------------------------------------------------------------
def print_corrections(corr_df: pd.DataFrame, console=None) -> None:
    """``corr_df``: output of sharpe_correction.corrections_table
    (cols sharpe_raw, sharpe_js, dsr, rank_raw, rank_js, rank_change)."""
    from rich.table import Table
    from rich.text import Text

    con = _console(console)
    table = Table(title="Sharpe Multiple-Testing Correction", title_style="bold",
                  caption="JS = James-Stein shrunk · DSR = Deflated Sharpe Ratio (prob. true Sharpe>0 after N trials)")
    if corr_df is None or corr_df.empty:
        table.add_column("info"); table.add_row("(no data)"); con.print(table); return

    for col in ("Strategy", "Freq"):
        table.add_column(col, style="cyan", no_wrap=True)
    for col in ("Sharpe", "Sharpe(JS)", "ΔRank", "DSR"):
        table.add_column(col, justify="right")

    for idx, row in corr_df.iterrows():
        strat, freq = idx if isinstance(idx, tuple) else (idx, "")
        dsr = row.get("dsr")
        dsr_style = "green" if _is_number(dsr) and dsr >= 0.9 else ("red" if _is_number(dsr) and dsr < 0.5 else "")
        rc = int(row.get("rank_change", 0))
        rc_txt = f"{rc:+d}" if rc else "0"
        table.add_row(
            str(strat), str(freq),
            f"{row['sharpe_raw']:.2f}" if _is_number(row.get("sharpe_raw")) else "-",
            f"{row['sharpe_js']:.2f}" if _is_number(row.get("sharpe_js")) else "-",
            Text(rc_txt, style="red" if rc < 0 else ("green" if rc > 0 else "")),
            Text(f"{dsr:.2f}" if _is_number(dsr) else "-", style=dsr_style),
        )
    con.print(table)


# --------------------------------------------------------------------------
# out-of-sample robustness (valid vs test)
# --------------------------------------------------------------------------
def print_oos(oos, console=None) -> None:
    """``oos``: runner.OOSResult. Shows valid-vs-test metrics side by side."""
    from rich.table import Table
    from rich.text import Text

    con = _console(console)
    con.print(f"[bold]OOS split:[/bold] {oos.split.summary()}")
    sel = oos.selection
    if sel is not None:
        con.print(f"[bold]Selected on valid ({oos.selection_metric}):[/bold] "
                  f"[green]{sel[0]} / {sel[1]}[/green]")

    rob = oos.robustness
    table = Table(title="Validation vs Test robustness", title_style="bold")
    for col in ("Strategy", "Freq"):
        table.add_column(col, style="cyan", no_wrap=True)
    for col in ("V.CAGR", "T.CAGR", "V.Sharpe", "T.Sharpe", "V.MaxDD", "T.MaxDD"):
        table.add_column(col, justify="right")

    if rob is None or rob.empty:
        table.add_row("(no data)", "", "", "", "", "", "", ""); con.print(table); return

    def _p(v):
        return f"{v*100:+.1f}%" if _is_number(v) else "-"

    def _r(v):
        return f"{v:.2f}" if _is_number(v) else "-"

    for idx, row in rob.iterrows():
        strat, freq = idx if isinstance(idx, tuple) else (idx, "")
        vsh, tsh = row.get("valid_sharpe"), row.get("test_sharpe")
        # flag fragility: test Sharpe much worse than valid
        frag = _is_number(vsh) and _is_number(tsh) and (tsh < vsh - 0.3)
        tsh_txt = Text(_r(tsh), style="red" if frag else "")
        table.add_row(
            str(strat), str(freq),
            _p(row.get("valid_cagr")), _p(row.get("test_cagr")),
            _r(vsh), tsh_txt,
            _p(row.get("valid_mdd")), _p(row.get("test_mdd")),
        )
    con.print(table)
