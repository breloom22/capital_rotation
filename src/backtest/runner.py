"""Run a whole suite: every strategy x every rebalance frequency, aligned to a
common start date so all equity curves are directly comparable."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data import MarketData
from ..strategy import get_strategy, ACTIVE_STRATEGIES, BENCHMARK_STRATEGIES
from .engine import BacktestEngine, BacktestResult


@dataclass
class SuiteResult:
    results: dict                       # (strategy_name, freq) -> BacktestResult
    metrics_table: pd.DataFrame         # MultiIndex (strategy, freq) x metrics
    labels: dict = field(default_factory=dict)   # strategy_name -> display label
    start_date: pd.Timestamp | None = None


def build_strategy(name: str, data: MarketData, config):
    cls = get_strategy(name)
    params = config.strategy_params(name)
    return cls(data, config, params)


def run_suite(
    data: MarketData,
    config,
    strategy_names: list[str] | None = None,
    freqs: list[str] | None = None,
    progress=None,
    window: tuple | None = None,
) -> SuiteResult:
    """Run every (strategy x freq). ``window=(start,end)`` restricts the
    EVALUATION period (e.g. an out-of-sample test slice); signals still see full
    history (trailing => look-ahead safe)."""
    strategy_names = strategy_names or (ACTIVE_STRATEGIES + BENCHMARK_STRATEGIES)
    freqs = freqs or config.rebalance.get("frequencies", ["monthly"])

    # anchor the global RNG so any (current or future) stochastic component is
    # reproducible from config.seed. The current pipeline is already deterministic.
    np.random.seed(config.seed)

    engine = BacktestEngine(data, config)
    instances = {n: build_strategy(n, data, config) for n in strategy_names}
    common = engine.common_start(list(instances.values()))
    if window is not None:
        start = max(pd.Timestamp(window[0]), common)
        end = pd.Timestamp(window[1])
    else:
        start, end = common, None

    results: dict = {}
    rows = []
    for n, strat in instances.items():
        strat.ensure_precomputed()
        for f in freqs:
            try:
                res = engine.run(strat, freq=f, start_date=start, end_date=end)
            except Exception as exc:
                if progress is not None:
                    progress(n, f, error=str(exc))
                continue
            results[(n, f)] = res
            rows.append({"strategy": strat.display_name(), "freq": f, **res.metrics})
            if progress is not None:
                progress(n, f, error=None)

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.set_index(["strategy", "freq"])
    return SuiteResult(
        results=results, metrics_table=table,
        labels={n: instances[n].display_name() for n in instances},
        start_date=start,
    )


@dataclass
class OOSResult:
    split: "object"                      # splits.Split
    valid: SuiteResult
    test: SuiteResult
    selection: tuple                     # (strategy_label, freq) chosen on valid
    selection_metric: str
    robustness: pd.DataFrame             # valid vs test for every config


def run_oos(
    data: MarketData,
    config,
    strategy_names: list[str] | None = None,
    freqs: list[str] | None = None,
    ratios: tuple = (0.6, 0.2, 0.2),
    select_by: str = "sharpe",
    progress=None,
) -> OOSResult:
    """Out-of-sample protocol: select the best config on the VALIDATION window,
    then report its (and everyone's) performance on the untouched TEST window.

    Returns an :class:`OOSResult` whose ``robustness`` table puts valid & test
    metrics side by side so fragile configs (great in valid, mediocre in test)
    are obvious.
    """
    from .splits import chronological_split

    split = chronological_split(data.dates, *ratios)
    v0, v1 = split.window("valid")
    t0, t1 = split.window("test")
    if v0 is None or t0 is None:
        raise ValueError("not enough history for a valid/test split")

    valid = run_suite(data, config, strategy_names, freqs, progress=progress, window=(v0, v1))
    test = run_suite(data, config, strategy_names, freqs, progress=progress, window=(t0, t1))

    # selection on validation
    vt = valid.metrics_table
    selection = None
    if select_by in vt.columns and not vt.empty:
        selection = vt[select_by].idxmax()

    # side-by-side robustness table (valid vs test) for shared metrics
    cols = ["cagr", "sharpe", "sortino", "mdd", "calmar"]
    vt2 = valid.metrics_table.add_prefix("valid_")
    tt2 = test.metrics_table.add_prefix("test_")
    keep = [f"valid_{c}" for c in cols] + [f"test_{c}" for c in cols]
    robustness = vt2.join(tt2, how="outer")
    robustness = robustness[[c for c in keep if c in robustness.columns]]
    if select_by in vt.columns:
        robustness = robustness.sort_values(f"valid_{select_by}", ascending=False)

    return OOSResult(
        split=split, valid=valid, test=test,
        selection=selection, selection_metric=select_by, robustness=robustness,
    )
