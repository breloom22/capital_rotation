"""Backtesting package: engine, metrics, suite runner, OOS, corrections."""
from . import metrics, sharpe_correction, splits
from .engine import BacktestEngine, BacktestResult, generate_rebalance_dates
from .runner import run_suite, run_oos, SuiteResult, OOSResult, build_strategy
from .splits import chronological_split, walk_forward_windows, Split

__all__ = [
    "metrics", "sharpe_correction", "splits",
    "BacktestEngine", "BacktestResult", "generate_rebalance_dates",
    "run_suite", "run_oos", "SuiteResult", "OOSResult", "build_strategy",
    "chronological_split", "walk_forward_windows", "Split",
]
