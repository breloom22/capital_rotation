"""Strategy package. Importing it registers every strategy in STRATEGY_REGISTRY."""
from .base import Strategy, register, get_strategy, STRATEGY_REGISTRY

# import side-effect: populate the registry
from . import (  # noqa: F401
    topn, risk_parity, momentum_score, regime_based, min_variance,
    regime_budget, benchmark,
)

#: default comparison suite. ``regime_budget`` is registered + available
#: (run it via --strategy regime_budget) but kept OUT of the default suite: it
#: reliably cuts drawdown yet did not improve risk-adjusted return on this
#: universe (Top-N already de-risks via its cash filter; min-var has no edge).
ACTIVE_STRATEGIES = ["topn", "risk_parity", "momentum_score", "regime_based", "min_variance"]
OPTIONAL_STRATEGIES = ["regime_budget"]
BENCHMARK_STRATEGIES = ["bench_6040", "bench_equal", "bench_bh"]

__all__ = [
    "Strategy", "register", "get_strategy", "STRATEGY_REGISTRY",
    "ACTIVE_STRATEGIES", "OPTIONAL_STRATEGIES", "BENCHMARK_STRATEGIES",
]
