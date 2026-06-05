"""Central configuration loader.

Loads the three YAML files under ``config/`` and exposes typed accessors used
everywhere else. Paths are resolved relative to the project root so the CLI can
be run from any working directory.

The whole project is config-driven: no asset list, parameter, cost or schedule
is hard-coded in the logic modules -- it all flows from here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# capital_rotation/  (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


@dataclass
class Asset:
    ticker: str
    name: str
    category: str
    tradable: bool = True


class Config:
    """Parsed view over the three config files."""

    def __init__(self, config_dir: Path | str = CONFIG_DIR):
        self.config_dir = Path(config_dir)
        self.assets_cfg = _load_yaml(self.config_dir / "assets.yaml")
        self.strategy_cfg = _load_yaml(self.config_dir / "strategy.yaml")
        self.backtest_cfg = _load_yaml(self.config_dir / "backtest.yaml")

        self.assets: list[Asset] = [
            Asset(
                ticker=a["ticker"],
                name=a.get("name", a["ticker"]),
                category=a.get("category", "other"),
                tradable=bool(a.get("tradable", True)),
            )
            for a in self.assets_cfg.get("assets", [])
        ]

    # ---- asset universe ----------------------------------------------------
    @property
    def tickers(self) -> list[str]:
        return [a.ticker for a in self.assets]

    @property
    def tradable_tickers(self) -> list[str]:
        return [a.ticker for a in self.assets if a.tradable]

    @property
    def categories(self) -> dict[str, str]:
        """ticker -> category."""
        return {a.ticker: a.category for a in self.assets}

    @property
    def category_members(self) -> dict[str, list[str]]:
        """category -> list of tickers (preserves config order)."""
        out: dict[str, list[str]] = {}
        for a in self.assets:
            out.setdefault(a.category, []).append(a.ticker)
        return out

    def tradable_in_category(self, category: str) -> list[str]:
        return [a.ticker for a in self.assets if a.category == category and a.tradable]

    @property
    def names(self) -> dict[str, str]:
        return {a.ticker: a.name for a in self.assets}

    @property
    def roles(self) -> dict[str, Any]:
        return self.assets_cfg.get("roles", {})

    # ---- paths -------------------------------------------------------------
    def _resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def raw_dir(self) -> Path:
        d = self._resolve(self.backtest_cfg.get("data", {}).get("raw_dir", "data/raw"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def output_dir(self) -> Path:
        d = self._resolve(self.backtest_cfg.get("data", {}).get("output_dir", "output"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- typed sub-sections ------------------------------------------------
    @property
    def backtest(self) -> dict[str, Any]:
        return self.backtest_cfg.get("backtest", {})

    @property
    def costs(self) -> dict[str, Any]:
        return self.backtest_cfg.get("costs", {})

    @property
    def rebalance(self) -> dict[str, Any]:
        return self.backtest_cfg.get("rebalance", {})

    @property
    def data_cfg(self) -> dict[str, Any]:
        return self.backtest_cfg.get("data", {})

    @property
    def momentum_cfg(self) -> dict[str, Any]:
        return self.strategy_cfg.get("momentum", {})

    def strategy_params(self, name: str) -> dict[str, Any]:
        return self.strategy_cfg.get("strategies", {}).get(name, {})

    @property
    def benchmarks_cfg(self) -> dict[str, Any]:
        return self.strategy_cfg.get("benchmarks", {})

    # ---- convenience -------------------------------------------------------
    @property
    def benchmark_ticker(self) -> str:
        return self.backtest.get("benchmark", "SPY")

    @property
    def trading_days(self) -> int:
        return int(self.backtest.get("trading_days", 252))

    @property
    def risk_free_rate(self) -> float:
        return float(self.backtest.get("risk_free_rate", 0.0))

    @property
    def seed(self) -> int:
        return int(self.backtest.get("seed", 42))

    @property
    def one_way_cost(self) -> float:
        return float(self.costs.get("commission", 0.0)) + float(self.costs.get("slippage", 0.0))


def load_config(config_dir: Path | str = CONFIG_DIR) -> Config:
    return Config(config_dir)
