"""Acceptance harness. ``python -m tests.check_module <name>`` exits 0 (PASS) or
1 (FAIL, with traceback). Names: momentum volume volatility correlation regime
rotation topn risk_parity momentum_score regime_based benchmark report tables
charts  -- or 'all'.

Each check asserts: correct output shapes, no NaN explosions, weights sum<=1,
and (where applicable) NO LOOK-AHEAD: a panel/weight computed on data truncated
at date T must equal the same value computed on the full panel. Trailing-window
code passes this exactly; future-peeking code fails it.
"""
from __future__ import annotations

import sys
import traceback
from io import StringIO

import numpy as np
import pandas as pd

from tests.fixtures import make_data

TOL = dict(check_exact=False, rtol=1e-6, atol=1e-9)


def _overlap_equal(full: pd.DataFrame, trunc: pd.DataFrame, cut) -> None:
    """Assert two panels agree on all rows <= cut (look-ahead test)."""
    a = full.loc[:cut].dropna(how="all")
    b = trunc.loc[:cut].dropna(how="all")
    common = a.index.intersection(b.index)
    assert len(common) > 10, "look-ahead test: too little overlap"
    pd.testing.assert_frame_equal(
        a.loc[common].sort_index(axis=1), b.loc[common].sort_index(axis=1), **TOL
    )


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def check_momentum():
    md, cfg = make_data()
    from src.analysis import momentum as M
    P = md.prices
    w = {252: 0.5, 126: 0.3, 63: 0.2}
    s = M.momentum_score(P, w)
    assert s.shape == P.shape, f"score shape {s.shape} != {P.shape}"
    cut = P.index[int(len(P) * 0.7)]
    _overlap_equal(M.momentum_score(P, w), M.momentum_score(P.loc[:cut], w), cut)
    tr = M.total_return(P, 63)
    assert tr.shape == P.shape
    rk = M.momentum_rank(s)
    assert rk.shape == P.shape
    am = M.absolute_momentum(P, 252)
    assert am.shape == P.shape
    M.dual_momentum(P, 252)
    # momentum over a known constant series is ~0
    print("momentum PASS")


def check_volume():
    md, cfg = make_data()
    from src.analysis import volume as V
    vr = V.volume_ratio(md.volume, 20)
    assert vr.shape == md.volume.shape
    cut = md.dates[int(len(md.dates) * 0.7)]
    _overlap_equal(V.volume_ratio(md.volume, 20), V.volume_ratio(md.volume.loc[:cut], 20), cut)
    V.dollar_volume_ratio(md.dollar_volume, 20)
    V.volume_trend(md.volume, 20)
    o = V.obv(md.close, md.volume)
    assert o.shape == md.close.shape
    mf = V.money_flow(md.close, md.volume, 20)
    assert mf.shape == md.close.shape
    print("volume PASS")


def check_volatility():
    md, cfg = make_data()
    from src.analysis import volatility as Vol
    rv = Vol.realized_vol(md.returns, 21, 252)
    assert rv.shape == md.returns.shape
    assert (rv.dropna() >= 0).all().all(), "vol must be non-negative"
    cut = md.dates[int(len(md.dates) * 0.7)]
    _overlap_equal(Vol.realized_vol(md.returns, 21, 252),
                   Vol.realized_vol(md.returns.loc[:cut], 21, 252), cut)
    a = Vol.atr(md.high, md.low, md.close, 14)
    assert a.shape == md.close.shape
    Vol.atr_pct(md.high, md.low, md.close, 14)
    vix = md.prices["^VIX"] if "^VIX" in md.prices else md.prices.iloc[:, 0]
    reg = Vol.vix_regime(vix)
    assert len(reg) == len(vix)
    assert isinstance(Vol.vol_regime_label(30.0), str)
    print("volatility PASS")


def check_correlation():
    md, cfg = make_data()
    from src.analysis import correlation as C
    cm = C.correlation_matrix(md.returns, 120)
    assert cm.shape[0] == cm.shape[1]
    rac = C.rolling_avg_correlation(md.returns, 120)
    assert rac.shape == md.returns.shape
    cut = md.dates[int(len(md.dates) * 0.7)]
    _overlap_equal(C.rolling_avg_correlation(md.returns, 120),
                   C.rolling_avg_correlation(md.returns.loc[:cut], 120), cut)
    cc = C.correlation_change(md.returns, 60)
    assert isinstance(cc, pd.Series)
    pca = C.pca_explained(md.returns, 252, 3)
    assert "explained_variance_ratio" in pca
    print("correlation PASS")


def check_regime():
    md, cfg = make_data()
    from src.analysis import regime as R
    rdf = R.classify_regime(md, cfg)
    assert "regime" in rdf.columns
    assert set(rdf["regime"].dropna().unique()) <= {"risk_on", "risk_off", "neutral"}
    cut = md.dates[int(len(md.dates) * 0.7)]
    # look-ahead on the score column
    full = R.classify_regime(md, cfg)
    trunc = R.classify_regime(md.slice(end=cut), cfg)
    _overlap_equal(full[["score"]], trunc[["score"]], cut)
    assert isinstance(R.regime_at(rdf, cut), str)
    print("regime PASS")


def check_rotation():
    md, cfg = make_data()
    from src.analysis import rotation as Rot
    rs = Rot.relative_strength(md.prices, 63)
    assert rs.shape == md.prices.shape
    io = Rot.inflow_outflow(md.close, md.volume, 20)
    assert io.shape == md.close.shape
    cr = Rot.category_rotation(md, cfg, 63)
    assert "status" in cr.columns
    ll = Rot.lead_lag(md.returns, md.tradable[0], md.tradable[1], 10)
    assert "best_lag" in ll
    Rot.cycle_length(md.returns, 252)
    print("rotation PASS")


# --------------------------------------------------------------------------
# strategies
# --------------------------------------------------------------------------
def _strategy_check(name: str):
    md, cfg = make_data()
    from src.strategy import get_strategy
    from src.backtest import BacktestEngine
    cls = get_strategy(name)
    strat = cls(md, cfg, cfg.strategy_params(name))
    strat.ensure_precomputed()
    dates = md.dates[max(strat.warmup, 260):]
    sample = list(dates[:: max(1, len(dates) // 6)])[:6]
    for d in sample:
        w = strat.target_weights(d)
        assert isinstance(w, pd.Series), f"{name}: weights not a Series"
        if len(w):
            assert w.notna().all(), f"{name}: NaN weight at {d.date()}"
            assert (w >= -1e-9).all(), f"{name}: negative weight at {d.date()}"
            assert w.sum() <= 1.0 + 1e-6, f"{name}: weights sum {w.sum():.4f} > 1 at {d.date()}"
            assert set(w.index) <= set(md.tradable_present()), f"{name}: weight on non-tradable"
    # look-ahead: weights at an early date unchanged when future data removed
    d = sample[2]
    strat2 = cls(md.slice(end=d), cfg, cfg.strategy_params(name))
    strat2.ensure_precomputed()
    w_full, w_trunc = strat.target_weights(d), strat2.target_weights(d)
    common = w_full.index.union(w_trunc.index)
    a = w_full.reindex(common).fillna(0.0)
    b = w_trunc.reindex(common).fillna(0.0)
    assert np.allclose(a.values, b.values, atol=1e-6), f"{name}: LOOK-AHEAD in target_weights"
    # full engine run
    eng = BacktestEngine(md, cfg)
    res = eng.run(strat, freq="monthly", start_date=md.dates[max(strat.warmup, 260)])
    assert res.equity.iloc[-1] > 0 and np.isfinite(res.metrics["sharpe"])
    # recorded allocation must add up: weights + cash == 1 at every rebalance
    inv = res.weights.sum(axis=1) + res.cash_weight.reindex(res.weights.index)
    assert np.allclose(inv.dropna().to_numpy(), 1.0, atol=1e-6), \
        f"{name}: weights + cash != 1 (recording bug)"
    # cash fraction must never go negative (no accidental leverage)
    assert (res.cash_weight.dropna() >= -1e-6).all(), f"{name}: negative cash weight"
    print(f"{name} PASS  (final={res.metrics['final_equity']:.0f}, sharpe={res.metrics['sharpe']:.2f})")


def check_topn():           _strategy_check("topn")
def check_risk_parity():    _strategy_check("risk_parity")
def check_momentum_score(): _strategy_check("momentum_score")
def check_regime_based():   _strategy_check("regime_based")
def check_min_variance():   _strategy_check("min_variance")
def check_regime_budget():  _strategy_check("regime_budget")


def check_erc():
    """risk_parity with method='erc' must also satisfy the strategy contract."""
    md, cfg = make_data()
    from src.strategy import get_strategy
    from src.backtest import BacktestEngine
    params = {**cfg.strategy_params("risk_parity"), "method": "erc"}
    strat = get_strategy("risk_parity")(md, cfg, params)
    strat.ensure_precomputed()
    d = md.dates[max(strat.warmup, 300)]
    w = strat.target_weights(d)
    assert isinstance(w, pd.Series) and w.notna().all() and w.sum() <= 1 + 1e-6
    res = BacktestEngine(md, cfg).run(strat, freq="monthly", start_date=md.dates[max(strat.warmup, 300)])
    assert res.equity.iloc[-1] > 0
    print("erc PASS")


def check_benchmark():
    for n in ("bench_6040", "bench_equal", "bench_bh"):
        _strategy_check(n)
    print("benchmark PASS")


# --------------------------------------------------------------------------
# report / viz
# --------------------------------------------------------------------------
def _small_suite():
    md, cfg = make_data()
    from src.backtest import run_suite
    suite = run_suite(md, cfg, ["bench_equal", "bench_6040"], ["monthly"])
    return md, cfg, suite


def check_report():
    md, cfg, suite = _small_suite()
    from src.backtest import report
    tbl = report.comparison_table(suite, sort_by="sharpe")
    assert isinstance(tbl, pd.DataFrame) and len(tbl) >= 2
    bp = report.best_per_strategy(suite, "sharpe")
    assert isinstance(bp, pd.DataFrame)
    files = report.export(suite, cfg.output_dir / "_test_export", formats=("csv", "json"))
    assert files and all(p.exists() for p in files)
    print("report PASS")


def check_tables():
    md, cfg, suite = _small_suite()
    from src.visualization import tables
    from rich.console import Console
    buf = StringIO()
    con = Console(file=buf, width=160)
    tables.print_metrics_table(suite, console=con)
    from src.analysis import correlation, regime, rotation
    tables.print_correlation(correlation.correlation_matrix(md.returns, 120), names=cfg.names, console=con)
    rdf = regime.classify_regime(md, cfg)
    tables.print_regime(rdf["regime"].iloc[-1], float(rdf["score"].iloc[-1]), console=con)
    tables.print_rotation(rotation.category_rotation(md, cfg, 63), console=con)
    assert len(buf.getvalue()) > 50, "tables produced no output"
    print("tables PASS")


def check_charts():
    md, cfg, suite = _small_suite()
    from src.visualization import charts
    out = cfg.output_dir / "_test_charts"
    p1 = charts.equity_curve(suite, out)
    p2 = charts.drawdown_chart(suite, out)
    from src.analysis import correlation
    p3 = charts.correlation_heatmap(correlation.correlation_matrix(md.returns, 120), out, names=cfg.names)
    for p in (p1, p2, p3):
        assert p and p.exists(), f"chart not written: {p}"
    print("charts PASS")


# --------------------------------------------------------------------------
# new features: corrections / OOS / overlay
# --------------------------------------------------------------------------
def check_corrections():
    md, cfg = make_data()
    from src.backtest import run_suite, sharpe_correction
    suite = run_suite(md, cfg, ["topn", "risk_parity", "bench_equal", "bench_bh"], ["monthly", "quarterly"])
    ct = sharpe_correction.corrections_table(suite, 252)
    assert {"sharpe_raw", "sharpe_js", "dsr", "rank_change"} <= set(ct.columns)
    dsr = ct["dsr"].dropna()
    assert ((dsr >= 0) & (dsr <= 1)).all(), "DSR must be a probability in [0,1]"
    raw_spread = ct["sharpe_raw"].max() - ct["sharpe_raw"].min()
    js_spread = ct["sharpe_js"].max() - ct["sharpe_js"].min()
    assert js_spread <= raw_spread + 1e-9, "JS must not widen the spread"
    print("corrections PASS")


def check_oos():
    md, cfg = make_data()
    from src.backtest import run_oos, chronological_split, walk_forward_windows
    sp = chronological_split(md.dates, 0.6, 0.2, 0.2)
    assert len(sp.train) and len(sp.valid) and len(sp.test)
    assert sp.train[-1] < sp.valid[0] <= sp.valid[-1] < sp.test[0], "splits must be ordered/disjoint"
    wf = walk_forward_windows(md.dates, n_folds=3, train_frac=0.5, embargo=1)
    assert len(wf) >= 1 and wf[0][0] < wf[0][1] <= wf[0][2]
    oos = run_oos(md, cfg, ["topn", "bench_equal"], ["monthly"])
    assert oos.selection is not None and not oos.robustness.empty
    print("oos PASS")


def check_overlay():
    md, cfg = make_data()
    from src.backtest import BacktestEngine
    from src.strategy import get_strategy
    start = md.dates[300]
    s = get_strategy("topn")(md, cfg, cfg.strategy_params("topn"))
    r_off = BacktestEngine(md, cfg).run(s, freq="monthly", start_date=start)
    # enable caps only (isolate from corr gate)
    cfg.backtest_cfg["risk_overlay"] = {"enabled": True, "max_single_weight": 0.1,
                                        "max_gross": 1.0, "corr_spike": {"enabled": False}}
    s2 = get_strategy("topn")(md, cfg, cfg.strategy_params("topn"))
    r_on = BacktestEngine(md, cfg).run(s2, freq="monthly", start_date=start)
    assert r_on.weights.max().max() <= 0.1 + 0.02, f"cap not enforced: {r_on.weights.max().max()}"
    assert abs(r_off.metrics["final_equity"] - r_on.metrics["final_equity"]) > 1, "overlay had no effect"
    print("overlay PASS")


CHECKS = {
    "momentum": check_momentum, "volume": check_volume, "volatility": check_volatility,
    "correlation": check_correlation, "regime": check_regime, "rotation": check_rotation,
    "topn": check_topn, "risk_parity": check_risk_parity,
    "momentum_score": check_momentum_score, "regime_based": check_regime_based,
    "min_variance": check_min_variance, "erc": check_erc,
    "regime_budget": check_regime_budget,
    "benchmark": check_benchmark, "report": check_report, "tables": check_tables,
    "charts": check_charts,
    "corrections": check_corrections, "oos": check_oos, "overlay": check_overlay,
}

ANALYSIS = ["momentum", "volume", "volatility", "correlation", "regime", "rotation"]
STRATEGIES = ["topn", "risk_parity", "momentum_score", "regime_based", "min_variance",
              "erc", "regime_budget", "benchmark"]
VIZ = ["report", "tables", "charts"]
FEATURES = ["corrections", "oos", "overlay"]


def main(argv):
    targets = argv or ["all"]
    if targets == ["all"]:
        targets = ANALYSIS + STRATEGIES + VIZ + FEATURES
    failed = []
    for name in targets:
        if name not in CHECKS:
            print(f"unknown check '{name}'"); failed.append(name); continue
        try:
            CHECKS[name]()
        except Exception:
            print(f"{name} FAIL")
            traceback.print_exc()
            failed.append(name)
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print(f"\nALL PASS: {targets}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
