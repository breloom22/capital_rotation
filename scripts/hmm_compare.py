"""Decisive head-to-head: does HMM-regime budgeting on an edge base beat plain Top-N?"""
import time
from src.config import load_config
from src.data import build_market_data, storage
from src.backtest import BacktestEngine, chronological_split
from src.strategy import get_strategy

cfg = load_config()
frames = storage.load_many(cfg.raw_dir, cfg.tickers)
md = build_market_data(frames, categories=cfg.categories, tradable=cfg.tradable_tickers,
                       freq="daily", start=cfg.backtest.get("start"), end=cfg.backtest.get("end"),
                       min_obs=cfg.data_cfg.get("min_obs", 100))
eng = BacktestEngine(md, cfg)

NF = {"absolute_filter": False}          # edge base: Top-N with cash filter OFF (always invested)
RB = cfg.strategy_params("regime_budget")


def mk(name, params):
    s = get_strategy(name)(md, cfg, params); s.ensure_precomputed(); return s


t0 = time.time()
strats = {
 "A Top-N (plain)":            mk("topn", cfg.strategy_params("topn")),
 "B Top-N no-filter (edge)":   mk("topn", {**cfg.strategy_params("topn"), **NF}),
 "C HMM-Budget(edge)":         mk("regime_budget", {**RB, "base": "topn", "base_params": NF,
                                                    "regime_source": "hmm", "mode": "continuous"}),
 "D Rule-Budget(edge)":        mk("regime_budget", {**RB, "base": "topn", "base_params": NF,
                                                    "regime_source": "rule", "mode": "continuous"}),
}
print("precompute %.0fs" % (time.time() - t0))
start = eng.common_start(list(strats.values()))


def row(s, a, b=None):
    m = eng.run(s, freq="monthly", start_date=a, end_date=b).metrics
    return m["cagr"], m["sharpe"], m["sortino"], m["mdd"], m["calmar"], m["final_equity"]


print("\n=== FULL PERIOD (monthly, from %s) ===" % start.date())
print(f'{"strategy":26s} {"CAGR":>7} {"Sharpe":>7} {"Sortino":>7} {"MaxDD":>8} {"Calmar":>7} {"Final$":>12}')
for k, s in strats.items():
    c, sh, so, dd, ca, fe = row(s, start)
    print(f"{k:26s} {c:6.1%} {sh:7.2f} {so:7.2f} {dd:7.1%} {ca:7.2f} {fe:12,.0f}")

sp = chronological_split(md.dates, 0.6, 0.2, 0.2)
for win in ["valid", "test"]:
    a, b = sp.window(win)
    print(f"\n=== {win.upper()} ({a.date()}..{b.date()}) ===")
    for k, s in strats.items():
        c, sh, so, dd, ca, fe = row(s, max(a, start), b)
        print(f"{k:26s} CAGR={c:6.1%} Sharpe={sh:5.2f} Sortino={so:5.2f} MaxDD={dd:7.1%} Calmar={ca:5.2f}")
print("\nDONE")
