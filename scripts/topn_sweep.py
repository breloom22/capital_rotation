"""Top-N parameter sensitivity sweep + robustness analysis (OOS + Deflated Sharpe).

Deterministic grid (no agents): for each (N, lookback, filter) run a monthly
backtest over full / validation / test windows, then judge robustness with the
same rigor tools the project ships -- valid->test stability and the multiple-
testing-corrected (Deflated) Sharpe.
"""
import numpy as np
import pandas as pd

from src.config import load_config
from src.data import build_market_data, storage
from src.backtest import BacktestEngine, chronological_split
from src.backtest.sharpe_correction import (
    james_stein_shrinkage, sharpe_sampling_variance, deflated_sharpe_ratio,
)
from src.strategy import get_strategy

cfg = load_config()
frames = storage.load_many(cfg.raw_dir, cfg.tickers)
md = build_market_data(frames, categories=cfg.categories, tradable=cfg.tradable_tickers,
                       freq="daily", start=cfg.backtest.get("start"), end=cfg.backtest.get("end"),
                       min_obs=cfg.data_cfg.get("min_obs", 100))
eng = BacktestEngine(md, cfg)
sp = chronological_split(md.dates, 0.6, 0.2, 0.2)

# lookback variant -> (weights dict, skip_recent days)
LOOKBACKS = {
    "blend_12_6_3": ({252: 0.5, 126: 0.3, 63: 0.2}, 0),   # current default
    "mom_12m":      ({252: 1.0}, 0),
    "mom_12_1":     ({252: 1.0}, 21),                      # classic 12-1 momentum
    "mom_6m":       ({126: 1.0}, 0),
    "mom_3m":       ({63: 1.0}, 0),
    "blend_12_6":   ({252: 0.5, 126: 0.5}, 0),
}
NS = [3, 5, 7, 10]
FILTERS = [True, False]

rows = []
_orig_skip = cfg.strategy_cfg.get("momentum", {}).get("skip_recent_days", 0)
common = None
for lbname, (weights, skip) in LOOKBACKS.items():
    cfg.strategy_cfg["momentum"]["skip_recent_days"] = skip
    for n in NS:
        for filt in FILTERS:
            params = {"n": n, "lookback_weights": weights, "absolute_filter": filt, "min_assets": 1}
            s = get_strategy("topn")(md, cfg, params)
            s.ensure_precomputed()
            if common is None:
                common = eng.common_start([s])
            try:
                rfull = eng.run(s, freq="monthly", start_date=common)
                rf = rfull.metrics
                va = eng.run(s, freq="monthly", start_date=max(sp.window("valid")[0], common),
                             end_date=sp.window("valid")[1]).metrics
                te = eng.run(s, freq="monthly", start_date=max(sp.window("test")[0], common),
                             end_date=sp.window("test")[1]).metrics
            except Exception as exc:
                print("skip", lbname, n, filt, exc); continue
            rows.append({
                "lookback": lbname, "n": n, "filt": "on" if filt else "off",
                "full_cagr": rf["cagr"], "full_sharpe": rf["sharpe"], "full_mdd": rf["mdd"],
                "full_calmar": rf["calmar"], "valid_sharpe": va["sharpe"], "test_sharpe": te["sharpe"],
                "test_mdd": te["mdd"], "n_periods": rf["n_periods"], "returns": rfull.returns,
            })
cfg.strategy_cfg["momentum"]["skip_recent_days"] = _orig_skip

df = pd.DataFrame(rows)
print(f"swept {len(df)} configs (monthly, full from {common.date()})\n")

# --- Deflated Sharpe across all swept configs (multiple-testing correction) ---
sh = df["full_sharpe"].to_numpy()
sig2 = float(np.nanmean([sharpe_sampling_variance(s, n) for s, n in zip(sh, df["n_periods"])]))
df["sharpe_js"] = james_stein_shrinkage(sh, sig2)
sr_p = sh / np.sqrt(252)
var_tr = float(np.nanvar(sr_p))
df["dsr"] = [deflated_sharpe_ratio(p, r, len(df), var_tr) for p, r in zip(sr_p, df["returns"])]

# === 1) top configs by VALIDATION Sharpe, with TEST (robustness) ===
print("=== Top 12 by VALIDATION Sharpe (then their TEST) ===")
top = df.sort_values("valid_sharpe", ascending=False).head(12)
print(f'{"lookback":13s} {"N":>2} {"filt":>4} | {"full_Sh":>7} {"valid_Sh":>8} {"test_Sh":>7} {"full_MDD":>8} {"test_MDD":>8} {"DSR":>5}')
for _, r in top.iterrows():
    print(f'{r.lookback:13s} {r.n:2d} {r.filt:>4} | {r.full_sharpe:7.2f} {r.valid_sharpe:8.2f} '
          f'{r.test_sharpe:7.2f} {r.full_mdd:7.1%} {r.test_mdd:7.1%} {r.dsr:5.2f}')

# === 2) marginal robustness by each knob (mean over the other knobs) ===
print("\n=== Marginal mean Sharpe by knob (full / valid / test) ===")
for knob in ["n", "lookback", "filt"]:
    g = df.groupby(knob)[["full_sharpe", "valid_sharpe", "test_sharpe", "full_mdd"]].mean()
    print(f"\n-- by {knob} --")
    for k, row in g.iterrows():
        print(f'  {str(k):13s}: full={row.full_sharpe:.2f} valid={row.valid_sharpe:.2f} '
              f'test={row.test_sharpe:.2f} MDD={row.full_mdd:.1%}')

# === 3) best by full-period Calmar & test Sharpe ===
print("\n=== Best full-period Sharpe / Calmar / test Sharpe ===")
for col in ["full_sharpe", "full_calmar", "test_sharpe"]:
    r = df.loc[df[col].idxmax()]
    print(f'  max {col:12s}: {r.lookback}/N={r.n}/filt={r.filt}  '
          f'(full_Sh={r.full_sharpe:.2f} test_Sh={r.test_sharpe:.2f} MDD={r.full_mdd:.1%} DSR={r.dsr:.2f})')

# current default for reference
cur = df[(df.lookback == "blend_12_6_3") & (df.n == 5) & (df["filt"] == "on")]
if len(cur):
    r = cur.iloc[0]
    print(f'\n  CURRENT default (blend_12_6_3/N=5/filt=on): full_Sh={r.full_sharpe:.2f} '
          f'valid_Sh={r.valid_sharpe:.2f} test_Sh={r.test_sharpe:.2f} MDD={r.full_mdd:.1%} DSR={r.dsr:.2f}')
print("\nDONE")
