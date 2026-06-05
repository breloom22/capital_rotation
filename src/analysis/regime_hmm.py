"""HMM-based market regime detection (look-ahead safe).

A statistical alternative to the rule-based ``regime.classify_regime``. A Gaussian
HMM is fit on market-level features and the per-date regime is the argmax of the
**forward-filtered** state probability -- P(S_t | y_1..y_t), which uses only past
observations (the forbidden alternative, forward-backward smoothing, would leak
the future).

Two guards make it backtest-safe (ported from regime-portfolio-HMM-RL):
1. **Forward filter only** (``_hmm.forward_filter``), never smoothing.
2. **Expanding-window refit** on an absolute calendar with **train-only feature
   normalisation**: at each quarterly anchor the model is refit on data strictly
   before the anchor, and used to label only the following segment. So the model
   that labels date t was trained on data < t, and the features it consumes are
   z-scored with train-window stats.

Output matches the rule-based contract: a DataFrame indexed by date with columns
['regime','score'], regime in {'risk_on','neutral','risk_off'}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _hmm


def _rolling_avg_corr(returns: pd.DataFrame, window: int) -> pd.Series:
    """Trailing average pairwise correlation of the sleeve (look-ahead safe).

    NaN-robust: within each window, drop columns that are too sparse (e.g. an
    asset not yet listed) rather than discarding the whole row -- otherwise a
    single late-inception member would wipe out all earlier history."""
    out = pd.Series(np.nan, index=returns.index)
    n = len(returns)
    thresh = max(3, window // 2)
    for i in range(window - 1, n):
        block = returns.iloc[i - window + 1:i + 1].dropna(axis=1, thresh=thresh)
        if block.shape[1] < 2:
            continue
        c = block.corr().to_numpy()
        if c.shape[0] < 2:
            continue
        m = ~np.eye(c.shape[0], dtype=bool)
        v = np.nanmean(c[m])
        out.iloc[i] = v
    return out


def build_market_features(data, config, vol_window: int = 20, dd_window: int = 40) -> pd.DataFrame:
    """Market-level feature panel for the HMM (all trailing -> look-ahead safe).

    Built on the RISK-ON sleeve (config roles) plus VIX, NOT the full mixed
    universe -- a cross-sectional mean over bonds/gold/FX is not a coherent
    'market'. Column 0 is ``mkt_ret`` so the HMM states can be canonically sorted
    by mean return.
    """
    roles = config.roles
    cats = config.categories
    # market sleeve = risk-on roles, EXCLUDING crypto (late inception would
    # truncate the feature history). Long-history equities make a coherent
    # "market" whose return/vol/correlation define the regime.
    sleeve = [t for t in roles.get("risk_on", [])
              if t in data.returns.columns and cats.get(t) != "crypto"]
    if len(sleeve) < 2:
        sleeve = [t for t in data.tradable_present() if cats.get(t) != "crypto"][:6]
    R = data.returns[sleeve]

    mkt_ret = R.mean(axis=1)
    mkt_vol = mkt_ret.rolling(vol_window, min_periods=max(5, vol_window // 2)).std() * np.sqrt(252)
    disp = R.std(axis=1)
    avg_corr = _rolling_avg_corr(R, vol_window)
    wealth = (1.0 + mkt_ret.fillna(0.0)).cumprod()
    dd = wealth / wealth.rolling(dd_window, min_periods=max(5, dd_window // 2)).max() - 1.0

    vix_t = roles.get("vix", "^VIX")
    vix = data.prices[vix_t] if vix_t in data.prices.columns else pd.Series(np.nan, index=data.dates)

    feats = pd.DataFrame({
        "mkt_ret": mkt_ret, "mkt_vol": mkt_vol, "avg_corr": avg_corr,
        "disp": disp, "dd": dd, "vix": vix,
    })
    return feats


def _state_label_map(order: np.ndarray) -> dict:
    """Map HMM state indices (given ascending-mean-return ``order``) to
    {'risk_off','neutral','risk_on'} by return rank (bottom/top third)."""
    k = len(order)
    labels = {}
    for rank, state in enumerate(order):
        if rank < max(1, k // 3):
            labels[state] = "risk_off"
        elif rank >= k - max(1, k // 3):
            labels[state] = "risk_on"
        else:
            labels[state] = "neutral"
    return labels


def classify_regime_hmm(data, config, n_states: int | None = None,
                        refit_freq: int = 126, min_train: int = 504,
                        train_window: int = 2520, burn_in: int = 200,
                        n_iter: int = 30, seed: int = 42,
                        n_states_range=(2, 3, 4)) -> pd.DataFrame:
    """Per-date HMM regime with rolling-window refit. Returns DataFrame indexed
    by ``data.dates`` with columns ['regime','score'] (score in [-1,1], >0 risk-on).

    ``train_window`` caps each refit's history (default 10y) -- bounds compute and
    keeps the model adapted to recent regime structure. Refit cadence
    ``refit_freq`` (default semi-annual). Look-ahead safe: train = data strictly
    before the segment, normalisation uses train stats, inference is forward-only."""
    feats_all = build_market_features(data, config).dropna()
    if len(feats_all) < min_train + refit_freq:
        return pd.DataFrame({"regime": "neutral", "score": 0.0}, index=data.dates)

    fidx = feats_all.index
    X = feats_all.to_numpy(dtype=np.float64)
    n = len(fidx)

    regime = pd.Series("neutral", index=fidx, dtype=object)
    score = pd.Series(0.0, index=fidx, dtype=float)

    anchors = list(range(min_train, n, refit_freq))
    if not anchors:
        anchors = [min_train]

    # choose n_states ONCE on the first train window (BIC), then reuse (faithful + fast)
    if n_states is None:
        train0 = X[max(0, anchors[0] - train_window):anchors[0]]
        mu0, sd0 = train0.mean(0), train0.std(0) + 1e-9
        best_k, best_bic = n_states_range[0], np.inf
        for k in n_states_range:
            try:
                p = _hmm.fit_hmm((train0 - mu0) / sd0, k, n_iter=n_iter, seed=seed)
                b = _hmm.bic((train0 - mu0) / sd0, p)
                if b < best_bic:
                    best_bic, best_k = b, k
            except Exception:
                continue
        n_states = best_k

    for ai, anchor in enumerate(anchors):
        seg_hi = anchors[ai + 1] if ai + 1 < len(anchors) else n
        train = X[max(0, anchor - train_window):anchor]   # capped history before segment
        mu, sd = train.mean(0), train.std(0) + 1e-9
        try:
            params = _hmm.fit_hmm((train - mu) / sd, n_states, n_iter=n_iter, seed=seed)
        except Exception:
            continue
        order = _hmm.mean_return_order(params, col=0)
        lab = _state_label_map(order)
        on_states = [s for s, v in lab.items() if v == "risk_on"]
        off_states = [s for s, v in lab.items() if v == "risk_off"]

        # forward-filter a trailing window ending at seg_hi (burn-in converges the
        # filter; every row uses only past obs -> look-ahead safe), label [anchor:seg_hi]
        lo = max(0, anchor - burn_in)
        Xn = (X[lo:seg_hi] - mu) / sd
        filt = _hmm.forward_filter(Xn, params)          # (seg_hi-lo, K)
        seg = filt[anchor - lo:seg_hi - lo]             # rows for [anchor, seg_hi)
        seg_states = seg.argmax(axis=1)
        seg_regime = [lab[s] for s in seg_states]
        seg_score = seg[:, on_states].sum(1) - seg[:, off_states].sum(1)

        regime.iloc[anchor:seg_hi] = seg_regime
        score.iloc[anchor:seg_hi] = seg_score

    out = pd.DataFrame({"regime": regime, "score": score})
    out = out.reindex(data.dates)
    out["regime"] = out["regime"].fillna("neutral")
    out["score"] = out["score"].fillna(0.0)
    return out
