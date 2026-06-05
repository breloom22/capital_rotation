"""Build comparison reports and export backtest results.

Consumes a ``SuiteResult`` (src/backtest/runner.py). No look-ahead concerns
here -- this is pure post-processing of completed backtests.

PUBLIC API (the CLI imports these exact names):
    comparison_table(suite, sort_by="sharpe", ascending=False) -> pd.DataFrame
    best_per_strategy(suite, metric="sharpe") -> pd.DataFrame
    export(suite, output_dir, formats=("csv", "json")) -> list[Path]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .runner import SuiteResult


def comparison_table(suite: SuiteResult, sort_by: str = "sharpe", ascending: bool = False) -> pd.DataFrame:
    """Tidy metrics table (index = [strategy, freq]); sorted by ``sort_by`` if
    present. Returns the raw numeric DataFrame (formatting happens in tables.py)."""
    table = suite.metrics_table
    if table is None or table.empty:
        return pd.DataFrame() if table is None else table.copy()
    out = table.copy()
    if sort_by in out.columns:
        out = out.sort_values(by=sort_by, ascending=ascending, kind="mergesort")
    return out


def best_per_strategy(suite: SuiteResult, metric: str = "sharpe") -> pd.DataFrame:
    """For each strategy, the rebalance frequency that maximises ``metric``."""
    table = suite.metrics_table
    if table is None or table.empty:
        return pd.DataFrame() if table is None else table.copy()

    # The index is a MultiIndex (strategy, freq). Group by the strategy level
    # and, within each group, keep the row whose ``metric`` is largest.
    if metric not in table.columns:
        # Nothing to optimise on: keep the first freq row per strategy.
        keep = []
        seen = set()
        for key in table.index:
            strat = key[0] if isinstance(key, tuple) else key
            if strat not in seen:
                seen.add(strat)
                keep.append(key)
        return table.loc[keep]

    strat_level = table.index.get_level_values(0)
    col = table[metric]
    keep_idx = []
    for strat in pd.unique(strat_level):
        group = col[strat_level == strat]
        valid = group.dropna()
        chosen = (valid if len(valid) else group)
        # idxmax returns the full MultiIndex tuple of the winning row.
        keep_idx.append(chosen.idxmax())

    return table.loc[keep_idx]


def export(suite: SuiteResult, output_dir, formats=("csv", "json")) -> list[Path]:
    """Write the metrics table and each strategy's equity curve to ``output_dir``.
    Returns the list of files written. Support 'csv' and 'json'."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    fmts = set(formats)
    table = suite.metrics_table
    if table is None:
        table = pd.DataFrame()
    # JSON serialises inf as null while CSV writes the literal 'inf', so the two
    # exports would disagree (e.g. profit_factor of a no-loss strategy). Map all
    # non-finite values to NaN so CSV (empty), JSON (null) and the table agree.
    table = table.replace([np.inf, -np.inf], np.nan)

    # --- metrics table ------------------------------------------------------
    if "csv" in fmts:
        csv_path = out_dir / "metrics.csv"
        table.to_csv(csv_path)
        written.append(csv_path)
    if "json" in fmts:
        json_path = out_dir / "metrics.json"
        # reset_index so the MultiIndex (strategy, freq) survives the round-trip
        table.reset_index().to_json(json_path, orient="records", indent=2)
        written.append(json_path)

    # --- equity curves ------------------------------------------------------
    for key, res in suite.results.items():
        if isinstance(key, tuple) and len(key) == 2:
            name, freq = key
        else:
            name, freq = getattr(res, "name", key), getattr(res, "freq", "")
        equity = getattr(res, "equity", None)
        if equity is None or len(equity) == 0:
            continue
        safe_name = str(name).replace("/", "_").replace("\\", "_")
        safe_freq = str(freq).replace("/", "_").replace("\\", "_")
        eq_path = out_dir / f"equity_{safe_name}_{safe_freq}.csv"
        curve = equity.copy()
        curve.name = "equity"
        curve.to_frame().to_csv(eq_path, index_label="Date")
        written.append(eq_path)

    return written


# --------------------------------------------------------------------------
# markdown report
# --------------------------------------------------------------------------
def _pct(v, signed=True):
    if v is None or not np.isfinite(v):
        return "-"
    return (f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%")


def _ratio(v):
    if v is None or not np.isfinite(v):
        return "-"
    return f"{v:.2f}"


def _money(v):
    if v is None or not np.isfinite(v):
        return "-"
    return f"${v:,.0f}"


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join("---" for _ in headers) + " |\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


# curated columns for the comparison table: (key, header, formatter)
_REPORT_COLS = [
    ("total_return", "Total Return", lambda v: _pct(v)),
    ("cagr", "CAGR", lambda v: _pct(v)),
    ("sharpe", "Sharpe", _ratio),
    ("sortino", "Sortino", _ratio),
    ("mdd", "MaxDD", lambda v: _pct(v)),
    ("calmar", "Calmar", _ratio),
    ("win_rate", "Win Rate", lambda v: _pct(v, signed=False)),
    ("avg_turnover", "Avg Turnover", lambda v: _pct(v, signed=False)),
    ("final_equity", "Final Equity", _money),
]


def _comparison_rows(table: pd.DataFrame, sort_by: str = "cagr") -> list[list]:
    df = table.copy()
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False, kind="mergesort")
    rows = []
    for idx, row in df.iterrows():
        strat, freq = (idx if isinstance(idx, tuple) else (idx, ""))
        cells = [str(strat), str(freq)]
        cells += [fmt(row.get(key)) for key, _, fmt in _REPORT_COLS]
        rows.append(cells)
    return rows


def markdown_report(suite: SuiteResult, data, config, output_dir,
                    generated: str = "") -> Path:
    """Write a self-contained Markdown report (REPORT.md) that consolidates the
    backtest comparison, key observations, current market snapshot and links to
    the charts in ``output_dir``. ``generated`` is an ISO timestamp string
    (passed in by the caller so this function stays deterministic)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = suite.metrics_table.replace([np.inf, -np.inf], np.nan)

    as_of = data.dates.max()
    start = suite.start_date
    ppy = 252 if data.freq == "daily" else 52
    L: list[str] = []
    A = L.append

    # ---- header ------------------------------------------------------------
    A("# Capital Rotation — 백테스트 결과 보고서\n")
    if generated:
        A(f"_생성: {generated}_\n")
    bt = config.backtest
    costs = config.costs
    A("## 개요\n")
    A(_md_table(
        ["항목", "값"],
        [
            ["분석 기간", f"{(start.date() if start is not None else '-')} → {as_of.date()}"],
            ["거래 자산", f"{len(data.tradable_present())} tradable / {len(data.tickers)} total"],
            ["초기 자본", _money(bt.get('initial_capital', 100000))],
            ["거래 비용", f"{costs.get('commission',0)*100:.2f}% 수수료 + {costs.get('slippage',0)*100:.2f}% 슬리피지 (편도)"],
            ["무위험 수익률", _pct(bt.get('risk_free_rate', 0.0), signed=False)],
            ["벤치마크", config.benchmark_ticker],
            ["리밸런싱 주기", ", ".join(config.rebalance.get('frequencies', []))],
            ["전략 × 주기 조합", str(len(suite.results))],
        ],
    ))

    # ---- comparison --------------------------------------------------------
    A("\n## 1. 전략 성과 비교 (CAGR 내림차순)\n")
    headers = ["Strategy", "Freq"] + [h for _, h, _ in _REPORT_COLS]
    A(_md_table(headers, _comparison_rows(table, "cagr")))

    # multiple-testing correction (overfitting check)
    try:
        from .sharpe_correction import corrections_table
        ct = corrections_table(suite, ppy)
        A("\n### 1b. Sharpe 다중검정 보정 (과적합 점검)\n")
        A("_DSR = 탐색한 모든 설정 수를 감안한 '진짜 Sharpe>0' 확률 "
          "(≥0.9 강건 / ≤0.5 운과 구별 곤란). JS = James-Stein 수축 "
          "(긴 일간 표본에선 추정이 정밀해 거의 no-op)._\n")
        crows = []
        for idx, r in ct.iterrows():
            s, f = idx if isinstance(idx, tuple) else (idx, "")
            crows.append([str(s), str(f), _ratio(r.get("sharpe_raw")),
                          _ratio(r.get("sharpe_js")), _ratio(r.get("dsr"))])
        A(_md_table(["Strategy", "Freq", "Sharpe", "Sharpe(JS)", "DSR"], crows))
    except Exception:
        pass

    # ---- best per strategy -------------------------------------------------
    A("\n## 2. 전략별 최적 리밸런싱 (Sharpe 기준)\n")
    best = best_per_strategy(suite, "sharpe")
    A(_md_table(headers, _comparison_rows(best, "cagr")))

    # ---- key observations --------------------------------------------------
    A("\n## 3. 핵심 관찰\n")
    A(_observations(table, config))

    # ---- market snapshot ---------------------------------------------------
    A("\n## 4. 현재 시장 스냅샷  (as of " + str(as_of.date()) + ")\n")
    A(_market_snapshot(data, config, ppy))

    # ---- charts ------------------------------------------------------------
    A("\n## 5. 차트\n")
    for fname, title in [
        ("equity_curve.png", "누적 수익 곡선 (Equity Curve)"),
        ("regime_equity.png", "레짐 음영 누적수익 (Risk-Off 구간 음영)"),
        ("drawdown.png", "드로다운 (Drawdown)"),
        ("monthly_heatmap.png", "월별 수익률 히트맵 — Composite Score"),
        ("correlation_heatmap.png", "자산 상관관계 히트맵"),
        ("rotation.png", "카테고리 자금 순환"),
    ]:
        if (out_dir / fname).exists():
            A(f"### {title}\n\n![{title}]({fname})\n")

    # ---- methodology -------------------------------------------------------
    A("\n## 6. 방법론 & 주의사항\n")
    A(
        "- **Look-ahead bias 방지**: 모든 시그널은 후행 윈도우만 사용하고, 엔진은 "
        "수익률을 먼저 반영한 뒤 리밸런싱하여 `t`에 결정된 비중은 `t+1`부터 수익을 냅니다.\n"
        "- **거래 비용**은 편도 수수료+슬리피지를 매 리밸런싱 거래 명목금액에 부과합니다.\n"
        "- **현금**은 수익률 0%로 가정합니다.\n"
        "- 모든 전략은 동일 시작일로 정렬되어 곡선이 직접 비교 가능합니다.\n"
        "- ⚠️ **교육·연구용**입니다. 과거 성과는 미래를 보장하지 않으며, yfinance 데이터는 "
        "프로덕션 트레이딩에 부적합합니다. 과적합 위험에 유의하세요.\n"
    )

    path = out_dir / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def _observations(table: pd.DataFrame, config) -> str:
    t = table.dropna(subset=["cagr"])
    if t.empty:
        return "_(결과 없음)_\n"
    lines = []

    def _name(idx):
        return f"**{idx[0]}** ({idx[1]})" if isinstance(idx, tuple) else f"**{idx}**"

    best_cagr = t["cagr"].idxmax()
    best_sharpe = t["sharpe"].idxmax()
    best_calmar = t["calmar"].idxmax()
    low_mdd = t["mdd"].idxmax()  # mdd is negative; max == shallowest
    lines.append(f"- 🥇 **최고 수익(CAGR)**: {_name(best_cagr)} — "
                 f"CAGR {_pct(t.loc[best_cagr,'cagr'])}, Sharpe {_ratio(t.loc[best_cagr,'sharpe'])}")
    lines.append(f"- 📈 **최고 위험조정수익(Sharpe)**: {_name(best_sharpe)} — "
                 f"Sharpe {_ratio(t.loc[best_sharpe,'sharpe'])}, MDD {_pct(t.loc[best_sharpe,'mdd'])}")
    lines.append(f"- 🛡️ **최고 Calmar(수익/낙폭)**: {_name(best_calmar)} — "
                 f"Calmar {_ratio(t.loc[best_calmar,'calmar'])}")
    lines.append(f"- 🌊 **최저 낙폭(MDD)**: {_name(low_mdd)} — MDD {_pct(t.loc[low_mdd,'mdd'])}")

    # vs Buy & Hold benchmark
    bh = t[t.index.get_level_values(0).str.contains("Buy", case=False)]
    if not bh.empty:
        bh_cagr = bh["cagr"].max()
        beat = t[t["cagr"] > bh_cagr]
        lines.append(f"- 📊 Buy & Hold SPY 최고 CAGR({_pct(bh_cagr)})를 능가한 조합: "
                     f"**{len(beat)} / {len(t)}**")
    return "\n".join(lines) + "\n"


def _market_snapshot(data, config, ppy: int) -> str:
    from ..analysis import momentum, regime, rotation, correlation
    import numpy as _np
    out = []

    # regime
    try:
        rdf = regime.classify_regime(data, config)
        label = str(rdf["regime"].iloc[-1]); score = float(rdf["score"].iloc[-1])
        emoji = {"risk_on": "🟢", "risk_off": "🔴"}.get(label, "🟡")
        out.append(f"**시장 레짐**: {emoji} `{label}` (score {score:+.2f})\n")
    except Exception:
        pass

    # momentum top 10
    try:
        tr = data.tradable_present()
        sc = momentum.momentum_score(data.prices[tr], config.momentum_cfg.get("lookback_weights"))
        latest = sc.ffill().iloc[-1].dropna().sort_values(ascending=False).head(10)
        rows = [[i + 1, tk, config.names.get(tk, tk), f"{v:+.3f}"]
                for i, (tk, v) in enumerate(latest.items())]
        out.append("**모멘텀 상위 10**\n")
        out.append(_md_table(["#", "Ticker", "Name", "Score"], rows))
    except Exception:
        pass

    # category rotation
    try:
        rot = rotation.category_rotation(data, config, 63)
        rows = [[c, _pct(r["rel_strength"]), f"{r['flow']:+.3f}", r["status"]]
                for c, r in rot.iterrows()]
        out.append("\n**카테고리 자금 순환**\n")
        out.append(_md_table(["Category", "Rel.Strength", "Flow", "Status"], rows))
    except Exception:
        pass

    # diversification (avg correlation)
    try:
        cm = correlation.correlation_matrix(data.returns, 120)
        off = cm.where(~_np.eye(len(cm), dtype=bool))
        avg = off.mean(axis=1).sort_values()
        best = avg.head(5); worst = avg.tail(5)
        out.append("\n**분산투자 관점 (평균 상관, 120일)**\n")
        out.append("- 최고 분산처(낮은 상관): " +
                   ", ".join(f"{config.names.get(t,t)} ({v:+.2f})" for t, v in best.items()))
        out.append("- 최고 동조화(높은 상관): " +
                   ", ".join(f"{config.names.get(t,t)} ({v:+.2f})" for t, v in worst.items()))
        pca = correlation.pca_explained(data.returns, 252, 3)
        evr = pca.get("explained_variance_ratio", [])
        if len(evr):
            out.append("- PCA(252일): " +
                       ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(evr)))
    except Exception:
        pass

    return "\n".join(out) + "\n"
