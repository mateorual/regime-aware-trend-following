"""Research pipeline: one call reproduces every table and figure for a config.

Strategy grid (all share data, sizing and costs; only the gate differs)::

    benchmark                long-only, equal risk across live assets
    {family}_base            trend signal                      family in {tsmom, donchian, kalman}
    {family}_hmm             + walk-forward HMM gate (filtered, tradeable)
    {family}_vol             + naive volatility-regime gate (control)
    {family}_lookahead       + full-sample smoothed HMM gate (NOT tradeable; bias measurement)
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import plotting as P
from . import regimes as R
from . import signals as S
from .config import Config
from .data import load_market_data
from .portfolio import ewma_covariance, regime_gate, risk_scale, simulate, unit_risk_weights

log = logging.getLogger(__name__)

FAMILIES = ["tsmom", "donchian", "kalman"]
VARIANTS = ["base", "hmm", "vol", "lookahead"]
PCT_COLS = {"CAGR", "Volatility", "Max drawdown", "Time invested", "Turnover (1-way, p.a.)",
            "Share turbulent", "Agreement with hindsight", "Calm vol", "Turbulent vol"}


def strategy_label(name: str) -> str:
    if name == "benchmark":
        return P.VARIANT_LABELS["benchmark"]
    fam, var = name.split("_", 1)
    suffix = {"base": "", "hmm": " + HMM gate", "vol": " + vol gate",
              "lookahead": " + HMM gate (look-ahead)"}[var]
    return P.FAMILY_LABELS[fam] + suffix


def markdown_table(df: pd.DataFrame, pct_cols: set[str] | None = None, decimals: int = 2) -> str:
    pct_cols = pct_cols or set()

    def fmt(col, v):
        if isinstance(v, str):
            return v
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return f"{v:.1%}" if col in pct_cols else f"{v:.{decimals}f}"

    head = "| | " + " | ".join(map(str, df.columns)) + " |"
    sep = "|:---|" + "|".join(["---:"] * len(df.columns)) + "|"
    rows = ["| " + str(i) + " | " + " | ".join(fmt(c, df.loc[i, c]) for c in df.columns) + " |"
            for i in df.index]
    return "\n".join([head, sep, *rows])


def _cached_regimes(cfg: Config, data, data_root: Path, workers: int | None
                    ) -> dict[str, R.RegimeResult]:
    """HMM fits are the expensive step and depend only on returns + HMM settings: cache them."""
    lr = data.log_returns
    key = hashlib.sha1(repr((cfg.hmm, cfg.tickers, str(lr.index[-1]),
                             float(np.nansum(lr.values)))).encode()).hexdigest()[:12]
    path = data_root / cfg.name / f"hmm_regimes_{key}.pkl"
    if path.exists():
        log.info("[%s] loading cached HMM regimes (%s)", cfg.name, path.name)
        with path.open("rb") as fh:
            return pickle.load(fh)
    log.info("[%s] fitting walk-forward HMMs for %d assets", cfg.name, len(cfg.tickers))
    hmm = R.run_hmm_regimes(lr, cfg.hmm, workers)
    with path.open("wb") as fh:
        pickle.dump(hmm, fh)
    return hmm


# --------------------------------------------------------------------------- #
def run(cfg: Config, data_root: str | Path = "data", results_root: str | Path = "results",
        refresh: bool = False, workers: int | None = None) -> dict:
    t0 = time.time()
    P.apply_style()
    out = Path(results_root) / cfg.name
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    pc, ev = cfg.portfolio, cfg.evaluation
    bps = pc.transaction_cost_bps

    # ---- data, signals ----------------------------------------------------- #
    data = load_market_data(cfg, data_root, refresh)
    vol = S.ewma_volatility(data.returns, pc.asset_vol_halflife)
    kal_sig, kal_level = S.kalman_trend(data.close, vol, cfg.signals.kalman_trend_period,
                                        cfg.signals.kalman_full_position_sharpe)
    signals = {
        "tsmom": S.tsmom(data.close, cfg.signals.tsmom_lookback),
        "donchian": S.donchian(data.close, data.high, data.low, cfg.signals.donchian_entry,
                               cfg.signals.donchian_exit),
        "kalman": kal_sig,
    }

    # ---- regimes ----------------------------------------------------------- #
    hmm = _cached_regimes(cfg, data, Path(data_root), workers)
    idx = data.close.index
    p_hmm = pd.DataFrame({t: hmm[t].p_turbulent for t in cfg.tickers}).reindex(idx)
    p_smooth = pd.DataFrame({t: hmm[t].p_turbulent_smoothed for t in cfg.tickers}).reindex(idx)
    p_vol = pd.DataFrame({t: R.volatility_regime(data.log_returns[t].dropna(), cfg.vol_control)
                          for t in cfg.tickers}).reindex(idx)
    log.info("[%s] regimes done (%.0fs)", cfg.name, time.time() - t0)

    # ---- one common live-asset mask for every strategy --------------------- #
    history = data.returns.notna().cumsum() >= pc.min_history_days
    active = history & vol.notna() & p_hmm.notna() & p_vol.notna() & p_smooth.notna()
    for s in signals.values():
        active &= s.notna()

    # ---- strategies -------------------------------------------------------- #
    cov = ewma_covariance(data.excess_returns, pc.portfolio_vol_halflife)
    excess, rf = data.excess_returns, data.risk_free
    results = {}
    ones = pd.DataFrame(1.0, index=idx, columns=cfg.tickers)
    u = unit_risk_weights(ones, vol, active)
    buf = pc.rebalance_buffer
    results["benchmark"] = simulate("benchmark", u, risk_scale(u, cov, pc.portfolio_vol_target,
                                                               pc.max_gross_leverage),
                                    excess, rf, buf)
    for fam, s in signals.items():
        u_base = unit_risk_weights(s, vol, active)
        k = risk_scale(u_base, cov, pc.portfolio_vol_target, pc.max_gross_leverage)
        gated = {"base": s, "hmm": regime_gate(s, p_hmm), "vol": regime_gate(s, p_vol),
                 "lookahead": regime_gate(s, p_smooth)}
        for v, g in gated.items():
            name = f"{fam}_{v}"
            results[name] = simulate(name, unit_risk_weights(g, vol, active), k, excess, rf,
                                     buf)
    evald = {n: r.slice(ev.start) for n, r in results.items()}
    excess_df = pd.DataFrame({n: r.excess(bps) for n, r in evald.items()})
    total_df = pd.DataFrame({n: r.total(bps) for n, r in evald.items()})
    n_live = active.loc[ev.start:].sum(axis=1)

    # ---- performance ------------------------------------------------------- #
    perf = pd.DataFrame({n: M.performance_summary(total_df[n], excess_df[n], r.weights, r.traded)
                         for n, r in evald.items()}).T

    pairs = []
    for fam in FAMILIES:
        pairs += [(f"{fam}_{v}", f"{fam}_base") for v in ("hmm", "vol", "lookahead")]
        pairs += [(f"{fam}_hmm", f"{fam}_vol"), (f"{fam}_base", "benchmark")]
    single, paired = M.bootstrap_sharpe(excess_df, pairs, ev.bootstrap_block,
                                        ev.bootstrap_samples, ev.seed)
    crisis = M.crisis_returns(total_df, ev.crises)
    cost = pd.DataFrame({b: {n: M.sharpe_ratio(r.excess(b)) for n, r in evald.items()}
                         for b in ev.cost_sensitivity_bps}).T.rename_axis("cost_bps")

    # ---- regime diagnostics ------------------------------------------------ #
    reg_rows = {}
    for t in cfg.tickers:
        pf = p_hmm[t].loc[ev.start:].dropna()
        ps = p_smooth[t].reindex(pf.index)
        pv = p_vol[t].reindex(pf.index)
        fits = hmm[t].fits.loc[ev.start:] if len(hmm[t].fits) else pd.DataFrame()
        if pf.empty or fits.empty:
            continue
        stay_c, stay_t = fits["stay_0"].median(), fits[f"stay_{cfg.hmm.n_states - 1}"].median()
        reg_rows[t] = {
            "Share turbulent": float((pf > 0.5).mean()),
            "Switches / yr (HMM real time)": R.switches_per_year(pf),
            "Switches / yr (HMM hindsight)": R.switches_per_year(ps),
            "Switches / yr (vol control)": R.switches_per_year(pv),
            "Agreement with hindsight": float(((pf > 0.5) == (ps > 0.5)).mean()),
            "Calm vol": float(fits["vol_0"].median()),
            "Turbulent vol": float(fits[f"vol_{cfg.hmm.n_states - 1}"].median()),
            "Calm duration (days)": float(1 / (1 - stay_c)),
            "Turbulent duration (days)": float(1 / (1 - stay_t)),
        }
    regime_table = pd.DataFrame(reg_rows).T

    asset = ev.anatomy_asset
    feats = R.build_features(data.log_returns[asset].dropna(), cfg.hmm.features)
    X = ((feats - feats.mean()) / feats.std(ddof=0)).values
    bic_rows = {}
    for k_states in range(1, 5):
        sub = type(cfg.hmm)(**{**cfg.hmm.__dict__, "n_states": k_states})
        m = R.fit_hmm(X, sub, n_init=cfg.hmm.n_init, seed=cfg.hmm.seed)
        bic_rows[k_states] = {"log-likelihood": float(m.score(X)), "BIC": R.bic(m, X)}
    bic_table = pd.DataFrame(bic_rows).T.rename_axis("states")
    bic_table["dBIC vs best"] = bic_table["BIC"] - bic_table["BIC"].min()
    log.info("[%s] analysis done (%.0fs)", cfg.name, time.time() - t0)

    # ---- figures ----------------------------------------------------------- #
    title = cfg.title
    P.wealth_by_family(total_df, FAMILIES, figs / "wealth.png", f"{title}: cumulative wealth")
    P.delta_sharpe(paired, FAMILIES, figs / "delta_sharpe.png",
                   f"{title}: does a regime gate improve the Sharpe ratio?")
    P.lookahead_bars(single["Sharpe"], FAMILIES, figs / "lookahead.png",
                     f"{title}: the cost of hindsight")
    pf = p_hmm[asset].dropna()
    sw = {"hmm": R.switches_per_year(pf),
          "vol": R.switches_per_year(p_vol[asset].reindex(pf.index)),
          "lookahead": R.switches_per_year(p_smooth[asset].reindex(pf.index))}
    P.regime_map(data.close[asset], p_hmm[asset], p_smooth[asset], p_vol[asset],
                 figs / "regimes.png", f"{asset}: HMM regimes in real time vs. in hindsight",
                 asset, sw)
    P.signal_anatomy(data.close[asset], data.high[asset], data.low[asset], kal_level[asset],
                     {f: signals[f][asset] for f in FAMILIES}, p_hmm[asset],
                     cfg.signals.donchian_entry, ev.anatomy_window, figs / "signal_anatomy.png",
                     f"{asset}: anatomy of the three trend signals", asset)
    if not crisis.empty:
        P.crisis_bars(crisis, FAMILIES, figs / "crises.png", f"{title}: behaviour in crises")
    P.cost_sensitivity(cost, FAMILIES, figs / "cost_sensitivity.png",
                       f"{title}: sensitivity to trading costs", bps)

    # ---- tables ------------------------------------------------------------ #
    perf.to_csv(out / "performance.csv")
    single.to_csv(out / "sharpe_ci.csv")
    paired.to_csv(out / "paired_tests.csv")
    crisis.to_csv(out / "crises.csv")
    cost.to_csv(out / "cost_sensitivity.csv")
    regime_table.to_csv(out / "regime_diagnostics.csv")
    bic_table.to_csv(out / "bic.csv")
    excess_df.to_csv(out / "excess_returns.csv", float_format="%.8f")
    pd.DataFrame({t: hmm[t].fits["loglik_per_obs"] for t in cfg.tickers if len(hmm[t].fits)}
                 ).to_csv(out / "hmm_fit_loglik.csv", float_format="%.6f")

    summary = {
        "config": cfg.name,
        "evaluation_period": [str(excess_df.index[0].date()), str(excess_df.index[-1].date())],
        "live_assets": [int(n_live.min()), int(n_live.max())],
        "transaction_cost_bps": bps,
        "performance": perf.round(6).to_dict(orient="index"),
        "paired_tests": {f"{a} vs {b}": v for (a, b), v in
                         paired.round(6).to_dict(orient="index").items()},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _report(cfg, out, perf, single, paired, crisis, cost, regime_table, bic_table, excess_df,
            n_live)
    log.info("[%s] finished in %.0fs -> %s", cfg.name, time.time() - t0, out)
    return summary


def _report(cfg, out, perf, single, paired, crisis, cost, regime_table, bic_table, excess_df,
            n_live) -> None:
    order = ["benchmark"] + [f"{f}_{v}" for f in FAMILIES for v in VARIANTS]
    perf_cols = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar",
                 "Skew (monthly)"]
    trade_cols = ["Avg gross leverage", "Time invested", "Turnover (1-way, p.a.)"]
    pt = paired.copy()
    pt.index = [f"{strategy_label(a)} vs {strategy_label(b)}" for a, b in pt.index]
    sc = single.reindex(order).rename(index=strategy_label)
    cr = crisis[[c for c in order if c in crisis.columns]].T.rename(index=strategy_label)
    cost_t = cost.T.reindex(order).rename(index=strategy_label)
    cost_t.columns = [f"{c:g} bps" for c in cost_t.columns]
    lines = [
        f"# {cfg.title} - results",
        "",
        f"_Auto-generated by `regime-trend run {cfg.source_path.as_posix() if cfg.source_path else ''}`._",
        "",
        f"* Evaluation period: **{excess_df.index[0].date()} -> {excess_df.index[-1].date()}**; "
        f"live assets {int(n_live.min())}-{int(n_live.max())} (unbalanced panel)",
        f"* HMM: {cfg.hmm.n_states} states on {', '.join(cfg.hmm.features)}; "
        f"{cfg.hmm.train_days}-day rolling window, refit every {cfg.hmm.refit_every} days, "
        "filtered probabilities",
        f"* Sizing: {cfg.portfolio.portfolio_vol_target:.0%} ex-ante vol target set by each base "
        f"signal; costs {cfg.portfolio.transaction_cost_bps:g} bps per unit traded",
        "",
        "![Wealth](figures/wealth.png)",
        "",
        "## Performance (net of costs)",
        "",
        markdown_table(perf.reindex(order)[perf_cols].rename(index=strategy_label), PCT_COLS),
        "",
        "## Trading profile",
        "",
        markdown_table(perf.reindex(order)[trade_cols].rename(index=strategy_label), PCT_COLS),
        "",
        "## Do the gates help? Paired block-bootstrap Sharpe differences",
        "",
        markdown_table(pt),
        "",
        "![Delta Sharpe](figures/delta_sharpe.png)",
        "",
        "Sharpe ratios with 95% intervals:",
        "",
        markdown_table(sc),
        "",
        "## Look-ahead bias",
        "",
        "![Look-ahead](figures/lookahead.png)",
        "",
        "## Crisis behaviour (compounded total return)",
        "",
        markdown_table(cr, set(cr.columns)) if not cr.empty else "_no crisis in sample_",
        "",
        "![Crises](figures/crises.png)",
        "",
        "## Regime diagnostics (evaluation period)",
        "",
        markdown_table(regime_table, PCT_COLS, decimals=1),
        "",
        "![Regimes](figures/regimes.png)",
        "",
        f"Model selection on {cfg.evaluation.anatomy_asset} (full sample, in-sample):",
        "",
        markdown_table(bic_table, decimals=1),
        "",
        "## Cost sensitivity (Sharpe)",
        "",
        markdown_table(cost_t),
        "",
        "![Costs](figures/cost_sensitivity.png)",
        "",
        "![Anatomy](figures/signal_anatomy.png)",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
