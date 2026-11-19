import numpy as np
import pandas as pd
import pytest

from regime_trend import metrics as M
from regime_trend.portfolio import (
    ewma_covariance,
    regime_gate,
    risk_scale,
    simulate,
    unit_risk_weights,
)

IDX = pd.bdate_range("2021-01-01", periods=5)


def test_regime_gate_keeps_longs_in_calm_and_shorts_in_turbulence():
    sig = pd.DataFrame({"a": [1.0, 1.0, -1.0, -1.0, 0.5]}, index=IDX)
    p = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0, 0.4]}, index=IDX)
    g = regime_gate(sig, p)["a"].values
    np.testing.assert_allclose(g, [1.0, 0.0, 0.0, -1.0, 0.3])


def test_unit_risk_weights_split_risk_across_live_assets():
    sig = pd.DataFrame({"a": 1.0, "b": -1.0, "c": 1.0}, index=IDX)
    vol = pd.DataFrame({"a": 0.10, "b": 0.20, "c": 0.40}, index=IDX)
    active = pd.DataFrame({"a": True, "b": True, "c": False}, index=IDX)
    w = unit_risk_weights(sig, vol, active).iloc[0]
    np.testing.assert_allclose(w.values, [1 / 0.10 / 2, -1 / 0.20 / 2, 0.0])


def test_risk_scale_hits_the_volatility_target():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.01, (600, 3)), columns=list("abc"),
                        index=pd.bdate_range("2020-01-01", periods=600))
    cov = ewma_covariance(rets, halflife=60)
    u = pd.DataFrame(rng.normal(0, 1, (600, 3)), index=rets.index, columns=rets.columns)
    k = risk_scale(u, cov, target_vol=0.10, max_gross=100.0)
    t = 400
    w = (u.iloc[t] * k.iloc[t]).values
    assert np.sqrt(w @ cov[t] @ w) == pytest.approx(0.10)
    capped = risk_scale(u, cov, target_vol=0.10, max_gross=0.5)
    assert (np.abs(u.mul(capped, axis=0)).sum(axis=1) <= 0.5 + 1e-12).all()


def test_ewma_covariance_is_psd_and_symmetric():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.01, (300, 4)))
    cov = ewma_covariance(rets, 30)
    assert np.allclose(cov[-1], cov[-1].T)
    assert np.linalg.eigvalsh(cov[-1]).min() >= -1e-15


def test_simulate_applies_one_day_lag_costs_and_cash():
    w = pd.DataFrame({"a": [0.0, 1.0, 1.0, 0.0, 0.0]}, index=IDX)
    ex = pd.DataFrame({"a": [0.05, 0.02, -0.01, 0.03, 0.04]}, index=IDX)
    rf = pd.Series(0.0001, index=IDX)
    res = simulate("x", w, pd.Series(1.0, index=IDX), ex, rf)
    # the position set at the close of day 1 earns the returns of days 2 and 3 only
    np.testing.assert_allclose(res.excess_gross.values, [0.0, 0.0, -0.01, 0.03, 0.0])
    np.testing.assert_allclose(res.traded.values, [0, 1, 0, 1, 0])
    net = res.excess(10)
    assert net.iloc[1] == pytest.approx(-0.001)
    assert res.total(0).iloc[2] == pytest.approx(-0.01 + 0.0001)


def test_buffer_skips_small_resizes_but_executes_entries_exits_and_flips():
    from regime_trend.portfolio import apply_buffer

    idx = pd.bdate_range("2021-01-01", periods=7)
    target = pd.DataFrame({"a": [1.0, 1.1, 0.95, 1.4, -1.0, 0.0, 0.0]}, index=idx)
    held = apply_buffer(target, 0.25)["a"].values
    # +10% and -5% resizes are ignored; +40% resize, flip and exit all trade
    np.testing.assert_allclose(held, [1.0, 1.0, 1.0, 1.4, -1.0, 0.0, 0.0])
    pd.testing.assert_frame_equal(apply_buffer(target, 0.0), target)


def test_crisis_returns_and_drawdown():
    idx = pd.bdate_range("2020-01-01", periods=10)
    total = pd.DataFrame({"s": [0.01] * 5 + [-0.02] * 5}, index=idx)
    table = M.crisis_returns(total, {"down": (str(idx[5].date()), str(idx[9].date()))})
    assert table.loc["down", "s"] == pytest.approx(0.98**5 - 1)
    assert M.drawdown(total["s"]).min() == pytest.approx(0.98**5 - 1)


def test_paired_bootstrap_is_reproducible_and_detects_a_clear_edge():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2010-01-01", periods=2500)
    base = rng.normal(0.0002, 0.01, len(idx))
    better = base + 0.0004  # same noise, higher mean -> a large paired edge
    ex = pd.DataFrame({"base": base, "better": better}, index=idx)
    s1, p1 = M.bootstrap_sharpe(ex, [("better", "base")], n_samples=300, seed=5)
    s2, p2 = M.bootstrap_sharpe(ex, [("better", "base")], n_samples=300, seed=5)
    pd.testing.assert_frame_equal(p1, p2)
    row = p1.loc[("better", "base")]
    assert row["CI low"] > 0 and row["p (dSharpe <= 0)"] == 0.0
    assert (s1["CI low"] < s1["Sharpe"]).all()


def test_performance_summary_fields():
    idx = pd.bdate_range("2015-01-01", periods=756)
    rng = np.random.default_rng(3)
    ex = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    w = pd.DataFrame({"a": 0.5}, index=idx)
    out = M.performance_summary(ex, ex, w, pd.Series(0.01, index=idx))
    assert out["Avg gross leverage"] == pytest.approx(0.5)
    assert out["Time invested"] == pytest.approx(1.0)
    assert out["Turnover (1-way, p.a.)"] == pytest.approx(0.5 * 0.01 * 252)
