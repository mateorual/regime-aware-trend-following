import numpy as np
import pandas as pd
import pytest

from regime_trend import signals as S
from regime_trend.kalman import LocalLinearTrend, slope_noise_ratio


# --------------------------------------------------------------------------- #
# Kalman filter
# --------------------------------------------------------------------------- #
def test_slope_ratio_decreases_with_horizon():
    qs = [slope_noise_ratio(p) for p in (21, 63, 126, 252)]
    assert all(a > b for a, b in zip(qs, qs[1:], strict=False))


def test_kalman_recovers_a_linear_trend():
    rng = np.random.default_rng(0)
    t = np.arange(1500)
    y = 4.0 + 0.0008 * t + rng.normal(0, 0.002, len(t))
    out = LocalLinearTrend.from_period(126).filter(y)
    assert out.slope[-1] == pytest.approx(0.0008, rel=0.05)
    assert out.level[-1] == pytest.approx(y[-1], abs=0.01)


def test_kalman_is_causal():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(0, 0.01, 800))
    kf = LocalLinearTrend.from_period(63)
    full = kf.filter(y)
    part = kf.filter(y[:500])
    np.testing.assert_allclose(full.slope[:500], part.slope)
    np.testing.assert_allclose(full.level[:500], part.level)


def test_kalman_bridges_missing_observations():
    y = np.linspace(0, 1, 300)
    y[100:110] = np.nan
    out = LocalLinearTrend.from_period(63).filter(y)
    assert np.isfinite(out.slope).all()
    assert np.isnan(out.innovations[100:110]).all()


def test_kalman_steady_state_variance_shrinks():
    out = LocalLinearTrend.from_period(126).filter(np.zeros(2000))
    assert out.slope_var[-1] < out.slope_var[10] and out.slope_var[-1] > 0


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def _frame(values, name="X"):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame({name: values}, index=idx)


def test_tsmom_sign():
    close = _frame(np.r_[np.linspace(100, 120, 300), np.linspace(120, 90, 300)])
    sig = S.tsmom(close, 252)["X"]
    assert sig.iloc[:252].isna().all()
    assert sig.iloc[260] == 1 and sig.iloc[-1] == -1


def test_donchian_enters_holds_and_exits():
    flat = np.full(80, 100.0)
    up = np.linspace(100, 130, 40)  # breakout -> long
    drift = np.full(30, 130.0)  # inside the channel -> keep holding
    down = np.linspace(129, 110, 15)  # breaks the 20-day low -> exit
    close = _frame(np.r_[flat, up, drift, down])
    pos = S.donchian(close, close, close, entry=55, exit_=20)["X"]
    assert pos.iloc[79] == 0
    assert pos.iloc[85] == 1  # entered on the breakout
    assert pos.iloc[80 + 40 + 25] == 1  # still held while drifting sideways
    assert pos.iloc[-1] in (0, -1)  # exited on the breakdown


def test_donchian_channel_excludes_the_current_bar():
    close = _frame(np.full(100, 100.0))
    high = close.copy()
    high.iloc[-1] = 500.0  # today's intraday spike must not raise today's channel
    close.iloc[-1] = 101.0  # close above the previous 55-day high -> breakout
    pos = S.donchian(close, high, close, entry=55, exit_=20)["X"]
    assert pos.iloc[-1] == 1


def test_signals_do_not_look_ahead():
    rng = np.random.default_rng(3)
    close = _frame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 700))))
    vol = S.ewma_volatility(close.pct_change(), 30)
    k = 500
    full = S.donchian(close, close, close)["X"].iloc[:k]
    part = S.donchian(close.iloc[:k], close.iloc[:k], close.iloc[:k])["X"]
    pd.testing.assert_series_equal(full, part)
    kf_full = S.kalman_trend(close, vol)[0]["X"].iloc[:k]
    kf_part = S.kalman_trend(close.iloc[:k], vol.iloc[:k])[0]["X"]
    pd.testing.assert_series_equal(kf_full, kf_part)


def test_kalman_signal_direction_and_bounds():
    close = _frame(np.r_[np.linspace(100, 200, 400), np.linspace(200, 100, 400)])
    noise = np.random.default_rng(4).normal(0, 0.005, len(close))
    close = close * np.exp(noise)[:, None]
    vol = S.ewma_volatility(close.pct_change(), 30)
    sig = S.kalman_trend(close, vol, 126, 0.5)[0]["X"]
    assert sig.iloc[350] > 0.5 and sig.iloc[-1] < -0.5
    assert sig.dropna().between(-1, 1).all()
