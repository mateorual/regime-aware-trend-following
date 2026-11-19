"""Trend signals. Every signal at date t uses information up to the close of t only.

All signals are expressed as a desired *direction and conviction* in [-1, 1];
position sizing (volatility targeting) is applied later and identically to all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .kalman import LocalLinearTrend


def ewma_volatility(returns: pd.DataFrame, halflife: int, periods: int = 252) -> pd.DataFrame:
    """Annualised exponentially-weighted volatility, using returns up to and including t."""
    return returns.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(periods)


def tsmom(close: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Time-series momentum (Moskowitz, Ooi & Pedersen, 2012): sign of the trailing return."""
    past = close / close.shift(lookback) - 1.0
    return np.sign(past).where(past.notna())


def donchian(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, entry: int = 55,
             exit_: int = 20) -> pd.DataFrame:
    """Donchian channel breakout as a *stateful* position (the 'turtle' rule).

    * enter long when the close breaks above the highest high of the previous
      ``entry`` days; exit when it breaks below the lowest low of the previous
      ``exit_`` days (mirror image for shorts);
    * the position is **held** between entry and exit.

    The channels are shifted by one day, so today's bar is compared with the
    *previous* N days and never with itself.
    """
    if exit_ >= entry:
        raise ValueError("exit channel should be shorter than the entry channel")
    hi_entry = high.rolling(entry).max().shift(1)
    lo_entry = low.rolling(entry).min().shift(1)
    hi_exit = high.rolling(exit_).max().shift(1)
    lo_exit = low.rolling(exit_).min().shift(1)

    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for col in close:
        c = close[col].values
        he, le = hi_entry[col].values, lo_entry[col].values
        hx, lx = hi_exit[col].values, lo_exit[col].values
        pos, state = np.full(len(c), np.nan), 0.0
        for t in range(len(c)):
            if np.isnan(he[t]) or np.isnan(c[t]):
                continue
            if state > 0 and c[t] < lx[t]:
                state = 0.0
            elif state < 0 and c[t] > hx[t]:
                state = 0.0
            if c[t] > he[t]:
                state = 1.0
            elif c[t] < le[t]:
                state = -1.0
            pos[t] = state
        out[col] = pos
    return out


def kalman_trend(close: pd.DataFrame, vol: pd.DataFrame, period_days: int = 126,
                 full_position_sharpe: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""Kalman local-linear-trend signal.

    The filtered daily slope of the log price, annualised and divided by the
    asset's volatility, is a *trend Sharpe ratio*
    :math:`\widehat{SR}_t = 252\,\hat\beta_{t|t} / \hat\sigma_t`. The position is
    :math:`\mathrm{clip}(\widehat{SR}_t / SR^*, -1, 1)`, so conviction scales with
    trend strength and saturates at :math:`SR^*`.

    Returns (signal, filtered level in price units).
    """
    kf = LocalLinearTrend.from_period(period_days)
    sig = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    level = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for col in close:
        out = kf.filter(np.log(close[col].values))
        trend_sharpe = 252.0 * out.slope / vol[col].values
        sig[col] = np.clip(trend_sharpe / full_position_sharpe, -1.0, 1.0)
        level[col] = np.exp(out.level)
    return sig.where(vol.notna()), level
