r"""Local-linear-trend Kalman filter for log prices, written from scratch.

State-space model (Harvey, 1989) for the log price :math:`y_t`:

.. math::
    y_t &= \mu_t + \varepsilon_t,            & \varepsilon_t &\sim N(0, \sigma^2_\varepsilon) \\
    \mu_{t+1} &= \mu_t + \beta_t + \eta_t,   & \eta_t &\sim N(0, q_\eta\,\sigma^2_\varepsilon) \\
    \beta_{t+1} &= \beta_t + \zeta_t,        & \zeta_t &\sim N(0, q_\zeta\,\sigma^2_\varepsilon)

:math:`\mu_t` is the trend level and :math:`\beta_t` the trend *slope*, the drift per day.
The filtered slope :math:`\hat\beta_{t|t}` is the trading signal.

**Calibration.** Only the signal-to-noise ratios :math:`q` affect the filtered
state, so :math:`\sigma^2_\varepsilon` is normalised to 1. Estimating :math:`q` by
maximum likelihood on daily prices is uninformative: daily returns are close to
white noise, so the likelihood pushes :math:`q_\zeta \to 0` and the slope
collapses to the full-sample drift. We therefore choose the trend *horizon*
explicitly. With :math:`q_\eta = 0` (an integrated random walk) the smoother is the
Hodrick-Prescott filter, whose gain halves at period :math:`P` when

.. math:: q_\zeta = \left(2 - 2\cos\tfrac{2\pi}{P}\right)^2 .

So the single hyper-parameter is an interpretable period in days, not an opaque
variance ratio. The filter is causal: :math:`\hat\beta_{t|t}` uses
:math:`y_1,\dots,y_t` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def slope_noise_ratio(period_days: float) -> float:
    """HP-filter correspondence: q_zeta giving a half-gain cut-off at ``period_days``."""
    if period_days <= 2:
        raise ValueError("period_days must exceed 2")
    return float((2.0 - 2.0 * np.cos(2.0 * np.pi / period_days)) ** 2)


@dataclass
class KalmanOutput:
    level: np.ndarray  # filtered level  mu_{t|t}
    slope: np.ndarray  # filtered slope  beta_{t|t}   (per period)
    slope_var: np.ndarray  # Var(beta_{t|t}) in units of sigma^2_eps
    innovations: np.ndarray  # one-step prediction errors y_t - mu_{t|t-1}


class LocalLinearTrend:
    """Two-state Kalman filter. Missing observations (NaN) are handled by prediction only."""

    T = np.array([[1.0, 1.0], [0.0, 1.0]])

    def __init__(self, q_level: float = 0.0, q_slope: float = 1e-6, diffuse: float = 1e6):
        if q_level < 0 or q_slope <= 0:
            raise ValueError("noise ratios must be q_level >= 0, q_slope > 0")
        self.Q = np.diag([q_level, q_slope])
        self.diffuse = diffuse

    @classmethod
    def from_period(cls, period_days: float) -> LocalLinearTrend:
        return cls(q_level=0.0, q_slope=slope_noise_ratio(period_days))

    def filter(self, y: np.ndarray) -> KalmanOutput:
        y = np.asarray(y, dtype=float)
        n = len(y)
        level = np.full(n, np.nan)
        slope = np.full(n, np.nan)
        slope_var = np.full(n, np.nan)
        innov = np.full(n, np.nan)

        first = np.flatnonzero(~np.isnan(y))
        if first.size == 0:
            return KalmanOutput(level, slope, slope_var, innov)
        t0 = first[0]
        a = np.array([y[t0], 0.0])  # diffuse prior around the first price, zero drift
        P = np.diag([self.diffuse, self.diffuse])
        T, Q = self.T, self.Q
        started = False
        for t in range(t0, n):
            if started:  # predict
                a = T @ a
                P = T @ P @ T.T + Q
            started = True
            if not np.isnan(y[t]):  # update with H = 1, Z = [1, 0]
                v = y[t] - a[0]
                F = P[0, 0] + 1.0
                K = P[:, 0] / F
                a = a + K * v
                P = P - np.outer(K, P[0, :])
                innov[t] = v
            level[t], slope[t], slope_var[t] = a[0], a[1], P[1, 1]
        return KalmanOutput(level, slope, slope_var, innov)
