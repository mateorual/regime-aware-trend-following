r"""From signals to P&L: regime gating, risk scaling, execution lag and costs.

Pipeline for one signal family (e.g. Donchian), all decisions at the close of t:

1. **Gate** (optional): keep long conviction in proportion to P(calm) and short
   conviction in proportion to P(turbulent):
   :math:`g_{i,t} = s^+_{i,t}\,(1-p_{i,t}) + s^-_{i,t}\,p_{i,t}`.
   A hard 0/1 regime gives the classic "longs only in calm, shorts only in
   turbulent" filter. The probabilistic version uses the HMM's uncertainty.
2. **Unit-risk weights**: :math:`u_{i,t} = g_{i,t} / (\hat\sigma_{i,t} N_t)`.
   Each active asset gets the same risk budget, and :math:`N_t` assets are active.
3. **Portfolio risk scaling**: :math:`k_t = \sigma^* / \sqrt{u_t^\top\hat\Sigma_t u_t}`
   using an ex-ante EWMA covariance, computed from the **ungated base** book and
   applied unchanged to its gated variants. A gate can therefore only *remove* risk;
   it never re-levers the surviving positions, so the comparison isolates the
   filter's value.
4. **Execution**: a no-trade buffer suppresses small resizing trades; weights
   decided at t earn the excess return of t+1; cost is ``bps x sum|W_t - W_{t-1}|``;
   collateral earns the risk-free rate (futures-style accounting).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def regime_gate(signal: pd.DataFrame, p_turbulent: pd.DataFrame) -> pd.DataFrame:
    p = p_turbulent.reindex_like(signal)
    return signal.clip(lower=0) * (1.0 - p) + signal.clip(upper=0) * p


def unit_risk_weights(signal: pd.DataFrame, asset_vol: pd.DataFrame, active: pd.DataFrame
                      ) -> pd.DataFrame:
    n_active = active.sum(axis=1).replace(0, np.nan)
    w = (signal / asset_vol).where(active, 0.0).fillna(0.0)
    return w.div(n_active, axis=0).fillna(0.0)


def ewma_covariance(returns: pd.DataFrame, halflife: int, periods: int = 252) -> np.ndarray:
    """Zero-mean (RiskMetrics) EWMA covariance path, shape (T, N, N); missing returns -> 0."""
    x = returns.fillna(0.0).values
    lam = 0.5 ** (1.0 / halflife)
    T, N = x.shape
    out = np.empty((T, N, N))
    S = np.zeros((N, N))
    for t in range(T):
        S = lam * S + (1.0 - lam) * np.outer(x[t], x[t])
        out[t] = S
    return out * periods


def risk_scale(unit_weights: pd.DataFrame, cov_path: np.ndarray, target_vol: float,
               max_gross: float) -> pd.Series:
    """k_t such that the ex-ante volatility of k_t * u_t equals the target."""
    u = unit_weights.values
    var = np.einsum("ti,tij,tj->t", u, cov_path, u)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(var > 0, target_vol / np.sqrt(var), 0.0)
    gross = np.abs(u).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.minimum(k, np.where(gross > 0, max_gross / gross, 0.0))
    return pd.Series(k, index=unit_weights.index, name="risk_scale")


@dataclass
class StrategyResult:
    name: str
    weights: pd.DataFrame  # final positions W_t (decided at close t)
    excess_gross: pd.Series  # daily excess return before costs
    traded: pd.Series  # sum |W_t - W_{t-1}|
    risk_free: pd.Series

    def excess(self, cost_bps: float) -> pd.Series:
        return (self.excess_gross - self.traded * cost_bps / 1e4).rename(self.name)

    def total(self, cost_bps: float) -> pd.Series:
        return (self.excess(cost_bps) + self.risk_free).rename(self.name)

    def slice(self, start) -> StrategyResult:
        return StrategyResult(self.name, self.weights.loc[start:], self.excess_gross.loc[start:],
                              self.traded.loc[start:], self.risk_free.loc[start:])


def apply_buffer(target: pd.DataFrame, buffer: float) -> pd.DataFrame:
    """No-trade band: trade an asset only when its position is off target by more than
    ``buffer`` x max(|target|, |current|).

    Entries, exits and sign flips always trade (the deviation is >= 100% of the
    position); day-to-day resizing caused by noisy volatility estimates does not.
    ``buffer = 0`` reproduces daily rebalancing to target.
    """
    if buffer <= 0:
        return target
    T = target.fillna(0.0).values
    W = np.empty_like(T)
    cur = np.zeros(T.shape[1])
    for t in range(len(T)):
        tgt = T[t]
        trade = np.abs(tgt - cur) > buffer * np.maximum(np.abs(tgt), np.abs(cur))
        cur = np.where(trade, tgt, cur)
        W[t] = cur
    return pd.DataFrame(W, index=target.index, columns=target.columns)


def simulate(name: str, unit_weights: pd.DataFrame, scale: pd.Series,
             excess_returns: pd.DataFrame, risk_free: pd.Series,
             buffer: float = 0.0) -> StrategyResult:
    W = apply_buffer(unit_weights.mul(scale, axis=0).fillna(0.0), buffer)
    gross = (W.shift(1) * excess_returns.fillna(0.0)).sum(axis=1)
    traded = W.diff().abs().sum(axis=1)
    traded.iloc[0] = 0.0
    return StrategyResult(name, W, gross.rename(name), traded.rename(name),
                          risk_free.reindex(W.index).fillna(0.0))
