"""Performance statistics, crisis windows and paired block-bootstrap inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown(total_returns: pd.Series) -> pd.Series:
    wealth = (1.0 + total_returns).cumprod()
    return wealth / wealth.cummax() - 1.0


def sharpe_ratio(excess: np.ndarray | pd.Series, periods: int = 252) -> float:
    ex = np.asarray(excess, dtype=float)
    sd = ex.std(ddof=1)
    return float(np.sqrt(periods) * ex.mean() / sd) if sd > 0 else np.nan


def performance_summary(total: pd.Series, excess: pd.Series, weights: pd.DataFrame,
                        traded: pd.Series, periods: int = 252) -> dict[str, float]:
    years = len(total) / periods
    cagr = float((1.0 + total).prod() ** (1.0 / years) - 1.0)
    vol = float(excess.std(ddof=1) * np.sqrt(periods))
    downside = float(np.sqrt(np.mean(np.minimum(excess.values, 0.0) ** 2)) * np.sqrt(periods))
    mdd = float(drawdown(total).min())
    monthly = (1.0 + excess).resample("ME").prod() - 1.0
    gross = weights.abs().sum(axis=1)
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe_ratio(excess, periods),
        "Sortino": float(excess.mean() * periods / downside) if downside > 0 else np.nan,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Skew (monthly)": float(monthly.skew()),
        "Avg gross leverage": float(gross.mean()),
        "Time invested": float((gross > 1e-3).mean()),
        "Turnover (1-way, p.a.)": float(0.5 * traded.sum() / years),
    }


def crisis_returns(total: pd.DataFrame, crises: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Compounded total return of every column over each named window."""
    rows = {}
    for name, (start, end) in crises.items():
        window = total.loc[start:end]
        if len(window) >= 2:  # windows outside the sample are skipped, not reported as 0%
            rows[name] = (1.0 + window).prod() - 1.0
    return pd.DataFrame(rows).T


def block_bootstrap_indices(n: int, block: int, n_samples: int, seed: int) -> np.ndarray:
    """Circular block bootstrap (Politis & Romano, 1992), shape (n_samples, n)."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_samples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_samples, -1)[:, :n]


def bootstrap_sharpe(excess: pd.DataFrame, pairs: list[tuple[str, str]], block: int = 21,
                     n_samples: int = 2000, seed: int = 7, periods: int = 252,
                     alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sharpe CIs per column and paired differences for each (strategy, reference) pair.

    All columns are resampled with the same indices, preserving cross-correlation,
    which is what makes the paired difference test powerful.
    """
    ex = excess.values
    idx = block_bootstrap_indices(len(ex), block, n_samples, seed)
    sr = np.empty((n_samples, ex.shape[1]))
    for s in range(0, n_samples, 100):
        sample = ex[idx[s:s + 100]]
        sr[s:s + 100] = np.sqrt(periods) * sample.mean(axis=1) / sample.std(axis=1, ddof=1)
    point = np.sqrt(periods) * ex.mean(axis=0) / ex.std(axis=0, ddof=1)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    single = pd.DataFrame({"Sharpe": point,
                           "CI low": np.percentile(sr, lo, axis=0),
                           "CI high": np.percentile(sr, hi, axis=0)}, index=excess.columns)
    cols = list(excess.columns)
    rows = {}
    for a, b in pairs:
        ia, ib = cols.index(a), cols.index(b)
        d = sr[:, ia] - sr[:, ib]
        rows[(a, b)] = {"dSharpe": point[ia] - point[ib],
                        "CI low": np.percentile(d, lo), "CI high": np.percentile(d, hi),
                        "p (dSharpe <= 0)": float((d <= 0).mean())}
    paired = pd.DataFrame(rows).T
    paired.index = pd.MultiIndex.from_tuples(paired.index, names=["strategy", "reference"])
    return single, paired
