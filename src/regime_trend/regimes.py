r"""Market-regime detection with Gaussian hidden Markov models.

Three regime series are produced for every asset:

1. **Walk-forward HMM (real time).** Re-estimated monthly on a rolling window
   using only past data. The regime probability on day *t* is the **filtered**
   probability :math:`P(s_t \mid x_1,\dots,x_t)`, propagated with the forward
   recursion

   .. math:: \alpha_t \propto \big(\alpha_{t-1}^\top A\big) \odot b(x_t),

   where :math:`A` is the transition matrix and :math:`b` the Gaussian emission densities.
2. **Volatility control.** "Turbulent" whenever 21-day realised volatility is above
   its trailing 3-year median. It has no latent states and no estimation, and asks
   whether the HMM adds anything beyond "volatility is high".
3. **Full-sample HMM (look-ahead, infeasible).** One model fitted on the whole
   history, with **smoothed** probabilities :math:`P(s_t \mid x_1,\dots,x_T)`
   from the forward-backward algorithm. It is used *only* to measure how much
   look-ahead flatters a backtest; it could never have been traded.

**State labelling.** HMM states are exchangeable: "state 0" means nothing, and
labels can swap between refits. After every fit the states are ordered by the
variance of the return feature. The lowest-variance state is *calm* and the highest
is *turbulent*. Variance is estimated far more precisely than the mean, so this
labelling is much more stable than ranking states by mean return.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from .config import HMMConfig, VolControlConfig

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def build_features(log_returns: pd.Series, names: tuple[str, ...]) -> pd.DataFrame:
    """HMM observation vector. Every feature at t uses data up to t only."""
    r = log_returns.dropna()
    cols = {}
    for name in names:
        if name == "return":
            cols[name] = r
        elif name == "log_vol":  # log 21-day realised volatility
            cols[name] = np.log(r.rolling(21).std())
        else:
            raise ValueError(f"Unknown HMM feature '{name}'")
    return pd.DataFrame(cols).dropna()


# --------------------------------------------------------------------------- #
# HMM primitives
# --------------------------------------------------------------------------- #
def _model(cfg: HMMConfig, seed: int, warm: GaussianHMM | None = None) -> GaussianHMM:
    m = GaussianHMM(n_components=cfg.n_states, covariance_type="full", n_iter=cfg.n_iter,
                    tol=cfg.tol, random_state=seed, init_params="" if warm else "stmc",
                    params="stmc", min_covar=1e-4)
    if warm is not None:
        m.startprob_ = warm.startprob_.copy()
        m.transmat_ = warm.transmat_.copy()
        m.means_ = warm.means_.copy()
        m.covars_ = warm.covars_.copy()
    return m


def fit_hmm(X: np.ndarray, cfg: HMMConfig, warm: GaussianHMM | None = None,
            n_init: int = 1, seed: int = 0) -> GaussianHMM:
    """Best-likelihood fit over random restarts (+ a warm start from the previous fit)."""
    candidates = [_model(cfg, seed + k) for k in range(n_init)]
    if warm is not None:
        candidates.append(_model(cfg, seed, warm))
    best, best_score = None, -np.inf
    for m in candidates:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m.fit(X)
                score = m.score(X)
            except (ValueError, np.linalg.LinAlgError):
                continue
        if np.isfinite(score) and score > best_score:
            best, best_score = m, score
    if best is None:
        raise RuntimeError("All HMM fits failed")
    return best


def state_order(model: GaussianHMM) -> np.ndarray:
    """State indices sorted from calm (lowest return variance) to turbulent."""
    return np.argsort(model.covars_[:, 0, 0])


def log_emissions(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """log b_k(x_t) for every observation and state, shape (T, K)."""
    return np.column_stack([
        multivariate_normal.logpdf(X, mean=model.means_[k], cov=model.covars_[k],
                                   allow_singular=True).reshape(len(X))
        for k in range(model.n_components)
    ])


def forward_filter(log_b: np.ndarray, transmat: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Filtered state probabilities continuing from ``prior`` = P(s_{t0} | x_{1..t0})."""
    log_A = np.log(np.clip(transmat, 1e-300, None))
    log_alpha = np.log(np.clip(prior, 1e-300, None))
    out = np.empty_like(log_b)
    for t in range(len(log_b)):
        log_pred = logsumexp(log_alpha[:, None] + log_A, axis=0)  # one-step prediction
        log_alpha = log_pred + log_b[t]
        log_alpha -= logsumexp(log_alpha)
        out[t] = np.exp(log_alpha)
    return out


def bic(model: GaussianHMM, X: np.ndarray) -> float:
    """Bayesian information criterion with an explicit parameter count."""
    k, d = model.n_components, X.shape[1]
    n_params = (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) / 2
    return float(-2.0 * model.score(X) + n_params * np.log(len(X)))


# --------------------------------------------------------------------------- #
# Regime series
# --------------------------------------------------------------------------- #
@dataclass
class RegimeResult:
    ticker: str
    p_turbulent: pd.Series  # filtered, walk-forward (tradeable)
    p_turbulent_smoothed: pd.Series  # full-sample smoothed (look-ahead, NOT tradeable)
    fits: pd.DataFrame  # one row per refit: state statistics and diagnostics


def walk_forward_hmm(log_returns: pd.Series, cfg: HMMConfig) -> tuple[pd.Series, pd.DataFrame]:
    feats = build_features(log_returns, cfg.features)
    vals, dates = feats.values, feats.index
    n = len(vals)
    p = np.full(n, np.nan)
    records = []
    if n < cfg.min_train_days:
        return pd.Series(p, index=dates), pd.DataFrame()

    refits = list(range(cfg.min_train_days - 1, n, cfg.refit_every))
    model = None
    for j, i in enumerate(refits):
        lo = max(0, i - cfg.train_days + 1)
        train = vals[lo: i + 1]
        mu, sd = train.mean(axis=0), train.std(axis=0)
        Xtr = (train - mu) / sd
        model = fit_hmm(Xtr, cfg, warm=model,
                        n_init=cfg.n_init if model is None else 1, seed=cfg.seed + j)
        order = state_order(model)
        turb = order[-1]
        alpha = model.predict_proba(Xtr)[-1]  # last smoothed row == filtered at t_i
        p[i] = alpha[turb]
        end = refits[j + 1] if j + 1 < len(refits) else n - 1
        if end > i:
            Xoos = (vals[i + 1: end + 1] - mu) / sd
            p[i + 1: end + 1] = forward_filter(log_emissions(model, Xoos), model.transmat_,
                                               alpha)[:, turb]
        ann_mu = (model.means_[order, 0] * sd[0] + mu[0]) * 252
        ann_vol = np.sqrt(model.covars_[order, 0, 0]) * sd[0] * np.sqrt(252)
        stay = np.diag(model.transmat_)[order]
        rec = {"date": dates[i], "loglik_per_obs": float(model.score(Xtr) / len(Xtr)),
               "converged": bool(model.monitor_.converged)}
        for s in range(cfg.n_states):
            rec[f"mean_{s}"] = float(ann_mu[s])
            rec[f"vol_{s}"] = float(ann_vol[s])
            rec[f"stay_{s}"] = float(stay[s])
        records.append(rec)
    return pd.Series(p, index=dates, name="p_turbulent"), pd.DataFrame(records).set_index("date")


def full_sample_smoothed(log_returns: pd.Series, cfg: HMMConfig) -> pd.Series:
    """LOOK-AHEAD benchmark: one fit on all data, forward-backward smoothed probabilities."""
    feats = build_features(log_returns, cfg.features)
    X = ((feats - feats.mean()) / feats.std(ddof=0)).values
    model = fit_hmm(X, cfg, n_init=cfg.n_init, seed=cfg.seed)
    post = model.predict_proba(X)
    return pd.Series(post[:, state_order(model)[-1]], index=feats.index, name="p_smoothed")


def volatility_regime(log_returns: pd.Series, cfg: VolControlConfig) -> pd.Series:
    """Naive control: turbulent (1) when realised vol exceeds its trailing median."""
    rv = log_returns.rolling(cfg.window).std()
    med = rv.rolling(cfg.median_window, min_periods=cfg.median_window).median()
    return (rv > med).astype(float).where(med.notna()).rename("p_turbulent_vol")


def _job(args) -> RegimeResult:
    ticker, series, cfg = args
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    p, fits = walk_forward_hmm(series, cfg)
    smoothed = full_sample_smoothed(series, cfg)
    return RegimeResult(ticker, p, smoothed, fits)


def run_hmm_regimes(log_returns: pd.DataFrame, cfg: HMMConfig, workers: int | None = None
                    ) -> dict[str, RegimeResult]:
    """Fit every asset's regime model, in parallel across assets."""
    jobs = [(t, log_returns[t].dropna(), cfg) for t in log_returns.columns]
    if workers == 1:
        results = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_job, jobs))
    return {r.ticker: r for r in results}


def switches_per_year(p_turbulent: pd.Series, threshold: float = 0.5) -> float:
    s = (p_turbulent.dropna() > threshold).astype(int)
    return float(s.diff().abs().sum() / (len(s) / 252)) if len(s) > 1 else np.nan
