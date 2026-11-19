# Methodology

Every number in the README can be traced to an equation below and a function in `src/regime_trend/`.

- [1. Research design](#1-research-design)
- [2. Data](#2-data)
- [3. Trend signals](#3-trend-signals)
- [4. Regime models](#4-regime-models)
- [5. Gating, sizing and execution](#5-gating-sizing-and-execution)
- [6. Evaluation](#6-evaluation)
- [7. Pitfalls this design avoids](#7-pitfalls-this-design-avoids)
- [References](#references)

---

## 1. Research design

**Question.** Does conditioning trend-following positions on a hidden-Markov regime
estimate improve risk-adjusted returns *after costs and without look-ahead*?

**Experiment grid.** Three trend signals × four gates, plus a long-only benchmark, all
sharing one data set, one live-asset mask, one sizing rule and one cost model:

| Gate | Information used | Tradeable? | Role |
|---|---|:---:|---|
| none (base) | – | yes | reference |
| HMM, walk-forward, filtered | $x_1..x_t$, re-estimated monthly | yes | **the hypothesis** |
| Volatility regime | trailing realised vol vs. its trailing median | yes | *control*: is the HMM more than "vol is high"? |
| HMM, full-sample, smoothed | $x_1..x_T$ | **no** | measures look-ahead bias |

Two universes test generality: an 18-ETF multi-asset panel (2005–2026 evaluation) and
SPY alone (1996–2026), which contains three equity bear markets.

## 2. Data

Daily adjusted OHLC from Yahoo Finance, cached locally as parquet. The risk-free rate is the
13-week T-bill (^IRX).

* **Unbalanced panel.** Each ETF enters after `min_history_days` = 756 days of history and
  once every input (signals, all three regime series) is defined. Every strategy uses the
  same live-asset mask. Evaluation of the multi-asset book starts in July 2005, with the four
  oldest equity ETFs live. Treasuries and credit join within weeks, and all 18 ETFs are live by
  2010.
* **Hygiene.** Runs of ≥5 identical closes are masked as vendor back-fill. Short gaps (≤5 days)
  are forward-filled, but never before an asset's inception.

## 3. Trend signals

All signals live in $[-1, 1]$ and use information up to the close of *t*.

**Time-series momentum** (Moskowitz, Ooi & Pedersen, 2012):
$s_t = \operatorname{sign}(P_t/P_{t-252} - 1)$.

**Donchian breakout** (the "turtle" rule) as a *stateful* position. Go long when
$C_t > \max(H_{t-55..t-1})$ and exit when $C_t < \min(L_{t-20..t-1})$; shorts mirror this.
The position is **held** between entry and exit. The channels exclude the current bar.
(A common implementation bug is to treat the breakout as a one-day event. The resulting strategy is
almost never invested yet still trades constantly.)

**Kalman local-linear trend** on log prices $y_t$ (Harvey, 1989):

$$y_t = \mu_t + \varepsilon_t,\qquad \mu_{t+1} = \mu_t + \beta_t + \eta_t,\qquad \beta_{t+1} = \beta_t + \zeta_t,$$

with $\operatorname{Var}(\eta)=q_\eta\sigma^2_\varepsilon$ and $\operatorname{Var}(\zeta)=q_\zeta\sigma^2_\varepsilon$. The filter is
implemented from scratch (`kalman.py`): diffuse initialisation, NaN-aware prediction, and a
standard covariance update.

*Calibration.* Only the ratios $q$ affect the filtered state. Maximum likelihood on daily
prices is uninformative here, because daily returns are close to white noise and the likelihood drives
$q_\zeta \to 0$, collapsing the slope to the full-sample drift. Instead the horizon is chosen explicitly.
With $q_\eta = 0$ the model's smoother is the Hodrick–Prescott filter, whose gain halves at
period $P$ when $q_\zeta = (2 - 2\cos(2\pi/P))^2$. We use $P$ = 126 days (≈6-month trends).
The signal is a trend Sharpe ratio, clipped:

$$s_t = \operatorname{clip}\!\Big(\frac{252\,\hat\beta_{t|t}/\hat\sigma_t}{0.5},\,-1,\,1\Big).$$

## 4. Regime models

**Observation.** The daily log return, standardised with the training window's mean and
standard deviation. This is a Hamilton-style switching-variance model (Hamilton, 1989; Ang & Timmermann, 2012).

**Model.** A 2-state Gaussian HMM with full covariances
(`hmmlearn`), fitted by EM. The first fit uses 5 random restarts; later monthly refits keep the
better of a warm start from the previous parameters and a fresh random start.

**Walk-forward protocol.** The first fit comes after 756 observations. The model is re-estimated every 21 days on a
rolling 2,520-day window. Between refits the parameters are frozen and the regime probability is
propagated with the **forward recursion**

$$\alpha_t \propto \big(\alpha_{t-1}^\top A\big)\odot b(x_t),\qquad \alpha_{t_0} = P(s_{t_0}\mid x_{1..t_0}),$$

implemented in log space (`regimes.forward_filter`). A unit test checks it against hmmlearn's
forward-backward output to 1e-8, since the last smoothed row equals the filtered probability.

**Labelling.** States are exchangeable and can swap between refits. After every fit they are
ordered by the variance of the return feature: lowest = *calm*, highest = *turbulent*. Variance is
estimated far more precisely than the mean, so this is much more stable than ranking by mean
return.

**Volatility control.** Turbulent ⇔ the 21-day realised volatility exceeds its trailing 756-day
median. It has no latent state and no fitting.

**Look-ahead benchmark.** A single HMM fitted on the full sample, with forward-backward
**smoothed** probabilities $P(s_t\mid x_{1..T})$. This is what an in-sample notebook computes when it
calls `model.fit(X); model.predict(X)`.

**Model selection.** The BIC on SPY is reported for 1–4 states (in-sample, diagnostic only).

## 5. Gating, sizing and execution

**Gate.** With $p_t = P(\text{turbulent})$:
$g_t = s_t^+\,(1-p_t) + s_t^-\,p_t$. Long conviction is kept in proportion to the probability of a
calm regime, short conviction in proportion to the probability of a turbulent regime. With a 0/1
regime this is exactly the rule "longs only in calm markets, shorts only in turbulent ones".

**Unit-risk weights.** $u_{i,t} = g_{i,t}/(\hat\sigma_{i,t} N_t)$, where $\hat\sigma$ is an EWMA
volatility (30-day half-life) and $N_t$ the number of live assets.

**Portfolio risk.** $k_t = \sigma^*/\sqrt{u_t^\top\hat\Sigma_t u_t}$, with $\sigma^*$ = 10% and $\hat\Sigma_t$ an
EWMA covariance (60-day half-life). Gross leverage is capped. **$k_t$ is computed from the
ungated base book and reused for every gated variant**, so a gate can only remove risk. It never
re-levers the surviving positions, which isolates the filter's value.

**No-trade buffer.** Volatility estimates move every day, so rebalancing exactly to target
generates turnover that is pure noise: about 25× gross per year in this universe, costing about 2.5% a year at
5 bps. As practitioners do, an asset is traded only when its position is more than 25% away
from target, $|W^{target}_{i,t} - W_{i,t-1}| > 0.25\max(|W^{target}_{i,t}|, |W_{i,t-1}|)$.
Entries, exits and sign flips always pass this test. The buffer is identical for every strategy.

**Execution.** Weights decided at the close of *t* earn the excess return of *t+1*. Costs are
`bps × Σ|W_t − W_{t−1}|` (5 bps for the ETF panel, 2 bps for SPY). Collateral earns the T-bill
rate (futures-style accounting), so Sharpe ratios are computed on excess returns.

## 6. Evaluation

| Statistic | Definition |
|---|---|
| Sharpe | $\sqrt{252}\,\bar r^{ex}/\operatorname{sd}(r^{ex})$, net of costs |
| Sortino, max drawdown, Calmar, CAGR | standard; computed on total returns |
| Skew (monthly) | skewness of monthly excess returns; trend following should be positive |
| Avg gross leverage, time invested, turnover | from the final positions $W_t$ |
| Crisis return | compounded total return over named windows (GFC, COVID, 2022, …) |

**Inference.** A circular block bootstrap (Politis & Romano, 1992) with 21-day blocks and 2,000 resamples.
All strategies are resampled with **the same indices**, so each paired difference
$\Delta SR = SR_{gated} - SR_{base}$ keeps the very high correlation between a gated strategy and
its base. That is what gives the test its power. The reported $p$ is the one-sided share of
resamples with $\Delta SR \le 0$.

**Regime diagnostics.** Share of time turbulent, switches per year (real-time vs. hindsight vs.
control), agreement between real-time and hindsight labels, state volatilities and implied durations
$1/(1-a_{kk})$.

## 7. Pitfalls this design avoids

| Pitfall | Consequence | Here |
|---|---|---|
| Fitting the HMM on the full sample and decoding with Viterbi / smoothing | Regimes "know" the future; Sharpe inflates massively | Walk-forward fits + forward filter; the look-ahead version is reported separately, as a measurement |
| Labelling states by in-sample mean return | Labels encode the very returns being traded; unstable across refits | Labels by variance |
| Predicting a regime from one observation (`model.predict(x_t)`) | Ignores the transition dynamics; the regime becomes a noisy function of a single return | Full forward recursion carries $\alpha_{t-1}$ |
| Breakout as a one-day event | Rarely invested, very high turnover | Stateful entry/exit channels |
| Kalman noise in price units (e.g. $R=10^{-3}$ on a \$300 stock) | Behaviour depends on the price level; not comparable across assets | Log prices, scale-free ratios, horizon calibration |
| Benchmarks without costs vs. strategy with costs | Unfair comparison | One cost model for all |
| Daily HMM refits with many EM iterations | Walk-forward never finishes | Monthly refits, warm starts, parallel across assets |
| Picking today's mega-cap winners | Survivorship / selection bias | Broad, rules-based ETF universe |
| Re-levering a gated book to full risk | Gate becomes a concentration bet | Risk scale fixed by the base book |
| Rebalancing to target daily | Volatility-estimate noise becomes turnover and cost | 25% no-trade buffer |

## References

- Ang, A. & Timmermann, A. (2012). Regime changes and financial markets. *Annual Review of Financial Economics*, 4.
- Hamilton, J. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2).
- Harvey, A. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Hodrick, R. & Prescott, E. (1997). Postwar U.S. business cycles. *Journal of Money, Credit and Banking*, 29(1).
- Hurst, B., Ooi, Y. H. & Pedersen, L. H. (2017). A century of evidence on trend-following investing. *Journal of Portfolio Management*, 44(1).
- Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2).
- Nystrup, P., Madsen, H. & Lindström, E. (2017). Long memory of financial time series and hidden Markov models with time-varying parameters. *Journal of Forecasting*, 36(8).
- Politis, D. & Romano, J. (1992). A circular block-resampling procedure for stationary data. In *Exploring the Limits of Bootstrap*. Wiley.
- Rabiner, L. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2).
