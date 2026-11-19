"""regime_trend: do hidden-Markov regime filters improve trend following?

Modules
-------
config      YAML experiment configuration -> typed dataclasses
data        OHLC / risk-free download with a local parquet cache, data hygiene
kalman      local-linear-trend Kalman filter (from scratch)
signals     trend signals: time-series momentum, Donchian breakout, Kalman trend
regimes     walk-forward Gaussian HMM with real-time (filtered) probabilities,
            a naive volatility-regime control, and a look-ahead benchmark
portfolio   regime gating, volatility targeting, costs, P&L
metrics     performance, crisis windows, paired block-bootstrap inference
plotting    one consistent visual system for every figure
experiments the research pipeline that produces every table and figure
"""

__version__ = "0.1.0"
