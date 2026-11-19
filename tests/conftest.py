"""Synthetic fixtures - the test suite never touches the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_trend.config import HMMConfig


def simulate_regimes(n: int = 3000, vols=(0.006, 0.02), means=(0.0005, -0.001),
                     stay=(0.99, 0.97), seed: int = 0) -> tuple[pd.Series, np.ndarray]:
    """Two-state Markov-switching daily returns with known hidden states."""
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        s = states[t - 1]
        states[t] = s if rng.random() < stay[s] else 1 - s
    r = rng.normal(np.take(means, states), np.take(vols, states))
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(r, index=idx, name="X"), states


@pytest.fixture(scope="session")
def regime_series():
    return simulate_regimes()


@pytest.fixture
def fast_hmm() -> HMMConfig:
    return HMMConfig(n_states=2, features=("return",), train_days=1500, min_train_days=750,
                     refit_every=126, n_iter=60, n_init=2, seed=1)
