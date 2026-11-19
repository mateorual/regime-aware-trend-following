import numpy as np
import pytest

from regime_trend import regimes as R
from regime_trend.config import VolControlConfig


def test_forward_filter_matches_hmmlearn(regime_series, fast_hmm):
    r, _ = regime_series
    X = ((r - r.mean()) / r.std()).values[:, None]
    m = R.fit_hmm(X, fast_hmm, n_init=1)
    alpha = m.predict_proba(X[:1000])[-1]
    ours = R.forward_filter(R.log_emissions(m, X[1000:]), m.transmat_, alpha)
    ref = m.predict_proba(X)[-1]  # the last smoothed row equals the filtered probability
    np.testing.assert_allclose(ours[-1], ref, atol=1e-8)


def test_states_are_ordered_calm_to_turbulent(regime_series, fast_hmm):
    r, _ = regime_series
    X = ((r - r.mean()) / r.std()).values[:, None]
    m = R.fit_hmm(X, fast_hmm, n_init=2)
    variances = m.covars_[R.state_order(m), 0, 0]
    assert np.all(np.diff(variances) > 0)


def test_walk_forward_hmm_recovers_simulated_regimes(regime_series, fast_hmm):
    r, states = regime_series
    p, fits = R.walk_forward_hmm(np.log1p(r), fast_hmm)
    valid = p.notna().values
    accuracy = np.mean((p.values[valid] > 0.5) == (states[valid] == 1))
    assert accuracy > 0.85
    assert (fits["vol_0"] < fits["vol_1"]).all()
    assert fits["converged"].all()


def test_walk_forward_hmm_has_no_look_ahead(regime_series, fast_hmm):
    r, _ = regime_series
    lr = np.log1p(r)
    p_full, _ = R.walk_forward_hmm(lr, fast_hmm)
    cut = 1600
    shocked = lr.copy()
    shocked.iloc[cut + 1:] = shocked.iloc[cut + 1:] * 5 + 0.01  # rewrite the future
    p_shock, _ = R.walk_forward_hmm(shocked, fast_hmm)
    np.testing.assert_allclose(p_full.iloc[: cut + 1], p_shock.iloc[: cut + 1])


def test_smoothed_probabilities_are_cleaner_than_filtered(regime_series, fast_hmm):
    r, _ = regime_series
    lr = np.log1p(r)
    p, _ = R.walk_forward_hmm(lr, fast_hmm)
    smooth = R.full_sample_smoothed(lr, fast_hmm).reindex(p.dropna().index)
    assert R.switches_per_year(smooth) <= R.switches_per_year(p.dropna())


def test_bic_prefers_two_states_on_switching_data(regime_series, fast_hmm):
    r, _ = regime_series
    X = ((r - r.mean()) / r.std()).values[:, None]
    one = R.fit_hmm(X, type(fast_hmm)(**{**fast_hmm.__dict__, "n_states": 1}))
    two = R.fit_hmm(X, fast_hmm, n_init=2)
    assert R.bic(two, X) < R.bic(one, X)


def test_volatility_regime_is_binary_and_causal(regime_series):
    r, _ = regime_series
    cfg = VolControlConfig(window=21, median_window=252)
    full = R.volatility_regime(r, cfg)
    part = R.volatility_regime(r.iloc[:1000], cfg)
    assert set(full.dropna().unique()) <= {0.0, 1.0}
    np.testing.assert_array_equal(full.iloc[:1000].values, part.values)


def test_unknown_feature_is_rejected(regime_series):
    with pytest.raises(ValueError):
        R.build_features(regime_series[0], ("volume",))
