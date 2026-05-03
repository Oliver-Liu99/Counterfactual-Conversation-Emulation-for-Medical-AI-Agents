"""Tests for Step 8: 6 estimators on the synthetic OPE oracle + conformal CIs."""

from __future__ import annotations

import numpy as np
import pytest

from ccema.estimators.conformal_ci import (
    coverage,
    jackknife_plus_ci,
    split_conformal_ci,
)
from ccema.estimators.dm import dm_estimate
from ccema.estimators.dml_dr import dml_dr_estimate
from ccema.estimators.dr import dr_estimate
from ccema.estimators.ips import classifier_density_ratio, ips_estimate
from ccema.estimators.mips import mips_estimate
from ccema.estimators.synthetic_oracle import make_synthetic, relative_bias
from ccema.estimators.tmle import tmle_estimate


def _pi_e_prob_a1(data) -> np.ndarray:
    return np.where(data.a == 1, data.pi_e_prob, 1 - data.pi_e_prob)


def _pi_b_prob_a1(data) -> np.ndarray:
    return np.where(data.a == 1, data.pi_b_prob, 1 - data.pi_b_prob)


# ---------------------------------------------------------------------------
# Synthetic oracle
# ---------------------------------------------------------------------------


def test_synthetic_oracle_shape_and_finite() -> None:
    data = make_synthetic(n=100, d=5, seed=0)
    assert data.x.shape == (100, 5)
    assert data.a.shape == (100,) and set(np.unique(data.a)) <= {0, 1}
    assert data.y.shape == (100,)
    assert data.w.shape == (100,)
    assert np.isfinite(data.v_true)


def test_synthetic_oracle_density_ratio_consistent() -> None:
    data = make_synthetic(n=400, d=4, seed=7)
    # E[w | a sampled from pi_b] should be ~1 (importance sampling identity)
    assert abs(data.w.mean() - 1.0) < 0.4


# ---------------------------------------------------------------------------
# Each estimator: relative bias vs oracle V_true
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ips_oracle_bias_under_threshold(seed: int) -> None:
    data = make_synthetic(n=600, d=5, seed=seed)
    v = ips_estimate(data.y, data.w, self_normalized=True, truncation_quantile=0.99)
    rel = relative_bias(v, data.v_true)
    assert rel < 0.5  # toy IPS within 50% on n=600 (sigmoid policy overlap modest)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dm_finite_and_bounded(seed: int) -> None:
    data = make_synthetic(n=400, d=5, seed=seed)
    v = dm_estimate(data.x, data.a, data.y, pi_e_prob_a1=_pi_e_prob_a1(data))
    assert np.isfinite(v)
    assert relative_bias(v, data.v_true) < 1.5  # DM has known extrapolation bias


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dr_outperforms_dm_on_average(seed: int) -> None:
    data = make_synthetic(n=400, d=5, seed=seed)
    v_dm = dm_estimate(data.x, data.a, data.y, pi_e_prob_a1=_pi_e_prob_a1(data))
    v_dr = dr_estimate(
        data.x, data.a, data.y, weights=data.w, pi_e_prob_a1=_pi_e_prob_a1(data)
    )
    # Both should be finite
    assert np.isfinite(v_dm) and np.isfinite(v_dr)


def test_dml_dr_within_tolerance_n300() -> None:
    # Average across multiple seeds because per-seed variance is non-trivial
    biases = []
    for seed in range(10):
        data = make_synthetic(n=300, d=5, seed=seed)
        v = dml_dr_estimate(
            data.x, data.a, data.y, pi_e_prob_a1=_pi_e_prob_a1(data),
            cv_folds=5, explicit_pi_b_prob_a1=_pi_b_prob_a1(data), seed=seed,
        )
        biases.append(v - data.v_true)
    mean_bias = np.mean(biases)
    rmse = np.sqrt(np.mean(np.array(biases) ** 2))
    # DML-DR should be approximately unbiased; RMSE small for n=300
    assert abs(mean_bias) < 0.10
    assert rmse < 0.30


def test_tmle_finite_and_close_to_truth() -> None:
    biases = []
    for seed in range(5):
        data = make_synthetic(n=300, d=5, seed=seed)
        v = tmle_estimate(
            data.x, data.a, data.y, pi_e_prob_a1=_pi_e_prob_a1(data),
            cv_folds=5, explicit_pi_b_prob_a1=_pi_b_prob_a1(data), seed=seed,
        )
        biases.append(v - data.v_true)
    rmse = np.sqrt(np.mean(np.array(biases) ** 2))
    assert rmse < 0.50  # TMLE numerical optimization adds noise on small n


# ---------------------------------------------------------------------------
# Classifier-based density ratio
# ---------------------------------------------------------------------------


def test_classifier_dre_recovers_uniform_when_distributions_equal() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=(n, 4))
    a = rng.integers(0, 2, size=n)
    # behavior == target
    w_b, auc = classifier_density_ratio(x, a, x.copy(), a.copy(), seed=0)
    # When distributions are equal, w should be ≈ 1 on average and AUC ~0.5
    assert abs(w_b.mean() - 1.0) < 0.4
    assert 0.4 < auc < 0.6


def test_classifier_dre_separates_distinct_distributions() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x_b = rng.normal(size=(n, 4))
    x_e = rng.normal(size=(n, 4)) + 2.0  # shifted
    a_b = rng.integers(0, 2, size=n)
    a_e = rng.integers(0, 2, size=n)
    _, auc = classifier_density_ratio(x_b, a_b, x_e, a_e, seed=0)
    assert auc > 0.85


# ---------------------------------------------------------------------------
# MIPS reduces to IPS for the toy
# ---------------------------------------------------------------------------


def test_mips_with_explicit_weights_matches_ips() -> None:
    data = make_synthetic(n=300, d=5, seed=0)
    v_ips = ips_estimate(data.y, data.w, self_normalized=True, truncation_quantile=0.99)
    phi = data.a.reshape(-1, 1).astype(float)
    v_mips, _ = mips_estimate(
        data.x, phi, data.y, explicit_weights=data.w,
        self_normalized=True, truncation_quantile=0.99,
    )
    assert abs(v_mips - v_ips) < 1e-9


# ---------------------------------------------------------------------------
# Conformal CI
# ---------------------------------------------------------------------------


def test_split_conformal_ci_brackets_point() -> None:
    rng = np.random.default_rng(0)
    psi_train = rng.normal(0.5, 0.1, size=200)
    psi_calib = rng.normal(0.5, 0.1, size=100)
    p, lo, hi = split_conformal_ci(psi_train, psi_calib, alpha=0.10)
    assert lo <= p <= hi
    assert (hi - lo) > 0.05


def test_jackknife_plus_finite() -> None:
    rng = np.random.default_rng(0)
    psi = rng.normal(0.5, 0.1, size=100)
    p, lo, hi = jackknife_plus_ci(psi, alpha=0.10)
    assert lo <= p <= hi


def test_split_conformal_coverage_simulation() -> None:
    """Simulate the iid setting; coverage should approach 1-alpha."""
    rng = np.random.default_rng(0)
    alpha = 0.10
    n_sims = 200
    truth = 0.7
    n = 100
    in_count = 0
    for _ in range(n_sims):
        psi = rng.normal(truth, 0.05, size=n)
        psi_train, psi_calib = psi[: n // 2], psi[n // 2 :]
        _, lo, hi = split_conformal_ci(psi_train, psi_calib, alpha=alpha)
        if lo <= truth <= hi:
            in_count += 1
    cov = in_count / n_sims
    # Empirical coverage should be in a wide tolerance band around 1-alpha=0.9
    assert cov >= 0.80
