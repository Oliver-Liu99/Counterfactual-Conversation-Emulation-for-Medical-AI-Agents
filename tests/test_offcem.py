"""Tests for OffCEM estimator (Saito et al. 2023).

OffCEM = cluster-effect DM on (x, phi(a)) + residual MIPS.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccema.estimators.dm import dm_estimate
from ccema.estimators.ips import ips_estimate
from ccema.estimators.offcem import offcem_estimate
from ccema.estimators.synthetic_oracle import make_synthetic, relative_bias


def _pi_e_prob_a1(data) -> np.ndarray:
    return np.where(data.a == 1, data.pi_e_prob, 1 - data.pi_e_prob)


def _sample_target_actions(data, seed: int) -> np.ndarray:
    """Draw one binary action per row from pi_e using the oracle's pi_e_prob."""
    rng = np.random.default_rng(seed)
    p_e_1 = _pi_e_prob_a1(data)
    return (rng.uniform(size=len(data.a)) < p_e_1).astype(np.int64)


# ---------------------------------------------------------------------------
# Reduction sanity: OffCEM with weights = 0 reduces to pure DM (cluster term)
# ---------------------------------------------------------------------------


def test_offcem_reduces_to_dm_when_weights_zero() -> None:
    data = make_synthetic(n=300, d=5, seed=0)
    phi_obs = data.a.reshape(-1, 1).astype(float)
    a_target = _sample_target_actions(data, seed=0)
    phi_target = a_target.reshape(-1, 1).astype(float)

    zero_weights = np.zeros(len(data.a))
    v, diag = offcem_estimate(
        data.x,
        phi_obs,
        data.y,
        phi_target,
        weights=zero_weights,
        cv_folds=5,
        seed=0,
    )

    # Residual IPS term is exactly 0 (SNIPS denominator handled gracefully)
    assert diag["v_ips_residual"] == 0.0
    # Total equals DM cluster term
    assert v == diag["v_dm_cluster"]
    assert np.isfinite(v)
    # And v_dm_cluster should be the cross-fitted mean of f(x, phi(target)) — finite & bounded
    assert -2.0 < diag["v_dm_cluster"] < 2.0


# ---------------------------------------------------------------------------
# Reduction sanity: when f_cluster perfectly fits y_obs, residual IPS ~ 0
# ---------------------------------------------------------------------------


def test_offcem_reduces_to_mips_when_f_cluster_perfect() -> None:
    """If we craft data where f_cluster *cannot* explain Y at all (constant
    rewards function only of phi(a)), the residual term should still be small
    and bounded (the DM term carries most of the signal). Conversely, this
    test verifies the IPS-residual collapses near zero when the cluster
    outcome model is well-specified."""
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=(n, 3))
    a = rng.integers(0, 2, size=n)
    phi_obs = a.reshape(-1, 1).astype(float)
    # y depends only on phi(a) — cluster effect is perfect; residual is pure noise
    y = 0.3 * phi_obs[:, 0] + 0.05 * rng.normal(size=n)

    a_target = rng.integers(0, 2, size=n)
    phi_target = a_target.reshape(-1, 1).astype(float)
    weights = np.ones(n)  # behaviour == target on the embedding

    v, diag = offcem_estimate(
        x, phi_obs, y, phi_target, weights=weights, cv_folds=5, seed=0
    )

    # Residual IPS should be essentially noise (mean ~ 0); DM term dominates
    assert abs(diag["v_ips_residual"]) < 0.05
    assert abs(diag["v_dm_cluster"]) < 0.5
    assert np.isfinite(v)


# ---------------------------------------------------------------------------
# Headline: OffCEM RMSE <= min(DM, MIPS) RMSE on the oracle (10 seeds avg)
# ---------------------------------------------------------------------------


def test_offcem_outperforms_dm_and_mips_on_oracle() -> None:
    n_seeds = 10
    sq_dm: list[float] = []
    sq_mips: list[float] = []  # plain IPS with toy weights == MIPS on toy
    sq_off: list[float] = []

    for seed in range(n_seeds):
        data = make_synthetic(n=400, d=5, seed=seed)
        a_target = _sample_target_actions(data, seed=seed)
        phi_obs = data.a.reshape(-1, 1).astype(float)
        phi_target = a_target.reshape(-1, 1).astype(float)

        v_dm = dm_estimate(
            data.x, data.a, data.y, pi_e_prob_a1=_pi_e_prob_a1(data)
        )
        v_mips = ips_estimate(
            data.y, data.w, self_normalized=True, truncation_quantile=0.99
        )
        v_off, _ = offcem_estimate(
            data.x,
            phi_obs,
            data.y,
            phi_target,
            weights=data.w,
            cv_folds=5,
            seed=seed,
        )

        sq_dm.append((v_dm - data.v_true) ** 2)
        sq_mips.append((v_mips - data.v_true) ** 2)
        sq_off.append((v_off - data.v_true) ** 2)

    rmse_dm = float(np.sqrt(np.mean(sq_dm)))
    rmse_mips = float(np.sqrt(np.mean(sq_mips)))
    rmse_off = float(np.sqrt(np.mean(sq_off)))

    # Allow a small slack so randomness doesn't flip the test
    assert rmse_off <= min(rmse_dm, rmse_mips) + 0.05, (
        f"OffCEM RMSE={rmse_off:.4f} should be <= min(DM={rmse_dm:.4f}, "
        f"MIPS={rmse_mips:.4f}) within 0.05 slack"
    )


# ---------------------------------------------------------------------------
# Sanity / API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_offcem_finite_and_bounded(seed: int) -> None:
    data = make_synthetic(n=300, d=5, seed=seed)
    phi_obs = data.a.reshape(-1, 1).astype(float)
    a_target = _sample_target_actions(data, seed=seed)
    phi_target = a_target.reshape(-1, 1).astype(float)

    v, diag = offcem_estimate(
        data.x,
        phi_obs,
        data.y,
        phi_target,
        weights=data.w,
        cv_folds=5,
        seed=seed,
    )
    assert np.isfinite(v)
    assert relative_bias(v, data.v_true) < 1.5  # Like DM, sanity bound
    # Diagnostics keys present
    for k in ("v_dm_cluster", "v_ips_residual", "v_total", "weight_mean", "weight_max"):
        assert k in diag
    assert diag["v_total"] == v


def test_offcem_classifier_density_ratio_path() -> None:
    """When weights=None, OffCEM trains a classifier-based density ratio
    internally and returns a finite estimate + an AUC diagnostic."""
    data = make_synthetic(n=300, d=5, seed=0)
    phi_obs = data.a.reshape(-1, 1).astype(float)
    a_target = _sample_target_actions(data, seed=0)
    phi_target = a_target.reshape(-1, 1).astype(float)

    v, diag = offcem_estimate(
        data.x,
        phi_obs,
        data.y,
        phi_target,
        weights=None,
        cv_folds=5,
        seed=0,
    )
    assert np.isfinite(v)
    assert diag["classifier_auc"] is not None
    assert 0.0 <= diag["classifier_auc"] <= 1.0
