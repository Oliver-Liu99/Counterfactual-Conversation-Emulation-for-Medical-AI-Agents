"""Tests for the DM with KL-control penalty (Jaques et al. 2019).

See ``ccema.estimators.dm.dm_kl_estimate`` for the estimator under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccema.estimators.dm import dm_estimate, dm_kl_estimate
from ccema.estimators.synthetic_oracle import make_synthetic


def _pi_e_actions_from_oracle(data) -> np.ndarray:
    """Use a deterministic agent policy: flip clinician 50% of the time
    based on a feature, so embeddings *can* differ — but on the toy
    binary action space the embedding remains 1-hot inside {0, 1}.
    """
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=len(data.a))


def test_dm_kl_reduces_to_dm_when_beta_zero() -> None:
    """With beta=0 the penalty vanishes and DM-KL equals plain DM trained
    on the same featurization (1-hot binary action embedding)."""
    data = make_synthetic(n=300, d=5, seed=0)
    a_target = _pi_e_actions_from_oracle(data)

    # 1-d "embedding" = the action itself (clinician)
    phi_obs = data.a.reshape(-1, 1).astype(float)
    phi_target = a_target.reshape(-1, 1).astype(float)

    v_kl, diag = dm_kl_estimate(
        x=data.x,
        a_obs=data.a,
        y_obs=data.y,
        phi_a_target=phi_target,
        phi_a_obs=phi_obs,
        beta=0.0,
        seed=42,
    )
    v_dm = dm_estimate(
        data.x, data.a, data.y, pi_e_actions=a_target, seed=42,
    )
    assert np.isfinite(v_kl)
    assert abs(v_kl - v_dm) < 1e-9
    assert diag["penalty_mean"] == 0.0


def test_dm_kl_penalizes_far_actions() -> None:
    """Synthetic where agent embeddings are pushed far from any clinician
    embedding — penalty should be > 0 and V_hat should shrink relative
    to beta=0."""
    rng = np.random.default_rng(0)
    n, d_x, d_phi = 200, 4, 3
    x = rng.normal(size=(n, d_x))
    # Clinician embeddings in a tight cluster near origin
    phi_obs = rng.normal(scale=0.1, size=(n, d_phi))
    # Outcome depends on x and embedding norm
    y = (x.sum(axis=1) + np.linalg.norm(phi_obs, axis=1)
         + rng.normal(scale=0.1, size=n))
    # Agent embeddings far from the clinician cluster
    phi_target = rng.normal(scale=0.1, size=(n, d_phi)) + np.array([5.0, 5.0, 5.0])

    v_zero, diag_zero = dm_kl_estimate(
        x=x, a_obs=phi_obs, y_obs=y, phi_a_target=phi_target,
        phi_a_obs=phi_obs, beta=0.0, seed=0,
    )
    v_kl, diag_kl = dm_kl_estimate(
        x=x, a_obs=phi_obs, y_obs=y, phi_a_target=phi_target,
        phi_a_obs=phi_obs, beta=1.0, seed=0,
    )
    assert diag_kl["penalty_mean"] > 0.0
    assert diag_kl["distance_p99"] > 0.0
    # KL-penalized value strictly below the unpenalized one
    assert v_kl < v_zero


def test_dm_kl_finite_and_bounded() -> None:
    """Sanity: outputs are finite for typical inputs; diagnostics
    contain the expected keys."""
    data = make_synthetic(n=200, d=5, seed=1)
    phi_obs = data.a.reshape(-1, 1).astype(float)
    a_target = (1 - data.a).astype(int)  # opposite policy
    phi_target = a_target.reshape(-1, 1).astype(float)

    v_kl, diag = dm_kl_estimate(
        x=data.x,
        a_obs=data.a,
        y_obs=data.y,
        phi_a_target=phi_target,
        phi_a_obs=phi_obs,
        beta=1.0,
        seed=7,
    )
    assert np.isfinite(v_kl)
    for key in ("penalty_mean", "distance_p99", "distance_median", "beta"):
        assert key in diag
    assert np.isfinite(diag["penalty_mean"])
    assert np.isfinite(diag["distance_p99"])


def test_dm_kl_phi_a_obs_defaults_to_a_obs_2d() -> None:
    """When phi_a_obs is None, a_obs is treated as the embedding."""
    rng = np.random.default_rng(2)
    n, d_x, d_phi = 80, 3, 2
    x = rng.normal(size=(n, d_x))
    phi_obs = rng.normal(size=(n, d_phi))
    y = x.sum(axis=1) + rng.normal(scale=0.1, size=n)
    phi_target = rng.normal(size=(n, d_phi))

    v_a, _ = dm_kl_estimate(
        x=x, a_obs=phi_obs, y_obs=y, phi_a_target=phi_target,
        phi_a_obs=None, beta=0.5, seed=0,
    )
    v_b, _ = dm_kl_estimate(
        x=x, a_obs=phi_obs, y_obs=y, phi_a_target=phi_target,
        phi_a_obs=phi_obs, beta=0.5, seed=0,
    )
    assert abs(v_a - v_b) < 1e-9


def test_dm_kl_dimension_mismatch_raises() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 3))
    phi_obs = rng.normal(size=(10, 2))
    phi_target = rng.normal(size=(10, 4))
    y = rng.normal(size=10)
    with pytest.raises(ValueError):
        dm_kl_estimate(
            x=x, a_obs=phi_obs, y_obs=y,
            phi_a_target=phi_target, phi_a_obs=phi_obs,
        )
