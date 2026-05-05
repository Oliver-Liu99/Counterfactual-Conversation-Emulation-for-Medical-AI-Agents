"""Direct Method estimator: V_hat = mean_i f_hat(x_i, a_i^agent).

Also includes a KL-control variant (``dm_kl_estimate``) following
Jaques et al. 2019, "Way Off-Policy Batch Deep Reinforcement Learning of
Implicit Human Preferences in Dialog" (https://arxiv.org/abs/1907.00456),
which penalizes outcome-model predictions on agent actions that depart
from the clinician (behavior) action distribution.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def fit_outcome_model(x: np.ndarray, a: np.ndarray, y: np.ndarray, seed: int = 42):
    """Fit f(x, a) -> y on observed (x_i, a_i, y_i)."""
    xa = np.hstack([x, a.reshape(-1, 1).astype(float)])
    model = RandomForestRegressor(n_estimators=100, random_state=seed, min_samples_leaf=2)
    model.fit(xa, y)
    return model


def dm_estimate(
    x: np.ndarray,
    a_obs: np.ndarray,
    y_obs: np.ndarray,
    pi_e_actions: np.ndarray | None = None,
    pi_e_prob_a1: np.ndarray | None = None,
    seed: int = 42,
) -> float:
    """V_hat = E_x[E_{a~pi_e}[f(x, a)]].

    Two evaluation modes:
        - pi_e_actions provided: deterministic agent actions (1 sample per x)
        - pi_e_prob_a1 provided: stochastic; predict for both a=0 and a=1
          and weight by pi_e
    """
    model = fit_outcome_model(x, a_obs, y_obs, seed=seed)
    if pi_e_actions is not None:
        xa = np.hstack([x, pi_e_actions.reshape(-1, 1).astype(float)])
        return float(model.predict(xa).mean())

    if pi_e_prob_a1 is None:
        raise ValueError("Provide either pi_e_actions or pi_e_prob_a1")

    x_a0 = np.hstack([x, np.zeros((len(x), 1))])
    x_a1 = np.hstack([x, np.ones((len(x), 1))])
    f0 = model.predict(x_a0)
    f1 = model.predict(x_a1)
    return float(((1 - pi_e_prob_a1) * f0 + pi_e_prob_a1 * f1).mean())


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1).astype(float)
    return arr.astype(float)


def dm_kl_estimate(
    x: np.ndarray,
    a_obs: np.ndarray,
    y_obs: np.ndarray,
    phi_a_target: np.ndarray,
    phi_a_obs: np.ndarray | None = None,
    beta: float = 1.0,
    seed: int = 42,
) -> tuple[float, dict]:
    """DM with KL-control penalty (Jaques et al. 2019).

    Penalizes predictions where the agent action embedding is far from any
    clinician action embedding seen at training time, as a simplified
    proxy for the KL constraint between agent and clinician policies in
    "Way Off-Policy Batch Deep Reinforcement Learning of Implicit Human
    Preferences in Dialog" (Jaques et al. 2019,
    https://arxiv.org/abs/1907.00456).

    Pipeline:
        1. Train ``f_hat(x, phi)`` on observed (clinician) data with
           ``RandomForestRegressor``.
        2. For each evaluation point i, compute base prediction
           ``f_hat(x_i, phi_a_target[i])``.
        3. Compute Euclidean distance ``d_i`` from ``phi_a_target[i]`` to
           the nearest clinician embedding in ``phi_a_obs``.
        4. Normalize distances by their median (robust scale) to obtain
           ``d_i_normalized``.
        5. Penalized prediction:
           ``f_KL(x_i) = f_hat(x_i, phi_a_target[i]) - beta * d_i_normalized``.
        6. Return ``(mean(f_KL), diagnostics)``.

    Parameters
    ----------
    x : np.ndarray, shape (n, d_x)
        Context features.
    a_obs : np.ndarray
        Observed (clinician) action; either a 1d index/scalar or an
        embedding. If ``phi_a_obs`` is None, ``a_obs`` is treated as the
        clinician embedding directly (with reshape to 2d when 1d).
    y_obs : np.ndarray, shape (n,)
        Observed outcomes.
    phi_a_target : np.ndarray, shape (n, d_phi)
        Agent action embeddings on which to score.
    phi_a_obs : np.ndarray | None
        Clinician action embeddings. If None, ``a_obs`` is reused.
    beta : float
        KL penalty strength. ``beta=0`` recovers vanilla DM (with the
        same outcome model and embedding featurization).
    seed : int
        Random forest seed.

    Returns
    -------
    v_hat : float
        Mean penalized prediction.
    diagnostics : dict
        ``{"penalty_mean": float, "distance_p99": float,
            "distance_median": float, "beta": float}``.
    """
    if phi_a_obs is None:
        phi_a_obs_2d = _ensure_2d(a_obs)
    else:
        phi_a_obs_2d = _ensure_2d(phi_a_obs)
    phi_a_target_2d = _ensure_2d(phi_a_target)

    if phi_a_obs_2d.shape[1] != phi_a_target_2d.shape[1]:
        raise ValueError(
            "phi_a_obs and phi_a_target must share embedding dimension; "
            f"got {phi_a_obs_2d.shape[1]} vs {phi_a_target_2d.shape[1]}"
        )

    # Train outcome model f(x, phi) on observed (clinician) tuples
    xa_train = np.hstack([np.asarray(x, dtype=float), phi_a_obs_2d])
    model = RandomForestRegressor(n_estimators=100, random_state=seed, min_samples_leaf=2)
    model.fit(xa_train, np.asarray(y_obs, dtype=float))

    # Base predictions on agent embeddings
    xa_target = np.hstack([np.asarray(x, dtype=float), phi_a_target_2d])
    f_base = model.predict(xa_target)

    # Distance to nearest clinician embedding (Euclidean)
    # diff: (n_target, n_obs, d_phi)
    diff = phi_a_target_2d[:, None, :] - phi_a_obs_2d[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))  # (n_target, n_obs)
    nearest = dists.min(axis=1)  # (n_target,)

    # Normalize by median (robust scale). If all zero, keep zero.
    med = float(np.median(nearest))
    if med > 0:
        nearest_norm = nearest / med
    else:
        nearest_norm = nearest.copy()

    f_kl = f_base - beta * nearest_norm
    v_hat = float(f_kl.mean())
    diagnostics = {
        "penalty_mean": float((beta * nearest_norm).mean()),
        "distance_p99": float(np.quantile(nearest, 0.99)),
        "distance_median": med,
        "beta": float(beta),
    }
    return v_hat, diagnostics
