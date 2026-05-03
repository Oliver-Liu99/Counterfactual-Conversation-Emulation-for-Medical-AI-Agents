"""Direct Method estimator: V_hat = mean_i f_hat(x_i, a_i^agent)."""

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
