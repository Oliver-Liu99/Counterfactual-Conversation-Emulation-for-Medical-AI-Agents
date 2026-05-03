"""Vanilla Doubly Robust estimator.

V_hat^DR = mean_i [ f(x_i, a_i^agent) + w_i * (y_i - f(x_i, a_i^obs)) ]

Outcome model and density ratio are estimated *on the same dataset* — this
is the paper baseline that DML cross-fit improves on.
"""

from __future__ import annotations

import numpy as np

from ccema.estimators.dm import fit_outcome_model


def dr_estimate(
    x: np.ndarray,
    a_obs: np.ndarray,
    y_obs: np.ndarray,
    weights: np.ndarray,
    pi_e_actions: np.ndarray | None = None,
    pi_e_prob_a1: np.ndarray | None = None,
    seed: int = 42,
    truncation_quantile: float | None = 0.99,
) -> float:
    model = fit_outcome_model(x, a_obs, y_obs, seed=seed)

    xa_obs = np.hstack([x, a_obs.reshape(-1, 1).astype(float)])
    f_obs = model.predict(xa_obs)

    if pi_e_actions is not None:
        xa_e = np.hstack([x, pi_e_actions.reshape(-1, 1).astype(float)])
        f_e = model.predict(xa_e)
    elif pi_e_prob_a1 is not None:
        x_a0 = np.hstack([x, np.zeros((len(x), 1))])
        x_a1 = np.hstack([x, np.ones((len(x), 1))])
        f0 = model.predict(x_a0)
        f1 = model.predict(x_a1)
        f_e = (1 - pi_e_prob_a1) * f0 + pi_e_prob_a1 * f1
    else:
        raise ValueError("Provide pi_e_actions or pi_e_prob_a1")

    w = np.asarray(weights, dtype=np.float64)
    if truncation_quantile is not None and 0 < truncation_quantile < 1:
        cap = float(np.quantile(w, truncation_quantile))
        w = np.clip(w, 0, cap)

    return float(np.mean(f_e + w * (y_obs - f_obs)))
