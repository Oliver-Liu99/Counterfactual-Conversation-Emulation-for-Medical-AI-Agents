"""DML Cross-Fit Doubly Robust (Chernozhukov et al. 2018).

Removes nuisance-overfitting first-order bias by sample-splitting:
estimate outcome model and propensity on fold k^c, evaluate DR on fold k,
average across folds. Yields a sqrt(n)-consistent + asymptotically normal
estimator under standard regularity conditions.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def dml_dr_estimate(
    x: np.ndarray,
    a_obs: np.ndarray,
    y_obs: np.ndarray,
    pi_e_prob_a1: np.ndarray,
    cv_folds: int = 5,
    seed: int = 42,
    explicit_pi_b_prob_a1: np.ndarray | None = None,
    truncation_quantile: float | None = 0.99,
) -> float:
    """Cross-fit DR estimator for binary actions.

    explicit_pi_b_prob_a1: when provided (e.g. toy oracle), skip the
    propensity model and use the true propensity. Useful for verifying
    that DR variance comes from outcome model misspecification, not
    propensity estimation.
    """
    rng_kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    n = len(x)
    psi = np.zeros(n)

    for fold, (train_idx, test_idx) in enumerate(rng_kf.split(x)):
        xa_train = np.hstack([x[train_idx], a_obs[train_idx].reshape(-1, 1).astype(float)])
        outcome_model = RandomForestRegressor(n_estimators=80, random_state=seed + fold, min_samples_leaf=2)
        outcome_model.fit(xa_train, y_obs[train_idx])

        # Predict outcome under target policy on test fold
        x_test = x[test_idx]
        x_a0 = np.hstack([x_test, np.zeros((len(x_test), 1))])
        x_a1 = np.hstack([x_test, np.ones((len(x_test), 1))])
        f0 = outcome_model.predict(x_a0)
        f1 = outcome_model.predict(x_a1)
        f_e = (1 - pi_e_prob_a1[test_idx]) * f0 + pi_e_prob_a1[test_idx] * f1

        a_test = a_obs[test_idx]
        xa_test = np.hstack([x_test, a_test.reshape(-1, 1).astype(float)])
        f_obs_test = outcome_model.predict(xa_test)

        # Propensity model
        if explicit_pi_b_prob_a1 is not None:
            pb1 = np.clip(explicit_pi_b_prob_a1[test_idx], 1e-3, 1 - 1e-3)
        else:
            prop_model = LogisticRegression(max_iter=2000, C=1.0, random_state=seed + fold)
            prop_model.fit(x[train_idx], a_obs[train_idx])
            pb1 = np.clip(prop_model.predict_proba(x_test)[:, 1], 1e-3, 1 - 1e-3)

        # Density ratio at observed action
        pe_obs = np.where(a_test == 1, pi_e_prob_a1[test_idx], 1 - pi_e_prob_a1[test_idx])
        pb_obs = np.where(a_test == 1, pb1, 1 - pb1)
        w = pe_obs / pb_obs

        if truncation_quantile is not None and 0 < truncation_quantile < 1:
            cap = float(np.quantile(w, truncation_quantile))
            w = np.clip(w, 0, cap)

        psi[test_idx] = f_e + w * (y_obs[test_idx] - f_obs_test)

    return float(psi.mean())
