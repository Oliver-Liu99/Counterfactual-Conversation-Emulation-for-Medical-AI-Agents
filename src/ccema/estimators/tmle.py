"""Targeted Maximum Likelihood Estimation (van der Laan 2006).

For binary outcomes / continuous outcomes, TMLE updates the outcome model
toward the EIF score so the resulting estimator simultaneously satisfies
the outcome moment condition and the propensity moment condition. For
continuous Y in [0, 1] (our rubric scores), the canonical update is:

    H_i = w_i  (clever covariate)
    fit logistic update:  y_i = sigmoid(logit(f(x_i, a_i)) + eps * H_i)
    estimate eps via MLE
    apply update only at the target-policy "evaluation point"

We use the ATT-style targeting since π_e is not necessarily a flip — we
want E_x[E_{a~pi_e}[Y]], and update f via a one-step logistic submodel
with the clever covariate H.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def tmle_estimate(
    x: np.ndarray,
    a_obs: np.ndarray,
    y_obs: np.ndarray,
    pi_e_prob_a1: np.ndarray,
    cv_folds: int = 5,
    seed: int = 42,
    explicit_pi_b_prob_a1: np.ndarray | None = None,
    truncation_quantile: float | None = 0.99,
    y_range: tuple[float, float] = (-1.5, 1.5),
) -> float:
    """One-step TMLE for the policy value V(pi_e) on continuous outcomes.

    Outcome y is rescaled to [0, 1] via min-max bounds, the logistic
    submodel update is applied, and the estimate is unscaled at the end.
    """
    n = len(x)

    # Cross-fitted nuisance estimates
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    f_obs = np.empty(n)
    f0 = np.empty(n)
    f1 = np.empty(n)
    pb1 = np.empty(n)

    for fold, (tr, te) in enumerate(kf.split(x)):
        xa_tr = np.hstack([x[tr], a_obs[tr].reshape(-1, 1).astype(float)])
        outcome_model = RandomForestRegressor(n_estimators=80, random_state=seed + fold, min_samples_leaf=2)
        outcome_model.fit(xa_tr, y_obs[tr])

        x_te = x[te]
        f_obs[te] = outcome_model.predict(np.hstack([x_te, a_obs[te].reshape(-1, 1).astype(float)]))
        f0[te] = outcome_model.predict(np.hstack([x_te, np.zeros((len(x_te), 1))]))
        f1[te] = outcome_model.predict(np.hstack([x_te, np.ones((len(x_te), 1))]))

        if explicit_pi_b_prob_a1 is not None:
            pb1[te] = np.clip(explicit_pi_b_prob_a1[te], 1e-3, 1 - 1e-3)
        else:
            prop_model = LogisticRegression(max_iter=2000, C=1.0, random_state=seed + fold)
            prop_model.fit(x[tr], a_obs[tr])
            pb1[te] = np.clip(prop_model.predict_proba(x_te)[:, 1], 1e-3, 1 - 1e-3)

    # Density ratio at observed action
    pe_obs = np.where(a_obs == 1, pi_e_prob_a1, 1 - pi_e_prob_a1)
    pb_obs = np.where(a_obs == 1, pb1, 1 - pb1)
    w = pe_obs / pb_obs
    if truncation_quantile is not None and 0 < truncation_quantile < 1:
        cap = float(np.quantile(w, truncation_quantile))
        w = np.clip(w, 0, cap)

    # Rescale outcomes / fitted values to [0, 1]
    y_lo, y_hi = y_range
    span = y_hi - y_lo
    y_s = np.clip((y_obs - y_lo) / span, 1e-3, 1 - 1e-3)
    f_obs_s = np.clip((f_obs - y_lo) / span, 1e-3, 1 - 1e-3)
    f0_s = np.clip((f0 - y_lo) / span, 1e-3, 1 - 1e-3)
    f1_s = np.clip((f1 - y_lo) / span, 1e-3, 1 - 1e-3)

    # Logistic submodel: update f_obs by sigmoid(logit(f_obs) + eps * w)
    def neg_loglik(eps: float) -> float:
        upd = _sigmoid(_logit(f_obs_s) + eps * w)
        upd = np.clip(upd, 1e-6, 1 - 1e-6)
        return float(-(y_s * np.log(upd) + (1 - y_s) * np.log(1 - upd)).sum())

    result = minimize_scalar(neg_loglik, bounds=(-10.0, 10.0), method="bounded")
    eps_hat = float(result.x)

    # Apply update at evaluation point: H_eval = pi_e_prob_a1 / pb1 (for a=1) etc.
    H_a1 = pi_e_prob_a1 / pb1
    H_a0 = (1 - pi_e_prob_a1) / (1 - pb1)
    if truncation_quantile is not None:
        cap = float(np.quantile(np.r_[H_a0, H_a1], truncation_quantile))
        H_a0 = np.clip(H_a0, 0, cap)
        H_a1 = np.clip(H_a1, 0, cap)

    f0_star = _sigmoid(_logit(f0_s) + eps_hat * H_a0)
    f1_star = _sigmoid(_logit(f1_s) + eps_hat * H_a1)
    f_e_star = (1 - pi_e_prob_a1) * f0_star + pi_e_prob_a1 * f1_star

    v_scaled = float(f_e_star.mean())
    return v_scaled * span + y_lo
