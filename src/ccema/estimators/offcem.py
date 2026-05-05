"""OffCEM: Off-Policy Evaluation via Conjunct Effect Modeling (Saito et al. 2023).

Reference
---------
    Saito, Ren, Joachims (2023). "Off-Policy Evaluation for Large Action
    Spaces via Conjunct Effect Modeling." ICML 2023.
    https://proceedings.mlr.press/v202/saito23b/saito23b.pdf

Idea
----
Decompose the reward into:

    Y = q_phi(x, phi(a)) + delta(x, a)        (cluster effect + residual)

Estimate the cluster effect q_phi via a Direct Method on (x, phi(a)) -> y
(i.e. an outcome model that can ONLY see the embedding, not the raw action).
Estimate the residual via marginalized importance sampling (MIPS-style) in
the embedding space.

The OffCEM estimator is then:

    V_hat = (1/n) * sum_i [
        f_cluster(x_i, phi(a_i^target))                               # DM term
      + w_marginal(x_i, phi(a_i^obs)) * (y_i - f_cluster(x_i, phi(a_i^obs)))   # IPS residual
    ]

with optional SNIPS self-normalization on the IPS residual term and 99-quantile
weight truncation.

Edge cases / sanity:
    - weights == 0          ->   V_hat = E[f_cluster(x, phi(target))]   (pure DM)
    - f_cluster perfect     ->   residual term ~= 0 in expectation       (pure DM)
    - w_marginal = pi_e/pi_b on phi  AND  f_cluster = 0   ->   reduces to MIPS

API mirrors the existing estimators in src/ccema/estimators/{dm,ips,dr,mips,dml_dr,tmle}.py.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from ccema.estimators.ips import classifier_density_ratio


def _truncate(w: np.ndarray, q: float | None) -> np.ndarray:
    """Hájek-style truncation at the q-th quantile of w."""
    if q is None or not (0 < q < 1):
        return w
    cap = float(np.quantile(w, q))
    return np.clip(w, 0.0, cap)


def _hash_embedding(phi: np.ndarray, seed: int = 0) -> np.ndarray:
    """Project a (n, d_phi) embedding to a 1-d scalar via a fixed random
    projection — used only to feed `classifier_density_ratio`, which expects
    a 1-d action input (matching how mips_estimate handles multi-dim phi)."""
    if phi.shape[1] == 1:
        return phi[:, 0].astype(float)
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(phi.shape[1], 1))
    return (phi @ proj).flatten()


def offcem_estimate(
    x: np.ndarray,
    phi_a_obs: np.ndarray,
    y_obs: np.ndarray,
    phi_a_target: np.ndarray,
    weights: np.ndarray | None = None,
    cv_folds: int = 5,
    self_normalized: bool = True,
    truncation_quantile: float | None = 0.99,
    seed: int = 42,
) -> tuple[float, dict]:
    """OffCEM (Saito 2023): cluster-effect DM + residual MIPS.

    Parameters
    ----------
    x : (n, d_x)
        Per-sample context features.
    phi_a_obs : (n, d_phi)
        Embedding of the observed (behavior / clinician) action.
    y_obs : (n,)
        Observed reward.
    phi_a_target : (n, d_phi)
        Embedding of the action drawn by the target policy (agent).
    weights : (n,) or None
        Marginal density ratio w(x, phi) = p(phi | x, pi_e) / p(phi | x, pi_b)
        evaluated at the observed phi_a_obs. If None, fitted via
        `classifier_density_ratio`.
    cv_folds : int
        K for KFold cross-fitting of f_cluster.
    self_normalized : bool
        Apply SNIPS-style normalization on the residual IPS term.
    truncation_quantile : float | None
        Hájek truncation quantile for w. None disables.
    seed : int
        RNG seed.

    Returns
    -------
    v_hat : float
        Point estimate.
    diagnostics : dict
        {
            "v_dm_cluster":  float,   # mean f_cluster(x, phi(a^target))
            "v_ips_residual": float,  # SNIPS-mean of w * (y - f_cluster(x, phi(a^obs)))
            "v_total":       float,   # = v_hat
            "classifier_auc": float | None,
            "weight_mean":   float,
            "weight_max":    float,
        }
    """
    x = np.asarray(x, dtype=np.float64)
    phi_a_obs = np.asarray(phi_a_obs, dtype=np.float64)
    if phi_a_obs.ndim == 1:
        phi_a_obs = phi_a_obs.reshape(-1, 1)
    phi_a_target = np.asarray(phi_a_target, dtype=np.float64)
    if phi_a_target.ndim == 1:
        phi_a_target = phi_a_target.reshape(-1, 1)
    y_obs = np.asarray(y_obs, dtype=np.float64)

    n = len(x)
    if cv_folds < 2 or cv_folds > n:
        cv_folds = max(2, min(5, n))

    # ------------------------------------------------------------------
    # 1) Cross-fit the cluster outcome model f_cluster(x, phi) -> y
    # ------------------------------------------------------------------
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    f_target = np.empty(n)
    f_obs = np.empty(n)
    for fold, (tr, te) in enumerate(kf.split(x)):
        x_tr_phi = np.hstack([x[tr], phi_a_obs[tr]])
        model = RandomForestRegressor(
            n_estimators=100, random_state=seed + fold, min_samples_leaf=2
        )
        model.fit(x_tr_phi, y_obs[tr])

        x_te_target = np.hstack([x[te], phi_a_target[te]])
        x_te_obs = np.hstack([x[te], phi_a_obs[te]])
        f_target[te] = model.predict(x_te_target)
        f_obs[te] = model.predict(x_te_obs)

    # ------------------------------------------------------------------
    # 2) Marginal density ratio w(x, phi(a^obs))
    # ------------------------------------------------------------------
    classifier_auc: float | None = None
    if weights is None:
        a_b = _hash_embedding(phi_a_obs, seed=seed)
        a_e = _hash_embedding(phi_a_target, seed=seed)
        w, auc = classifier_density_ratio(x_b=x, a_b=a_b, x_e=x, a_e=a_e, seed=seed)
        classifier_auc = float(auc)
    else:
        w = np.asarray(weights, dtype=np.float64).copy()

    w = _truncate(w, truncation_quantile)

    # ------------------------------------------------------------------
    # 3) DM cluster term
    # ------------------------------------------------------------------
    v_dm_cluster = float(f_target.mean())

    # ------------------------------------------------------------------
    # 4) IPS residual term (SNIPS by default)
    # ------------------------------------------------------------------
    residual = y_obs - f_obs
    if self_normalized:
        denom = w.sum()
        if denom == 0:
            v_ips_residual = 0.0
        else:
            v_ips_residual = float((w * residual).sum() / denom)
    else:
        v_ips_residual = float((w * residual).mean())

    v_total = v_dm_cluster + v_ips_residual

    diagnostics = {
        "v_dm_cluster": v_dm_cluster,
        "v_ips_residual": v_ips_residual,
        "v_total": v_total,
        "classifier_auc": classifier_auc,
        "weight_mean": float(np.asarray(w).mean()),
        "weight_max": float(np.asarray(w).max()),
    }
    return v_total, diagnostics
