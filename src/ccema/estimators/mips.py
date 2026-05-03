"""Marginalized IPS over action embeddings (Saito & Joachims 2022).

Standard IPS in NLP action spaces fails because the action a is too
high-dimensional for the density ratio to be reliably estimated. MIPS
projects a -> phi(a) and assumes the embedding is *reward-sufficient*
(common-embedding-support condition); the density ratio is then estimated
in the lower-dimensional embedding space.

For the toy oracle, phi(a) = a (binary), so MIPS reduces to IPS. The
real value of MIPS comes when phi is the CCE projection from Step 6.
"""

from __future__ import annotations

import numpy as np

from ccema.estimators.ips import classifier_density_ratio, ips_estimate


def mips_estimate(
    x: np.ndarray,
    phi_a_obs: np.ndarray,  # (n, d_phi) embedding of observed actions
    y_obs: np.ndarray,
    phi_a_target: np.ndarray | None = None,  # (n, d_phi) embedding of pi_e samples
    explicit_weights: np.ndarray | None = None,
    self_normalized: bool = True,
    truncation_quantile: float | None = 0.99,
    seed: int = 42,
) -> tuple[float, float]:
    """Marginalized IPS estimate.

    Returns (v_hat, classifier_auc). When `explicit_weights` is given (e.g.
    in the toy oracle with known propensities), the classifier step is
    skipped and we report AUC = nan.
    """
    if explicit_weights is not None:
        v = ips_estimate(
            y_obs, explicit_weights, self_normalized=self_normalized,
            truncation_quantile=truncation_quantile,
        )
        return v, float("nan")

    if phi_a_target is None:
        raise ValueError("Provide phi_a_target or explicit_weights")

    # Train classifier on (x, phi_a_obs) labelled 0 vs (x, phi_a_target) labelled 1
    w, auc = classifier_density_ratio(
        x_b=x, a_b=phi_a_obs[:, 0] if phi_a_obs.shape[1] == 1 else _hash_action(phi_a_obs),
        x_e=x, a_e=phi_a_target[:, 0] if phi_a_target.shape[1] == 1 else _hash_action(phi_a_target),
        seed=seed,
    )
    # The above only handles 1-d embeddings; for multi-dim, hash to scalar
    # (an approximation acceptable for the unit-test toy). Real Step 9
    # pipeline calls classifier_density_ratio directly with full embeddings.
    v = ips_estimate(
        y_obs, w, self_normalized=self_normalized, truncation_quantile=truncation_quantile
    )
    return v, auc


def _hash_action(phi: np.ndarray) -> np.ndarray:
    """Hash multi-dim action embedding to a scalar for the toy classifier."""
    # Simple linear projection — preserves enough signal for the unit test
    rng = np.random.default_rng(0)
    proj = rng.normal(size=(phi.shape[1], 1))
    return (phi @ proj).flatten()
