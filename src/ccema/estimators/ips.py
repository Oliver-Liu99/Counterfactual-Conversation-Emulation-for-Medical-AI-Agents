"""Inverse Propensity Score estimator with classifier-based density ratio.

For NLP action spaces, the density ratio w(x, a) = pi_e(a|x) / pi_b(a|x)
cannot be computed in closed form. We estimate it by training a classifier
to discriminate (x, a) ~ pi_e from (x, a) ~ pi_b and reading off the
ratio from the classifier's softmax (Sugiyama 2012).

For the toy synthetic oracle, the propensities are known and we can pass
the exact w in directly. The toy mode validates the IPS *math*; the
classifier mode validates the *DRE pipeline*.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


def classifier_density_ratio(
    x_b: np.ndarray,
    a_b: np.ndarray,
    x_e: np.ndarray,
    a_e: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Train a logistic classifier to distinguish behavior vs target samples.

    Returns the per-sample density ratio w_i evaluated AT the *behavior*
    samples (x_b, a_b), and the classifier AUC for diagnostics.
    """
    Xb = np.hstack([x_b, a_b.reshape(-1, 1).astype(float)])
    Xe = np.hstack([x_e, a_e.reshape(-1, 1).astype(float)])
    X_all = np.vstack([Xb, Xe])
    y_all = np.r_[np.zeros(len(Xb)), np.ones(len(Xe))]

    n_pos = int(y_all.sum())
    n_neg = int(len(y_all) - n_pos)
    if n_pos < 5 or n_neg < 5:
        # Too small to fit a calibrated classifier — fall back to plain logistic
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_all, y_all)
        # Calibrated probabilities not available; use raw predict_proba
        p = clf.predict_proba(Xb)[:, 1]
    else:
        cv_folds = min(5, n_pos, n_neg)
        clf = CalibratedClassifierCV(LogisticRegression(max_iter=2000), method="isotonic", cv=cv_folds)
        clf.fit(X_all, y_all)
        p = clf.predict_proba(Xb)[:, 1]

    p = np.clip(p, 1e-3, 1 - 1e-3)
    n_b = len(Xb)
    n_e = len(Xe)
    w = (p / (1 - p)) * (n_b / n_e)

    # AUC on full set
    from ccema.embeddings.diagnostics import roc_auc

    scores = clf.predict_proba(X_all)[:, 1]
    auc = float(roc_auc(scores, y_all.astype(int)))

    return w, auc


def ips_estimate(
    y_obs: np.ndarray,
    weights: np.ndarray,
    self_normalized: bool = True,
    truncation_quantile: float | None = 0.99,
) -> float:
    """V_hat^IPS = sum w_i y_i / [n  or  sum w_i] (self-normalized = SNIPS).

    truncation_quantile clips w at the qth quantile (Hájek-style).
    """
    w = np.asarray(weights, dtype=np.float64)
    if truncation_quantile is not None and 0 < truncation_quantile < 1:
        cap = float(np.quantile(w, truncation_quantile))
        w = np.clip(w, 0, cap)

    if self_normalized:
        denom = w.sum()
        if denom == 0:
            return float("nan")
        return float((w * y_obs).sum() / denom)
    return float((w * y_obs).mean())
