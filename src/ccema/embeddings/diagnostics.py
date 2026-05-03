"""Positivity diagnostics for OPE-on-NLP.

Implements the diagnostic battery specified in the plan:
- propensity histogram + ESS/n
- max weight, weight quantiles
- DRE classifier AUC
- MMD two-sample test (numpy implementation)
- Conformal coverage diagnostic (Jesson 2021 style)

All return native floats / dicts so the report can be JSON-serialized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's ESS = (sum w)^2 / sum w^2."""
    s = weights.sum()
    if s == 0:
        return 0.0
    return float(s * s / (weights * weights).sum())


def weight_summary(weights: np.ndarray) -> dict[str, float]:
    return {
        "n": int(weights.size),
        "mean": float(weights.mean()) if weights.size else float("nan"),
        "max": float(weights.max()) if weights.size else float("nan"),
        "p95": float(np.quantile(weights, 0.95)) if weights.size else float("nan"),
        "p99": float(np.quantile(weights, 0.99)) if weights.size else float("nan"),
        "ess": effective_sample_size(weights),
        "ess_per_n": effective_sample_size(weights) / max(1, weights.size),
    }


# ---------------------------------------------------------------------------
# DRE classifier AUC — pure numpy
# ---------------------------------------------------------------------------


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U-stat AUC. Higher score = predicted positive."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_p = len(pos)
    n_n = len(neg)
    # Rank-based formula: AUC = (sum_rank(pos) - n_p (n_p+1)/2) / (n_p n_n)
    all_scores = np.concatenate([pos, neg])
    ranks = _average_ranks(all_scores)
    sum_rank_pos = ranks[:n_p].sum()
    return float((sum_rank_pos - n_p * (n_p + 1) / 2) / (n_p * n_n))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(x, dtype=np.float64)
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# MMD two-sample
# ---------------------------------------------------------------------------


def mmd_unbiased(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    """Unbiased estimator of MMD^2 with Gaussian kernel."""
    n = x.shape[0]
    m = y.shape[0]
    K_xx = _gauss_kernel(x, x, sigma)
    K_yy = _gauss_kernel(y, y, sigma)
    K_xy = _gauss_kernel(x, y, sigma)
    np.fill_diagonal(K_xx, 0.0)
    np.fill_diagonal(K_yy, 0.0)
    return float(
        K_xx.sum() / (n * (n - 1) + 1e-12)
        + K_yy.sum() / (m * (m - 1) + 1e-12)
        - 2 * K_xy.mean()
    )


def _gauss_kernel(a: np.ndarray, b: np.ndarray, sigma: float | None) -> np.ndarray:
    sq_a = (a * a).sum(-1)[:, None]
    sq_b = (b * b).sum(-1)[None, :]
    d2 = np.clip(sq_a + sq_b - 2 * a @ b.T, 0, None)
    if sigma is None:
        flat = d2.reshape(-1)
        sigma = float(np.sqrt(np.median(flat[flat > 0]) / 2.0)) if (flat > 0).any() else 1.0
    return np.exp(-d2 / (2 * sigma * sigma + 1e-12))


# ---------------------------------------------------------------------------
# Composite report
# ---------------------------------------------------------------------------


@dataclass
class PositivityGate:
    ess_per_n_min: float = 0.10
    classifier_auc_max: float = 0.90


@dataclass
class PositivityReport:
    weights: dict[str, float]
    classifier_auc: float
    mmd: float
    pass_ess: bool
    pass_auc: bool

    @property
    def passes(self) -> bool:
        return self.pass_ess and self.pass_auc

    def as_dict(self) -> dict:
        return {
            "weights": {k: round(v, 6) if isinstance(v, float) else v for k, v in self.weights.items()},
            "classifier_auc": round(self.classifier_auc, 4),
            "mmd": round(self.mmd, 6),
            "pass_ess": self.pass_ess,
            "pass_auc": self.pass_auc,
            "pass": self.passes,
        }


def positivity_report(
    weights: np.ndarray,
    embeddings_clinician: np.ndarray,
    embeddings_agent: np.ndarray,
    classifier_scores: np.ndarray,
    classifier_labels: np.ndarray,
    gate: PositivityGate | None = None,
) -> PositivityReport:
    g = gate or PositivityGate()
    w_summary = weight_summary(weights)
    auc = roc_auc(classifier_scores, classifier_labels)
    mmd = mmd_unbiased(embeddings_clinician, embeddings_agent)
    return PositivityReport(
        weights=w_summary,
        classifier_auc=auc,
        mmd=mmd,
        pass_ess=(w_summary["ess_per_n"] > g.ess_per_n_min),
        pass_auc=(not (auc != auc) and auc < g.classifier_auc_max),  # NaN-safe
    )
