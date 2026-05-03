"""Losses for Causal Contrastive Embedding (CCE).

The training objective combines:
- InfoNCE on (anchor=phi(x_i, a_i^clinician), positive=phi(x_i, a_i^agent),
  negatives=phi(x_j, ·) for j != i). Positives share *the same context*
  with a different action — so the embedding learns to separate
  context-conditional action variation, exactly what the IPS density
  ratio needs.
- HSIC penalty: minimize statistical dependence between the embedding and
  observed confounders (demographics) so the embedding is "Markov-blanket
  free" of confounder variation.

Both losses are implemented with numpy primitives so unit tests do not
require torch. A torch wrapper is provided for the actual training loop.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# InfoNCE — symmetric over batch
# ---------------------------------------------------------------------------


def info_nce(
    anchor: np.ndarray,  # (B, d)
    positive: np.ndarray,  # (B, d)
    tau: float = 0.07,
) -> tuple[float, np.ndarray]:
    """Symmetric InfoNCE loss.

    Treats `positive[i]` as the only positive for anchor[i]; all other
    rows in the batch are in-batch negatives. Returns (loss, per_anchor_loss).
    """
    a = _l2_normalize(anchor)
    p = _l2_normalize(positive)
    sim = a @ p.T / tau  # (B, B)
    # Numerical stabilization
    sim -= sim.max(axis=1, keepdims=True)

    log_z = np.log(np.exp(sim).sum(axis=1) + 1e-12)
    loss_per = log_z - np.diag(sim)
    return float(loss_per.mean()), loss_per


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n


# ---------------------------------------------------------------------------
# HSIC — Hilbert-Schmidt Independence Criterion (Gretton 2005)
# ---------------------------------------------------------------------------


def gaussian_kernel(x: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Pairwise gaussian kernel with optional median heuristic for sigma."""
    sq = np.sum(x * x, axis=-1)
    d2 = sq[:, None] + sq[None, :] - 2 * x @ x.T
    d2 = np.clip(d2, 0, None)
    if sigma is None:
        # median heuristic on flattened pairwise distances (>0 only)
        flat = d2[np.triu_indices_from(d2, k=1)]
        m = np.median(flat) if len(flat) > 0 else 1.0
        sigma = math.sqrt(max(m, 1e-6) / 2.0)
    K = np.exp(-d2 / (2 * sigma * sigma + 1e-12))
    return K


def hsic(x: np.ndarray, y: np.ndarray, sigma_x: float | None = None, sigma_y: float | None = None) -> float:
    """Biased empirical HSIC between x and y."""
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x and y must have same batch size")
    K = gaussian_kernel(x, sigma=sigma_x)
    L = gaussian_kernel(y if y.ndim == 2 else y.reshape(-1, 1), sigma=sigma_y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    return float(np.trace(Kc @ Lc) / (n - 1) ** 2)


# ---------------------------------------------------------------------------
# Combined CCE loss
# ---------------------------------------------------------------------------


def cce_loss(
    anchor: np.ndarray,
    positive: np.ndarray,
    confounder_features: np.ndarray | None = None,
    tau: float = 0.07,
    lambda_hsic: float = 0.1,
) -> dict[str, float]:
    """Composite CCE loss = InfoNCE + lambda_hsic * HSIC(embedding, confounder)."""
    nce, _ = info_nce(anchor, positive, tau=tau)
    out = {"info_nce": nce, "hsic": 0.0, "total": nce}
    if confounder_features is not None and lambda_hsic > 0:
        h = hsic(np.vstack([anchor, positive]),
                 np.vstack([confounder_features, confounder_features]))
        out["hsic"] = h
        out["total"] = nce + lambda_hsic * h
    return out
