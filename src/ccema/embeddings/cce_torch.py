"""PyTorch implementation of Causal Contrastive Embedding (CCE) training.

This module mirrors the API of `cce.py` (numpy) but uses `torch.nn` and
`torch.optim.AdamW` so the InfoNCE + HSIC objective is differentiated
properly via autograd. The key fix relative to the numpy path: the
gradient of `F.normalize` is included automatically, which is the main
source of bias in the hand-rolled numpy gradient.

The trained `CCEProjection` exposes a `.W` attribute (numpy ndarray of
the final weight) and a `.history` list of per-epoch loss dicts so that
downstream code (which uses `proj.W` and `proj.history`) can swap the
backend without changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ccema.embeddings.cce import CCEConfig


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class CCEProjection(nn.Module):
    """Single linear projection input_dim -> projection_dim, no bias.

    Initialized with the same fan-in scaling as the numpy version so that
    the two backends start from a comparable initial loss.
    """

    def __init__(self, input_dim: int, projection_dim: int = 128, seed: int = 42):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.linear = nn.Linear(input_dim, projection_dim, bias=False)
        # Match numpy init: N(0, 1/sqrt(input_dim))
        scale = 1.0 / max(1.0, math.sqrt(input_dim))
        gen = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.normal(0.0, scale, size=self.linear.weight.shape, generator=gen)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @property
    def W(self) -> np.ndarray:
        # nn.Linear stores weight as (out, in); return (in, out) to match numpy.
        return self.linear.weight.detach().cpu().numpy().T.copy()


# ---------------------------------------------------------------------------
# Differentiable losses
# ---------------------------------------------------------------------------


def info_nce_torch(anchor: torch.Tensor, positive: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE with proper L2-normalization gradients."""
    a = F.normalize(anchor, p=2, dim=-1, eps=1e-12)
    p = F.normalize(positive, p=2, dim=-1, eps=1e-12)
    logits = a @ p.t() / tau
    labels = torch.arange(a.shape[0], device=a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_p = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_p)


def _gaussian_kernel_torch(x: torch.Tensor, sigma: torch.Tensor | float | None = None) -> torch.Tensor:
    sq = (x * x).sum(dim=-1)
    d2 = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * x @ x.t()
    d2 = torch.clamp(d2, min=0.0)
    if sigma is None:
        # Median heuristic on strict upper triangle
        n = x.shape[0]
        if n > 1:
            iu = torch.triu_indices(n, n, offset=1, device=x.device)
            flat = d2[iu[0], iu[1]]
            # Detach the median to keep sigma a constant; we don't want
            # the kernel bandwidth to backprop into the projection.
            med = torch.median(flat).detach()
        else:
            med = torch.tensor(1.0, device=x.device)
        sigma = torch.sqrt(torch.clamp(med, min=1e-6) / 2.0)
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=x.device)
    return torch.exp(-d2 / (2.0 * sigma * sigma + 1e-12))


def hsic_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Biased empirical HSIC (differentiable in x)."""
    n = x.shape[0]
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    K = _gaussian_kernel_torch(x)
    L = _gaussian_kernel_torch(y)
    H = torch.eye(n, device=x.device) - torch.ones((n, n), device=x.device) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    return torch.trace(Kc @ Lc) / max((n - 1) ** 2, 1)


# ---------------------------------------------------------------------------
# Result wrapper — compatible with NumpyProjection interface (.W, .history)
# ---------------------------------------------------------------------------


class _TorchTrainResult:
    """Lightweight stand-in for `NumpyProjection` exposing .W and .history.

    Matches the attributes downstream code reads (.W, .history). Holds a
    reference to the underlying `CCEProjection` for callers who want the
    live module.
    """

    def __init__(self, module: CCEProjection, history: list[dict]):
        self.module = module
        self.history = history

    @property
    def W(self) -> np.ndarray:
        return self.module.W


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def train_cce_torch(
    anchor_emb: np.ndarray,
    positive_emb: np.ndarray,
    config: CCEConfig,
    confounder_features: np.ndarray | None = None,
) -> _TorchTrainResult:
    """Train CCE projection with autograd-backed InfoNCE (+ optional HSIC).

    Returns an object with `.W` (numpy ndarray, shape input_dim x projection_dim)
    and `.history` (list of per-epoch loss dicts), matching `train_cce_numpy`.
    """
    if anchor_emb.shape != positive_emb.shape:
        raise ValueError("anchor_emb and positive_emb must have identical shape")

    torch.manual_seed(int(config.seed))

    # Same input normalization as the numpy path — keeps starting losses
    # comparable across backends.
    anchor_emb = anchor_emb - anchor_emb.mean(0, keepdims=True)
    positive_emb = positive_emb - positive_emb.mean(0, keepdims=True)
    a_scale = float(np.linalg.norm(anchor_emb, axis=-1).mean()) + 1e-12
    anchor_emb = anchor_emb / a_scale
    positive_emb = positive_emb / a_scale

    a_t = torch.from_numpy(np.ascontiguousarray(anchor_emb)).float()
    p_t = torch.from_numpy(np.ascontiguousarray(positive_emb)).float()
    if confounder_features is not None:
        cf = confounder_features
        if cf.ndim == 1:
            cf = cf.reshape(-1, 1)
        c_t: torch.Tensor | None = torch.from_numpy(np.ascontiguousarray(cf)).float()
    else:
        c_t = None

    module = CCEProjection(
        input_dim=config.input_dim,
        projection_dim=config.projection_dim,
        seed=config.seed,
    )
    optim = torch.optim.AdamW(
        module.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )

    n = a_t.shape[0]
    g = torch.Generator().manual_seed(int(config.seed))
    history: list[dict] = []

    for epoch in range(config.epochs):
        idx = torch.randperm(n, generator=g)
        epoch_nce = 0.0
        epoch_hsic = 0.0
        n_batches = 0

        for start in range(0, n, config.batch_size):
            b_idx = idx[start : start + config.batch_size]
            if b_idx.numel() < 2:
                continue
            a_b = a_t[b_idx]
            p_b = p_t[b_idx]

            optim.zero_grad()
            za = module(a_b)
            zp = module(p_b)
            nce = info_nce_torch(za, zp, tau=config.tau)
            loss = nce
            h_val = 0.0
            if c_t is not None and config.lambda_hsic > 0:
                c_b = c_t[b_idx]
                h = hsic_torch(za, c_b)
                loss = loss + config.lambda_hsic * h
                h_val = float(h.detach().item())
            loss.backward()
            # Modest gradient clipping for stability on small batches
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=5.0)
            optim.step()

            epoch_nce += float(nce.detach().item())
            epoch_hsic += h_val
            n_batches += 1

        history.append(
            {
                "epoch": epoch + 1,
                "info_nce": epoch_nce / max(1, n_batches),
                "hsic": epoch_hsic / max(1, n_batches),
            }
        )

    module.eval()
    return _TorchTrainResult(module, history)
