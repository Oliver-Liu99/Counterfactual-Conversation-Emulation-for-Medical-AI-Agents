"""Tests for the PyTorch CCE training path.

These tests are skipped automatically if torch is not installed, so the
suite still runs in numpy-only environments.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ccema.embeddings.cce import CCEConfig, cce_sanity_check  # noqa: E402
from ccema.embeddings.cce_torch import (  # noqa: E402
    CCEProjection,
    hsic_torch,
    info_nce_torch,
    train_cce_torch,
)


def _make_structured_pair(n: int = 80, d_in: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    anchor = rng.normal(size=(n, d_in)).astype(np.float32)
    positive = anchor + 0.05 * rng.normal(size=(n, d_in)).astype(np.float32)
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    other = anchor[perm]
    return anchor, positive, other


def test_cce_projection_forward_shapes() -> None:
    mod = CCEProjection(input_dim=16, projection_dim=8, seed=0)
    x = torch.randn(5, 16)
    y = mod(x)
    assert y.shape == (5, 8)
    assert mod.W.shape == (16, 8)


def test_info_nce_torch_lower_for_aligned_pair() -> None:
    rng = np.random.default_rng(0)
    a = torch.from_numpy(rng.normal(size=(8, 16)).astype(np.float32))
    p_aligned = a.clone()
    p_random = torch.from_numpy(rng.normal(size=(8, 16)).astype(np.float32))
    loss_aligned = float(info_nce_torch(a, p_aligned, tau=0.07).item())
    loss_random = float(info_nce_torch(a, p_random, tau=0.07).item())
    assert loss_aligned < loss_random


def test_hsic_torch_finite_and_nonneg_for_independent_data() -> None:
    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.normal(size=(40, 4)).astype(np.float32))
    y = torch.from_numpy(rng.normal(size=(40, 2)).astype(np.float32))
    h = float(hsic_torch(x, y).item())
    assert np.isfinite(h)
    # Biased empirical HSIC is non-negative by construction
    assert h >= -1e-6


def test_train_cce_torch_loss_decreases() -> None:
    anchor, positive, _ = _make_structured_pair(n=64, d_in=32, seed=0)
    cfg = CCEConfig(
        input_dim=32,
        projection_dim=8,
        epochs=10,
        batch_size=16,
        seed=0,
        learning_rate=1e-2,
    )
    res = train_cce_torch(anchor, positive, cfg)
    assert len(res.history) == 10
    first = res.history[0]["info_nce"]
    last = res.history[-1]["info_nce"]
    assert last < first


def test_train_cce_torch_sanity_gap_positive() -> None:
    anchor, positive, other = _make_structured_pair(n=80, d_in=32, seed=1)
    cfg = CCEConfig(
        input_dim=32,
        projection_dim=16,
        epochs=20,
        batch_size=16,
        seed=1,
        learning_rate=1e-2,
    )
    res = train_cce_torch(anchor, positive, cfg)
    pa = anchor @ res.W
    pp = positive @ res.W
    po = other @ res.W
    sanity = cce_sanity_check(pa, pp, po)
    assert sanity["gap"] > 0.0


def test_train_cce_torch_W_finite() -> None:
    anchor, positive, _ = _make_structured_pair(n=40, d_in=16, seed=2)
    cfg = CCEConfig(
        input_dim=16,
        projection_dim=8,
        epochs=5,
        batch_size=16,
        seed=2,
        learning_rate=1e-2,
    )
    res = train_cce_torch(anchor, positive, cfg)
    W = res.W
    assert isinstance(W, np.ndarray)
    assert W.shape == (16, 8)
    assert np.all(np.isfinite(W))


def test_train_cce_torch_with_confounder_runs() -> None:
    rng = np.random.default_rng(3)
    n, d_in = 40, 16
    anchor, positive, _ = _make_structured_pair(n=n, d_in=d_in, seed=3)
    confounder = rng.normal(size=(n, 3)).astype(np.float32)
    cfg = CCEConfig(
        input_dim=d_in,
        projection_dim=8,
        epochs=4,
        batch_size=16,
        seed=3,
        learning_rate=1e-2,
        lambda_hsic=0.1,
    )
    res = train_cce_torch(anchor, positive, cfg, confounder_features=confounder)
    assert len(res.history) == 4
    assert all(np.isfinite(h["info_nce"]) for h in res.history)
    assert all(np.isfinite(h["hsic"]) for h in res.history)
