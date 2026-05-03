"""Tests for embeddings: losses, backbone wrappers, CCE training, diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from ccema.embeddings.backbones import DeterministicHashEncoder, make_encoder
from ccema.embeddings.cce import (
    CCEConfig,
    NumpyProjection,
    cce_sanity_check,
    train_cce_numpy,
)
from ccema.embeddings.diagnostics import (
    PositivityGate,
    effective_sample_size,
    mmd_unbiased,
    positivity_report,
    roc_auc,
    weight_summary,
)
from ccema.embeddings.losses import _l2_normalize, hsic, info_nce


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def test_info_nce_loss_zero_when_paired_perfectly() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 16))
    p = a.copy()  # identical → max similarity at i==i
    loss, _ = info_nce(a, p, tau=0.07)
    # Loss is approximately log(B) with very small noise; we just check
    # it's small relative to a random pairing.
    rand_loss, _ = info_nce(a, rng.normal(size=(8, 16)), tau=0.07)
    assert loss < rand_loss


def test_info_nce_finite() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(16, 32))
    p = rng.normal(size=(16, 32))
    loss, _ = info_nce(a, p)
    assert np.isfinite(loss)


def test_hsic_independent_data_near_zero() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(64, 4))
    y = rng.normal(size=(64, 2))  # statistically independent
    h = hsic(x, y)
    # Empirical HSIC has finite-sample bias but should be small
    assert abs(h) < 0.05


def test_hsic_dependent_data_positive() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(64, 4))
    y = (x[:, :2] + 0.01 * rng.normal(size=(64, 2)))  # strongly dependent
    h = hsic(x, y)
    h_indep = hsic(x, rng.normal(size=(64, 2)))
    assert h > h_indep


def test_l2_normalize_unit_norm() -> None:
    x = np.array([[3.0, 4.0], [0.0, 1.0]])
    n = _l2_normalize(x)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0)


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------


def test_hash_encoder_deterministic() -> None:
    enc = DeterministicHashEncoder(dim=64)
    a = enc.encode(["hello", "world"])
    b = enc.encode(["hello", "world"])
    assert np.array_equal(a, b)
    assert a.shape == (2, 64)


def test_hash_encoder_distinguishes_inputs() -> None:
    enc = DeterministicHashEncoder(dim=32)
    a = enc.encode(["hello"])
    b = enc.encode(["world"])
    assert not np.allclose(a, b)


def test_make_encoder_hash() -> None:
    enc = make_encoder("hash")
    assert isinstance(enc, DeterministicHashEncoder)


def test_make_encoder_unknown_raises() -> None:
    with pytest.raises(ValueError):
        make_encoder("not_a_real_backbone")


# ---------------------------------------------------------------------------
# CCE training
# ---------------------------------------------------------------------------


def test_numpy_projection_adam_step_changes_W() -> None:
    proj = NumpyProjection(input_dim=8, output_dim=4, seed=0)
    W0 = proj.W.copy()
    grad = np.ones_like(proj.W) * 0.1
    proj.step(grad, lr=0.01)
    assert not np.allclose(proj.W, W0)


def test_train_cce_numpy_decreases_loss() -> None:
    rng = np.random.default_rng(0)
    n, d_in = 64, 32
    # Create a structured signal: anchor ≈ positive, both differ from "other"
    anchor = rng.normal(size=(n, d_in)).astype(np.float32)
    positive = anchor + 0.1 * rng.normal(size=(n, d_in)).astype(np.float32)
    cfg = CCEConfig(input_dim=d_in, projection_dim=8, epochs=5, batch_size=16, seed=0, learning_rate=1e-2)
    proj = train_cce_numpy(anchor, positive, cfg)
    assert len(proj.history) == 5
    first = proj.history[0]["info_nce"]
    last = proj.history[-1]["info_nce"]
    assert last < first  # loss should decrease


def test_train_cce_numpy_sanity_separates_same_x_from_diff_x() -> None:
    rng = np.random.default_rng(1)
    n, d_in = 80, 32
    anchor = rng.normal(size=(n, d_in)).astype(np.float32)
    positive = anchor + 0.05 * rng.normal(size=(n, d_in)).astype(np.float32)
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    other = anchor[perm]

    cfg = CCEConfig(input_dim=d_in, projection_dim=16, epochs=10, batch_size=16, seed=1, learning_rate=1e-2)
    proj = train_cce_numpy(anchor, positive, cfg)

    pa = anchor @ proj.W
    pp = positive @ proj.W
    po = other @ proj.W
    sanity = cce_sanity_check(pa, pp, po)
    assert sanity["gap"] > 0  # same-x cosine > diff-x cosine


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_ess_uniform_weights_equals_n() -> None:
    w = np.ones(50)
    assert effective_sample_size(w) == pytest.approx(50.0)


def test_ess_one_weight_dominant() -> None:
    w = np.array([1.0] * 49 + [1000.0])
    assert effective_sample_size(w) < 5  # heavily concentrated


def test_weight_summary_has_expected_keys() -> None:
    s = weight_summary(np.array([0.5, 1.0, 1.5, 2.0]))
    assert "ess" in s and "ess_per_n" in s and "max" in s and "p99" in s


def test_roc_auc_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_random_labels_near_half() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=200)
    labels = rng.integers(0, 2, size=200)
    auc = roc_auc(scores, labels)
    assert 0.4 < auc < 0.6


def test_mmd_near_zero_for_iid_same_distribution() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 8))
    y = rng.normal(size=(80, 8))  # iid from same distribution
    val = mmd_unbiased(x, y)
    # Unbiased estimator has finite-sample noise; tolerance scales as ~1/sqrt(n)
    assert abs(val) < 0.1


def test_mmd_larger_when_shifted() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 8))
    y_close = rng.normal(size=(80, 8))
    y_shifted = rng.normal(size=(80, 8)) + 3.0
    # Fix sigma so the comparison isn't washed out by the median heuristic
    sigma = 2.0
    assert mmd_unbiased(x, y_shifted, sigma=sigma) > mmd_unbiased(x, y_close, sigma=sigma)
    assert mmd_unbiased(x, y_shifted, sigma=sigma) > 0.1


def test_positivity_report_passes_for_overlapping_distributions() -> None:
    rng = np.random.default_rng(0)
    n = 80
    weights = rng.uniform(0.5, 2.0, size=n)  # benign weights
    emb_a = rng.normal(size=(n, 16))
    emb_b = rng.normal(size=(n, 16))  # overlapping
    scores = rng.uniform(0.45, 0.55, size=n * 2)  # near 0.5 AUC
    labels = np.r_[np.zeros(n), np.ones(n)]
    rep = positivity_report(weights, emb_a, emb_b, scores, labels.astype(int))
    assert rep.pass_ess
    assert rep.pass_auc
    assert rep.passes


def test_positivity_report_fails_when_classifier_perfect() -> None:
    rng = np.random.default_rng(0)
    n = 50
    weights = rng.uniform(0.5, 2.0, size=n)
    emb_a = rng.normal(size=(n, 16))
    emb_b = rng.normal(size=(n, 16)) + 5.0
    # Perfectly separable scores → AUC = 1.0
    scores = np.r_[np.zeros(n), np.ones(n)]
    labels = np.r_[np.zeros(n), np.ones(n)].astype(int)
    rep = positivity_report(weights, emb_a, emb_b, scores, labels, gate=PositivityGate(classifier_auc_max=0.9))
    assert not rep.pass_auc
    assert not rep.passes
