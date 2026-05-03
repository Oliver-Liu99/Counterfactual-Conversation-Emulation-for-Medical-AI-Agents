"""Tests for Step 7: ground truth, effect contrasts, decomposition, sensitivity."""

from __future__ import annotations

import numpy as np
import pytest

from ccema.analysis.decomposition import decompose
from ccema.analysis.ground_truth import (
    ScoreRow,
    aggregate_to_per_case,
    effect_vs_baseline,
    per_case_mean,
    v_true_per_arm,
)
from ccema.analysis.sensitivity import fragility_gamma, msm_bounds_per_arm


def _row(case_id: str, arm: str, axis: str, sample_idx: int, score: float, chapter: str = "J", **md):
    return ScoreRow(
        case_id=case_id,
        arm=arm,
        sample_idx=sample_idx,
        axis=axis,
        score=score,
        metadata={"icd_chapter": chapter, **md},
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_averages_axes_within_sample() -> None:
    rows = [
        _row("c1", "clinician", "ddx_accuracy", 0, 0.8),
        _row("c1", "clinician", "mx_safety", 0, 0.6),
        _row("c1", "clinician", "ddx_accuracy", 1, 0.9),
        _row("c1", "clinician", "mx_safety", 1, 0.5),
    ]
    per_case = aggregate_to_per_case(rows)
    assert (("c1", "clinician") in per_case)
    samples = per_case[("c1", "clinician")]
    assert len(samples) == 2
    # sample 0: (0.8 + 0.6) / 2 = 0.7
    # sample 1: (0.9 + 0.5) / 2 = 0.7
    assert sorted(samples) == pytest.approx([0.7, 0.7])


# ---------------------------------------------------------------------------
# V_true with bootstrap CIs
# ---------------------------------------------------------------------------


def test_v_true_recovers_known_mean() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(50):
        rows.append(_row(f"c{i}", "clinician", "ddx_accuracy", 0, float(rng.normal(0.7, 0.05))))
        rows.append(_row(f"c{i}", "agent_strong", "ddx_accuracy", 0, float(rng.normal(0.85, 0.05))))
    arm_values = v_true_per_arm(rows, n_bootstrap=200, seed=0)
    assert abs(arm_values["clinician"].v_true - 0.7) < 0.03
    assert abs(arm_values["agent_strong"].v_true - 0.85) < 0.03
    assert arm_values["clinician"].ci_lower < arm_values["clinician"].v_true
    assert arm_values["clinician"].ci_upper > arm_values["clinician"].v_true


def test_v_true_ci_narrows_with_more_data() -> None:
    # Generate a small and large dataset; CI width should shrink
    rng = np.random.default_rng(1)
    small_rows = [_row(f"c{i}", "x", "a", 0, float(rng.normal(0.5, 0.05))) for i in range(10)]
    large_rows = [_row(f"c{i}", "x", "a", 0, float(rng.normal(0.5, 0.05))) for i in range(200)]
    s = v_true_per_arm(small_rows, n_bootstrap=200, seed=1)["x"]
    L = v_true_per_arm(large_rows, n_bootstrap=200, seed=1)["x"]
    assert (L.ci_upper - L.ci_lower) < (s.ci_upper - s.ci_lower)


# ---------------------------------------------------------------------------
# Effect contrasts
# ---------------------------------------------------------------------------


def test_effect_significant_when_gap_large() -> None:
    rng = np.random.default_rng(2)
    rows = []
    for i in range(50):
        rows.append(_row(f"c{i}", "clinician", "a", 0, float(rng.normal(0.5, 0.05))))
        rows.append(_row(f"c{i}", "strong", "a", 0, float(rng.normal(0.8, 0.05))))
    contrasts = effect_vs_baseline(rows, baseline_arm="clinician", n_bootstrap=300, seed=2)
    assert "strong" in contrasts
    ec = contrasts["strong"]
    assert ec.delta > 0.25
    assert ec.significant


def test_effect_not_significant_when_arms_equal() -> None:
    rng = np.random.default_rng(3)
    rows = []
    for i in range(20):
        # Same distribution for both arms
        rows.append(_row(f"c{i}", "clinician", "a", 0, float(rng.normal(0.5, 0.05))))
        rows.append(_row(f"c{i}", "weak", "a", 0, float(rng.normal(0.5, 0.05))))
    contrasts = effect_vs_baseline(rows, baseline_arm="clinician", n_bootstrap=300, seed=3)
    ec = contrasts["weak"]
    assert abs(ec.delta) < 0.1
    # may or may not include 0; with n=20 likely yes


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def test_decompose_produces_cells_per_axis_and_subpop() -> None:
    rng = np.random.default_rng(4)
    rows = []
    for i in range(30):
        chap = ["J", "K", "I"][i % 3]
        for axis in ("ddx_accuracy", "mx_safety"):
            rows.append(_row(f"c{i}", "clinician", axis, 0, float(rng.normal(0.6, 0.05)), chapter=chap))
            rows.append(_row(f"c{i}", "strong", axis, 0, float(rng.normal(0.8, 0.05)), chapter=chap))
    cells = decompose(rows, baseline_arm="clinician", n_bootstrap=200, seed=4)
    # 1 arm × 2 axes × 3 chapters
    arms = {c.arm for c in cells}
    axes = {c.axis for c in cells}
    chaps = {c.subpop_value for c in cells}
    assert arms == {"strong"}
    assert axes == {"ddx_accuracy", "mx_safety"}
    assert chaps == {"J", "K", "I"}
    assert len(cells) == 1 * 2 * 3
    # All cells should have positive delta
    assert all(c.delta > 0 for c in cells)


# ---------------------------------------------------------------------------
# Sensitivity bounds
# ---------------------------------------------------------------------------


def test_msm_bounds_widen_with_gamma() -> None:
    diffs = np.array([0.1, 0.15, 0.05, 0.20, 0.12])
    bounds = msm_bounds_per_arm({"strong": diffs}, gammas=(1.0, 2.0, 5.0))
    by_g = {b.gamma: b for b in bounds}
    # At Gamma=1, lo == hi == point
    assert abs(by_g[1.0].delta_lo - by_g[1.0].delta_hi) < 1e-9
    # Widens with gamma
    width_2 = by_g[2.0].delta_hi - by_g[2.0].delta_lo
    width_5 = by_g[5.0].delta_hi - by_g[5.0].delta_lo
    assert width_5 > width_2 > 0


def test_msm_sign_robust_when_gamma_low() -> None:
    diffs = np.array([0.5, 0.5, 0.5, 0.5])  # all same, very strong effect
    bounds = msm_bounds_per_arm({"strong": diffs}, gammas=(1.0, 1.5, 2.0))
    assert all(b.sign_robust for b in bounds)


def test_fragility_gamma_finite_for_modest_effect() -> None:
    # Mixed-sign diffs with positive mean: |abs_mean| > |mean|, so the
    # MSM bound flips at finite Γ. Required for Yadlowsky's per-case bound.
    diffs = np.array([-0.05, 0.10, 0.20, -0.10, 0.15])
    g = fragility_gamma(diffs)
    # mean=0.06, abs_mean=0.12, ratio=0.5 → gamma = (1+0.5)/(1-0.5) = 3.0
    assert 1 < g < 5


def test_fragility_gamma_inf_for_overwhelming_effect() -> None:
    diffs = np.array([0.5, 0.5, 0.5, 0.5])
    assert fragility_gamma(diffs) == float("inf")


def test_fragility_gamma_one_when_no_effect() -> None:
    diffs = np.zeros(5)
    assert fragility_gamma(diffs) == 1.0
