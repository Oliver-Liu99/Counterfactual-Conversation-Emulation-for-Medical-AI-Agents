"""Ground truth construction for V_true(pi_agent) per arm.

Per the plan Step 7:
    For each arm a in {clinician, strong, weak, open_medical}:
        For each case i with sample_idx s in 0..n_samples-1:
            y_{i,s,a} = mean over rubric axes of debate-judge Beta(...).mean
    V_true(arm) = mean_i mean_s y_{i,s,a}
    True effect = V_true(arm) - V_true(clinician)
    Bootstrap CIs over cases.

This module operates on already-computed scores (from Step 4 + Step 5);
it does not call any LLM. The caller passes in a long-format score table.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Long-format score record
# ---------------------------------------------------------------------------


@dataclass
class ScoreRow:
    """One judge score for one (case, arm, sample, axis)."""

    case_id: str
    arm: str
    sample_idx: int
    axis: str
    score: float  # in [0, 1] — typically Beta posterior mean
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_to_per_case(
    rows: Sequence[ScoreRow],
) -> dict[tuple[str, str], list[float]]:
    """Returns {(case_id, arm): [overall_means_per_sample]}.

    For each (case, arm), averages across axes within a sample, then
    returns the list across sample_idx values.
    """
    # First group by (case, arm, sample) and average over axes
    by_sample: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in rows:
        by_sample[(r.case_id, r.arm, r.sample_idx)].append(r.score)

    # Then group across samples per (case, arm)
    per_case: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (cid, arm, _s), axis_scores in by_sample.items():
        per_case[(cid, arm)].append(sum(axis_scores) / len(axis_scores))

    return per_case


def per_case_mean(per_case: dict[tuple[str, str], list[float]]) -> dict[tuple[str, str], float]:
    """Mean across sample_idx for each (case, arm)."""
    return {k: float(np.mean(v)) for k, v in per_case.items()}


# ---------------------------------------------------------------------------
# V_true and bootstrap CIs
# ---------------------------------------------------------------------------


@dataclass
class ArmValue:
    arm: str
    n_cases: int
    v_true: float
    se: float
    ci_lower: float
    ci_upper: float


def v_true_per_arm(
    rows: Sequence[ScoreRow],
    arms: Sequence[str] | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, ArmValue]:
    """Compute V_true and bootstrap CI for each arm in `arms`."""
    rng = np.random.default_rng(seed)
    per_case = aggregate_to_per_case(rows)
    case_means = per_case_mean(per_case)
    arms = arms or sorted({r.arm for r in rows})

    out: dict[str, ArmValue] = {}
    for arm in arms:
        per_arm_cases = sorted(c for (c, a) in case_means if a == arm)
        if not per_arm_cases:
            continue
        values = np.array([case_means[(c, arm)] for c in per_arm_cases])
        v = float(values.mean())

        # Bootstrap over cases
        n = len(values)
        boots = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boots[b] = values[idx].mean()
        lo = float(np.quantile(boots, alpha / 2))
        hi = float(np.quantile(boots, 1 - alpha / 2))
        se = float(boots.std(ddof=1))

        out[arm] = ArmValue(arm=arm, n_cases=n, v_true=v, se=se, ci_lower=lo, ci_upper=hi)
    return out


# ---------------------------------------------------------------------------
# Effect contrasts
# ---------------------------------------------------------------------------


@dataclass
class EffectContrast:
    arm: str
    baseline: str
    n_cases: int
    delta: float
    ci_lower: float
    ci_upper: float
    significant: bool  # CI excludes 0


def effect_vs_baseline(
    rows: Sequence[ScoreRow],
    baseline_arm: str = "clinician",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, EffectContrast]:
    """Per-arm Delta = V_true(arm) - V_true(baseline) with paired bootstrap."""
    rng = np.random.default_rng(seed)
    per_case = aggregate_to_per_case(rows)
    case_means = per_case_mean(per_case)

    arms = sorted({a for _, a in case_means} - {baseline_arm})

    # Build paired arrays for each arm: cases that have both arm + baseline
    out: dict[str, EffectContrast] = {}
    for arm in arms:
        common = sorted(c for (c, a) in case_means if a == arm and (c, baseline_arm) in case_means)
        if not common:
            continue
        a_vals = np.array([case_means[(c, arm)] for c in common])
        b_vals = np.array([case_means[(c, baseline_arm)] for c in common])
        diffs = a_vals - b_vals
        delta = float(diffs.mean())

        n = len(common)
        boots = np.empty(n_bootstrap)
        for k in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boots[k] = diffs[idx].mean()
        lo = float(np.quantile(boots, alpha / 2))
        hi = float(np.quantile(boots, 1 - alpha / 2))

        out[arm] = EffectContrast(
            arm=arm,
            baseline=baseline_arm,
            n_cases=n,
            delta=delta,
            ci_lower=lo,
            ci_upper=hi,
            significant=(lo > 0 or hi < 0),
        )
    return out
