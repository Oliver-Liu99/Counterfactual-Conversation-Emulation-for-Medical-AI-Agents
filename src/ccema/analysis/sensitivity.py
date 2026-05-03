"""Marginal Sensitivity Model bounds (Tan 2006; Yadlowsky 2018; Kallus & Zhou 2021).

For policy-value contrast Δ = V(π_agent) - V(π_b), the logistic-MSM
parameterized by Γ ≥ 1 bounds the worst-case Δ when an unmeasured
confounder shifts the propensity within [1/Γ, Γ] of the observed.

For paired per-case differences d_i = y_i^agent - y_i^clin (after weighting
or matching), the worst-case lower bound at confounder strength Γ is

    Δ_lo(Γ) = mean_i d_i - (Γ - 1) * |d_i| / (Γ + 1)        (one-sided)
    Δ_hi(Γ) = mean_i d_i + (Γ - 1) * |d_i| / (Γ + 1)

This is the *simple* per-case bound used by Yadlowsky 2018 §4.1; full
sharp bounds require a constrained optimization. For the paper we use
this conservative version which is monotone in Γ and analytic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class SensitivityBound:
    arm: str
    gamma: float
    delta_point: float
    delta_lo: float
    delta_hi: float
    sign_robust: bool  # True if 0 not in [lo, hi]


def msm_bounds_per_arm(
    paired_diffs: dict[str, np.ndarray],
    gammas: Sequence[float] = (1.0, 1.5, 2.0, 3.0),
) -> list[SensitivityBound]:
    """Compute MSM bounds for a dict of {arm: paired_diffs_array}."""
    out: list[SensitivityBound] = []
    for arm, diffs in paired_diffs.items():
        if len(diffs) == 0:
            continue
        point = float(diffs.mean())
        for gamma in gammas:
            margin = float(((gamma - 1) / (gamma + 1)) * np.abs(diffs).mean())
            lo = point - margin
            hi = point + margin
            out.append(
                SensitivityBound(
                    arm=arm,
                    gamma=float(gamma),
                    delta_point=point,
                    delta_lo=lo,
                    delta_hi=hi,
                    sign_robust=(lo > 0 or hi < 0),
                )
            )
    return out


def fragility_gamma(diffs: np.ndarray, gamma_max: float = 5.0, n_steps: int = 50) -> float:
    """Smallest Γ ≥ 1 at which the bound straddles 0 (sign flips).

    Returns ∞ if the sign holds at gamma_max; 1.0 if even at no
    confounding the diff is not significant in sign.
    """
    if len(diffs) == 0:
        return float("nan")
    point = diffs.mean()
    abs_mean = np.abs(diffs).mean()
    if abs_mean == 0:
        return 1.0
    # Solve point ± (g-1)/(g+1) * abs_mean = 0
    if point > 0:
        # margin > point  →  (g-1)/(g+1) > point/abs_mean
        ratio = point / abs_mean
        if ratio >= 1.0:
            return float("inf")
        gamma = (1 + ratio) / (1 - ratio)
        return float(gamma) if gamma <= gamma_max else float("inf")
    if point < 0:
        ratio = -point / abs_mean
        if ratio >= 1.0:
            return float("inf")
        gamma = (1 + ratio) / (1 - ratio)
        return float(gamma) if gamma <= gamma_max else float("inf")
    return 1.0
