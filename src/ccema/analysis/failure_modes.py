"""Failure-mode ablations + "When to trust emulation" diagnostic.

The 8-item upgraded checklist (extends the paper's original 5):

    1. Agent capability gap (strong vs weak rubric mean diff > 0.10)
    2. Positivity (ESS/n > 0.10 AND classifier AUC < 0.90)
    3. Conversation length impact (estimate variance not monotonic in length)
    4. Rubric stability (debate vs single judge ρ > 0.70)
    5. Embedding choice (CCE > raw on positivity AUC)
    6. Conformal CI width (< 4 × point estimate)             [NEW]
    7. Sensitivity bound holds at Γ = 2 (effect sign unchanged) [NEW]
    8. Process-reward dispersion (per-axis stdev > 0.05)        [NEW]

This module implements machine-readable checks that ingest already-
computed Step 4-9 outputs and produce a pass/fail diagnostic dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrustChecklistItem:
    name: str
    label: str
    value: float
    threshold: float
    op: str  # ">" or "<"
    passed: bool
    notes: str = ""


@dataclass
class TrustReport:
    items: list[TrustChecklistItem] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for i in self.items if i.passed)

    @property
    def n_total(self) -> int:
        return len(self.items)

    @property
    def all_passed(self) -> bool:
        return self.n_passed == self.n_total

    def as_dict(self) -> dict:
        return {
            "n_passed": self.n_passed,
            "n_total": self.n_total,
            "all_passed": self.all_passed,
            "items": [
                {
                    "name": i.name,
                    "label": i.label,
                    "value": float(i.value),
                    "threshold": float(i.threshold),
                    "op": i.op,
                    "passed": i.passed,
                    "notes": i.notes,
                }
                for i in self.items
            ],
        }


def _check(name: str, label: str, value: float, threshold: float, op: str, notes: str = "") -> TrustChecklistItem:
    if op == ">":
        passed = value > threshold
    elif op == "<":
        passed = value < threshold
    elif op == ">=":
        passed = value >= threshold
    elif op == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"unknown op {op}")
    return TrustChecklistItem(name=name, label=label, value=value, threshold=threshold, op=op, passed=passed, notes=notes)


def trust_report(
    capability_gap: float,
    ess_per_n: float,
    classifier_auc: float,
    rubric_correlation: float,
    cce_vs_raw_auc_drop: float,
    conformal_ci_width: float,
    point_estimate: float,
    msm_sign_robust_at_gamma2: bool,
    process_reward_stdev: float,
    *,
    capability_threshold: float = 0.10,
    ess_threshold: float = 0.10,
    auc_max: float = 0.90,
    rubric_threshold: float = 0.70,
    cce_drop_threshold: float = 0.0,
    ci_width_to_point_max: float = 4.0,
    process_dispersion_threshold: float = 0.05,
) -> TrustReport:
    items = []
    items.append(_check(
        "agent_capability", "Strong vs weak rubric gap > threshold",
        capability_gap, capability_threshold, ">"
    ))
    items.append(_check(
        "positivity_ess", "ESS / n > threshold (overlap)",
        ess_per_n, ess_threshold, ">"
    ))
    items.append(_check(
        "positivity_auc", "DRE classifier AUC < threshold",
        classifier_auc, auc_max, "<"
    ))
    items.append(_check(
        "rubric_stability", "Judge vs human Spearman ρ > threshold",
        rubric_correlation, rubric_threshold, ">"
    ))
    items.append(_check(
        "embedding_cce", "CCE reduces classifier AUC vs raw (drop > threshold)",
        cce_vs_raw_auc_drop, cce_drop_threshold, ">"
    ))
    ratio = (conformal_ci_width / abs(point_estimate)) if abs(point_estimate) > 1e-6 else float("inf")
    items.append(_check(
        "conformal_ci_width",
        "Conformal CI width / |point estimate| < threshold",
        ratio, ci_width_to_point_max, "<",
        notes="ratio = 2 * half_width / |V_hat|",
    ))
    items.append(TrustChecklistItem(
        name="msm_robust",
        label="Effect sign holds at Γ = 2",
        value=1.0 if msm_sign_robust_at_gamma2 else 0.0,
        threshold=1.0,
        op=">=",
        passed=msm_sign_robust_at_gamma2,
    ))
    items.append(_check(
        "process_reward_dispersion",
        "Per-axis stdev > threshold (avoids collapsed rubric)",
        process_reward_stdev, process_dispersion_threshold, ">"
    ))
    return TrustReport(items=items)
