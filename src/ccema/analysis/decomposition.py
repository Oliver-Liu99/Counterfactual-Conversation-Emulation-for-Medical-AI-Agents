"""Path-specific effect decomposition (U4).

For each rubric axis k and each subpopulation s defined by metadata
(icd_chapter, age_band, comorbidity_count), compute the conditional
effect contrast between an arm and the clinician baseline.

This is *not* full mediation analysis (no path coefficients) — it is
the simpler axis-and-subpopulation decomposition that is most useful
for the paper's main figure (heatmap of arm × subpop × axis).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ccema.analysis.ground_truth import ScoreRow


@dataclass
class CellResult:
    arm: str
    baseline: str
    axis: str
    subpop_key: str
    subpop_value: str
    n_cases: int
    delta: float
    ci_lower: float
    ci_upper: float


def decompose(
    rows: Sequence[ScoreRow],
    baseline_arm: str = "clinician",
    subpop_keys: Sequence[str] = ("icd_chapter",),
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[CellResult]:
    """Decompose per (arm, axis, subpop_key, subpop_value)."""
    rng = np.random.default_rng(seed)

    # Group by (case, arm, axis) -> mean across samples
    by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    case_subpops: dict[str, dict[str, str]] = {}
    for r in rows:
        by_key[(r.case_id, r.arm, r.axis)].append(r.score)
        if r.case_id not in case_subpops:
            case_subpops[r.case_id] = {k: r.metadata.get(k, "_unk") for k in subpop_keys}

    case_arm_axis = {k: float(np.mean(v)) for k, v in by_key.items()}

    # Identify arms and axes
    arms = sorted({a for (_, a, _) in case_arm_axis} - {baseline_arm})
    axes = sorted({ax for (_, _, ax) in case_arm_axis})

    results: list[CellResult] = []
    for arm in arms:
        for axis in axes:
            # paired diffs grouped by subpop key/value
            for skey in subpop_keys:
                vals_by_sval: dict[str, list[float]] = defaultdict(list)
                for cid, subpops in case_subpops.items():
                    sval = subpops[skey]
                    a_key = (cid, arm, axis)
                    b_key = (cid, baseline_arm, axis)
                    if a_key in case_arm_axis and b_key in case_arm_axis:
                        vals_by_sval[sval].append(case_arm_axis[a_key] - case_arm_axis[b_key])

                for sval, diffs in vals_by_sval.items():
                    if len(diffs) < 2:
                        continue
                    arr = np.array(diffs)
                    delta = float(arr.mean())
                    n = len(arr)
                    boots = np.empty(n_bootstrap)
                    for k in range(n_bootstrap):
                        idx = rng.integers(0, n, size=n)
                        boots[k] = arr[idx].mean()
                    lo = float(np.quantile(boots, alpha / 2))
                    hi = float(np.quantile(boots, 1 - alpha / 2))
                    results.append(
                        CellResult(
                            arm=arm,
                            baseline=baseline_arm,
                            axis=axis,
                            subpop_key=skey,
                            subpop_value=sval,
                            n_cases=n,
                            delta=delta,
                            ci_lower=lo,
                            ci_upper=hi,
                        )
                    )
    return results
