"""Step 7 driver: Ground truth + effect decomposition + sensitivity bounds.

Pipeline:
    1. Load agent samples (Step 5) and judge scores (Step 4) — or
       reproduce them in mock mode.
    2. Compute V_true(arm) per arm with bootstrap CIs.
    3. Run path-specific decomposition by ICD chapter × axis.
    4. Run marginal sensitivity model bounds at Γ ∈ {1, 1.5, 2, 3}.
    5. Emit a JSON summary; print the headline pass/fail vs Week-5
       milestone (effect significant under Γ = 1).

Usage:
    python scripts/07_ground_truth.py --mock --n 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ccema.analysis.decomposition import decompose
from ccema.analysis.ground_truth import (
    ScoreRow,
    aggregate_to_per_case,
    effect_vs_baseline,
    per_case_mean,
    v_true_per_arm,
)
from ccema.analysis.sensitivity import fragility_gamma, msm_bounds_per_arm
from ccema.data.loaders import load_dataset
from ccema.utils.config import load_config
from ccema.utils.seeding import set_global_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval_config_v1.yaml")
    p.add_argument("--mock", action="store_true", help="generate synthetic mock scores")
    p.add_argument("--n", type=int, default=30, help="encounters in mock mode")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--gammas", default="1.0,1.5,2.0,3.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/step7_ground_truth")
    return p.parse_args()


def make_mock_score_rows(n_cases: int, n_samples: int, seed: int) -> list[ScoreRow]:
    """Generate mock long-format ScoreRows with a real strong > clinician > weak gap.

    Used for smoke-testing the ground-truth pipeline without API calls.
    """
    rng = np.random.default_rng(seed)
    encs = load_dataset("synthetic", n_encounters=n_cases, seed=seed)
    arm_means = {"clinician": 0.65, "agent_strong": 0.78, "agent_weak": 0.50}
    arm_sd = 0.06
    axes = ("ddx_accuracy", "mx_safety", "mx_practicality", "mx_cost")

    rows: list[ScoreRow] = []
    for enc in encs:
        for arm, mu in arm_means.items():
            for s in range(n_samples):
                for axis in axes:
                    # Per-arm shift varies slightly by axis (so decomposition has signal)
                    axis_shift = {"ddx_accuracy": 0.02, "mx_safety": 0.0,
                                  "mx_practicality": -0.03, "mx_cost": -0.01}[axis]
                    score = float(np.clip(rng.normal(mu + axis_shift, arm_sd), 0.0, 1.0))
                    rows.append(
                        ScoreRow(
                            case_id=enc.case_id,
                            arm=arm,
                            sample_idx=s,
                            axis=axis,
                            score=score,
                            metadata={
                                "icd_chapter": enc.metadata["icd_chapter"],
                                "age_band": enc.metadata["age_band"],
                            },
                        )
                    )
    return rows


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    cfg = load_config(REPO_ROOT / args.config)
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gammas = [float(g) for g in args.gammas.split(",") if g.strip()]

    if args.mock:
        rows = make_mock_score_rows(args.n, args.n_samples, args.seed)
        print(f"Mock mode: {len(rows)} score rows over {args.n} cases × 3 arms × "
              f"{args.n_samples} samples × 4 axes")
    else:
        raise NotImplementedError(
            "Real scoring pipeline (load Step 5 outputs + run Step 4 debate judge) "
            "not implemented yet — re-run with --mock or wire to outputs/step5_*"
        )

    # 1. V_true per arm
    arm_values = v_true_per_arm(rows, n_bootstrap=cfg["estimators"]["bootstrap"]["n_samples"], seed=args.seed)
    print("\nV_true per arm:")
    for arm, av in arm_values.items():
        print(f"  {arm}: V={av.v_true:.4f} 95% CI=[{av.ci_lower:.4f}, {av.ci_upper:.4f}] (n={av.n_cases})")

    # 2. Effect contrasts vs clinician
    contrasts = effect_vs_baseline(rows, baseline_arm="clinician", seed=args.seed)
    print("\nDelta vs clinician (95% CI):")
    for arm, ec in contrasts.items():
        sig = " *significant" if ec.significant else ""
        print(f"  {arm}: Δ={ec.delta:+.4f} CI=[{ec.ci_lower:+.4f}, {ec.ci_upper:+.4f}]{sig}")

    # 3. Decomposition
    cells = decompose(rows, baseline_arm="clinician", subpop_keys=("icd_chapter",), seed=args.seed)
    print(f"\nDecomposition cells: {len(cells)} (arm × axis × ICD chapter)")

    # 4. Sensitivity bounds (paired diffs in overall_mean per arm vs clinician)
    per_case = aggregate_to_per_case(rows)
    case_means = per_case_mean(per_case)
    paired = {}
    for arm in {a for (_, a) in case_means} - {"clinician"}:
        common = sorted(c for (c, a) in case_means if a == arm and (c, "clinician") in case_means)
        diffs = np.array([case_means[(c, arm)] - case_means[(c, "clinician")] for c in common])
        paired[arm] = diffs
    bounds = msm_bounds_per_arm(paired, gammas=gammas)

    print(f"\nMarginal Sensitivity Model bounds at Γ ∈ {gammas}:")
    for sb in bounds:
        flag = "robust" if sb.sign_robust else "FLIPS"
        print(f"  {sb.arm} @ Γ={sb.gamma:.1f}: Δ∈[{sb.delta_lo:+.4f}, {sb.delta_hi:+.4f}] {flag}")

    fragility = {arm: fragility_gamma(diffs) for arm, diffs in paired.items()}
    print("\nFragility Γ (sign flips at):")
    for arm, g in fragility.items():
        gs = f"{g:.2f}" if g != float("inf") else "∞"
        print(f"  {arm}: {gs}")

    # Persist
    summary = {
        "v_true": {arm: av.__dict__ for arm, av in arm_values.items()},
        "contrasts": {arm: ec.__dict__ for arm, ec in contrasts.items()},
        "decomposition": [c.__dict__ for c in cells],
        "sensitivity_bounds": [sb.__dict__ for sb in bounds],
        "fragility_gamma": fragility,
        "milestone_week5": {
            arm: contrasts[arm].significant
            for arm in contrasts
        },
    }
    out = out_dir / "ground_truth.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
