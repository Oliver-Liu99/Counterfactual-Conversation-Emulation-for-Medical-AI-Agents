"""Step 10 driver: ablation matrix + 'When to trust' diagnostic.

Produces:
    outputs/step10/ablations.json   — raw ablation values
    outputs/step10/trust_report.json — 8-item checklist with pass/fail
    docs/when_to_trust.md            — human-readable upgraded checklist

Mock mode plugs in plausible synthetic values so the report renders
without API calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ccema.analysis.failure_modes import trust_report
from ccema.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval_config_v1.yaml")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--output-dir", default="outputs/step10")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO_ROOT / args.config)

    if not args.mock:
        raise NotImplementedError(
            "Real-data Step 10 wires real Step 4-9 outputs into the trust "
            "diagnostic; re-run with --mock for now."
        )

    # Ablation values — mock plausible numbers based on the project's
    # success milestones; real-data run replaces these from Step 4-9 outputs
    ablation = {
        "judge_modes": {
            "single": {"spearman_rho": 0.68, "kappa": 0.39, "icc": 0.71, "bias": 0.06},
            "ensemble_3": {"spearman_rho": 0.74, "kappa": 0.44, "icc": 0.78, "bias": 0.04},
            "debate": {"spearman_rho": 0.79, "kappa": 0.51, "icc": 0.82, "bias": 0.03},
        },
        "embedding": {
            "raw_medcpt": {"classifier_auc": 0.92, "ess_per_n": 0.07},
            "cce_medcpt": {"classifier_auc": 0.83, "ess_per_n": 0.18},
            "raw_bge_m3":  {"classifier_auc": 0.94, "ess_per_n": 0.06},
            "cce_bge_m3":  {"classifier_auc": 0.86, "ess_per_n": 0.14},
        },
        "estimators_rmse": {
            "DM": 0.067, "IPS": 0.068, "DR": 0.056,
            "MIPS": 0.054, "DML-DR": 0.053, "TMLE": 0.051,
        },
        "agent_arms": {
            "strong_mean": 0.78, "weak_mean": 0.50, "open_medical_mean": 0.71,
            "capability_gap": 0.28,
        },
        "subpopulation_heterogeneity": {
            "max_axis_delta": 0.21,  # mx_practicality
            "min_axis_delta": -0.04,  # mx_cost
            "axes_with_heterogeneous_signs": ["mx_cost"],
        },
    }

    # Run trust checklist with summary values
    rep = trust_report(
        capability_gap=ablation["agent_arms"]["capability_gap"],
        ess_per_n=ablation["embedding"]["cce_medcpt"]["ess_per_n"],
        classifier_auc=ablation["embedding"]["cce_medcpt"]["classifier_auc"],
        rubric_correlation=ablation["judge_modes"]["debate"]["spearman_rho"],
        cce_vs_raw_auc_drop=(
            ablation["embedding"]["raw_medcpt"]["classifier_auc"]
            - ablation["embedding"]["cce_medcpt"]["classifier_auc"]
        ),
        conformal_ci_width=0.05,
        point_estimate=0.13,
        msm_sign_robust_at_gamma2=True,
        process_reward_stdev=0.09,
    )

    print("\nTrust diagnostic (8 items):")
    for item in rep.items:
        flag = "PASS" if item.passed else "FAIL"
        print(f"  [{flag}] {item.label}: {item.value:.4f} {item.op} {item.threshold:.4f}")
    print(f"\n  {rep.n_passed}/{rep.n_total} items passed")
    print(f"  All passed: {rep.all_passed}")

    (out_dir / "ablations.json").write_text(json.dumps(ablation, indent=2))
    (out_dir / "trust_report.json").write_text(json.dumps(rep.as_dict(), indent=2))
    print(f"\nWrote {(out_dir / 'ablations.json').relative_to(REPO_ROOT)}")
    print(f"Wrote {(out_dir / 'trust_report.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
