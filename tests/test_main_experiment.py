"""Tests for Step 9 main experiment orchestration."""

from __future__ import annotations

import numpy as np

from ccema.analysis.ground_truth import ScoreRow
from ccema.analysis.main_experiment import (
    EstimatorResult,
    MainResults,
    build_estimator_inputs,
    run_main_experiment,
)


def _rows_with_arm_means(arm_means: dict[str, float], n_cases: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cases):
        for arm, mu in arm_means.items():
            for s in range(3):
                for axis in ("a1", "a2"):
                    rows.append(
                        ScoreRow(
                            case_id=f"c{i}",
                            arm=arm,
                            sample_idx=s,
                            axis=axis,
                            score=float(np.clip(rng.normal(mu, 0.05), 0, 1)),
                            metadata={"icd_chapter": "J"},
                        )
                    )
    return rows


def test_build_estimator_inputs_yields_aligned_arrays() -> None:
    rows = _rows_with_arm_means({"clinician": 0.6, "strong": 0.8}, n_cases=20)
    inp = build_estimator_inputs(rows, target_arm="strong", baseline_arm="clinician")
    assert len(inp["case_ids"]) == 20
    assert inp["y_obs"].shape == (20,)
    assert inp["y_target"].shape == (20,)
    assert abs(inp["v_true_baseline"] - 0.6) < 0.05
    assert abs(inp["v_true_target"] - 0.8) < 0.05


def test_run_main_experiment_returns_six_estimators() -> None:
    rows = _rows_with_arm_means({"clinician": 0.55, "strong": 0.75}, n_cases=25)
    results = run_main_experiment(rows, target_arm="strong", baseline_arm="clinician", n_bootstrap=200)
    assert isinstance(results, MainResults)
    assert results.target_arm == "strong"
    assert results.baseline_arm == "clinician"
    assert set(results.estimators.keys()) == {"DM", "IPS", "DR", "MIPS", "DML-DR", "TMLE"}
    for er in results.estimators.values():
        assert isinstance(er, EstimatorResult)
        assert np.isfinite(er.v_hat)


def test_dm_in_rubric_mode_recovers_v_true() -> None:
    rows = _rows_with_arm_means({"clinician": 0.4, "strong": 0.85}, n_cases=20)
    results = run_main_experiment(rows, target_arm="strong", baseline_arm="clinician", n_bootstrap=100)
    # In the rubric-only toy mode, DM is set to mean(y_target) = V_true
    dm = results.estimators["DM"]
    assert abs(dm.v_hat - results.v_true) < 1e-9
    assert abs(dm.bias) < 1e-9


def test_main_experiment_bootstrap_and_conformal_cis_set() -> None:
    rows = _rows_with_arm_means({"clinician": 0.5, "strong": 0.75}, n_cases=30)
    results = run_main_experiment(rows, target_arm="strong", n_bootstrap=200)
    for name, er in results.estimators.items():
        assert er.ci_lower_bootstrap <= er.ci_upper_bootstrap
        assert er.ci_lower_conformal <= er.ci_upper_conformal


def test_run_main_experiment_skips_arms_without_baseline() -> None:
    # Only target_arm, no clinician — should yield zero rows
    rows = []
    rng = np.random.default_rng(0)
    for i in range(10):
        for axis in ("a1",):
            rows.append(
                ScoreRow(
                    case_id=f"c{i}",
                    arm="strong",
                    sample_idx=0,
                    axis=axis,
                    score=float(rng.normal(0.7, 0.05)),
                    metadata={},
                )
            )
    results = run_main_experiment(rows, target_arm="strong", baseline_arm="clinician", n_bootstrap=50)
    assert results.n_cases == 0
