"""Tests for active calibration sampling acquisition + stratified selection."""

from __future__ import annotations

import math

import pytest

from ccema.analysis.active_sampling import (
    CaseRecord,
    SamplingConfig,
    acquisition_score,
    beta_entropy,
    case_arm_disagreement,
    case_uncertainty,
    select_calibration_set,
)
from ccema.judges.beta_score import BetaScore, MultiAxisScore


def make_score(arm_means: dict[str, float], conc: float = 10.0) -> dict[str, MultiAxisScore]:
    return {
        arm: MultiAxisScore(
            axis_scores={
                "ddx_accuracy": BetaScore.from_mean_concentration(m, conc),
                "mx_safety": BetaScore.from_mean_concentration(m, conc),
                "mx_practicality": BetaScore.from_mean_concentration(m, conc),
                "mx_cost": BetaScore.from_mean_concentration(m, conc),
            }
        )
        for arm, m in arm_means.items()
    }


def make_case(case_id: str, arm_means: dict[str, float], chapter: str, conc: float = 10.0) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        x="x",
        arm_texts={arm: f"text_{arm}" for arm in arm_means},
        arm_scores=make_score(arm_means, conc=conc),
        metadata={"icd_chapter": chapter},
    )


def test_beta_entropy_is_finite_for_typical_inputs() -> None:
    h = beta_entropy(2.0, 3.0)
    assert math.isfinite(h)


def test_case_uncertainty_higher_for_lower_concentration() -> None:
    low_conc = make_case("c1", {"a": 0.5}, "J", conc=4.0)
    high_conc = make_case("c2", {"a": 0.5}, "J", conc=50.0)
    assert case_uncertainty(low_conc) > case_uncertainty(high_conc)


def test_case_disagreement_zero_when_arms_agree() -> None:
    c = make_case("c1", {"a": 0.6, "b": 0.6}, "J")
    assert case_arm_disagreement(c) == pytest.approx(0.0)


def test_case_disagreement_picks_up_arm_difference() -> None:
    c = make_case("c1", {"clin": 0.4, "agent": 0.9}, "J")
    assert case_arm_disagreement(c) == pytest.approx(0.5)


def test_acquisition_score_combines_signals() -> None:
    quiet = make_case("c1", {"a": 0.5, "b": 0.5}, "J", conc=50.0)
    loud = make_case("c2", {"a": 0.2, "b": 0.9}, "J", conc=4.0)
    assert acquisition_score(loud) > acquisition_score(quiet)


def test_select_calibration_set_respects_target_size() -> None:
    cases = [
        make_case(f"c{i}", {"clin": 0.4 + (i % 4) * 0.1, "agent": 0.6 + (i % 5) * 0.05},
                  chapter=["J", "K", "I", "M", "N", "F", "G", "R", "S"][i % 9])
        for i in range(60)
    ]
    cfg = SamplingConfig(n_total=20, min_chapters=8, seed=1)
    sel = select_calibration_set(cases, cfg)
    assert len(sel) == 20


def test_select_calibration_set_meets_min_chapters() -> None:
    cases = [
        make_case(f"c{i}", {"clin": 0.5, "agent": 0.5},
                  chapter=["J", "K", "I", "M", "N", "F", "G", "R", "S", "Z"][i % 10])
        for i in range(50)
    ]
    cfg = SamplingConfig(n_total=15, min_chapters=8, seed=1)
    sel = select_calibration_set(cases, cfg)
    chapters = {c.metadata["icd_chapter"] for c in sel}
    assert len(chapters) >= 8


def test_select_calibration_set_prefers_high_acquisition() -> None:
    # 10 cases: half with big disagreement+uncertainty, half quiet
    loud = [
        make_case(f"loud_{i}", {"clin": 0.2, "agent": 0.9}, "J", conc=3.5)
        for i in range(10)
    ]
    quiet = [
        make_case(f"quiet_{i}", {"clin": 0.55, "agent": 0.55}, "J", conc=80.0)
        for i in range(10)
    ]
    cfg = SamplingConfig(n_total=5, min_chapters=1, seed=42)
    sel = select_calibration_set(loud + quiet, cfg)
    n_loud = sum(1 for c in sel if c.case_id.startswith("loud_"))
    assert n_loud == 5  # all loud cases preferred


def test_select_calibration_set_deterministic_under_seed() -> None:
    cases = [
        make_case(f"c{i}", {"clin": (i * 0.1) % 1.0 or 0.1, "agent": ((i + 3) * 0.13) % 1.0 or 0.2},
                  chapter=["J", "K", "I", "M", "N", "F", "G", "R", "S"][i % 9])
        for i in range(40)
    ]
    cfg = SamplingConfig(n_total=10, min_chapters=8, seed=99)
    sel_a = select_calibration_set(cases, cfg)
    sel_b = select_calibration_set(cases, cfg)
    assert [c.case_id for c in sel_a] == [c.case_id for c in sel_b]
