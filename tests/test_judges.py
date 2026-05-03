"""Tests for judge components: BetaScore, prompts, calibration metrics, debate."""

from __future__ import annotations

import json

import pytest

from ccema.judges.beta_score import BetaScore, MultiAxisScore
from ccema.judges.calibration import (
    CalibrationGates,
    bland_altman,
    calibration_report,
    discretize,
    icc_2_1,
    pearson_r,
    quadratic_weighted_kappa,
    spearman_rho,
)
from ccema.judges.debate import DebateJudge
from ccema.judges.llm_client import MockLLMClient
from ccema.judges.rubric import DEFAULT_RUBRIC
from ccema.judges.single import SingleJudge, parse_beta_json


# ---------------------------------------------------------------------------
# BetaScore
# ---------------------------------------------------------------------------


def test_beta_mean_and_concentration() -> None:
    s = BetaScore(alpha=7.0, beta=3.0)
    assert s.mean == pytest.approx(0.7)
    assert s.concentration == pytest.approx(10.0)
    assert s.variance > 0


def test_beta_invalid_args() -> None:
    with pytest.raises(ValueError):
        BetaScore(alpha=0, beta=1)
    with pytest.raises(ValueError):
        BetaScore(alpha=1, beta=-0.1)


def test_beta_from_mean_concentration_roundtrip() -> None:
    s = BetaScore.from_mean_concentration(mean=0.65, concentration=20)
    assert s.mean == pytest.approx(0.65)
    assert s.concentration == pytest.approx(20)


def test_multiaxis_aggregation() -> None:
    m = MultiAxisScore(
        axis_scores={
            "ddx": BetaScore(8, 2),
            "safety": BetaScore(9, 1),
            "practicality": BetaScore(5, 5),
        }
    )
    assert m.means["ddx"] == pytest.approx(0.8)
    assert m.overall_mean == pytest.approx((0.8 + 0.9 + 0.5) / 3)
    assert m.disagreement() == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Robust JSON parsing
# ---------------------------------------------------------------------------


def test_parse_beta_json_clean() -> None:
    s = parse_beta_json('{"alpha": 6, "beta": 4, "rationale": "ok"}')
    assert s.mean == pytest.approx(0.6)


def test_parse_beta_json_with_markdown_fence() -> None:
    raw = '```json\n{"alpha": 6, "beta": 4}\n```'
    s = parse_beta_json(raw)
    assert s.mean == pytest.approx(0.6)


def test_parse_beta_json_with_prose_around() -> None:
    raw = 'Sure, here you go: {"alpha": 9, "beta": 1, "rationale": "great"} done.'
    s = parse_beta_json(raw)
    assert s.alpha == 9
    assert s.beta == 1


# ---------------------------------------------------------------------------
# Calibration metrics — sanity checks on small known sequences
# ---------------------------------------------------------------------------


def test_spearman_perfect_positive() -> None:
    a = [0.1, 0.3, 0.5, 0.7, 0.9]
    b = [0.0, 0.2, 0.4, 0.6, 0.8]
    assert spearman_rho(a, b) == pytest.approx(1.0)


def test_spearman_perfect_negative() -> None:
    a = [1, 2, 3, 4, 5]
    b = [5, 4, 3, 2, 1]
    assert spearman_rho(a, b) == pytest.approx(-1.0)


def test_pearson_matches_spearman_when_linear() -> None:
    a = [0.1, 0.2, 0.3, 0.4, 0.5]
    b = [0.2, 0.4, 0.6, 0.8, 1.0]
    assert pearson_r(a, b) == pytest.approx(1.0)
    assert spearman_rho(a, b) == pytest.approx(1.0)


def test_discretize_bounds() -> None:
    assert discretize([0.0, 0.5, 0.99, 1.0], n_bins=5) == [0, 2, 4, 4]


def test_quadratic_kappa_perfect() -> None:
    xs = [0.05, 0.25, 0.45, 0.65, 0.85]
    assert quadratic_weighted_kappa(xs, xs, n_bins=5) == pytest.approx(1.0)


def test_quadratic_kappa_random() -> None:
    a = [0.1] * 5 + [0.9] * 5
    b = [0.9] * 5 + [0.1] * 5  # perfectly inverted in 5 bins
    k = quadratic_weighted_kappa(a, b, n_bins=5)
    assert k < 0  # negative kappa for systematic disagreement


def test_icc_2_1_high_agreement() -> None:
    judge = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.6]
    human = [0.12, 0.32, 0.49, 0.71, 0.88, 0.21, 0.58]
    icc = icc_2_1(judge, human)
    assert icc > 0.9


def test_icc_2_1_low_agreement() -> None:
    judge = [0.1, 0.3, 0.5, 0.7, 0.9]
    human = [0.9, 0.7, 0.5, 0.3, 0.1]  # inverted
    icc = icc_2_1(judge, human)
    assert icc < 0


def test_bland_altman_zero_bias_when_identical() -> None:
    xs = [0.1, 0.2, 0.3]
    ba = bland_altman(xs, xs)
    assert ba.bias == 0
    assert ba.sd_diff == 0


def test_bland_altman_signed_bias() -> None:
    judge = [0.5, 0.5, 0.5]
    human = [0.4, 0.4, 0.4]
    ba = bland_altman(judge, human)
    assert ba.bias == pytest.approx(0.10)


def test_calibration_report_passes_high_agreement() -> None:
    # construct judge ≈ human with small noise
    human = [i / 99 for i in range(0, 100, 5)]
    judge = [h + 0.01 for h in human]
    rep = calibration_report(judge, human)
    assert rep.spearman_rho > 0.99
    assert rep.pass_spearman
    assert abs(rep.bland_altman_bias) < 0.05
    assert rep.pass_bias


def test_calibration_report_fails_when_biased() -> None:
    human = [i / 99 for i in range(0, 100, 5)]
    judge = [h + 0.20 for h in human]  # large positive bias
    rep = calibration_report(judge, human)
    assert rep.pass_spearman  # ranks unchanged
    assert not rep.pass_bias  # bias too large


# ---------------------------------------------------------------------------
# Single + Debate judges with mock LLM
# ---------------------------------------------------------------------------


def test_single_judge_mock_returns_expected_score() -> None:
    client = MockLLMClient(return_alpha_beta=(8.0, 2.0))
    judge = SingleJudge(client=client)
    score = judge.score_axis("x", "a", DEFAULT_RUBRIC.criteria[0])
    assert score.mean == pytest.approx(0.8)


def test_single_judge_scores_all_axes() -> None:
    client = MockLLMClient(return_alpha_beta=(7.0, 3.0))
    judge = SingleJudge(client=client)
    ms = judge.score_all_axes("x", "a", DEFAULT_RUBRIC)
    assert set(ms.axes) == set(DEFAULT_RUBRIC.axes)
    assert all(round(s.mean, 3) == 0.7 for s in ms.axis_scores.values())


def test_debate_judge_three_role_call_paths() -> None:
    """Verify defender/prosecutor/judge are each called once per axis."""

    class CountingMock(MockLLMClient):
        def __init__(self, name: str, ab: tuple[float, float] | None = None):
            super().__init__(name=name, return_alpha_beta=ab)
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            return super().complete(*args, **kwargs)

    defender = CountingMock("defender")
    prosecutor = CountingMock("prosecutor")
    judge_client = CountingMock("judge", ab=(6.0, 4.0))

    debate = DebateJudge(
        defender=defender,
        prosecutor=prosecutor,
        judge=judge_client,
        defender_prompt="Defender for {criterion}",
        prosecutor_prompt="Prosecutor for {criterion}",
        judge_prompt="Judge for {criterion}",
    )
    ms = debate.score_all_axes("x", "a", DEFAULT_RUBRIC)
    n_axes = len(DEFAULT_RUBRIC.criteria)
    assert defender.calls == n_axes
    assert prosecutor.calls == n_axes
    assert judge_client.calls == n_axes
    assert all(round(s.mean, 3) == 0.6 for s in ms.axis_scores.values())


def test_debate_judge_traces() -> None:
    debate = DebateJudge(
        defender=MockLLMClient("d"),
        prosecutor=MockLLMClient("p"),
        judge=MockLLMClient("j", return_alpha_beta=(5.0, 5.0)),
        defender_prompt="d {criterion}",
        prosecutor_prompt="p {criterion}",
        judge_prompt="j {criterion}",
        return_traces=True,
    )
    debate.score_all_axes("x", "a", DEFAULT_RUBRIC)
    assert len(debate.traces) == len(DEFAULT_RUBRIC.criteria)
    assert all(t.score.mean == pytest.approx(0.5) for t in debate.traces)
