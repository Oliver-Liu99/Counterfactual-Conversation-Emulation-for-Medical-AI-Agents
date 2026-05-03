"""High-level runner: score a list of (x, a) pairs across a rubric.

Used by:
- Step 3 driver (single-judge scoring of full D for active sampling)
- Step 4 driver (debate-judge scoring of calibration set)
- Step 7 driver (debate-judge scoring of full D for ground truth)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ccema.judges.beta_score import MultiAxisScore
from ccema.judges.rubric import Rubric


class _AxisScorer(Protocol):
    def score_all_axes(self, x: str, a: str, rubric: Rubric) -> MultiAxisScore: ...


@dataclass
class ScoringResult:
    case_id: str
    arm: str  # e.g. "clinician", "agent_strong", "agent_weak"
    sample_idx: int
    score: MultiAxisScore


def score_pairs(
    judge: _AxisScorer,
    pairs: Iterable[tuple[str, str, str, str, int]],  # (case_id, arm, x, a, sample_idx)
    rubric: Rubric,
) -> list[ScoringResult]:
    """Score every (x, a) pair under every rubric axis.

    Linear in pairs * len(rubric.criteria); the caller is responsible for
    batching, retry, and persistence.
    """
    out: list[ScoringResult] = []
    for case_id, arm, x, a, sample_idx in pairs:
        out.append(
            ScoringResult(
                case_id=case_id,
                arm=arm,
                sample_idx=sample_idx,
                score=judge.score_all_axes(x, a, rubric),
            )
        )
    return out


def collapse_to_means(results: Sequence[ScoringResult]) -> dict[tuple[str, str, int], dict[str, float]]:
    """Collapse Beta posteriors to per-axis means keyed by (case_id, arm, sample_idx)."""
    return {(r.case_id, r.arm, r.sample_idx): r.score.means for r in results}
