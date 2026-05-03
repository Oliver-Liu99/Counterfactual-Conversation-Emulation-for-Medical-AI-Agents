"""Rubric: 4-axis process reward (U2).

Each axis maps to one criterion definition + a one-line description used in
the debate prompt template substitution. The 4 axes mirror the BIDMC rubric
(Brodeur 2026): DDx accuracy / Mx safety / Mx practicality / Mx cost.

Subdividing into more granular criteria is straightforward — extend
`Rubric.criteria` with additional `Criterion` records.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Criterion:
    axis_id: str  # e.g. "ddx_accuracy"
    label: str  # human-readable
    definition: str  # plugged into {criterion_definition} in prompts
    weight: float = 1.0


@dataclass(frozen=True)
class Rubric:
    name: str
    version: str
    criteria: tuple[Criterion, ...]

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(c.axis_id for c in self.criteria)

    def get(self, axis_id: str) -> Criterion:
        for c in self.criteria:
            if c.axis_id == axis_id:
                return c
        raise KeyError(axis_id)


DEFAULT_RUBRIC = Rubric(
    name="ccema_v1",
    version="2026-05-03",
    criteria=(
        Criterion(
            axis_id="ddx_accuracy",
            label="Differential Diagnosis Accuracy",
            definition=(
                "The differential includes the most likely diagnosis given x, "
                "the ranking is clinically defensible, and at least one "
                "important rule-out is acknowledged."
            ),
        ),
        Criterion(
            axis_id="mx_safety",
            label="Management Plan Safety",
            definition=(
                "The plan avoids medications/procedures contraindicated by the "
                "patient's allergies, comorbidities, or pregnancy status, and "
                "addresses red-flag features in x."
            ),
        ),
        Criterion(
            axis_id="mx_practicality",
            label="Management Plan Practicality",
            definition=(
                "The plan is executable in a typical primary-care setting "
                "without exotic referrals, esoteric labs, or unrealistic "
                "follow-up cadence."
            ),
        ),
        Criterion(
            axis_id="mx_cost",
            label="Management Plan Cost-Awareness",
            definition=(
                "The plan prefers generics, defers low-yield imaging when "
                "history/exam are sufficient, and avoids unnecessary "
                "duplicative testing."
            ),
        ),
    ),
)
