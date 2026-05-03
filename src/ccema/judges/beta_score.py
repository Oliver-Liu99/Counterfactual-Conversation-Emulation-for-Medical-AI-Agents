"""Beta-posterior helpers for judge outputs.

A judge per-criterion output is `Beta(alpha, beta)` over [0, 1]. We use
the posterior mean as the point estimate and `alpha + beta` as the
concentration. Helpers convert between (alpha, beta), (mean, concentration),
and aggregated multi-axis vectors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BetaScore:
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(f"Beta requires alpha,beta > 0; got {self.alpha}, {self.beta}")

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def concentration(self) -> float:
        return self.alpha + self.beta

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @classmethod
    def from_mean_concentration(cls, mean: float, concentration: float) -> "BetaScore":
        if not (0 < mean < 1):
            raise ValueError(f"mean must be in (0, 1); got {mean}")
        if concentration <= 2:
            raise ValueError(f"concentration must be > 2; got {concentration}")
        a = mean * concentration
        b = (1 - mean) * concentration
        return cls(alpha=a, beta=b)


@dataclass
class MultiAxisScore:
    """Score vector across rubric axes for one (case, agent_arm) pair."""

    axis_scores: dict[str, BetaScore]

    @property
    def axes(self) -> list[str]:
        return list(self.axis_scores.keys())

    @property
    def means(self) -> dict[str, float]:
        return {k: v.mean for k, v in self.axis_scores.items()}

    @property
    def overall_mean(self) -> float:
        if not self.axis_scores:
            return float("nan")
        return sum(v.mean for v in self.axis_scores.values()) / len(self.axis_scores)

    def disagreement(self) -> float:
        """Range across axis means (max - min). Used for Step 4 ablation."""
        means = [v.mean for v in self.axis_scores.values()]
        if not means:
            return 0.0
        return max(means) - min(means)
