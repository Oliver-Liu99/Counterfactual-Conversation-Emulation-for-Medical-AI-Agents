"""π_agent: an evaluated medical AI agent.

Wraps an LLMClient + system prompt + sampling config; produces n_samples
A&P outputs per encounter context x.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ccema.agents.schemas import (
    AgentOutput,
    SchemaError,
    looks_like_transcript_reference,
    validate_pi_agent_output,
)
from ccema.judges.llm_client import LLMClient


@dataclass
class AgentSample:
    """One sample from π_agent for one (case, sample_idx)."""

    case_id: str
    arm: str  # e.g. "strong", "weak", "open_medical"
    sample_idx: int
    output: AgentOutput | None  # None if validation failed
    error: str | None = None


@dataclass
class PiAgent:
    arm: str  # "strong" | "weak" | "open_medical" | "scaffolded"
    client: LLMClient
    system_prompt: str
    temperature: float = 0.2
    max_tokens: int = 2048
    n_samples: int = 5
    leakage_check: bool = True
    schema_strict: bool = True

    @classmethod
    def from_prompt_file(cls, arm: str, client: LLMClient, prompt_path: str | Path, **kw) -> "PiAgent":
        return cls(arm=arm, client=client, system_prompt=Path(prompt_path).read_text(), **kw)

    def sample_one(self, case_id: str, x: str, sample_idx: int) -> AgentSample:
        try:
            resp = self.client.complete(
                system=self.system_prompt,
                user=f"Patient pre-decision context (x):\n\n{x}",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                json_mode=True,
            )
            out = validate_pi_agent_output(resp.text, strict=self.schema_strict)
            if self.leakage_check and looks_like_transcript_reference(out):
                return AgentSample(
                    case_id=case_id,
                    arm=self.arm,
                    sample_idx=sample_idx,
                    output=None,
                    error="transcript_reference",
                )
            return AgentSample(case_id=case_id, arm=self.arm, sample_idx=sample_idx, output=out)
        except SchemaError as e:
            return AgentSample(
                case_id=case_id, arm=self.arm, sample_idx=sample_idx, output=None, error=f"schema:{e}"
            )

    def sample_n(self, case_id: str, x: str) -> list[AgentSample]:
        return [self.sample_one(case_id, x, i) for i in range(self.n_samples)]


# ---------------------------------------------------------------------------
# Capability-gap test (Step 5 verification)
# ---------------------------------------------------------------------------


@dataclass
class CapabilityGapResult:
    n_cases: int
    strong_mean: float
    weak_mean: float
    gap: float
    threshold: float
    passes: bool
    per_case: list[dict] = field(default_factory=list)


def capability_gap(
    strong_scores: list[float],
    weak_scores: list[float],
    threshold: float = 0.10,
) -> CapabilityGapResult:
    """Compute mean(strong) - mean(weak) and check it exceeds threshold."""
    if len(strong_scores) != len(weak_scores):
        raise ValueError("strong_scores and weak_scores must align case-by-case")
    n = len(strong_scores)
    if n == 0:
        return CapabilityGapResult(0, 0.0, 0.0, 0.0, threshold, False)
    s_mean = sum(strong_scores) / n
    w_mean = sum(weak_scores) / n
    gap = s_mean - w_mean
    return CapabilityGapResult(
        n_cases=n,
        strong_mean=s_mean,
        weak_mean=w_mean,
        gap=gap,
        threshold=threshold,
        passes=(gap > threshold),
        per_case=[
            {"strong": s, "weak": w, "diff": s - w}
            for s, w in zip(strong_scores, weak_scores)
        ],
    )
