"""Multi-agent debate judge (U2).

Triadic debate per criterion:
  defender   — argues the criterion IS satisfied
  prosecutor — argues the criterion is NOT satisfied
  judge      — sees both arguments + (x, a) and outputs Beta(alpha, beta)

Each role is parameterized by a LLMClient; the recommended config uses
Anthropic Claude as defender, Google Gemini as prosecutor, and OpenAI GPT
as judge to neutralize self-preference bias when a^agent is GPT-generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ccema.judges.beta_score import BetaScore, MultiAxisScore
from ccema.judges.llm_client import LLMClient
from ccema.judges.rubric import Criterion, Rubric
from ccema.judges.single import parse_beta_json


@dataclass
class DebateTraces:
    """Captured debate transcripts for a single criterion."""

    criterion_id: str
    defender_text: str
    prosecutor_text: str
    judge_text: str
    score: BetaScore


def _format_advocate_user(x: str, a: str, criterion: Criterion) -> str:
    return (
        f"Criterion: {criterion.label}\n"
        f"Definition: {criterion.definition}\n\n"
        f"x (patient pre-decision context):\n{x}\n\n"
        f"a (candidate assessment + plan):\n{a}\n\n"
        "Provide your argument as 4 short bullet points only."
    )


def _format_judge_user(
    x: str,
    a: str,
    criterion: Criterion,
    defender_text: str,
    prosecutor_text: str,
) -> str:
    return (
        f"Rubric criterion: {criterion.label}\n"
        f"Definition: {criterion.definition}\n\n"
        f"Patient pre-decision context (x):\n{x}\n\n"
        f"Candidate assessment + plan (a):\n{a}\n\n"
        f"Defender's argument (claims criterion is satisfied):\n{defender_text}\n\n"
        f"Prosecutor's argument (claims criterion is not satisfied):\n{prosecutor_text}\n\n"
        "Output VALID JSON only:\n"
        '{"rationale": "<≤5 sentences>", "alpha": <float >0>, "beta": <float >0>}'
    )


@dataclass
class DebateJudge:
    defender: LLMClient
    prosecutor: LLMClient
    judge: LLMClient
    defender_prompt: str
    prosecutor_prompt: str
    judge_prompt: str
    max_tokens_advocate: int = 256
    max_tokens_judge: int = 512
    temperature: float = 0.0
    return_traces: bool = False
    traces: list[DebateTraces] = field(default_factory=list)

    @classmethod
    def from_prompt_files(
        cls,
        defender: LLMClient,
        prosecutor: LLMClient,
        judge: LLMClient,
        defender_path: str | Path,
        prosecutor_path: str | Path,
        judge_path: str | Path,
        **kw,
    ) -> "DebateJudge":
        return cls(
            defender=defender,
            prosecutor=prosecutor,
            judge=judge,
            defender_prompt=Path(defender_path).read_text(),
            prosecutor_prompt=Path(prosecutor_path).read_text(),
            judge_prompt=Path(judge_path).read_text(),
            **kw,
        )

    def score_axis(self, x: str, a: str, criterion: Criterion) -> BetaScore:
        # Resolve {criterion} placeholder in role prompts
        d_sys = self.defender_prompt.format(criterion=criterion.label)
        p_sys = self.prosecutor_prompt.format(criterion=criterion.label)
        j_sys = self.judge_prompt.format(criterion=criterion.label)

        advocate_user = _format_advocate_user(x, a, criterion)

        d_resp = self.defender.complete(
            system=d_sys,
            user=advocate_user,
            max_tokens=self.max_tokens_advocate,
            temperature=self.temperature,
        )
        p_resp = self.prosecutor.complete(
            system=p_sys,
            user=advocate_user,
            max_tokens=self.max_tokens_advocate,
            temperature=self.temperature,
        )

        j_user = _format_judge_user(x, a, criterion, d_resp.text, p_resp.text)
        j_resp = self.judge.complete(
            system=j_sys,
            user=j_user,
            max_tokens=self.max_tokens_judge,
            temperature=self.temperature,
            json_mode=True,
        )

        score = parse_beta_json(j_resp.text)

        if self.return_traces:
            self.traces.append(
                DebateTraces(
                    criterion_id=criterion.axis_id,
                    defender_text=d_resp.text,
                    prosecutor_text=p_resp.text,
                    judge_text=j_resp.text,
                    score=score,
                )
            )
        return score

    def score_all_axes(self, x: str, a: str, rubric: Rubric) -> MultiAxisScore:
        return MultiAxisScore(
            axis_scores={c.axis_id: self.score_axis(x, a, c) for c in rubric.criteria}
        )
