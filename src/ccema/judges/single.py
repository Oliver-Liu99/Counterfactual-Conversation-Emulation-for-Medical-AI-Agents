"""Single-LLM judge (paper baseline + Step 3 active-sampling driver).

Asks one judge model to score (x, a) on a single rubric criterion. Output
is parsed as JSON with {alpha, beta, rationale} and wrapped in BetaScore.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ccema.judges.beta_score import BetaScore, MultiAxisScore
from ccema.judges.llm_client import LLMClient
from ccema.judges.rubric import Criterion, Rubric


SINGLE_JUDGE_SYSTEM = (
    "You are an impartial expert clinician serving as a judge on whether a "
    "proposed assessment+plan satisfies a rubric criterion. Output a Beta "
    "posterior over [0,1] reflecting your belief the criterion is satisfied."
)


def _format_user(x: str, a: str, criterion: Criterion) -> str:
    return (
        f"Rubric criterion: {criterion.label}\n"
        f"Definition: {criterion.definition}\n\n"
        f"Patient pre-decision context (x):\n{x}\n\n"
        f"Candidate assessment + plan (a):\n{a}\n\n"
        "Output VALID JSON only, no markdown:\n"
        '{"rationale": "<≤5 sentences>", "alpha": <float >0>, "beta": <float >0>}'
    )


def parse_beta_json(text: str) -> BetaScore:
    """Robustly parse JSON from possibly-noisy LLM output."""
    # Strip markdown fences
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"```\s*$", "", s)
    # First-pass: try whole string
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        # Find the first {...} block
        match = re.search(r"\{.*?\}", s, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"No JSON found in: {text!r}") from None
        obj = json.loads(match.group(0))

    a = float(obj["alpha"])
    b = float(obj["beta"])
    return BetaScore(alpha=a, beta=b)


@dataclass
class SingleJudge:
    client: LLMClient
    system_prompt: str = SINGLE_JUDGE_SYSTEM
    max_tokens: int = 512
    temperature: float = 0.0

    def score_axis(self, x: str, a: str, criterion: Criterion) -> BetaScore:
        resp = self.client.complete(
            system=self.system_prompt,
            user=_format_user(x, a, criterion),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            json_mode=True,
        )
        return parse_beta_json(resp.text)

    def score_all_axes(self, x: str, a: str, rubric: Rubric) -> MultiAxisScore:
        scores: dict[str, BetaScore] = {}
        for c in rubric.criteria:
            scores[c.axis_id] = self.score_axis(x, a, c)
        return MultiAxisScore(axis_scores=scores)
