"""Strict JSON schema validator for π_agent output.

Pure-Python validator (no jsonschema/pydantic dependency for portability).
Returns a validated, normalized dict on success or raises SchemaError with a
specific message. Use `validate_pi_agent_output(text)` as the entry point.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


class SchemaError(ValueError):
    """Raised when an agent output violates the locked schema."""


_VALID_PROBABILITY = {"high", "moderate", "low"}


@dataclass
class AgentOutput:
    """Normalized π_agent output."""

    differential_diagnosis: list[dict]
    working_diagnosis: str
    assessment: str
    plan: dict
    uncertainty_notes: str
    raw: str

    def to_assessment_plan_text(self) -> str:
        """Render to the canonical (a) text passed to the rubric judge."""
        ddx = "\n".join(
            f"- {d['dx']} ({d.get('icd10') or 'no ICD'}, {d['probability']})"
            for d in self.differential_diagnosis
        )
        plan = self.plan
        plan_text = (
            f"Diagnostics: {'; '.join(plan.get('diagnostics', []))}\n"
            f"Therapeutics: {'; '.join(plan.get('therapeutics', []))}\n"
            f"Patient education: {'; '.join(plan.get('patient_education', []))}\n"
            f"Follow-up: {plan.get('follow_up', '')}\n"
            f"Red flags: {'; '.join(plan.get('red_flags', []))}"
        )
        return (
            f"## Differential\n{ddx}\n\n"
            f"## Working Diagnosis\n{self.working_diagnosis}\n\n"
            f"## Assessment\n{self.assessment}\n\n"
            f"## Plan\n{plan_text}\n\n"
            f"## Uncertainty\n{self.uncertainty_notes}"
        )


def _strip_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"```\s*$", "", s)
    return s


def _coerce_json(text: str) -> dict:
    s = _strip_fences(text)
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        # Try to extract first {...} block
        match = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if match is None:
            raise SchemaError(f"No JSON found in output: {text[:200]!r}") from e
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e2:
            raise SchemaError(f"Invalid JSON: {e2}") from e2


def validate_pi_agent_output(text: str, *, strict: bool = True) -> AgentOutput:
    """Parse + validate an agent's raw text into an AgentOutput.

    Raises SchemaError with a specific message on any violation.
    """
    obj = _coerce_json(text)
    if not isinstance(obj, dict):
        raise SchemaError(f"Top-level must be object, got {type(obj).__name__}")

    # differential_diagnosis
    ddx = obj.get("differential_diagnosis")
    if not isinstance(ddx, list):
        raise SchemaError("differential_diagnosis must be a list")
    if not (1 <= len(ddx) <= 5):
        raise SchemaError(f"differential_diagnosis must have 1–5 entries; got {len(ddx)}")
    for i, d in enumerate(ddx):
        if not isinstance(d, dict):
            raise SchemaError(f"differential_diagnosis[{i}] must be object")
        if "dx" not in d or not isinstance(d["dx"], str) or not d["dx"].strip():
            raise SchemaError(f"differential_diagnosis[{i}].dx must be non-empty string")
        prob = d.get("probability", "").lower()
        if prob not in _VALID_PROBABILITY:
            raise SchemaError(
                f"differential_diagnosis[{i}].probability must be one of {sorted(_VALID_PROBABILITY)}; got {prob!r}"
            )
        d["probability"] = prob
        d.setdefault("icd10", None)
        d.setdefault("supporting", [])
        d.setdefault("against", [])

    wd = obj.get("working_diagnosis")
    if not isinstance(wd, str) or not wd.strip():
        raise SchemaError("working_diagnosis must be non-empty string")

    assessment = obj.get("assessment")
    if not isinstance(assessment, str) or not assessment.strip():
        raise SchemaError("assessment must be non-empty string")

    plan = obj.get("plan")
    if not isinstance(plan, dict):
        raise SchemaError("plan must be object")
    for key in ("diagnostics", "therapeutics", "patient_education", "red_flags"):
        v = plan.get(key)
        if not isinstance(v, list) or any(not isinstance(s, str) for s in v):
            raise SchemaError(f"plan.{key} must be list[str]")
    follow_up = plan.get("follow_up")
    if not isinstance(follow_up, str):
        raise SchemaError("plan.follow_up must be string")

    if strict:
        # Plan must be non-empty in at least diagnostics/therapeutics
        if not (plan["diagnostics"] or plan["therapeutics"]):
            raise SchemaError("plan must have at least one diagnostic or therapeutic entry")

    uncertainty = obj.get("uncertainty_notes", "")
    if not isinstance(uncertainty, str):
        raise SchemaError("uncertainty_notes must be string")

    return AgentOutput(
        differential_diagnosis=ddx,
        working_diagnosis=wd,
        assessment=assessment,
        plan=plan,
        uncertainty_notes=uncertainty,
        raw=text,
    )


def looks_like_transcript_reference(output: AgentOutput) -> bool:
    """Sanity check: agent must not reference original transcript/note."""
    bad_phrases = (
        "as documented in the note",
        "the visit note",
        "according to the transcript",
        "in the original note",
        "from the chart",
        "as the clinician noted",
    )
    blob = f"{output.assessment}\n{output.uncertainty_notes}\n{json.dumps(output.plan)}".lower()
    return any(p in blob for p in bad_phrases)
