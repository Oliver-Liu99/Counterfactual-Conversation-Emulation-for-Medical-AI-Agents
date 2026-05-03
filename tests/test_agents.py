"""Tests for π_agent JSON schema validation, sampling, capability gap."""

from __future__ import annotations

import json

import pytest

from ccema.agents.pi_agent import PiAgent, capability_gap
from ccema.agents.schemas import (
    AgentOutput,
    SchemaError,
    looks_like_transcript_reference,
    validate_pi_agent_output,
)
from ccema.judges.llm_client import MockLLMClient


VALID = {
    "differential_diagnosis": [
        {
            "dx": "acute bronchitis",
            "icd10": "J20.9",
            "probability": "high",
            "supporting": ["productive cough"],
            "against": [],
        }
    ],
    "working_diagnosis": "acute bronchitis",
    "assessment": "Patient presents with productive cough; likely acute bronchitis.",
    "plan": {
        "diagnostics": ["consider CXR if persists"],
        "therapeutics": ["supportive care"],
        "patient_education": ["hydration", "cough hygiene"],
        "follow_up": "1 week",
        "red_flags": ["fever >39C", "dyspnea"],
    },
    "uncertainty_notes": "differential includes pneumonia",
}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_validate_clean_output() -> None:
    out = validate_pi_agent_output(json.dumps(VALID))
    assert isinstance(out, AgentOutput)
    assert out.working_diagnosis == "acute bronchitis"
    assert out.differential_diagnosis[0]["probability"] == "high"


def test_validate_strips_markdown_fence() -> None:
    text = "```json\n" + json.dumps(VALID) + "\n```"
    out = validate_pi_agent_output(text)
    assert out.assessment.startswith("Patient")


def test_validate_extracts_json_from_prose() -> None:
    text = "Here you go:\n" + json.dumps(VALID) + "\nLet me know if you need more."
    out = validate_pi_agent_output(text)
    assert out.working_diagnosis == "acute bronchitis"


def test_missing_field_raises() -> None:
    bad = dict(VALID)
    del bad["working_diagnosis"]
    with pytest.raises(SchemaError, match="working_diagnosis"):
        validate_pi_agent_output(json.dumps(bad))


def test_too_many_ddx_entries_raises() -> None:
    bad = dict(VALID)
    bad["differential_diagnosis"] = VALID["differential_diagnosis"] * 6
    with pytest.raises(SchemaError, match="1.5"):
        validate_pi_agent_output(json.dumps(bad))


def test_invalid_probability_label_raises() -> None:
    bad = json.loads(json.dumps(VALID))
    bad["differential_diagnosis"][0]["probability"] = "definitely"
    with pytest.raises(SchemaError, match="probability"):
        validate_pi_agent_output(json.dumps(bad))


def test_empty_plan_raises_in_strict_mode() -> None:
    bad = json.loads(json.dumps(VALID))
    bad["plan"]["diagnostics"] = []
    bad["plan"]["therapeutics"] = []
    with pytest.raises(SchemaError, match="diagnostic"):
        validate_pi_agent_output(json.dumps(bad), strict=True)


def test_empty_plan_ok_when_not_strict() -> None:
    bad = json.loads(json.dumps(VALID))
    bad["plan"]["diagnostics"] = []
    bad["plan"]["therapeutics"] = []
    out = validate_pi_agent_output(json.dumps(bad), strict=False)
    assert out.plan["therapeutics"] == []


def test_to_assessment_plan_text_renders_all_sections() -> None:
    out = validate_pi_agent_output(json.dumps(VALID))
    txt = out.to_assessment_plan_text()
    for section in ("Differential", "Working Diagnosis", "Assessment", "Plan", "Uncertainty"):
        assert section in txt


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------


def test_looks_like_transcript_reference_flags_phrase() -> None:
    out = validate_pi_agent_output(json.dumps(VALID))
    out.assessment = "As documented in the note, the patient has cough."
    assert looks_like_transcript_reference(out)


def test_looks_like_transcript_reference_clean() -> None:
    out = validate_pi_agent_output(json.dumps(VALID))
    assert not looks_like_transcript_reference(out)


# ---------------------------------------------------------------------------
# π_agent sampling
# ---------------------------------------------------------------------------


def test_sample_n_returns_n_samples() -> None:
    client = MockLLMClient(name="mock", canned_response=json.dumps(VALID))
    agent = PiAgent(arm="strong", client=client, system_prompt="sys", n_samples=3)
    samples = agent.sample_n("c1", "x text")
    assert len(samples) == 3
    assert all(s.output is not None for s in samples)
    assert all(s.error is None for s in samples)
    assert all(s.arm == "strong" for s in samples)
    assert {s.sample_idx for s in samples} == {0, 1, 2}


def test_invalid_json_marked_as_failed() -> None:
    client = MockLLMClient(name="mock", canned_response="not even json")
    agent = PiAgent(arm="weak", client=client, system_prompt="sys", n_samples=1)
    s = agent.sample_one("c1", "x", 0)
    assert s.output is None
    assert s.error is not None and s.error.startswith("schema:")


def test_transcript_reference_marked_as_failed() -> None:
    bad = json.loads(json.dumps(VALID))
    bad["assessment"] = "From the chart, patient has cough."
    client = MockLLMClient(name="mock", canned_response=json.dumps(bad))
    agent = PiAgent(arm="strong", client=client, system_prompt="sys", n_samples=1, leakage_check=True)
    s = agent.sample_one("c1", "x", 0)
    assert s.output is None
    assert s.error == "transcript_reference"


# ---------------------------------------------------------------------------
# Capability gap
# ---------------------------------------------------------------------------


def test_capability_gap_passes_when_strong_better() -> None:
    strong = [0.9, 0.85, 0.92, 0.88, 0.91]
    weak = [0.6, 0.55, 0.62, 0.58, 0.61]
    res = capability_gap(strong, weak, threshold=0.10)
    assert res.passes
    assert res.gap > 0.25


def test_capability_gap_fails_when_no_separation() -> None:
    same = [0.7] * 5
    res = capability_gap(same, same, threshold=0.10)
    assert not res.passes
    assert res.gap == 0


def test_capability_gap_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        capability_gap([0.5, 0.6], [0.4])
