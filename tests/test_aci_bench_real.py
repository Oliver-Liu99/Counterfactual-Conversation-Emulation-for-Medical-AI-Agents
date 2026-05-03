"""Smoke tests against a real local clone of ACI-Bench.

The tests are skipped automatically when the ACI-Bench data directory is not
present (e.g. on CI without a clone). Locally, after running

    git clone https://github.com/wyim/aci-bench data/raw/aci_bench

these tests exercise the CSV loader on the genuine corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ccema.data.aci_bench import (
    SPLIT_FILES,
    _bucket_sections,
    load_aci_bench,
    parse_note_sections,
)
from ccema.data.leakage import audit_dataset_layer1
from ccema.data.loaders import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
ACI_ROOT = REPO_ROOT / "data" / "raw" / "aci_bench"
TRAIN_CSV = ACI_ROOT / "data" / "challenge_data" / SPLIT_FILES["train"]


def _require_real_data() -> None:
    if not TRAIN_CSV.exists():
        pytest.skip(
            f"Real ACI-Bench data not found at {TRAIN_CSV}. "
            "Clone https://github.com/wyim/aci-bench into data/raw/aci_bench to enable."
        )


# ---------------------------------------------------------------------------
# Pure-Python parser unit tests (always run; no data dependency)
# ---------------------------------------------------------------------------


def test_parse_note_sections_extracts_canonical_headers() -> None:
    note = (
        "CHIEF COMPLAINT\n\nCough.\n\n"
        "HISTORY OF PRESENT ILLNESS\n\nThree days of productive cough.\n\n"
        "PHYSICAL EXAM\n\nLungs clear bilaterally.\n\n"
        "ASSESSMENT AND PLAN\n\nAcute bronchitis.\n• Plan: amoxicillin x7 days.\n"
    )
    sections = parse_note_sections(note)
    assert "CHIEF COMPLAINT" in sections
    assert "HISTORY OF PRESENT ILLNESS" in sections
    assert "PHYSICAL EXAM" in sections
    assert "ASSESSMENT AND PLAN" in sections
    assert "Cough." in sections["CHIEF COMPLAINT"]


def test_bucket_sections_routes_to_soap() -> None:
    sections = {
        "CHIEF COMPLAINT": "Cough.",
        "HISTORY OF PRESENT ILLNESS": "Three days.",
        "PHYSICAL EXAM": "Lungs clear.",
        "ASSESSMENT AND PLAN": "Acute bronchitis.\nPlan: amoxicillin.",
    }
    subj, obj, assess, plan = _bucket_sections(sections)
    assert "Cough." in subj
    assert "Three days." in subj
    assert "Lungs clear." in obj
    assert "bronchitis" in assess.lower()
    # The "Plan:" sub-line should split into the plan bucket.
    assert "amoxicillin" in plan.lower()


def test_bucket_sections_combined_block_without_plan_subline() -> None:
    """When ASSESSMENT AND PLAN has no separate `Plan:` sub-line, we route
    the entire block to assessment so it is not silently lost."""
    sections = {"ASSESSMENT AND PLAN": "Acute bronchitis. Will start amoxicillin."}
    _, _, assess, plan = _bucket_sections(sections)
    assert "bronchitis" in assess.lower()
    assert "amoxicillin" in assess.lower()
    assert plan == ""


def test_bucket_sections_handles_separate_assessment_and_plan() -> None:
    sections = {
        "ASSESSMENT": "Acute bronchitis.",
        "PLAN": "Amoxicillin x7 days.",
    }
    _, _, assess, plan = _bucket_sections(sections)
    assert assess == "Acute bronchitis."
    assert plan == "Amoxicillin x7 days."


# ---------------------------------------------------------------------------
# Tests requiring the real corpus
# ---------------------------------------------------------------------------


def test_load_aci_bench_real_train_returns_encounters() -> None:
    _require_real_data()
    encounters = load_aci_bench(ACI_ROOT, split="train")
    # Cap at 5 for smoke; underlying dataset is 67 encounters.
    sample = encounters[:5]
    assert len(sample) == 5
    for enc in sample:
        assert enc.source == "aci_bench"
        assert enc.case_id.startswith("aci_bench/")
        assert enc.case_id.count("/") == 2
        assert enc.x and enc.a_clinician
        assert "## Assessment" in enc.a_clinician or "## Plan" in enc.a_clinician
        assert enc.metadata["split"] == "train"
        assert "encounter_id" in enc.metadata


def test_load_aci_bench_real_via_dispatcher() -> None:
    _require_real_data()
    encounters = load_dataset("aci_bench", root=str(ACI_ROOT), split="train")
    assert len(encounters) > 0
    assert encounters[0].source == "aci_bench"


def test_load_aci_bench_real_layer1_audit_runs() -> None:
    _require_real_data()
    encounters = load_aci_bench(ACI_ROOT, split="train")[:5]
    reports = audit_dataset_layer1(encounters)
    assert len(reports) == 5
    # At least one of the five should be clean (sanity — corpus isn't 100% dirty).
    # We don't assert on the specific count to avoid coupling to corpus revision.
    assert any(r.is_clean for r in reports) or all(
        # Either some are clean, or all have *known* leakage modes (no surprise modes).
        not r.leaked_icd or r.leaked_dx or r.leaked_labs
        for r in reports
    )
