"""Tests for data loading, time-zero cuts, leakage probes, and holdout."""

from __future__ import annotations

import pytest

from ccema.data.holdout import make_leak_probe_holdout
from ccema.data.leakage import (
    audit_dataset_layer1,
    layer1_audit,
    layer2_audit,
    ngrams,
    summarize,
)
from ccema.data.loaders import load_dataset
from ccema.data.schema import Encounter
from ccema.data.synthetic import SyntheticConfig, make_synthetic_dataset
from ccema.data.time_zero import (
    cut_by_heuristic,
    cut_by_intent,
    strip_post_decision_artifacts,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_encounter_hash_stable() -> None:
    e = Encounter(case_id="t/0", source="t", x="hello", a_clinician="bye")
    assert e.x_hash == Encounter(case_id="t/0", source="t", x="hello", a_clinician="bye").x_hash
    assert len(e.x_hash) == 16


# ---------------------------------------------------------------------------
# Time-zero cuts
# ---------------------------------------------------------------------------


def test_cut_by_intent_finds_assessment_boundary() -> None:
    turns = [
        {"speaker": "patient", "text": "I have a cough.", "intent": "history"},
        {"speaker": "doctor", "text": "How long?", "intent": "history"},
        {"speaker": "patient", "text": "5 days.", "intent": "history"},
        {"speaker": "doctor", "text": "I think this is bronchitis.", "intent": "assessment"},
        {"speaker": "doctor", "text": "Let's start antibiotics.", "intent": "plan"},
    ]
    cut = cut_by_intent(turns)
    assert cut.cut_index == 3
    assert "5 days" in cut.pre_decision
    assert "bronchitis" not in cut.pre_decision
    assert "bronchitis" in cut.post_decision
    assert cut.method == "intent"


def test_cut_by_intent_no_marker() -> None:
    turns = [{"speaker": "patient", "text": "hi", "intent": "history"}]
    cut = cut_by_intent(turns)
    assert cut.cut_index == -1
    assert "hi" in cut.pre_decision


def test_cut_by_heuristic_assessment_phrase() -> None:
    text = (
        "DOCTOR: How long has this been going on?\n"
        "PATIENT: About a week.\n"
        "DOCTOR: My plan is to order labs.\n"
    )
    cut = cut_by_heuristic(text)
    assert cut.cut_index > 0
    assert "About a week" in cut.pre_decision
    assert "My plan" not in cut.pre_decision
    assert "My plan" in cut.post_decision


def test_strip_post_decision_artifacts_removes_icd_and_assessment_lines() -> None:
    raw = (
        "Chief complaint: cough.\n"
        "Assessment: acute bronchitis (J20.9)\n"
        "WBC: 12.3 mg/dL\n"
        "Plan: antibiotics.\n"
    )
    cleaned = strip_post_decision_artifacts(raw)
    assert "J20.9" not in cleaned
    assert "Assessment" not in cleaned
    assert "Plan:" not in cleaned
    assert "Chief complaint" in cleaned


# ---------------------------------------------------------------------------
# Layer-1 leakage probe
# ---------------------------------------------------------------------------


def test_layer1_clean_when_no_leakage() -> None:
    enc = Encounter(
        case_id="c1",
        source="t",
        x="cough for 5 days, productive sputum",
        a_clinician="Assessment: acute bronchitis. Plan: rest fluids.",
        working_diagnosis="acute bronchitis",
    )
    rep = layer1_audit(enc)
    assert rep.is_clean


def test_layer1_detects_dx_leak() -> None:
    enc = Encounter(
        case_id="c1",
        source="t",
        x="patient reports symptoms of acute bronchitis for 5 days",
        a_clinician="A&P",
        working_diagnosis="acute bronchitis",
    )
    rep = layer1_audit(enc)
    assert rep.leaked_dx
    assert not rep.is_clean


def test_layer1_does_not_flag_chronic_history() -> None:
    enc = Encounter(
        case_id="c_hist",
        source="t",
        x="patient has a history of acute bronchitis and presents today with cough",
        a_clinician="A&P",
        working_diagnosis="acute bronchitis",
    )
    rep = layer1_audit(enc)
    assert not rep.leaked_dx
    assert rep.dx_in_history  # captured for transparency
    assert rep.is_clean


def test_layer1_flags_active_diagnosis() -> None:
    enc = Encounter(
        case_id="c_active",
        source="t",
        x="the patient's acute bronchitis is severe today",
        a_clinician="A&P",
        working_diagnosis="acute bronchitis",
    )
    rep = layer1_audit(enc)
    assert rep.leaked_dx
    assert not rep.is_clean


def test_layer1_detects_icd_leak() -> None:
    enc = Encounter(
        case_id="c2",
        source="t",
        x="cough J20.9 in chart",
        a_clinician="A&P",
    )
    rep = layer1_audit(enc)
    assert rep.leaked_icd == ["J20.9"]


def test_layer1_detects_lab_leak() -> None:
    enc = Encounter(
        case_id="c3",
        source="t",
        x="WBC was 12.3 mg/dL today",
        a_clinician="A&P",
    )
    rep = layer1_audit(enc)
    assert any("12.3" in s for s in rep.leaked_labs)


def test_synthetic_with_injected_leakage_is_caught() -> None:
    cfg = SyntheticConfig(n_encounters=20, seed=1, inject_layer1_leakage_prob=1.0)
    encs = make_synthetic_dataset(cfg)
    reports = audit_dataset_layer1(encs)
    # Every encounter has injected dx leakage → none should be clean
    assert sum(r.is_clean for r in reports) == 0
    summary = summarize(reports)
    assert summary.layer1_failures.get("leaked_dx", 0) == len(encs)


# ---------------------------------------------------------------------------
# Layer-2 ngram overlap
# ---------------------------------------------------------------------------


def test_ngrams_basic() -> None:
    text = "the quick brown fox jumps over the lazy dog now"
    g = ngrams(text, n=3)
    assert "the quick brown" in g
    assert "lazy dog now" in g


def test_layer2_flags_when_ngram_present() -> None:
    enc = Encounter(
        case_id="c1",
        source="t",
        x="patient reports productive cough for five days with low grade fever",
        a_clinician="acute bronchitis",
    )
    corpus = {
        "fake_train": ngrams("patient reports productive cough for five days with low grade", n=8)
    }
    rep = layer2_audit(enc, corpus_ngrams=corpus, n=8)
    assert rep.flagged
    assert rep.ngram_match_count >= 1


def test_layer2_clean_when_disjoint() -> None:
    enc = Encounter(case_id="c1", source="t", x="totally different content here", a_clinician="x")
    corpus = {"fake": ngrams("nothing in common at all between texts", n=8)}
    rep = layer2_audit(enc, corpus_ngrams=corpus, n=8)
    assert not rep.flagged


# ---------------------------------------------------------------------------
# Holdout
# ---------------------------------------------------------------------------


def test_leak_probe_holdout_truncates() -> None:
    encs = make_synthetic_dataset(SyntheticConfig(n_encounters=30, seed=2))
    main, probe = make_leak_probe_holdout(encs, holdout_frac=0.20, seed=2)
    assert len(main) + len(probe) == len(encs)
    assert len(probe) == 6  # 20% of 30
    for p in probe:
        original_len = p.metadata["orig_len"]
        assert len(p.x) <= original_len
        assert p.case_id.endswith("_truncated")
        assert p.metadata["leak_probe"] is True


def test_leak_probe_holdout_deterministic() -> None:
    encs = make_synthetic_dataset(SyntheticConfig(n_encounters=20, seed=3))
    main_a, probe_a = make_leak_probe_holdout(encs, holdout_frac=0.10, seed=99)
    main_b, probe_b = make_leak_probe_holdout(encs, holdout_frac=0.10, seed=99)
    assert [e.case_id for e in probe_a] == [e.case_id for e in probe_b]


# ---------------------------------------------------------------------------
# Loader dispatcher
# ---------------------------------------------------------------------------


def test_load_dataset_synthetic() -> None:
    encs = load_dataset("synthetic", n_encounters=10, seed=4)
    assert len(encs) == 10
    assert all(e.source == "synthetic" for e in encs)


def test_load_dataset_unknown_raises() -> None:
    with pytest.raises(ValueError):
        load_dataset("not_a_dataset")


def test_load_dataset_real_requires_root() -> None:
    with pytest.raises(ValueError):
        load_dataset("aci_bench")
    with pytest.raises(ValueError):
        load_dataset("mts_dialog")
