"""Canonical data structures for clinical encounters."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Encounter:
    """A single clinician–patient encounter post time-zero cut.

    Attributes:
        case_id: stable identifier within `source`
        source: dataset name, e.g. "aci_bench", "mts_dialog", "synthetic"
        x: patient pre-decision context — chief complaint, HPI, exam, etc.
        a_clinician: clinician's final assessment + plan (from SOAP note)
        dialogue_pre: raw dialogue up to and including the time-zero boundary
        dialogue_post: raw dialogue after time zero (NOT used for evaluation;
            retained only for leakage auditing and ground-truth construction)
        working_diagnosis: extracted clinician working dx (audit only)
        icd10_codes: extracted ICD-10 codes (audit only — must NOT appear in x)
        metadata: free-form keys (icd_chapter, age_band, comorbidity_count, …)
    """

    case_id: str
    source: str
    x: str
    a_clinician: str
    dialogue_pre: str = ""
    dialogue_post: str = ""
    working_diagnosis: str = ""
    icd10_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def x_hash(self) -> str:
        return hashlib.sha256(self.x.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
