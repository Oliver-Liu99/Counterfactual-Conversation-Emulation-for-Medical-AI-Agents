"""ACI-Bench loader.

ACI-Bench (Yim et al. 2023, Sci Data) provides ambient-clinical-intelligence
encounters: paired full doctor-patient dialogue + a structured clinical note.

Data layout (clone https://github.com/wyim/aci-bench):

    aci_bench_root/
        data/
            challenge_data/
                train.csv
                valid.csv
                clinicalnlp_taskB_test1.csv
                clinicalnlp_taskC_test2.csv
                clef_taskC_test3.csv

Each CSV has columns: ``dataset, encounter_id, dialogue, note``.

The notes are NOT in clean SOAP form.  Instead, they use clinical headers
("CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "PHYSICAL EXAM",
"ASSESSMENT AND PLAN", etc.) which we group into the four canonical SOAP
buckets:

    Subjective ← CHIEF COMPLAINT, HPI, REVIEW OF SYSTEMS, *_HISTORY,
                 ALLERGIES, MEDICATIONS
    Objective  ← PHYSICAL EXAM(INATION), EXAM, VITALS(_REVIEWED), RESULTS
    Assessment ← ASSESSMENT, IMPRESSION  (first half of "ASSESSMENT AND PLAN")
    Plan       ← PLAN, INSTRUCTIONS       (second half of "ASSESSMENT AND PLAN")

x = subjective + objective + dialogue cut at heuristic time-zero.
a_clinician = assessment + plan.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterator

from ccema.data.schema import Encounter
from ccema.data.time_zero import (
    cut_by_heuristic,
    strip_post_decision_artifacts,
)


# Splits and their CSV filenames inside <root>/data/challenge_data/
SPLIT_FILES: dict[str, str] = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test1": "clinicalnlp_taskB_test1.csv",
    "test2": "clinicalnlp_taskC_test2.csv",
    "test3": "clef_taskC_test3.csv",
}

# Header categorisation. Keys are normalised (uppercased, single-space).
SUBJECTIVE_HEADERS: frozenset[str] = frozenset({
    "CHIEF COMPLAINT",
    "CC",
    "HISTORY OF PRESENT ILLNESS",
    "HPI",
    "REVIEW OF SYSTEMS",
    "ROS",
    "SOCIAL HISTORY",
    "MEDICAL HISTORY",
    "PAST HISTORY",
    "PAST MEDICAL HISTORY",
    "FAMILY HISTORY",
    "SURGICAL HISTORY",
    "ALLERGIES",
    "MEDICATIONS",
    "CURRENT MEDICATIONS",
})

OBJECTIVE_HEADERS: frozenset[str] = frozenset({
    "PHYSICAL EXAM",
    "PHYSICAL EXAMINATION",
    "EXAM",
    "VITALS",
    "VITALS REVIEWED",
    "RESULTS",
    "PROCEDURE",
})

ASSESSMENT_HEADERS: frozenset[str] = frozenset({
    "ASSESSMENT",
    "IMPRESSION",
})

PLAN_HEADERS: frozenset[str] = frozenset({
    "PLAN",
    "INSTRUCTIONS",
})

ASSESSMENT_AND_PLAN_HEADERS: frozenset[str] = frozenset({
    "ASSESSMENT AND PLAN",
    "ASSESSMENT & PLAN",
    "ASSESSMENT/PLAN",
    "A&P",
    "A AND P",
    "A/P",
})

# A header is a line with only uppercase letters / spaces / a few punctuation
# marks, optionally followed by a colon. Length capped to avoid grabbing
# all-caps sentences.
_HEADER_RE = re.compile(r"^[A-Z][A-Z &,/\-]{1,40}:?\s*$", flags=re.MULTILINE)


def _normalise_header(h: str) -> str:
    """Return upper-case, single-space, colon-stripped form."""
    return re.sub(r"\s+", " ", h.strip().rstrip(":")).upper()


def parse_note_sections(note: str) -> dict[str, str]:
    """Split a clinical note into a {header: body} dict.

    Headers are matched case-insensitively against the all-caps form. Bodies
    are everything between one header and the next.
    """
    if not note:
        return {}

    matches = list(_HEADER_RE.finditer(note))
    if not matches:
        return {"_BODY": note.strip()}

    sections: dict[str, str] = {}
    # Anything before the first header → "_PREAMBLE" (rare but possible).
    if matches[0].start() > 0:
        pre = note[: matches[0].start()].strip()
        if pre:
            sections["_PREAMBLE"] = pre

    for i, m in enumerate(matches):
        header = _normalise_header(m.group(0))
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(note)
        body = note[body_start:body_end].strip()
        if body:
            # If the same header appears twice (rare), concatenate.
            if header in sections:
                sections[header] = sections[header] + "\n\n" + body
            else:
                sections[header] = body
    return sections


def _split_assessment_and_plan(body: str) -> tuple[str, str]:
    """Best-effort split of an "ASSESSMENT AND PLAN" body into (assess, plan).

    ACI-Bench A&P blocks usually start with a one-paragraph patient summary,
    followed by per-problem entries. Each problem has bullet/sub-headers like
    "Medical Reasoning:" and "Medical Treatment:" — the treatment-side bullets
    constitute "plan", the reasoning-side constitutes "assessment". For the
    coarse SOAP mapping we don't need a perfect split: assigning the whole
    block to BOTH assessment and plan would double-count, so we instead route
    the entire block to assessment and leave plan empty when no separate "PLAN"
    header is present. Callers can join (assessment, plan) freely.
    """
    if not body:
        return "", ""
    # If the body has a "Plan:" line, split there.
    m = re.search(r"^\s*plan\s*:\s*", body, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return body[: m.start()].strip(), body[m.start():].strip()
    # Otherwise return whole body as assessment.
    return body.strip(), ""


def _bucket_sections(sections: dict[str, str]) -> tuple[str, str, str, str]:
    """Map raw sections into (subjective, objective, assessment, plan) text."""
    subj_parts: list[str] = []
    obj_parts: list[str] = []
    assess_parts: list[str] = []
    plan_parts: list[str] = []

    for header, body in sections.items():
        if header in SUBJECTIVE_HEADERS:
            subj_parts.append(f"### {header.title()}\n{body}")
        elif header in OBJECTIVE_HEADERS:
            obj_parts.append(f"### {header.title()}\n{body}")
        elif header in ASSESSMENT_HEADERS:
            assess_parts.append(body)
        elif header in PLAN_HEADERS:
            plan_parts.append(body)
        elif header in ASSESSMENT_AND_PLAN_HEADERS:
            a, p = _split_assessment_and_plan(body)
            if a:
                assess_parts.append(a)
            if p:
                plan_parts.append(p)
        # Headers that don't match any bucket (e.g. "_PREAMBLE", "HIV") are
        # dropped — they're rare and not load-bearing for SOAP-style x.

    return (
        "\n\n".join(subj_parts).strip(),
        "\n\n".join(obj_parts).strip(),
        "\n\n".join(assess_parts).strip(),
        "\n\n".join(plan_parts).strip(),
    )


def _first_sentence(text: str, max_chars: int = 200) -> str:
    """Return a best-effort first sentence (used as working_diagnosis)."""
    if not text:
        return ""
    # Strip a leading patient-summary paragraph if it looks like a recap
    # (e.g. starts with a name + "is a NN-year-old"). Take the next non-empty
    # paragraph as the diagnosis surface.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    candidate = ""
    for p in paragraphs:
        if re.search(r"\bis a \d{1,3}\s*-?year[- ]old\b", p, flags=re.IGNORECASE):
            continue
        candidate = p
        break
    if not candidate:
        candidate = paragraphs[0] if paragraphs else text
    # First sentence = up to first period / newline.
    sent = re.split(r"(?<=[.!?])\s|\n", candidate, maxsplit=1)[0].strip()
    sent = sent.rstrip(".:")
    return sent[:max_chars]


def load_aci_bench(
    root: str | Path,
    split: str = "train",
) -> list[Encounter]:
    """Load ACI-Bench encounters from a local clone of github.com/wyim/aci-bench.

    Args:
        root: path to the ACI-Bench repo root (the directory that contains
            ``data/challenge_data/<split>.csv``). Either the repo root or the
            ``data/challenge_data`` directory itself works.
        split: one of ``train``, ``valid``, ``test1``, ``test2``, ``test3``.

    Returns:
        A list of Encounter objects.

    Raises:
        FileNotFoundError: if the expected CSV is missing.
        ValueError: if ``split`` is not recognised.
    """
    if split not in SPLIT_FILES:
        raise ValueError(
            f"Unknown split {split!r}. Valid splits: {sorted(SPLIT_FILES)}"
        )

    csv_path = _resolve_csv(root, split)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ACI-Bench CSV not found at {csv_path}. "
            f"Clone https://github.com/wyim/aci-bench and pass its path as `root`."
        )

    encounters: list[Encounter] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            enc = _encounter_from_row(row, split=split)
            if enc is not None:
                encounters.append(enc)
    return encounters


def _resolve_csv(root: str | Path, split: str) -> Path:
    fname = SPLIT_FILES[split]
    root_path = Path(root)
    # Accept either the repo root or the .../data/challenge_data dir.
    candidates = [
        root_path / "data" / "challenge_data" / fname,
        root_path / "challenge_data" / fname,
        root_path / fname,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _encounter_from_row(row: dict[str, str], split: str) -> Encounter | None:
    encounter_id = (row.get("encounter_id") or "").strip()
    dataset = (row.get("dataset") or "unknown").strip()
    dialogue = (row.get("dialogue") or "").strip()
    note = (row.get("note") or "").strip()
    if not encounter_id or not dialogue or not note:
        return None

    sections = parse_note_sections(note)
    subjective, objective, assessment, plan = _bucket_sections(sections)

    cut = cut_by_heuristic(dialogue)

    x_parts: list[str] = []
    if subjective:
        x_parts.append(f"## Subjective\n{subjective}")
    if objective:
        x_parts.append(f"## Objective\n{objective}")
    if cut.pre_decision:
        x_parts.append(f"## Dialogue (pre-decision)\n{cut.pre_decision}")

    x = strip_post_decision_artifacts("\n\n".join(x_parts))

    a_parts: list[str] = []
    if assessment:
        a_parts.append(f"## Assessment\n{assessment}")
    if plan:
        a_parts.append(f"## Plan\n{plan}")
    a_clinician = "\n\n".join(a_parts).strip()

    if not x or not a_clinician:
        return None

    working_dx = _first_sentence(assessment) if assessment else ""

    return Encounter(
        case_id=f"aci_bench/{dataset}/{encounter_id}",
        source="aci_bench",
        x=x,
        a_clinician=a_clinician,
        dialogue_pre=cut.pre_decision,
        dialogue_post=cut.post_decision,
        working_diagnosis=working_dx,
        icd10_codes=[],  # ACI-Bench does not ship ICD codes.
        metadata={
            "dataset": dataset,
            "split": split,
            "encounter_id": encounter_id,
            "time_zero_method": cut.method,
            "time_zero_cut_index": cut.cut_index,
            "section_headers": sorted(sections.keys()),
        },
    )


def iter_aci_bench(
    root: str | Path, splits: tuple[str, ...] = ("train",)
) -> Iterator[Encounter]:
    for split in splits:
        yield from load_aci_bench(root, split=split)
