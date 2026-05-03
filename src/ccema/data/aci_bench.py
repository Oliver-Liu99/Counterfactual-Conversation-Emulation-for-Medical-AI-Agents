"""ACI-Bench loader.

ACI-Bench (Yim et al. 2023, Sci Data) provides 207 ambient-clinical-
intelligence encounters: paired full dialogue + structured SOAP note.
The 2025 release adds turn-level intent annotations enabling clean
time-zero cuts.

Data layout expected (download from
https://github.com/wyim/aci-bench):

    aci_bench_root/
        train/
            <case_id>.dialog.json     # turns with intent labels
            <case_id>.note.json       # SOAP sections
        valid/
            ...
        test/
            ...

Each .dialog.json is a list of {"speaker", "text", "intent"} dicts.
Each .note.json is {"subjective", "objective", "assessment", "plan"}.

If the 2025 intent release is unavailable, the loader falls back to
heuristic time-zero cuts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ccema.data.schema import Encounter
from ccema.data.time_zero import (
    cut_by_heuristic,
    cut_by_intent,
    strip_post_decision_artifacts,
)


def load_aci_bench(
    root: str | Path,
    split: str = "train",
    require_intent: bool = False,
) -> list[Encounter]:
    """Load ACI-Bench encounters from a local directory.

    Args:
        root: ACI-Bench data root (containing train/ valid/ test/)
        split: "train" | "valid" | "test"
        require_intent: if True, drop encounters lacking intent labels
    """
    root_path = Path(root) / split
    if not root_path.exists():
        raise FileNotFoundError(
            f"ACI-Bench split not found at {root_path}. "
            f"Download from https://github.com/wyim/aci-bench and unpack to {root}."
        )

    encounters: list[Encounter] = []
    for dialog_path in sorted(root_path.glob("*.dialog.json")):
        case_id = dialog_path.stem.removesuffix(".dialog")
        note_path = root_path / f"{case_id}.note.json"
        if not note_path.exists():
            continue

        with open(dialog_path) as f:
            turns = json.load(f)
        with open(note_path) as f:
            note = json.load(f)

        enc = _encounter_from_pair(case_id, turns, note, require_intent=require_intent)
        if enc is not None:
            encounters.append(enc)

    return encounters


def _encounter_from_pair(
    case_id: str,
    turns: list[dict],
    note: dict,
    require_intent: bool,
) -> Encounter | None:
    has_intent = bool(turns) and "intent" in turns[0]

    if has_intent:
        cut = cut_by_intent(turns, assessment_label="assessment")
    else:
        if require_intent:
            return None
        flat = "\n".join(f"{t.get('speaker', '?').upper()}: {t.get('text', '')}" for t in turns)
        cut = cut_by_heuristic(flat)

    # Subjective + Objective form x; Assessment + Plan form a_clinician.
    subjective = note.get("subjective", "").strip()
    objective = note.get("objective", "").strip()
    assessment = note.get("assessment", "").strip()
    plan = note.get("plan", "").strip()

    x_parts = []
    if subjective:
        x_parts.append(f"## Subjective\n{subjective}")
    if objective:
        x_parts.append(f"## Objective\n{objective}")
    if cut.pre_decision:
        x_parts.append(f"## Dialogue (pre-decision)\n{cut.pre_decision}")

    x = strip_post_decision_artifacts("\n\n".join(x_parts))

    a_parts = []
    if assessment:
        a_parts.append(f"## Assessment\n{assessment}")
    if plan:
        a_parts.append(f"## Plan\n{plan}")
    a_clinician = "\n\n".join(a_parts).strip()

    if not x or not a_clinician:
        return None

    return Encounter(
        case_id=f"aci_bench/{case_id}",
        source="aci_bench",
        x=x,
        a_clinician=a_clinician,
        dialogue_pre=cut.pre_decision,
        dialogue_post=cut.post_decision,
        working_diagnosis=note.get("working_diagnosis", ""),
        icd10_codes=note.get("icd10_codes", []) or [],
        metadata={
            "split": Path(case_id).stem,
            "time_zero_method": cut.method,
            "time_zero_cut_index": cut.cut_index,
        },
    )


def iter_aci_bench(root: str | Path, splits: tuple[str, ...] = ("train",)) -> Iterator[Encounter]:
    for split in splits:
        yield from load_aci_bench(root, split=split)
