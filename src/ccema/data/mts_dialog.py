"""MTS-Dialog loader.

MTS-Dialog (Abacha et al. EACL 2023) provides ~1700 dialogue–note snippet
pairs across SOAP sections. Each row is

    section_header,section_text,dialogue

with no turn-level intent annotations, so we apply heuristic time-zero
cuts.

Data layout expected (download from
https://github.com/abachaa/MTS-Dialog):

    mts_dialog_root/
        Main-Dataset/
            MTS-Dialog-TrainingSet.csv
            MTS-Dialog-ValidationSet.csv
            MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv
            MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from ccema.data.schema import Encounter
from ccema.data.time_zero import cut_by_heuristic, strip_post_decision_artifacts


_AP_SECTIONS = {"ASSESSMENT", "PLAN", "ASSESSMENT_AND_PLAN", "ASSESSMENT AND PLAN"}
_X_SECTIONS = {
    "CC",
    "HPI",
    "PASTMEDICALHX",
    "MEDICATIONS",
    "ALLERGY",
    "FAM/SOCHX",
    "ROS",
    "GENHX",
    "EXAM",
    "PHYSICALEXAM",
    "PHYSICAL EXAM",
}


def load_mts_dialog(
    root: str | Path,
    split: str = "training",
) -> list[Encounter]:
    """Load MTS-Dialog encounters from local CSVs.

    Args:
        root: MTS-Dialog repo root (must contain Main-Dataset/)
        split: "training" | "validation" | "test1" | "test2"

    Returns:
        Encounters keyed by `ID` column. The dataset is per-section, so
        we group rows by `ID` and reconstruct full encounters.
    """
    fname_map = {
        "training": "MTS-Dialog-TrainingSet.csv",
        "validation": "MTS-Dialog-ValidationSet.csv",
        "test1": "MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv",
        "test2": "MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv",
    }
    if split not in fname_map:
        raise ValueError(f"split must be one of {list(fname_map)}; got {split!r}")

    csv_path = Path(root) / "Main-Dataset" / fname_map[split]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MTS-Dialog file not found at {csv_path}. "
            f"Clone https://github.com/abachaa/MTS-Dialog into {root}."
        )

    # Group rows by ID (one ID = one encounter, multiple sections)
    by_id: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_id[row["ID"]].append(row)

    encounters: list[Encounter] = []
    for case_id, rows in by_id.items():
        enc = _encounter_from_rows(case_id, rows)
        if enc is not None:
            encounters.append(enc)
    return encounters


def _encounter_from_rows(case_id: str, rows: list[dict]) -> Encounter | None:
    x_chunks: list[str] = []
    a_chunks: list[str] = []
    dialogue_chunks: list[str] = []

    for row in rows:
        header = row.get("section_header", "").strip().upper()
        section_text = row.get("section_text", "").strip()
        dialogue = row.get("dialogue", "").strip()

        if dialogue:
            dialogue_chunks.append(dialogue)

        if header in _AP_SECTIONS:
            a_chunks.append(f"## {header.title()}\n{section_text}")
        elif header in _X_SECTIONS:
            x_chunks.append(f"## {header.title()}\n{section_text}")

    if not a_chunks or not (x_chunks or dialogue_chunks):
        return None

    full_dialogue = "\n\n".join(dialogue_chunks)
    cut = cut_by_heuristic(full_dialogue)

    x_parts = list(x_chunks)
    if cut.pre_decision:
        x_parts.append(f"## Dialogue (pre-decision)\n{cut.pre_decision}")
    x = strip_post_decision_artifacts("\n\n".join(x_parts))

    a_clinician = "\n\n".join(a_chunks).strip()

    return Encounter(
        case_id=f"mts_dialog/{case_id}",
        source="mts_dialog",
        x=x,
        a_clinician=a_clinician,
        dialogue_pre=cut.pre_decision,
        dialogue_post=cut.post_decision,
        metadata={
            "time_zero_method": cut.method,
            "time_zero_cut_index": cut.cut_index,
            "n_sections": len(rows),
        },
    )
