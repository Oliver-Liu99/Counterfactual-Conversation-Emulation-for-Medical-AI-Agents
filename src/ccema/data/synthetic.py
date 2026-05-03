"""Synthetic encounter generator for tests and the toy OPE oracle.

Real ACI-Bench / MTS-Dialog data is gated by license / institutional access.
This module generates structurally similar encounters with controlled
ground truth, so that:
1. unit tests for loaders / leakage probes / estimators run anywhere
2. Step 8's synthetic OPE benchmark has known w(x, a) and Y(x, a)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ccema.data.schema import Encounter

_TEMPLATES = [
    {
        "icd_chapter": "J",  # respiratory
        "presentations": [
            "cough productive of yellow sputum for 5 days, low-grade fever, "
            "no shortness of breath at rest. Smoker, 20 pack-years.",
            "wheezing and dry cough worse at night, recent URI 2 weeks ago, "
            "pet exposure.",
        ],
        "exam": "Lungs with scattered rhonchi bilaterally; no rales.",
        "plausible_dx": ["acute bronchitis", "asthma exacerbation", "viral URI"],
    },
    {
        "icd_chapter": "K",  # GI
        "presentations": [
            "epigastric burning post-prandial, worse with NSAIDs, no melena. "
            "Drinks 6 cups coffee daily.",
            "nausea and intermittent right upper quadrant pain after fatty meals, "
            "no jaundice, BMI 32.",
        ],
        "exam": "Abdomen soft, mild epigastric tenderness, no rebound or guarding.",
        "plausible_dx": ["GERD", "gastritis", "biliary colic"],
    },
    {
        "icd_chapter": "I",  # cardiovascular
        "presentations": [
            "intermittent palpitations on exertion, no syncope, no chest pain. "
            "Family history of atrial fibrillation.",
            "dyspnea on exertion progressing over 2 months, ankle swelling, "
            "history of poorly controlled hypertension.",
        ],
        "exam": "Heart regular rate, no murmurs; lungs clear; trace pedal edema.",
        "plausible_dx": [
            "paroxysmal atrial fibrillation",
            "heart failure with preserved EF",
            "anxiety with palpitations",
        ],
    },
]


@dataclass
class SyntheticConfig:
    n_encounters: int = 60
    seed: int = 42
    inject_layer1_leakage_prob: float = 0.0  # for testing the auditor


def make_synthetic_dataset(cfg: SyntheticConfig) -> list[Encounter]:
    rng = random.Random(cfg.seed)

    encounters: list[Encounter] = []
    for i in range(cfg.n_encounters):
        tmpl = rng.choice(_TEMPLATES)
        presentation = rng.choice(tmpl["presentations"])
        dx = rng.choice(tmpl["plausible_dx"])
        age = rng.randint(20, 84)
        sex = rng.choice(["M", "F"])

        x = (
            f"## Subjective\n"
            f"Age {age} {sex}. Chief complaint: {presentation}\n\n"
            f"## Objective\n{tmpl['exam']}\n\n"
            f"## Dialogue (pre-decision)\n"
            f"PATIENT: {presentation}\n"
            f"DOCTOR: How long have you had these symptoms?\n"
            f"PATIENT: About a week, getting worse.\n"
        )

        # Optionally inject leakage to test Layer-1 auditor
        if cfg.inject_layer1_leakage_prob > 0 and rng.random() < cfg.inject_layer1_leakage_prob:
            x += f"\nDOCTOR: I think this is {dx}. Let's start treatment."

        a_clinician = (
            f"## Assessment\n{age} {sex} with {dx}.\n\n"
            f"## Plan\n"
            f"- Confirm with appropriate diagnostics for {tmpl['icd_chapter']} chapter\n"
            f"- Symptom management\n"
            f"- Follow up in 2 weeks; return precautions reviewed."
        )

        encounters.append(
            Encounter(
                case_id=f"synthetic/{i:04d}",
                source="synthetic",
                x=x,
                a_clinician=a_clinician,
                dialogue_pre=x.split("## Dialogue (pre-decision)\n")[-1],
                dialogue_post="",
                working_diagnosis=dx,
                icd10_codes=[],
                metadata={
                    "icd_chapter": tmpl["icd_chapter"],
                    "age_band": _age_band(age),
                    "comorbidity_count": rng.randint(0, 3),
                    "synthetic": True,
                },
            )
        )

    return encounters


def _age_band(age: int) -> str:
    if age < 30:
        return "18-29"
    if age < 50:
        return "30-49"
    if age < 70:
        return "50-69"
    return "70+"
