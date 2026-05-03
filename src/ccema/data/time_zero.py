"""Time-zero principle: cut x to contain only pre-decision information.

We support two modes:
1. intent-based cut (ACI-Bench 2025 release adds turn-level intent labels)
2. heuristic cut (MTS-Dialog has no intent annotations; use marker phrases)

The output is a *pre-decision* dialogue that becomes part of x. The
post-decision portion is retained only for audit and ground-truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Heuristic markers that signal the clinician has shifted from gathering
# information to articulating an assessment / plan. Conservative — false
# positives cut more aggressively (safer wrt leakage).
ASSESSMENT_MARKERS: tuple[str, ...] = (
    r"\bassessment\b",
    r"\bimpression\b",
    r"\bI think (this|you have|it'?s)\b",
    r"\bmy plan\b",
    r"\bwe(?:'?ll| will) (start|prescribe|order|do)\b",
    r"\blet'?s (start|try|prescribe|order)\b",
    r"\bI'?m going to (start|prescribe|order|put you on)\b",
    r"\bdiagnos(is|ed) (with|of|as)\b",
    r"\bwe(?:'?re| are) going to\b",
    r"\bthe diagnosis is\b",
)

_ASSESSMENT_RE = re.compile("|".join(ASSESSMENT_MARKERS), flags=re.IGNORECASE)


@dataclass
class TimeZeroCut:
    """Result of a time-zero cut on a dialogue."""

    pre_decision: str
    post_decision: str
    cut_index: int  # turn index where cut occurred; -1 if no cut found
    method: str  # "intent" | "heuristic" | "fulltext"


def cut_by_intent(turns: list[dict], assessment_label: str = "assessment") -> TimeZeroCut:
    """Cut a turn list at the first turn labelled as assessment intent.

    Args:
        turns: list of {"speaker": str, "text": str, "intent": str}
        assessment_label: the intent string marking time zero

    Returns:
        TimeZeroCut with pre_decision = all turns *before* the marker.
    """
    cut_idx = -1
    for i, t in enumerate(turns):
        if t.get("intent", "").lower() == assessment_label.lower():
            cut_idx = i
            break

    if cut_idx == -1:
        # No assessment intent found → conservative: keep full dialogue as
        # pre-decision and mark for heuristic fallback in downstream code.
        pre = "\n".join(_render(t) for t in turns)
        return TimeZeroCut(pre_decision=pre, post_decision="", cut_index=-1, method="intent")

    pre_turns = turns[:cut_idx]
    post_turns = turns[cut_idx:]
    return TimeZeroCut(
        pre_decision="\n".join(_render(t) for t in pre_turns),
        post_decision="\n".join(_render(t) for t in post_turns),
        cut_index=cut_idx,
        method="intent",
    )


def cut_by_heuristic(dialogue_text: str) -> TimeZeroCut:
    """Cut a flat dialogue string at the first assessment-marker occurrence.

    Used as fallback when intent annotations are unavailable (MTS-Dialog).
    """
    match = _ASSESSMENT_RE.search(dialogue_text)
    if match is None:
        return TimeZeroCut(
            pre_decision=dialogue_text,
            post_decision="",
            cut_index=-1,
            method="heuristic",
        )

    # Cut at the start of the line containing the match — keep the full
    # pre-decision portion verbatim.
    line_start = dialogue_text.rfind("\n", 0, match.start()) + 1
    return TimeZeroCut(
        pre_decision=dialogue_text[:line_start].rstrip(),
        post_decision=dialogue_text[line_start:],
        cut_index=line_start,
        method="heuristic",
    )


def strip_post_decision_artifacts(text: str) -> str:
    """Remove post-decision fragments that may have leaked into x.

    Strips:
    - ICD-10 codes (e.g. "J45.909")
    - Lines beginning with "Assessment:" or "Plan:"
    - Lab result rows of the form "<test>: <value> <unit>" with numeric value
    """
    # ICD-10
    text = re.sub(r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?\b", "", text)

    # Section headers
    text = re.sub(
        r"^\s*(Assessment|Plan|Impression|Diagnosis)\s*:.*$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # Numeric lab result rows (heuristic — strip if line has number+unit)
    text = re.sub(
        r"^\s*[A-Za-z][\w\s]{0,30}:\s*\d+(?:\.\d+)?\s*(?:mg/dL|mmol/L|g/dL|/μL|U/L|mEq/L|%)\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    return _normalize_whitespace(text)


def _render(turn: dict) -> str:
    speaker = turn.get("speaker", "?").upper()
    return f"{speaker}: {turn.get('text', '').strip()}"


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
