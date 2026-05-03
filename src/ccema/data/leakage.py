"""Two-layer leakage audit.

Layer 1 — *in-encounter* leakage:
    Verifies that x does not contain post-decision content. Specifically:
      L1a) the working diagnosis string from a_clinician must not appear in x
      L1b) ICD-10 codes must not appear in x
      L1c) lab/test result values must not appear in x

Layer 2 — *training-set decontamination*:
    Detects whether the encounter text overlaps with the training corpora of
    candidate π_agent or judge models. We use:
      L2a) n-gram exact match (default n=8, threshold ≥ 1 match per encounter)
      L2b) sentence-embedding cosine similarity (default threshold 0.90)

We treat L1 as a hard constraint (auto-fix or drop the encounter); L2 as a
diagnostic flag (does not necessarily invalidate, but documents the risk).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from ccema.data.schema import Encounter


# ---------------------------------------------------------------------------
# Layer 1 — in-encounter leakage
# ---------------------------------------------------------------------------


@dataclass
class Layer1Report:
    case_id: str
    leaked_dx: bool = False
    leaked_icd: list[str] = field(default_factory=list)
    leaked_labs: list[str] = field(default_factory=list)
    # Diagnosis matches that occur in a "history of …" / chronic-comorbidity
    # context. These are NOT counted as leaks but are surfaced for human
    # triage. Each entry is a short context snippet around the match.
    dx_in_history: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (self.leaked_dx or self.leaked_icd or self.leaked_labs)


# Phrases that, when they immediately precede a working-dx mention in `x`,
# indicate a chronic / historical comorbidity rather than a leaked working
# diagnosis. Matched case-insensitively against ~50 chars of left context.
# Be conservative: phrases here must be unambiguous markers of "this is
# pre-existing context, not the clinician's current decision".
_HISTORY_MARKERS: tuple[str, ...] = (
    "history of",
    "h/o",
    "hx of",
    "hx:",
    "known",
    "with a diagnosis of",
    "with a known diagnosis of",
    "diagnosed with",
    "previously diagnosed with",
    "pmh:",
    "pmh of",
    "past medical history",
    "past history",
    "medical history",
    "significant for",  # almost always the tail of "PMH significant for ..."
    "chronic",
    "patient has known",
    "longstanding",
    "long-standing",
    "long standing",
    "established",
    "underlying",
    "pre-existing",
    "preexisting",
)

# Section headers (markdown ### or ALL-CAPS) under which any dx mention is
# considered historical context, not a leak. Matched against the most recent
# section header preceding the dx match.
_HISTORY_SECTION_HEADERS: tuple[str, ...] = (
    "past history",
    "past medical history",
    "medical history",
    "social history",
    "family history",
    "surgical history",
    "allergies",
    "current medications",
    "medications",
    "review of systems",
    "ros",
    "pmh",
)

# Matches a markdown ("### Foo") or ALL-CAPS-line section header.
_SECTION_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(?:#{2,4}\s*([^\n]+?)|([A-Z][A-Z &/\-]{2,40}))\s*:?\s*(?=\n|$)",
)

# How many characters of left context we look at when classifying a match.
_HISTORY_CONTEXT_CHARS = 50


def _nearest_preceding_section(text: str, match_start: int) -> str | None:
    """Return the title of the markdown / all-caps section header most
    recently opened before ``match_start`` (lower-cased, stripped). None
    if no header precedes the match.
    """
    last: str | None = None
    for m in _SECTION_HEADER_RE.finditer(text, 0, match_start):
        title = (m.group(1) or m.group(2) or "").strip().lower()
        if title:
            last = title
    return last


def _is_in_history_context(text: str, match_start: int) -> bool:
    """Return True iff the working-dx match at ``match_start`` is in a
    historical / chronic-comorbidity context.

    Two complementary signals:
      1. A short left-window (~50 chars) contains a history/chronic marker.
      2. The match falls under a "Past History" / "Medical History" /
         "Allergies" / etc. section header.
    """
    left_start = max(0, match_start - _HISTORY_CONTEXT_CHARS)
    left = text[left_start:match_start].lower()
    for marker in _HISTORY_MARKERS:
        if marker in left:
            return True

    section = _nearest_preceding_section(text, match_start)
    if section is not None:
        for hdr in _HISTORY_SECTION_HEADERS:
            if hdr in section:
                return True
    return False


def layer1_audit(enc: Encounter) -> Layer1Report:
    """Audit a single encounter for in-text leakage.

    For the working-diagnosis check (L1a) we now distinguish two cases:
      * the dx string is preceded by a history/chronic marker → recorded in
        ``dx_in_history`` and NOT flagged as a leak;
      * otherwise the encounter is flagged with ``leaked_dx = True``.
    """
    rep = Layer1Report(case_id=enc.case_id)

    # L1a — working diagnosis must not appear verbatim in pre-decision text,
    # *unless* it is clearly framed as historical / chronic comorbidity.
    # Skip if the surface form is too short / generic (single token of <=2
    # chars or pure digits) — those are extraction artifacts, not real
    # diagnoses, and would otherwise produce noisy false positives.
    if enc.working_diagnosis:
        wd = enc.working_diagnosis.strip()
        if wd and not (len(wd) <= 2 or wd.isdigit()):
            pattern = r"\b" + re.escape(wd) + r"\b"
            for m in re.finditer(pattern, enc.x, flags=re.IGNORECASE):
                if _is_in_history_context(enc.x, m.start()):
                    # Capture a short context window for transparency.
                    ctx_start = max(0, m.start() - _HISTORY_CONTEXT_CHARS)
                    ctx_end = min(len(enc.x), m.end() + 10)
                    rep.dx_in_history.append(enc.x[ctx_start:ctx_end])
                else:
                    rep.leaked_dx = True
                    # One unambiguous leak is enough; we still keep iterating
                    # so that any history-context matches further on are
                    # captured for the report, but we don't break.

    # L1b — ICD-10 codes must not appear
    icd_re = re.compile(r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?\b")
    found_icd = icd_re.findall(enc.x)
    rep.leaked_icd = sorted(set(found_icd))

    # L1c — labs (heuristic; high-precision check)
    lab_re = re.compile(
        r"\b\d+(?:\.\d+)?\s*(mg/dL|mmol/L|g/dL|/μL|U/L|mEq/L|ng/mL|IU/L)\b",
        re.IGNORECASE,
    )
    rep.leaked_labs = sorted({m.group(0) for m in lab_re.finditer(enc.x)})

    return rep


def audit_dataset_layer1(encounters: Iterable[Encounter]) -> list[Layer1Report]:
    return [layer1_audit(e) for e in encounters]


# ---------------------------------------------------------------------------
# Layer 2 — training-set decontamination
# ---------------------------------------------------------------------------


def ngrams(text: str, n: int = 8) -> set[str]:
    """Tokenize on whitespace + lowercase, return set of n-grams."""
    toks = text.lower().split()
    if len(toks) < n:
        return set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


@dataclass
class Layer2Report:
    case_id: str
    ngram_match_count: int = 0
    ngram_match_examples: list[str] = field(default_factory=list)
    cosine_max: float = 0.0
    cosine_max_corpus_id: str | None = None
    flagged: bool = False


def layer2_ngram_overlap(
    enc_text: str,
    corpus_ngrams: set[str],
    n: int = 8,
    max_examples: int = 3,
) -> tuple[int, list[str]]:
    """Count enc n-grams present in corpus_ngrams; return (count, examples)."""
    enc_ngrams = ngrams(enc_text, n=n)
    matches = enc_ngrams & corpus_ngrams
    examples = list(matches)[:max_examples]
    return len(matches), examples


def layer2_audit(
    enc: Encounter,
    corpus_ngrams: dict[str, set[str]] | None = None,
    semantic_similarity_fn=None,
    n: int = 8,
    cosine_threshold: float = 0.90,
) -> Layer2Report:
    """Audit a single encounter against named training corpora.

    Args:
        enc: encounter to audit
        corpus_ngrams: {corpus_id: set of n-grams} pre-computed
        semantic_similarity_fn: callable(enc_text, corpus_id) -> float in [0,1]
            (optional; if None, only n-gram check runs)
        n: n-gram size
        cosine_threshold: flag encounter if any cosine ≥ threshold
    """
    rep = Layer2Report(case_id=enc.case_id)

    text = enc.x + "\n" + enc.a_clinician

    if corpus_ngrams:
        for cid, cng in corpus_ngrams.items():
            count, examples = layer2_ngram_overlap(text, cng, n=n)
            if count > rep.ngram_match_count:
                rep.ngram_match_count = count
                rep.ngram_match_examples = examples

    if semantic_similarity_fn is not None and corpus_ngrams:
        for cid in corpus_ngrams:
            cos = float(semantic_similarity_fn(text, cid))
            if cos > rep.cosine_max:
                rep.cosine_max = cos
                rep.cosine_max_corpus_id = cid

    rep.flagged = (rep.ngram_match_count >= 1) or (rep.cosine_max >= cosine_threshold)
    return rep


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


@dataclass
class AuditSummary:
    n_encounters: int
    n_layer1_clean: int
    n_layer2_flagged: int
    layer1_failures: dict[str, int] = field(default_factory=dict)
    # Number of encounters where the working dx appeared only in a clearly
    # historical/chronic context (not counted as leaks).
    n_dx_in_history: int = 0


def summarize(
    layer1_reports: list[Layer1Report],
    layer2_reports: list[Layer2Report] | None = None,
) -> AuditSummary:
    failures: Counter[str] = Counter()
    n_clean = 0
    n_dx_in_history = 0
    for r in layer1_reports:
        if r.dx_in_history:
            n_dx_in_history += 1
        if r.is_clean:
            n_clean += 1
        else:
            if r.leaked_dx:
                failures["leaked_dx"] += 1
            if r.leaked_icd:
                failures["leaked_icd"] += 1
            if r.leaked_labs:
                failures["leaked_labs"] += 1

    n_flagged = sum(1 for r in (layer2_reports or []) if r.flagged)
    return AuditSummary(
        n_encounters=len(layer1_reports),
        n_layer1_clean=n_clean,
        n_layer2_flagged=n_flagged,
        layer1_failures=dict(failures),
        n_dx_in_history=n_dx_in_history,
    )
