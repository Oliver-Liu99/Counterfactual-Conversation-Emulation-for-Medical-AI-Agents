"""Active calibration sampling (U5).

Replaces stratified random sampling for the 100-case human-rated calibration
set. The acquisition criterion approximates mutual information between the
judge score and the (latent) human label, given x:

    I(y_human ; theta_judge | x)
        ≈ disagreement(judge_arms) * judge_uncertainty(case)

Intuition:
- Cases where strong/weak agents and clinician get *very different* scores
  carry the most signal for distinguishing rubric reliability.
- Cases where the judge's Beta posterior is broad (low concentration)
  also carry more information per labelled instance.

Stratified selection then enforces (a) at least `min_chapters` distinct
ICD chapters and (b) approximate clinician/agent balance for double-blind
labelling.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from ccema.judges.beta_score import MultiAxisScore


# ---------------------------------------------------------------------------
# Per-case record
# ---------------------------------------------------------------------------


@dataclass
class CaseRecord:
    """One case for active sampling.

    arm_scores maps arm_id -> MultiAxisScore (from a single judge run).
    metadata carries icd_chapter, age_band, etc., used for stratification.
    """

    case_id: str
    x: str
    arm_texts: dict[str, str]  # arm_id -> a (the candidate output to grade)
    arm_scores: dict[str, MultiAxisScore]
    metadata: dict


# ---------------------------------------------------------------------------
# Acquisition scoring
# ---------------------------------------------------------------------------


def beta_entropy(alpha: float, beta: float) -> float:
    """Differential entropy of Beta(alpha, beta) (nats).

    H = log B(a, b) - (a-1) psi(a) - (b-1) psi(b) + (a+b-2) psi(a+b)
    We approximate via Gauss series for psi to avoid scipy.
    """
    return _log_beta(alpha, beta) - (alpha - 1) * _digamma(alpha) - (beta - 1) * _digamma(beta) + (alpha + beta - 2) * _digamma(alpha + beta)


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _digamma(x: float) -> float:
    """Asymptotic series for psi(x); uses recurrence for x < 6."""
    result = 0.0
    while x < 6:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / (x * x)
    result += math.log(x) - 0.5 / x
    result -= inv * (1 / 12 - inv * (1 / 120 - inv / 252))
    return result


def case_uncertainty(case: CaseRecord) -> float:
    """Mean entropy across (arm, axis) Beta posteriors. Higher = less certain."""
    n_terms = 0
    total = 0.0
    for ms in case.arm_scores.values():
        for beta in ms.axis_scores.values():
            total += beta_entropy(beta.alpha, beta.beta)
            n_terms += 1
    return total / max(1, n_terms)


def case_arm_disagreement(case: CaseRecord) -> float:
    """Range of overall_mean across arms (max - min). Higher = more contested."""
    means = [ms.overall_mean for ms in case.arm_scores.values()]
    if len(means) < 2:
        return 0.0
    return max(means) - min(means)


def acquisition_score(case: CaseRecord, *, w_disagree: float = 1.0, w_uncertain: float = 1.0) -> float:
    """BALD-style acquisition: arm disagreement × judge uncertainty.

    Both terms are positive; cases with both = 0 cannot be informative.
    """
    return w_disagree * case_arm_disagreement(case) + w_uncertain * case_uncertainty(case)


# ---------------------------------------------------------------------------
# Stratified selection
# ---------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    n_total: int = 100
    min_chapters: int = 8
    stratify_keys: tuple[str, ...] = ("icd_chapter",)
    balance_arms: bool = True
    seed: int = 42
    w_disagree: float = 1.0
    w_uncertain: float = 1.0


def select_calibration_set(
    cases: Sequence[CaseRecord],
    cfg: SamplingConfig | None = None,
) -> list[CaseRecord]:
    """Pick `cfg.n_total` cases by acquisition score within stratification.

    Algorithm:
        1. Score every case.
        2. Bucket by stratify_keys.
        3. Round-robin through buckets in descending acquisition order until
           n_total reached or all buckets exhausted.
        4. Verify min_chapters; if not met, draft additional cases from
           under-represented chapters even at lower acquisition score.
        5. If balance_arms, ensure roughly equal counts per arm in the
           returned set (greedy swap).
    """
    cfg = cfg or SamplingConfig()
    rng = random.Random(cfg.seed)

    scored = [(c, acquisition_score(c, w_disagree=cfg.w_disagree, w_uncertain=cfg.w_uncertain))
              for c in cases]

    # Bucket
    buckets: dict[tuple, list[tuple[CaseRecord, float]]] = defaultdict(list)
    for c, s in scored:
        key = tuple(c.metadata.get(k, "_unk") for k in cfg.stratify_keys)
        buckets[key].append((c, s))
    for key in buckets:
        buckets[key].sort(key=lambda t: -t[1])

    # Round-robin
    selected: list[CaseRecord] = []
    bucket_iters = {k: iter(v) for k, v in buckets.items()}
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)

    while len(selected) < cfg.n_total:
        progress = False
        for key in bucket_keys:
            if len(selected) >= cfg.n_total:
                break
            try:
                c, _ = next(bucket_iters[key])
                selected.append(c)
                progress = True
            except StopIteration:
                continue
        if not progress:
            break

    # Enforce min_chapters
    chapters = {c.metadata.get("icd_chapter", "_unk") for c in selected}
    if len(chapters) < cfg.min_chapters:
        # Pull additional cases from missing chapters at any acquisition score
        missing = [k for k in buckets if k[0] not in chapters]  # icd_chapter is first key by convention
        rng.shuffle(missing)
        for key in missing:
            for c, _ in buckets[key]:
                if c not in selected:
                    selected.append(c)
                    chapters.add(c.metadata.get("icd_chapter", "_unk"))
                    break
            if len(chapters) >= cfg.min_chapters:
                break

    # Balance arms via greedy swap
    if cfg.balance_arms and selected:
        selected = _balance_arms(selected, cases, cfg.n_total, rng)

    # Trim to n_total in case we over-pulled for chapters
    return selected[: cfg.n_total]


def _balance_arms(
    selected: list[CaseRecord],
    pool: Sequence[CaseRecord],
    target_n: int,
    rng: random.Random,
) -> list[CaseRecord]:
    """Roughly equalize per-arm counts among selected cases.

    Each case's "primary arm" is the arm whose mean score is most extreme
    (farthest from 0.5) — that determines whether the case is "agent-favoured"
    or "clinician-favoured" for double-blind balancing.
    """

    def primary_arm(c: CaseRecord) -> str:
        if not c.arm_scores:
            return "_none"
        return max(c.arm_scores.items(), key=lambda kv: abs(kv[1].overall_mean - 0.5))[0]

    counts: dict[str, int] = defaultdict(int)
    for c in selected:
        counts[primary_arm(c)] += 1
    if not counts or len(counts) <= 1:
        return selected

    avg = target_n / len(counts)
    over = [a for a, n in counts.items() if n > math.ceil(avg)]
    under = [a for a, n in counts.items() if n < math.floor(avg)]

    selected_set = {c.case_id for c in selected}
    for under_arm in under:
        # Find pool cases whose primary arm == under_arm and not yet selected
        candidates = [c for c in pool if c.case_id not in selected_set and primary_arm(c) == under_arm]
        rng.shuffle(candidates)
        for c in candidates:
            # Try to remove a case from `over` to make room
            for over_arm in over:
                idx = next(
                    (i for i, sc in enumerate(selected) if primary_arm(sc) == over_arm),
                    None,
                )
                if idx is not None:
                    removed = selected.pop(idx)
                    selected.append(c)
                    selected_set.discard(removed.case_id)
                    selected_set.add(c.case_id)
                    counts[over_arm] -= 1
                    counts[under_arm] += 1
                    if counts[over_arm] <= math.ceil(avg):
                        over.remove(over_arm)
                    if counts[under_arm] >= math.floor(avg):
                        break
            if counts[under_arm] >= math.floor(avg):
                break

    return selected
