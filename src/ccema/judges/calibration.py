"""Calibration metrics for judge vs human ratings.

Implements the four-metric calibration gate from D001 / Step 4:
- Spearman ρ (main gate ≥ 0.70)
- Quadratic-weighted Cohen's κ (≥ 0.40)
- Intraclass Correlation Coefficient ICC(2,1) (≥ 0.75)
- Bland-Altman bias (|mean diff| ≤ 0.05)

All functions take parallel iterables of floats in [0, 1] and return
floats. Pure-Python fallbacks are provided for environments without
scipy/statsmodels so unit tests run anywhere.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Spearman ρ
# ---------------------------------------------------------------------------


def _ranks(xs: Sequence[float]) -> list[float]:
    """Average ranks (handles ties)."""
    n = len(xs)
    indexed = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[indexed[j + 1]] == xs[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based ranks
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("a and b must have equal length")
    if len(a) < 2:
        return float("nan")
    ra = _ranks(a)
    rb = _ranks(b)
    return _pearson(ra, rb)


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    num = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    den_a = math.sqrt(sum((a[i] - mean_a) ** 2 for i in range(n)))
    den_b = math.sqrt(sum((b[i] - mean_b) ** 2 for i in range(n)))
    if den_a == 0 or den_b == 0:
        return float("nan")
    return num / (den_a * den_b)


pearson_r = _pearson


# ---------------------------------------------------------------------------
# Quadratic-weighted Cohen's κ on discretized scores
# ---------------------------------------------------------------------------


def discretize(xs: Sequence[float], n_bins: int = 5) -> list[int]:
    """Map [0, 1] floats into n_bins equal-width buckets {0, ..., n_bins-1}."""
    out: list[int] = []
    for x in xs:
        idx = min(n_bins - 1, max(0, int(math.floor(x * n_bins))))
        out.append(idx)
    return out


def quadratic_weighted_kappa(
    a: Sequence[float],
    b: Sequence[float],
    n_bins: int = 5,
) -> float:
    if len(a) != len(b):
        raise ValueError("a and b must have equal length")
    if len(a) < 2:
        return float("nan")

    da = discretize(a, n_bins=n_bins)
    db = discretize(b, n_bins=n_bins)
    n = len(a)

    # Confusion matrix
    O = [[0.0] * n_bins for _ in range(n_bins)]
    for x, y in zip(da, db):
        O[x][y] += 1

    # Marginal histograms
    hist_a = [sum(O[i][j] for j in range(n_bins)) for i in range(n_bins)]
    hist_b = [sum(O[i][j] for i in range(n_bins)) for j in range(n_bins)]

    # Expected matrix under independence
    E = [[hist_a[i] * hist_b[j] / n for j in range(n_bins)] for i in range(n_bins)]

    # Quadratic weights, normalized
    norm = (n_bins - 1) ** 2
    W = [[((i - j) ** 2) / norm for j in range(n_bins)] for i in range(n_bins)]

    num = sum(W[i][j] * O[i][j] for i in range(n_bins) for j in range(n_bins))
    den = sum(W[i][j] * E[i][j] for i in range(n_bins) for j in range(n_bins))
    if den == 0:
        return float("nan")
    return 1.0 - num / den


# ---------------------------------------------------------------------------
# ICC(2,1) — Shrout & Fleiss two-way random, single rater absolute agreement
# ---------------------------------------------------------------------------


def icc_2_1(judge: Sequence[float], human: Sequence[float]) -> float:
    """ICC(2,1) for two raters on n subjects."""
    if len(judge) != len(human):
        raise ValueError("judge and human must have equal length")
    n = len(judge)
    if n < 2:
        return float("nan")
    k = 2  # raters

    # Per-subject means and grand mean
    subj_means = [(judge[i] + human[i]) / 2 for i in range(n)]
    grand_mean = sum(subj_means) / n
    rater_means = [sum(judge) / n, sum(human) / n]

    # Sum of squares
    ss_subj = k * sum((m - grand_mean) ** 2 for m in subj_means)
    ss_rater = n * sum((rm - grand_mean) ** 2 for rm in rater_means)
    ss_total = sum((judge[i] - grand_mean) ** 2 + (human[i] - grand_mean) ** 2 for i in range(n))
    ss_err = ss_total - ss_subj - ss_rater

    ms_subj = ss_subj / (n - 1) if n > 1 else 0.0
    ms_rater = ss_rater / (k - 1) if k > 1 else 0.0
    df_err = (n - 1) * (k - 1)
    ms_err = ss_err / df_err if df_err > 0 else 0.0

    denom = ms_subj + (k - 1) * ms_err + (k * (ms_rater - ms_err)) / n
    if denom == 0:
        return float("nan")
    return (ms_subj - ms_err) / denom


# ---------------------------------------------------------------------------
# Bland-Altman bias
# ---------------------------------------------------------------------------


@dataclass
class BlandAltman:
    bias: float
    limits_of_agreement: tuple[float, float]
    sd_diff: float


def bland_altman(judge: Sequence[float], human: Sequence[float]) -> BlandAltman:
    if len(judge) != len(human):
        raise ValueError("judge and human must have equal length")
    n = len(judge)
    diffs = [judge[i] - human[i] for i in range(n)]
    bias = sum(diffs) / n
    var = sum((d - bias) ** 2 for d in diffs) / max(1, n - 1)
    sd = math.sqrt(var)
    return BlandAltman(
        bias=bias,
        limits_of_agreement=(bias - 1.96 * sd, bias + 1.96 * sd),
        sd_diff=sd,
    )


# ---------------------------------------------------------------------------
# Composite report against locked-in gates
# ---------------------------------------------------------------------------


@dataclass
class CalibrationGates:
    spearman_min: float = 0.70
    kappa_min: float = 0.40
    icc_min: float = 0.75
    bland_altman_abs_max: float = 0.05


@dataclass
class CalibrationReport:
    n: int
    spearman_rho: float
    pearson_r: float
    kappa_quadratic: float
    icc_2_1: float
    bland_altman_bias: float
    bland_altman_loa: tuple[float, float]
    gates: CalibrationGates
    pass_spearman: bool
    pass_kappa: bool
    pass_icc: bool
    pass_bias: bool

    @property
    def all_pass(self) -> bool:
        return self.pass_spearman and self.pass_kappa and self.pass_icc and self.pass_bias

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "spearman_rho": round(self.spearman_rho, 4),
            "pearson_r": round(self.pearson_r, 4),
            "quadratic_weighted_kappa": round(self.kappa_quadratic, 4),
            "icc_2_1": round(self.icc_2_1, 4),
            "bland_altman_bias": round(self.bland_altman_bias, 4),
            "bland_altman_loa": [round(v, 4) for v in self.bland_altman_loa],
            "gates": {
                "spearman_min": self.gates.spearman_min,
                "kappa_min": self.gates.kappa_min,
                "icc_min": self.gates.icc_min,
                "bland_altman_abs_max": self.gates.bland_altman_abs_max,
            },
            "pass": {
                "spearman": self.pass_spearman,
                "kappa": self.pass_kappa,
                "icc": self.pass_icc,
                "bias": self.pass_bias,
                "all": self.all_pass,
            },
        }


def calibration_report(
    judge: Sequence[float],
    human: Sequence[float],
    gates: CalibrationGates | None = None,
    n_bins: int = 5,
) -> CalibrationReport:
    g = gates or CalibrationGates()
    rho = spearman_rho(judge, human)
    r = _pearson(judge, human)
    k = quadratic_weighted_kappa(judge, human, n_bins=n_bins)
    icc = icc_2_1(judge, human)
    ba = bland_altman(judge, human)

    return CalibrationReport(
        n=len(judge),
        spearman_rho=rho,
        pearson_r=r,
        kappa_quadratic=k,
        icc_2_1=icc,
        bland_altman_bias=ba.bias,
        bland_altman_loa=ba.limits_of_agreement,
        gates=g,
        pass_spearman=(rho >= g.spearman_min),
        pass_kappa=(k >= g.kappa_min),
        pass_icc=(icc >= g.icc_min),
        pass_bias=(abs(ba.bias) <= g.bland_altman_abs_max),
    )
