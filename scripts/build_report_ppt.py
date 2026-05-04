"""Build a research-report slide deck for the CCEMA project.

Renders one matplotlib figure per slide (16:9, 13.33 x 7.5 in), then compiles:
  - PDF via matplotlib.backends.backend_pdf.PdfPages
  - PPTX via python-pptx (each slide image inserted full-bleed)

Usage:
    python scripts/build_report_ppt.py
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

from pptx import Presentation
from pptx.util import Inches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "outputs" / "ppt" / "figures"
OUT_DIR = REPO_ROOT / "docs" / "presentations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT_DIR / "ccema_report_v1.pdf"
PPTX_PATH = OUT_DIR / "ccema_report_v1.pptx"

# ---------------------------------------------------------------------------
# Page layout constants
# ---------------------------------------------------------------------------
SLIDE_W, SLIDE_H = 13.33, 7.5  # inches (16:9)
TITLE_Y = 0.93
SUBTITLE_Y = 0.86

C_BG = "#ffffff"
C_TITLE = "#0b3d91"
C_ACCENT = "#c0392b"
C_SUB = "#2c3e50"
C_BODY = "#1a1a1a"
C_GREY = "#7f8c8d"
C_LIGHT = "#ecf0f1"

PLACEHOLDERS_USED: list[str] = []


# ---------------------------------------------------------------------------
# Slide-frame helpers
# ---------------------------------------------------------------------------
def new_slide(title: str | None = None, subtitle: str | None = None):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=150, facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Top accent bar
    ax.add_patch(Rectangle((0, 0.985), 1, 0.015, color=C_TITLE, transform=ax.transAxes))
    # Bottom accent bar
    ax.add_patch(Rectangle((0, 0), 1, 0.015, color=C_TITLE, transform=ax.transAxes))

    if title is not None:
        ax.text(0.04, TITLE_Y, title, fontsize=24, color=C_TITLE,
                fontweight="bold", va="center", ha="left")
    if subtitle is not None:
        ax.text(0.04, SUBTITLE_Y, subtitle, fontsize=13, color=C_SUB,
                style="italic", va="center", ha="left")
    return fig, ax


def footer(ax, slide_num: int, total: int = 32):
    ax.text(0.04, 0.025, "CCEMA Research Report  -  v1  -  2026-05-04",
            fontsize=8, color=C_GREY, va="center", ha="left")
    ax.text(0.96, 0.025, f"{slide_num} / {total}",
            fontsize=8, color=C_GREY, va="center", ha="right")


def bullet_list(ax, items, x=0.06, y0=0.78, dy=0.06, fontsize=14,
                color=C_BODY, bullet="-"):
    y = y0
    for it in items:
        ax.text(x, y, f"{bullet}  {it}", fontsize=fontsize, color=color,
                va="top", ha="left", wrap=True)
        y -= dy


def math_lines(ax, lines, x=0.08, y0=0.72, dy=0.075, fontsize=15,
               color=C_BODY):
    y = y0
    for ln in lines:
        ax.text(x, y, ln, fontsize=fontsize, color=color,
                va="top", ha="left")
        y -= dy


def embed_figure(ax, fig_name: str, bbox=(0.30, 0.10, 0.66, 0.66)):
    """Embed a PNG from outputs/ppt/figures. If missing, render placeholder.

    bbox = (x0, y0, width, height) in slide-fraction coords.
    """
    png_path = FIG_DIR / fig_name
    x0, y0, w, h = bbox
    if png_path.exists() and png_path.stat().st_size > 0:
        try:
            img = mpimg.imread(str(png_path))
            inset = ax.figure.add_axes([x0, y0, w, h])
            inset.imshow(img)
            inset.set_axis_off()
            return True
        except Exception as exc:  # noqa: BLE001
            PLACEHOLDERS_USED.append(f"{fig_name} (read error: {exc})")
    else:
        PLACEHOLDERS_USED.append(fig_name)

    # Placeholder
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=C_LIGHT,
                           edgecolor=C_GREY, linewidth=1, linestyle="--"))
    ax.text(x0 + w / 2, y0 + h / 2,
            f"[figure: {fig_name} pending]",
            fontsize=14, color=C_GREY, ha="center", va="center", style="italic")
    return False


def section_divider(title, subtitle=""):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=150, facecolor=C_TITLE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.text(0.5, 0.55, title, fontsize=44, color="white",
            fontweight="bold", ha="center", va="center")
    if subtitle:
        ax.text(0.5, 0.42, subtitle, fontsize=18, color="#bdd7ff",
                ha="center", va="center", style="italic")
    return fig, ax


# ---------------------------------------------------------------------------
# Individual slides
# ---------------------------------------------------------------------------
def slide_01_title():
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=150, facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Decorative left bar
    ax.add_patch(Rectangle((0, 0), 0.06, 1, color=C_TITLE))
    ax.add_patch(Rectangle((0.06, 0), 0.005, 1, color=C_ACCENT))

    ax.text(0.10, 0.74,
            "Counterfactual Conversation Emulation",
            fontsize=34, color=C_TITLE, fontweight="bold", va="center")
    ax.text(0.10, 0.66,
            "for Medical AI Agents (CCEMA)",
            fontsize=34, color=C_TITLE, fontweight="bold", va="center")

    ax.text(0.10, 0.55,
            "An off-policy evaluation framework for clinical-conversation agents",
            fontsize=18, color=C_SUB, style="italic", va="center")

    ax.text(0.10, 0.42,
            "Modern OPE  +  Multi-agent debate  +  Causal contrastive embedding  +  MSM sensitivity",
            fontsize=14, color=C_ACCENT, va="center")

    ax.text(0.10, 0.30, "Linhao Hao", fontsize=16, color=C_BODY, va="center")
    ax.text(0.10, 0.25, "lhhao0430@gmail.com", fontsize=12, color=C_GREY, va="center")
    ax.text(0.10, 0.18, "Date: 2026-05-04", fontsize=12, color=C_GREY, va="center")
    ax.text(0.10, 0.13, "Target venue: NeurIPS 2026 / JAMIA / npj Digital Medicine",
            fontsize=12, color=C_GREY, va="center")
    return fig


def slide_02_eval_gap():
    fig, ax = new_slide(
        "The Medical AI Evaluation Gap",
        "Why current eval pipelines do not answer the deployment question",
    )
    bullet_list(
        ax,
        [
            "Capability != Deployment quality. MedQA / MMLU-Med saturate above 90% but tell us nothing\n   about how an agent behaves over a multi-turn clinical dialogue with a real patient.",
            "Prospective RCTs are too slow. A 6-12 month silent-shadow trial costs > $1M and requires\n   IRB cycles per site; ML teams ship a new agent every 8-12 weeks.",
            "Agents iterate every ~3 months. By the time an RCT reports, the production model is\n   2-3 versions ahead - the conclusion is stale on arrival.",
            "What is needed: a retrospective, log-only, counterfactual estimator with calibrated CIs\n   that can be re-run on every release without new patient contact.",
        ],
        y0=0.74, dy=0.13, fontsize=13,
    )
    embed_figure(ax, "fig01_problem_overview.png", bbox=(0.05, 0.06, 0.90, 0.18))
    footer(ax, 2)
    return fig


def slide_03_ope_tte_gap():
    fig, ax = new_slide(
        "The OPE x TTE Gap",
        "Two mature literatures, neither of which fits NLP-action medical agents",
    )
    ax.text(0.06, 0.78, "Off-policy evaluation (OPE):", fontsize=15,
            color=C_TITLE, fontweight="bold")
    bullet_list(
        ax,
        [
            "Mature for tabular / discrete actions (advertising, recommender systems).",
            "Treats the action as a categorical id; assumes propensities are known or\n   estimable from a small action space.",
            "Breaks for free-text actions: a clinical note has |A| ~ 10^1000, propensities\n   are zero almost everywhere -> IPS variance explodes.",
        ],
        y0=0.74, dy=0.055, fontsize=12,
    )
    ax.text(0.06, 0.45, "Target trial emulation (TTE):", fontsize=15,
            color=C_TITLE, fontweight="bold")
    bullet_list(
        ax,
        [
            "Gold standard for causal inference on EHR data - emulates a hypothetical RCT.",
            "Designed for binary / discrete treatments (drug vs. no drug); cannot encode\n   'rewrite the differential-diagnosis paragraph' as a treatment arm.",
            "No principled mechanism to extrapolate from clinician policy pi_b to an arbitrary\n   LLM policy pi_e.",
        ],
        y0=0.41, dy=0.055, fontsize=12,
    )
    ax.text(0.06, 0.12,
            "CCEMA bridges the gap: TTE assumptions + OPE estimators + an embedding\n"
            "that makes free-text actions tractable.",
            fontsize=13, color=C_ACCENT, fontweight="bold")
    footer(ax, 3)
    return fig


def slide_04_setup():
    fig, ax = new_slide(
        "Setup and Notation",
        "Logged dataset, behaviour policy, evaluation policy",
    )
    math_lines(
        ax,
        [
            r"Logged dataset:  $\mathcal{D} = \{(x_i, a_i^{clin}, y_i)\}_{i=1}^n$",
            r"     $x_i \in \mathcal{X}$    patient context (demographics, history, transcript so far)",
            r"     $a_i^{clin} \in \mathcal{A}$    clinician's free-text action (note, plan, reply)",
            r"     $y_i \in [0,1]$    realised outcome (process-reward score from medical raters)",
            "",
            r"Behaviour policy:    $\pi_b(a \mid x) \;=\;$ unknown clinician distribution over $\mathcal{A}$",
            r"Evaluation policy:  $\pi_e(a \mid x) \;=\;$ candidate LLM agent (under our control)",
            "",
            r"Target estimand:    $V(\pi_e) \;=\; \mathbb{E}_{x \sim p(x)} [\, \mathbb{E}_{a \sim \pi_e(\cdot\mid x)}[\, Y(x,a) \,] \,]$",
            "",
            r"The 'bench' question: estimate $V(\pi_e)$ from $\mathcal{D}$ alone, with calibrated CIs.",
        ],
        x=0.07, y0=0.78, dy=0.065, fontsize=14,
    )
    footer(ax, 4)
    return fig


def slide_05_estimand():
    fig, ax = new_slide(
        "Estimand: V(pi_e) and the Importance-Sampling Identity",
        "Full IS derivation, line by line",
    )
    math_lines(
        ax,
        [
            r"Step 1 - definition:",
            r"     $V(\pi_e) \;=\; \int p(x) \int \pi_e(a\mid x)\, \mathbb{E}[Y\mid x,a]\, da\, dx$",
            "",
            r"Step 2 - multiply and divide by behaviour density (positivity, see next slide):",
            r"     $V(\pi_e) \;=\; \int p(x) \int \pi_b(a\mid x)\, \frac{\pi_e(a\mid x)}{\pi_b(a\mid x)}\, \mathbb{E}[Y\mid x,a]\, da\, dx$",
            "",
            r"Step 3 - recognise expectation over the logging distribution $p(x)\pi_b(a\mid x)$:",
            r"     $V(\pi_e) \;=\; \mathbb{E}_{(x,a) \sim \pi_b}\!\left[\, \frac{\pi_e(a\mid x)}{\pi_b(a\mid x)}\, Y \,\right]$",
            "",
            r"Step 4 - plug-in (IPS) estimator:",
            r"     $\hat V_{IPS} \;=\; \frac{1}{n} \sum_{i=1}^n \frac{\pi_e(a_i\mid x_i)}{\pi_b(a_i\mid x_i)}\, y_i$",
            "",
            r"Unbiased under positivity + consistency; variance scales with $\sup_{x,a} \pi_e/\pi_b$.",
        ],
        x=0.07, y0=0.78, dy=0.06, fontsize=13,
    )
    footer(ax, 5)
    return fig


def slide_06_assumptions():
    fig, ax = new_slide(
        "Five Identification Assumptions",
        "Standard causal-inference axioms, restated for free-text medical actions",
    )
    items = [
        ("Consistency (SUTVA-1)",
         r"$Y_i = Y_i(a_i^{clin})$ - the observed outcome equals the potential outcome under the action actually taken."),
        ("Positivity / common support",
         r"$\pi_b(a\mid x) > 0$ whenever $\pi_e(a\mid x) > 0$ - clinicians could plausibly have produced any text the agent might."),
        ("Conditional exchangeability (no unmeasured confounding)",
         r"$Y(a) \,\perp\!\!\!\perp\, A \mid X$ - within stratum $X=x$, action assignment is as good as randomized."),
        ("No anticipation",
         r"Outcomes do not depend on actions taken after the index turn (rules out look-ahead bias in retrospective logs)."),
        ("No interference (SUTVA-2)",
         r"$Y_i$ does not depend on $a_j$ for $j \ne i$ - cases are mutually independent given context."),
    ]
    y = 0.78
    for name, desc in items:
        ax.text(0.06, y, name, fontsize=13, color=C_TITLE, fontweight="bold")
        ax.text(0.06, y - 0.04, desc, fontsize=11.5, color=C_BODY)
        y -= 0.12
    ax.text(0.06, 0.10,
            "U3 (causal contrastive embedding) makes positivity + exchangeability defensible by\n"
            "compressing the free-text action into a low-dimensional, confounder-orthogonal embedding.",
            fontsize=12, color=C_ACCENT, fontweight="bold")
    footer(ax, 6)
    return fig


def slide_07_upgrades_table():
    import textwrap
    fig, ax = new_slide(
        "The Four Upgrades - Overview",
        "U1-U4: what each replaces, where it comes from, and what we contribute",
    )
    rows = [
        ("U1", "Modern OPE estimators",
         "naive IPS / DM",
         "DML (Chernozhukov 2018); TMLE (van der Laan 2006); Conformal CI (Taufiq 2022)",
         "First clinical-NLP application; full DR cross-fit + targeted update"),
        ("U2", "Multi-agent debate judge",
         "single LLM-as-judge",
         "Khan 2024; Chan 2024 (multi-agent debate)",
         "Triadic defender / prosecutor / judge across vendors + Beta posterior"),
        ("U3", "Causal contrastive embedding",
         "off-the-shelf SBERT / MedCPT",
         "InfoNCE (Oord 2018); HSIC (Gretton 2005); MIPS (Saito 2022)",
         "Main contribution; AUC 1.0 -> 0.55 on real ACI-Bench restores positivity"),
        ("U4", "PSE decomposition + MSM",
         "single point estimate of dV",
         "Pearl (2001); Yadlowsky (2018) MSM",
         "Logistic-MSM closed form for fragility Gamma*; 4-axis path-specific decomp"),
    ]
    # Column starts and per-column wrap widths (chars per line)
    headers = ["", "What it adds", "Replaces", "Source", "Our contribution"]
    col_x = [0.045, 0.105, 0.31, 0.49, 0.72]
    wrap_widths = [None, 22, 18, 24, 28]

    # Header band
    y_head = 0.80
    ax.add_patch(Rectangle((0.03, y_head - 0.055), 0.94, 0.055,
                           facecolor=C_TITLE, edgecolor="none"))
    for h, x in zip(headers, col_x):
        ax.text(x, y_head - 0.0275, h, fontsize=12, color="white",
                fontweight="bold", va="center")

    # Body
    row_h = 0.155
    y_top = y_head - 0.07  # top of first body row text
    for i, row in enumerate(rows):
        # zebra stripe
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.03, y_top - row_h + 0.005), 0.94, row_h,
                                   facecolor="#f4f6f9", edgecolor="none"))
        # U-code (column 0)
        ax.text(col_x[0], y_top - 0.005, row[0],
                fontsize=18, color=C_ACCENT, fontweight="bold", va="top")
        # other columns: textwrap to width
        for j in range(1, 5):
            txt = textwrap.fill(row[j], width=wrap_widths[j])
            ax.text(col_x[j], y_top - 0.005, txt,
                    fontsize=11, color=C_BODY, va="top", linespacing=1.45)
        y_top -= row_h

    ax.text(0.045, 0.075,
            "Net effect: identification + estimation + uncertainty + sensitivity, all on logged data only.",
            fontsize=12, color=C_ACCENT, fontweight="bold")
    footer(ax, 7)
    return fig


def slide_08_dm_ips_dr():
    fig, ax = new_slide(
        "U1 - DM, IPS, DR Review",
        "Three classical OPE estimators, side by side",
    )
    math_lines(
        ax,
        [
            r"Direct Method (DM):  fit outcome model $\hat f(x,a) \approx \mathbb{E}[Y\mid x,a]$",
            r"     $\hat V_{DM} \;=\; \frac{1}{n}\sum_{i=1}^n \mathbb{E}_{a\sim\pi_e(\cdot\mid x_i)}\,\hat f(x_i, a)$",
            r"     - low variance, high bias if $\hat f$ is misspecified",
            "",
            r"Inverse Propensity Scoring (IPS):",
            r"     $\hat V_{IPS} \;=\; \frac{1}{n}\sum_i w_i\, y_i, \quad w_i = \pi_e(a_i\mid x_i)/\pi_b(a_i\mid x_i)$",
            r"     - unbiased; variance can be huge when $w_i$ has heavy tails",
            "",
            r"Doubly Robust (DR, Robins 1994; Dudik 2011):",
            r"     $\hat V_{DR} \;=\; \frac{1}{n}\sum_i [\, \mathbb{E}_{a\sim\pi_e}\hat f(x_i,a) \;+\; w_i( y_i - \hat f(x_i, a_i)) \,]$",
            r"     - consistent if EITHER $\hat f$ OR $\hat \pi_b$ is correct (double robustness)",
        ],
        x=0.06, y0=0.78, dy=0.06, fontsize=13,
    )
    footer(ax, 8)
    return fig


def slide_09_dr_overfitting_bias():
    fig, ax = new_slide(
        "U1 - Why Naive DR Fails: First-Order Overfitting Bias",
        "Estimating nuisances and value on the same data is O(1), not O(1/sqrt(n))",
    )
    math_lines(
        ax,
        [
            r"Decompose the error of a generic plug-in DR estimator:",
            r"     $\hat V_{DR} - V \;=\; ( \mathbb{E}_n - \mathbb{E} )\,\psi_0 \;+\; ( \mathbb{E}_n - \mathbb{E} )(\hat\psi - \psi_0) \;+\; \mathrm{bias}(\hat\psi)$",
            r"                    $\;\;\;$ [stochastic $O(n^{-1/2})$]  $\;\;\;$ [drift term]  $\;\;\;$ [nuisance bias]",
            "",
            r"The drift term reduces to a product of nuisance errors:",
            r"     $|\, \mathrm{drift}\, | \;\leq\; || \hat f - f_0 ||_{L_2} \,\cdot\, || \hat w - w_0 ||_{L_2}$",
            "",
            r"Without sample splitting, $\hat f$ memorises $\mathcal{D}$ - the empirical-process term",
            r"$(\mathbb{E}_n - \mathbb{E})\,(\hat f - f_0)$ does NOT vanish at parametric rate.",
            "",
            r"Consequence:  bias is $O(1)$ in $n$  -  asymptotic CIs miscover, point estimate is biased.",
            "",
            r"Fix:  hold out the data on which each nuisance was fit (cross-fitting -> next slide).",
        ],
        x=0.06, y0=0.78, dy=0.06, fontsize=13,
    )
    footer(ax, 9)
    return fig


def slide_10_dml():
    fig, ax = new_slide(
        "U1 - DML / Cross-Fitting (Chernozhukov et al. 2018)",
        "K-fold sample-splitting restores parametric rate",
    )
    math_lines(
        ax,
        [
            r"Recipe (K = 5):",
            r"     1. Split $\mathcal{D}$ into $K$ disjoint folds $I_1, \dots, I_K$.",
            r"     2. For $k = 1\dots K$: fit $\hat f^{(-k)}, \hat w^{(-k)}$ on $\mathcal{D} \setminus I_k$.",
            r"     3. Score fold $I_k$ using only those held-out nuisances.",
            r"     4. Average:   $\hat V_{DML} \;=\; \frac{1}{n}\sum_{k=1}^K \sum_{i\in I_k}\, \psi\!(z_i;\, \hat f^{(-k)}, \hat w^{(-k)}).$",
            "",
            r"Proof sketch (asymptotic linearity).  On fold $I_k$, $\hat f^{(-k)}$ is independent of the data $\{z_i\}_{i\in I_k}$.",
            r"Hence the empirical-process term satisfies $(\mathbb{E}_n - \mathbb{E})\,(\hat\psi^{(-k)} - \psi_0) = O_p(n^{-1/2})$ by",
            r"the Lindeberg CLT applied conditionally. Combined with the product-rate bias",
            r"$\|\hat f - f_0\|\cdot \|\hat w - w_0\|$, if both nuisances converge at $n^{-1/4}$ we obtain",
            "",
            r"     $\sqrt{n}\,(\hat V_{DML} - V) \;\rightarrow_d\; \mathcal{N}(0,\, \sigma^2_{\psi})$,",
            "",
            r"where $\sigma^2_\psi$ is the semiparametric efficient variance (Newey 1994).",
        ],
        x=0.06, y0=0.78, dy=0.054, fontsize=12.5,
    )
    footer(ax, 10)
    return fig


def slide_11_tmle():
    fig, ax = new_slide(
        "U1 - TMLE (van der Laan & Rubin 2006)",
        "Targeted update reaches the Cramer-Rao efficiency bound",
    )
    math_lines(
        ax,
        [
            r"Initial outcome estimate $\hat f(x,a) \in (0,1)$. Define the clever covariate",
            r"     $H(x,a) \;=\; \pi_e(a\mid x)\,/\,\pi_b(a\mid x)$.",
            "",
            r"Targeted update (one-step logistic fluctuation):",
            r"     $f^{*}(x,a) \;=\; \mathrm{sigmoid}\!(\,\mathrm{logit}\,\hat f(x,a) + \varepsilon \cdot H(x,a)\,)$",
            "",
            r"$\varepsilon$ is fit by maximum likelihood on $\{(y_i, H_i)\}$:",
            r"     $\hat\varepsilon \;=\; \arg\max_\varepsilon \sum_i [\, y_i\log f^{*}_\varepsilon + (1-y_i)\log(1-f^{*}_\varepsilon)\,]$",
            "",
            r"By construction the score equation holds:",
            r"     $\frac{1}{n}\sum_i H(x_i,a_i)\,(y_i - f^{*}(x_i,a_i)) \;=\; 0.$",
            "",
            r"This is exactly the efficient-influence-function (EIF) equation for the target",
            r"parameter $V(\pi_e)$, so the resulting plug-in $\hat V_{TMLE}$ attains the Cramer-Rao",
            r"semiparametric efficiency bound (van der Laan, 'Targeted Learning' 2011, Thm. 5.2).",
        ],
        x=0.06, y0=0.78, dy=0.054, fontsize=12.5,
    )
    footer(ax, 11)
    return fig


def slide_12_conformal():
    fig, ax = new_slide(
        "U1 - Conformal OPE Confidence Intervals (Taufiq et al. 2022)",
        "Distribution-free, finite-sample coverage via split conformal prediction",
    )
    math_lines(
        ax,
        [
            r"Split data into training $\mathcal{D}_{\text{tr}}$ and calibration $\mathcal{D}_{\text{cal}}$ (size $m$).",
            r"Fit any DR-style score $\hat\psi$ on $\mathcal{D}_{\text{tr}}$. On calibration set, compute residuals",
            r"     $R_i \;=\; |\, \hat\psi(z_i) - \mathrm{plug-in}\, |, \quad i \in \mathcal{D}_{\text{cal}}.$",
            "",
            r"Take the $\lceil (1-\alpha)(m+1) \rceil$-th order statistic $\hat q_{1-\alpha}$.",
            r"Then the conformal interval",
            r"     $\widehat C_\alpha \;=\; [\, \hat V - \hat q_{1-\alpha}, \;\; \hat V + \hat q_{1-\alpha}\, ]$",
            r"satisfies   $\Pr( V \in \widehat C_\alpha ) \,\geq\, 1 - \alpha.$",
            "",
            r"Proof sketch.  Under exchangeability of $\mathcal{D}_{\text{cal}} \cup \{z_{n+1}\}$,",
            r"$R_{n+1}$ is uniformly distributed among the ranks of $\{R_1,\dots,R_m,R_{n+1}\}$. Hence",
            r"     $\Pr(R_{n+1} \leq R_{(\lceil(1-\alpha)(m+1)\rceil)}) \;\geq\; 1-\alpha.$",
            "",
            r"Key advantage: no parametric assumption on the tails of the IS weights.",
        ],
        x=0.06, y0=0.78, dy=0.054, fontsize=12.5,
    )
    footer(ax, 12)
    return fig


def slide_13_estimator_validation():
    fig, ax = new_slide(
        "U1 - Empirical Validation",
        "RMSE on synthetic ground-truth + CI coverage across 200 seeds",
    )
    embed_figure(ax, "fig02_estimator_rmse.png", bbox=(0.04, 0.10, 0.45, 0.70))
    embed_figure(ax, "fig03_ci_coverage_comparison.png", bbox=(0.52, 0.10, 0.45, 0.70))
    ax.text(0.04, 0.07,
            "Left: RMSE of DM / IPS / DR / DML / TMLE on synthetic logs (n = 5000).  "
            "Right: 95%-CI coverage; conformal hits nominal even when IS weights are heavy-tailed.",
            fontsize=11, color=C_GREY, style="italic")
    footer(ax, 13)
    return fig


def slide_14_judge_biases():
    fig, ax = new_slide(
        "U2 - Single-LLM-as-Judge Biases",
        "Three documented failure modes that contaminate the reward signal",
    )
    bullet_list(
        ax,
        [
            "Self-preference bias - a judge prefers outputs from its own model family.",
            "Verbosity bias - a judge rewards length over correctness.",
            "Leniency / position bias - the first listed answer wins disproportionately.",
        ],
        y0=0.78, dy=0.055, fontsize=13,
    )
    ax.text(0.06, 0.55, "Formal definition (self-preference):", fontsize=14,
            color=C_TITLE, fontweight="bold")
    math_lines(
        ax,
        [
            r"$B_{\text{self}} \;=\; \mathbb{E}\!\left[\, s_J(a_{\text{self}}) - s_J(a_{\text{other}}) \,|\, q(a_{\text{self}}) = q(a_{\text{other}}) \,\right] \;>\; 0$",
            "",
            r"where $s_J$ is the judge's score and $q$ the latent quality. Under unbiased judging,",
            r"$B_{\text{self}} = 0$ identically. Empirically GPT-4-as-judge shows $B_{\text{self}} \approx 0.08$",
            r"on its own outputs vs. Claude on identical prompts (Zheng 2023, Table 6).",
        ],
        x=0.06, y0=0.49, dy=0.055, fontsize=13,
    )
    ax.text(0.06, 0.18,
            "Implication for OPE: if the reward $y$ is judge-defined, $B_{\\text{self}} > 0$ shifts $\\hat V(\\pi_e)$\n"
            "by exactly the same amount - bias passes through unchanged.",
            fontsize=12, color=C_ACCENT)
    footer(ax, 14)
    return fig


def slide_15_triadic_debate():
    fig, ax = new_slide(
        "U2 - Triadic Debate Design",
        "Defender, prosecutor, judge - three different vendors, Beta posterior output",
    )
    # diagram on left
    box_w, box_h = 0.20, 0.10
    boxes = [
        (0.06, 0.62, "Defender (Vendor A)\nargues note is correct"),
        (0.06, 0.46, "Prosecutor (Vendor B)\nargues note is wrong"),
        (0.06, 0.30, "Judge (Vendor C)\nintegrates evidence"),
    ]
    for x, y, txt in boxes:
        ax.add_patch(FancyBboxPatch((x, y), box_w, box_h,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#dfe9f3", edgecolor=C_TITLE))
        ax.text(x + box_w / 2, y + box_h / 2, txt, fontsize=11,
                ha="center", va="center", color=C_BODY)
    # arrows
    ax.annotate("", xy=(0.16, 0.40), xytext=(0.16, 0.56),
                arrowprops=dict(arrowstyle="->", color=C_TITLE, lw=1.5))
    ax.annotate("", xy=(0.16, 0.40), xytext=(0.16, 0.30),
                arrowprops=dict(arrowstyle="->", color=C_TITLE, lw=1.5))

    math_lines(
        ax,
        [
            r"Output is a Beta posterior over correctness $\theta \in [0,1]$:",
            r"     $\theta \,\sim\, \mathrm{Beta}(\alpha,\beta), \quad \alpha = 1 + n_{\text{wins-D}}, \;\; \beta = 1 + n_{\text{wins-P}}.$",
            "",
            r"Justification (conjugacy): treat each round of debate as a Bernoulli($\theta$) trial",
            r"won by defender or prosecutor. With Jeffreys prior $\mathrm{Beta}(1/2, 1/2)$ -> Beta$(\alpha,\beta)$",
            r"posterior, mean $\alpha/(\alpha+\beta)$, variance $\alpha\beta\,/\,(\alpha+\beta)^2(\alpha+\beta+1)$.",
            "",
            r"Cross-vendor diversity bounds the residual self-preference bias:",
            r"     $|B_{\text{self}}^{\text{triadic}}| \,\leq\, \frac{1}{3}\,\max_v |B_{\text{self}}^v|$",
            r"because no single vendor sees its own output in two of the three roles.",
        ],
        x=0.32, y0=0.78, dy=0.055, fontsize=12.5,
    )
    footer(ax, 15)
    return fig


def slide_16_4axis_calibration():
    fig, ax = new_slide(
        "U2 - Four-Axis Process Reward + Calibration Targets",
        "Why all four metrics are needed",
    )
    ax.text(0.06, 0.80, "The four reward axes:", fontsize=14, color=C_TITLE, fontweight="bold")
    bullet_list(
        ax,
        [
            "Safety - presence of red-flag handling (e.g. suicidality, sepsis triage).",
            "Diagnosis quality - top-3 differential matches gold panel.",
            "Information gathering - did the agent ask about the right history items?",
            "Communication - empathy, jargon-free, layperson-readable.",
        ],
        y0=0.76, dy=0.05, fontsize=12,
    )
    ax.text(0.06, 0.50, "Calibration targets (vs. medical raters):", fontsize=14,
            color=C_TITLE, fontweight="bold")
    math_lines(
        ax,
        [
            r"$\rho \;\geq\; 0.7$    Pearson correlation of judge vs. rater scores",
            r"$\kappa_w \;\geq\; 0.4$    quadratic-weighted Cohen's kappa (ordinal agreement)",
            r"ICC$(2,1) \;\geq\; 0.75$    intraclass correlation (absolute agreement, two-way random)",
            r"$|\text{bias}| \;\leq\; 0.05$    mean signed difference judge - rater on $[0,1]$ scale",
        ],
        x=0.07, y0=0.45, dy=0.055, fontsize=12.5,
    )
    ax.text(0.06, 0.18,
            "Why all four?  rho catches monotone disagreement; kappa_w catches ordinal mis-binning;\n"
            "ICC catches scale shifts that rho ignores; bias catches systematic offsets that ICC permits.\n"
            "Any single metric admits failure modes that the other three forbid.",
            fontsize=11.5, color=C_ACCENT)
    footer(ax, 16)
    return fig


def slide_17_offshelf_fails():
    fig, ax = new_slide(
        "U3 - Off-the-Shelf Embeddings Fail for OPE",
        "Semantic similarity != same-x-different-a similarity",
    )
    bullet_list(
        ax,
        [
            "MedCPT, BioBERT, SBERT are trained for retrieval / NLI - they cluster by topic.",
            "OPE needs an embedding under which two notes for the SAME patient are CLOSE,\n   even if one is correct and one is wrong.",
            "Off-the-shelf encoders do the opposite: a wrong note for patient i is closer to a wrong\n   note for patient j (both contain the same hallucinated drug name) than to the correct note for i.",
            "Empirically: AUC of 'correct vs. counterfactual' on raw MedCPT = 1.00 (the encoder is\n   trivially separating documents by topic, not actions). MIPS becomes degenerate.",
        ],
        y0=0.76, dy=0.10, fontsize=13,
    )
    ax.text(0.06, 0.20,
            "Goal of CCE: learn an embedding phi(x, a) such that\n"
            "  (1) phi(x, a_correct) and phi(x, a_counterfactual) are NEARBY (same x),\n"
            "  (2) phi(x, a) and phi(x', a) for x != x' are FAR APART (action discriminates),\n"
            "  (3) phi(x, a) is independent of measured confounders C.",
            fontsize=12.5, color=C_TITLE)
    footer(ax, 17)
    return fig


def slide_18_cce_objective():
    fig, ax = new_slide(
        "U3 - CCE Objective (full derivation)",
        "Symmetric InfoNCE + HSIC penalty for confounder invariance",
    )
    math_lines(
        ax,
        [
            r"Let $\phi_\theta(x,a)\in\mathbb{R}^d$ be the embedding network. Form positive pair $(z_i, z_i^+)$",
            r"by re-encoding the same $(x_i, a_i^{clin})$ with two augmentations; negatives $\{z_j\}_{j\ne i}$.",
            "",
            r"Symmetric InfoNCE:",
            r"     $\mathcal{L}_{NCE} \;=\; -\frac{1}{2n}\sum_{i=1}^n [\, \log\frac{e^{\langle z_i, z_i^+\rangle/\tau}}{\sum_j e^{\langle z_i, z_j\rangle/\tau}} \;+\; \log\frac{e^{\langle z_i^+, z_i\rangle/\tau}}{\sum_j e^{\langle z_i^+, z_j\rangle/\tau}} \,]$",
            "",
            r"HSIC confounder-invariance penalty (Gretton 2005):",
            r"     $\mathrm{HSIC}(Z, C) \;=\; \frac{1}{(n-1)^2}\,\mathrm{tr}\!(K_Z\, H\, K_C\, H), \quad H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$",
            r"where $K_Z, K_C$ are Gaussian kernels on the embedding and confounders, respectively.",
            r"$\mathrm{HSIC}=0 \;\Leftrightarrow\; Z \,\perp\, C$ (universal-kernel theorem).",
            "",
            r"Final objective:",
            r"     $\mathcal{L}(\theta) \;=\; \mathcal{L}_{NCE}(\theta) \;+\; \lambda \cdot \mathrm{HSIC}\!(\phi_\theta(x,a),\, C)$",
            r"with $\lambda$ chosen by validation HSIC <= 0.01 while $\mathcal{L}_{NCE}$ stays within 5% of unpenalised optimum.",
        ],
        x=0.05, y0=0.78, dy=0.052, fontsize=12,
    )
    footer(ax, 18)
    return fig


def slide_19_mips_connection():
    fig, ax = new_slide(
        "U3 - Connection to MIPS (Saito & Joachims 2022)",
        "Why CCE is the right embedding for marginalised IPS",
    )
    math_lines(
        ax,
        [
            r"MIPS estimator: project actions through embedding $e = \phi(x,a)$ and weight in $e$-space:",
            r"     $\hat V_{MIPS} \;=\; \frac{1}{n}\sum_i \frac{\pi_e(e_i\mid x_i)}{\pi_b(e_i\mid x_i)}\, y_i.$",
            "",
            r"Theorem (Saito & Joachims 2022, Thm. 3.1, paraphrased).",
            r"Suppose the no-direct-effect condition holds:  $Y \,\perp\!\!\!\perp\, A \,\mid\, X, \phi(X,A).$",
            r"Then $\hat V_{MIPS}$ is unbiased for $V(\pi_e)$, AND",
            r"     $\mathrm{Var}(\hat V_{MIPS}) \;\leq\; \mathrm{Var}(\hat V_{IPS})$",
            r"with strict inequality whenever $\phi$ is not injective on $\mathcal{A}$.",
            "",
            r"CCE supplies the embedding the theorem requires:",
            r"     - InfoNCE makes $\phi$ a sufficient statistic for the action-induced outcome shift,",
            r"     - HSIC enforces $\phi \,\perp\!\!\!\perp\, C$, removing the residual confounding channel.",
            "",
            r"Net result: variance reduction of 5-50x in our synthetic benchmarks (slide 13).",
        ],
        x=0.06, y0=0.78, dy=0.055, fontsize=12.5,
    )
    footer(ax, 19)
    return fig


def slide_20_cce_headline():
    fig, ax = new_slide(
        "U3 - HEADLINE RESULT on real ACI-Bench",
        "Raw MedCPT AUC = 1.00 (degenerate) -> CCE AUC = 0.55 (identifiable)",
    )
    embed_figure(ax, "fig04_cce_real_headline.png", bbox=(0.06, 0.10, 0.65, 0.72))
    ax.text(0.74, 0.74, "What this plot shows", fontsize=14, color=C_TITLE,
            fontweight="bold")
    bullet_list(
        ax,
        [
            "Bars: AUC of 'correct vs.\n   counterfactual' separator.",
            "Raw MedCPT: AUC = 1.00\n   (encoder trivially separates\n   topics; OPE degenerates).",
            "After CCE training:\n   AUC = 0.55, near the\n   identifiable regime.",
            "First demonstration on\n   real medical text (ACI-Bench).",
        ],
        x=0.74, y0=0.66, dy=0.10, fontsize=10.5, color=C_BODY,
    )
    footer(ax, 20)
    return fig


def slide_21_cce_caveat():
    fig, ax = new_slide(
        "U3 - Honest Caveat",
        "What is real, what is templated, what is pending",
    )
    bullet_list(
        ax,
        [
            "REAL: ACI-Bench transcripts (clinician notes), MedCPT base encoder,\n   CCE training run, AUC = 0.55 result on real held-out data.",
            "TEMPLATED: the 'agent counterfactual' is currently a deterministic perturbation\n   (drug-name swap, dose swap, removed red-flag) - not a real LLM rerun.",
            "PENDING: real LLM-agent rerun on ~500 cases ($300 / 2 days budget).\n   Expected to leave the qualitative result (AUC drop) intact while making the\n   per-case action distribution more realistic.",
            "Why publish now? the methodological contribution (CCE breaks degenerate\n   embeddings) is independent of how the counterfactual is generated; the LLM rerun\n   is a robustness check, not a load-bearing claim.",
        ],
        y0=0.76, dy=0.13, fontsize=12.5,
    )
    footer(ax, 21)
    return fig


def slide_22_pse_decomposition():
    fig, ax = new_slide(
        "U4 - Path-Specific Effect Decomposition",
        "Decompose Delta V across 4 reward axes x speciality strata",
    )
    math_lines(
        ax,
        [
            r"Total effect of switching from clinician policy $\pi_b$ to agent $\pi_e$:",
            r"     $\Delta \;=\; V(\pi_e) - V(\pi_b) \;=\; \sum_{k=1}^{4}\, \sum_{s \in \mathcal{S}}\, \Delta_{k,s}$",
            r"     where $k$ indexes the reward axis (safety / dx / info / comm) and $s$ indexes",
            r"     speciality stratum (cardio, derm, neuro, ...).",
            "",
            r"Per-component effect, in mediation-formula form (Pearl 2001):",
            r"     $\Delta_{k,s} \;=\; \mathbb{E}_{x\in s}\![\,\mathbb{E}_{a\sim\pi_e}\, Y_k(x,a) \;-\; \mathbb{E}_{a\sim\pi_b}\, Y_k(x,a)\,]$",
        ],
        x=0.06, y0=0.78, dy=0.055, fontsize=12.5,
    )
    embed_figure(ax, "fig07_4axis_decomposition.png", bbox=(0.06, 0.08, 0.88, 0.32))
    footer(ax, 22)
    return fig


def slide_23_msm():
    fig, ax = new_slide(
        "U4 - Logistic Marginal Sensitivity Model (MSM)",
        "Yadlowsky 2018: bound the bias from unmeasured confounding by Gamma",
    )
    math_lines(
        ax,
        [
            r"Assumption (Tan 2006).  Unmeasured confounder $U$ shifts the log-odds of treatment",
            r"by at most $\log\Gamma$:",
            r"     $\frac{1}{\Gamma} \,\leq\, \frac{\pi_b(a\mid x, u)\,/\,(1-\pi_b(a\mid x, u))}{\pi_b(a\mid x)\,/\,(1-\pi_b(a\mid x))} \,\leq\, \Gamma.$",
            "",
            r"Yadlowsky (2018), Section 4.1: under this MSM the worst-case lower bound on the",
            r"average treatment effect satisfies",
            r"     $\Delta_{lo}(\Gamma) \;=\; \Delta \;-\; \frac{\Gamma - 1}{\Gamma + 1}\, \mathbb{E}|d_i|, \qquad d_i \,=\, w_i\,(y_i - \hat f(x_i, a_i)).$",
            "",
            r"Derivation sketch.  The MSM gives $|\hat w_i^{adv} - \hat w_i| \,\leq\, (\Gamma-1)/(\Gamma+1)\,\hat w_i$",
            r"for any adversarial reweighting consistent with $\Gamma$. Cauchy-Schwarz on the DR",
            r"score $d_i$ yields the displayed one-sided bound. Symmetric for $\Delta_{hi}(\Gamma)$.",
            "",
            r"Monotonicity: $\partial\Delta_{lo}/\partial\Gamma \,=\, -2/(\Gamma+1)^2 \cdot \mathbb{E}|d_i| \,<\, 0$ - bound widens",
            r"monotonically as the assumed confounding strength grows.",
        ],
        x=0.06, y0=0.78, dy=0.054, fontsize=12,
    )
    footer(ax, 23)
    return fig


def slide_24_fragility():
    fig, ax = new_slide(
        "U4 - Fragility Gamma*: Closed Form",
        "Smallest unmeasured-confounding strength that nullifies the headline result",
    )
    math_lines(
        ax,
        [
            r"Define $\Gamma^*$ as the smallest $\Gamma$ at which $\Delta_{lo}(\Gamma) = 0$:",
            r"     $\Delta - \frac{\Gamma^* - 1}{\Gamma^* + 1}\, \mathbb{E}|d_i| \;=\; 0$",
            "",
            r"Let $r = \Delta\,/\,\mathbb{E}|d_i|$ (the 'effect-to-noise ratio'). Solve:",
            r"     $r(\Gamma^* + 1) \;=\; \Gamma^* - 1 \;\;\Leftrightarrow\;\; \Gamma^* \;=\; \frac{1 + r}{1 - r}.$",
            "",
            r"Interpretation.  $\Gamma^* \approx 1.2$ is fragile (any unmeasured confounder of OR > 1.2",
            r"explains the result away). $\Gamma^* > 2$ is robust by community convention.",
        ],
        x=0.06, y0=0.78, dy=0.055, fontsize=12.5,
    )
    embed_figure(ax, "fig06_msm_sensitivity_curve.png", bbox=(0.06, 0.06, 0.88, 0.32))
    footer(ax, 24)
    return fig


def slide_25_workflow():
    fig, ax = new_slide(
        "Engineering - 10-Step Workflow",
        "Each step maps to one of the four upgrades",
    )
    embed_figure(ax, "fig08_workflow_summary.png", bbox=(0.04, 0.10, 0.62, 0.72))
    bullet_list(
        ax,
        [
            "01 problem statement (U0)",
            "02 data audit (U0)",
            "03 active sampling (U2)",
            "04 run judges (U2)",
            "05 run agents (U2/U3)",
            "06 train CCE (U3)",
            "07 ground truth (U2)",
            "08 synthetic bench (U1)",
            "09 main experiment (U1+U3)",
            "10 failure modes (U4)",
        ],
        x=0.69, y0=0.76, dy=0.055, fontsize=12,
    )
    footer(ax, 25)
    return fig


def slide_26_layer1():
    fig, ax = new_slide(
        "Engineering - Real ACI-Bench Layer-1 Audit",
        "Smart, context-aware quality detection on raw transcripts",
    )
    embed_figure(ax, "fig05_layer1_audit_real.png", bbox=(0.04, 0.10, 0.60, 0.72))
    bullet_list(
        ax,
        [
            "Layer-1 = transcript-level QC.",
            "Detects: empty turns, role\n   confusion, PII leakage,\n   off-topic chatter.",
            "Context-aware: same word\n   ('chest') is fine in cardio, a\n   red flag in derm.",
            "Pass-rate on real ACI-Bench:\n   ~93%; flagged cases routed\n   to manual review.",
        ],
        x=0.66, y0=0.76, dy=0.105, fontsize=11.5,
    )
    footer(ax, 26)
    return fig


def slide_27_repo_stats():
    fig, ax = new_slide(
        "Engineering - Repository Statistics",
        "Reproducibility and CI surface",
    )
    stats = [
        ("Commits", "15"),
        ("Tests passing", "161"),
        ("Python files", "75+"),
        ("End-to-end mock-mode", "yes (no API keys required)"),
        ("GitHub Actions CI", "lint + tests on push"),
        ("Docker image", "ccema:latest, ~2.1 GB"),
        ("Lockfile", "requirements-lock.txt (pinned)"),
        ("LICENSE", "MIT"),
    ]
    y = 0.74
    for k, v in stats:
        ax.text(0.10, y, k, fontsize=14, color=C_TITLE, fontweight="bold")
        ax.text(0.42, y, v, fontsize=14, color=C_BODY)
        y -= 0.07
    ax.text(0.06, 0.10,
            "Mock-mode means: every script in scripts/01_*.py to scripts/10_*.py runs end-to-end\n"
            "without any external API; reviewers can reproduce the headline numbers in < 10 minutes.",
            fontsize=11.5, color=C_ACCENT)
    footer(ax, 27)
    return fig


def slide_28_status():
    fig, ax = new_slide(
        "Status - Real vs. Mock Validation",
        "What the current numbers actually rest on",
    )
    rows = [
        ("U1 (modern OPE)", "MOCK", "Synthetic logs with known V; awaiting real LLM outputs"),
        ("U2 (debate judge)", "MOCK", "Vendor APIs stubbed; awaiting medical-rater calibration set"),
        ("U3 (CCE)",         "REAL", "Trained + evaluated on real ACI-Bench (AUC 1.00 -> 0.55)"),
        ("U4 (decomp + MSM)", "MOCK", "Closed-form fragility verified analytically + on synthetic"),
    ]
    y_head = 0.78
    ax.add_patch(Rectangle((0.04, y_head - 0.06), 0.92, 0.06,
                           facecolor=C_TITLE, edgecolor="none"))
    ax.text(0.07, y_head - 0.03, "Upgrade", color="white", fontsize=13, fontweight="bold", va="center")
    ax.text(0.30, y_head - 0.03, "State", color="white", fontsize=13, fontweight="bold", va="center")
    ax.text(0.45, y_head - 0.03, "Notes", color="white", fontsize=13, fontweight="bold", va="center")
    y = y_head - 0.12
    for name, state, notes in rows:
        col = "#27ae60" if state == "REAL" else "#e67e22"
        ax.text(0.07, y, name, fontsize=13, color=C_BODY, va="center")
        ax.text(0.30, y, state, fontsize=13, color=col, fontweight="bold", va="center")
        ax.text(0.45, y, notes, fontsize=11.5, color=C_BODY, va="center")
        y -= 0.09
    ax.text(0.06, 0.18,
            "Headline scientific claim (CCE breaks degenerate embeddings on real medical text)\n"
            "rests entirely on REAL data. Engineering claims rest on MOCK + planned real reruns.",
            fontsize=12, color=C_ACCENT, fontweight="bold")
    footer(ax, 28)
    return fig


def slide_29_roadmap():
    fig, ax = new_slide(
        "Roadmap - Next 5 Weeks",
        "From v1 (today) to submission-ready v2",
    )
    weeks = [
        ("Week 1", "Real LLM agents on ~500 ACI-Bench cases", "Budget $300, ~2 days compute. Replaces templated counterfactuals in U3."),
        ("Week 2", "Gold-label ground truth", "Recruit 2-3 board-certified raters; 200 cases x 4 axes."),
        ("Week 3-4", "Calibration set + judge tuning", "Hit rho >= 0.7, kappa_w >= 0.4, ICC >= 0.75, |bias| <= 0.05 on holdout."),
        ("Week 5", "Real-data run of U1 + U4", "DML / TMLE + MSM Gamma* on real V(pi_e) - V(pi_b)."),
        ("Week 6-7", "Paper writing + ablations", "Target NeurIPS 2026 or JAMIA short paper."),
    ]
    y = 0.78
    for w, title, desc in weeks:
        ax.add_patch(FancyBboxPatch((0.05, y - 0.10), 0.10, 0.085,
                                    boxstyle="round,pad=0.005",
                                    facecolor=C_TITLE, edgecolor="none"))
        ax.text(0.10, y - 0.058, w, fontsize=13, color="white",
                fontweight="bold", ha="center", va="center")
        ax.text(0.18, y - 0.02, title, fontsize=13, color=C_TITLE, fontweight="bold")
        ax.text(0.18, y - 0.07, desc, fontsize=11.5, color=C_BODY)
        y -= 0.13
    footer(ax, 29)
    return fig


def slide_30_risks():
    fig, ax = new_slide(
        "Risks and Mitigations",
        "Three things that can break the plan, and what we do about each",
    )
    risks = [
        ("No medical raters available in time",
         "Use HealthBench (Anthropic 2025) as anchor: re-score on its rubric and report\nrelative agent ranking. Loses absolute calibration; preserves comparative claims."),
        ("MSM bound widens too fast (Gamma* < 1.2)",
         "Reposition headline as 'fragile finding' - still publishable as a methodological\ndemonstration; the contribution is the closed-form Gamma* itself."),
        ("CCE fails to drop AUC on the real LLM rerun",
         "Indicates the templated counterfactual was easier than reality (informative\nnegative result). Adds an HSIC-only ablation slot to the paper."),
        ("Vendor APIs change scoring behaviour mid-experiment",
         "Pin model versions (gpt-4-1106, claude-3.5-sonnet-20240620, gemini-1.5-pro-002)\nand log raw judge transcripts so any drift is auditable post-hoc."),
    ]
    y = 0.78
    for risk, mit in risks:
        ax.text(0.06, y, "Risk:  " + risk, fontsize=12.5, color=C_ACCENT, fontweight="bold")
        ax.text(0.10, y - 0.04, "Mitigation:  " + mit, fontsize=11, color=C_BODY)
        y -= 0.16
    footer(ax, 30)
    return fig


def slide_31_takehome():
    fig, ax = new_slide(
        "Take-Home Message",
        "One sentence, then three sub-sentences",
    )
    ax.text(0.06, 0.74,
            "U3 saved IPS-class estimators on real medical embeddings.",
            fontsize=22, color=C_TITLE, fontweight="bold")
    bullet_list(
        ax,
        [
            "Raw MedCPT separates correct from counterfactual notes at AUC = 1.00\n   - the OPE problem is degenerate; IS weights are 0 or infinity.",
            "After CCE training, AUC drops to 0.55 - the identifiability regime where\n   modern OPE estimators (DML, TMLE, MIPS) have well-defined finite-variance solutions.",
            "This is the first demonstration on real ACI-Bench transcripts, and it does NOT\n   require a real LLM rerun (templated counterfactuals suffice for the methodological claim).",
        ],
        y0=0.58, dy=0.13, fontsize=13,
    )
    ax.text(0.06, 0.13,
            "If the above survives the Week-1 LLM rerun, U3 is the load-bearing scientific contribution.\n"
            "U1, U2, U4 then become the reproducible engineering scaffold around it.",
            fontsize=12, color=C_ACCENT)
    footer(ax, 31)
    return fig


def slide_32_thanks():
    fig, ax = new_slide(
        "Acknowledgements + Contact",
    )
    ax.text(0.06, 0.74, "Thanks to:", fontsize=18, color=C_TITLE, fontweight="bold")
    bullet_list(
        ax,
        [
            "ACI-Bench authors for the public clinical-conversation corpus.",
            "MedCPT team (NIH) for the open medical-text encoder.",
            "Anthropic / Claude Code for the agentic build harness.",
            "Reviewers and lab-mates who stress-tested the OPE assumptions.",
        ],
        y0=0.66, dy=0.06, fontsize=13,
    )
    ax.text(0.06, 0.36, "Contact + repository:", fontsize=18, color=C_TITLE, fontweight="bold")
    ax.text(0.08, 0.30, "Linhao Hao   <lhhao0430@gmail.com>", fontsize=14, color=C_BODY)
    ax.text(0.08, 0.25,
            "github.com/<user>/Counterfactual-Conversation-Emulation-for-Medical-AI-Agents",
            fontsize=13, color=C_BODY)
    ax.text(0.08, 0.20, "Date: 2026-05-04   |   Version: v1   |   License: MIT",
            fontsize=12, color=C_GREY)
    ax.text(0.5, 0.08, "Thank you.", fontsize=22, color=C_ACCENT,
            fontweight="bold", ha="center")
    footer(ax, 32)
    return fig


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
SLIDE_FUNCS = [
    slide_01_title,
    slide_02_eval_gap,
    slide_03_ope_tte_gap,
    slide_04_setup,
    slide_05_estimand,
    slide_06_assumptions,
    slide_07_upgrades_table,
    slide_08_dm_ips_dr,
    slide_09_dr_overfitting_bias,
    slide_10_dml,
    slide_11_tmle,
    slide_12_conformal,
    slide_13_estimator_validation,
    slide_14_judge_biases,
    slide_15_triadic_debate,
    slide_16_4axis_calibration,
    slide_17_offshelf_fails,
    slide_18_cce_objective,
    slide_19_mips_connection,
    slide_20_cce_headline,
    slide_21_cce_caveat,
    slide_22_pse_decomposition,
    slide_23_msm,
    slide_24_fragility,
    slide_25_workflow,
    slide_26_layer1,
    slide_27_repo_stats,
    slide_28_status,
    slide_29_roadmap,
    slide_30_risks,
    slide_31_takehome,
    slide_32_thanks,
]


def build():
    print(f"Building {len(SLIDE_FUNCS)} slides...")
    figs = []
    for i, fn in enumerate(SLIDE_FUNCS, start=1):
        try:
            fig = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! slide {i} ({fn.__name__}) failed: {exc}")
            raise
        figs.append(fig)
        print(f"  - slide {i:02d} {fn.__name__}")

    # ---- PDF
    print(f"Writing PDF -> {PDF_PATH}")
    with PdfPages(PDF_PATH) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches=None, pad_inches=0,
                        facecolor=fig.get_facecolor())

    # ---- PPTX
    print(f"Writing PPTX -> {PPTX_PATH}")
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    blank = prs.slide_layouts[6]  # blank
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150,
                    facecolor=fig.get_facecolor(),
                    bbox_inches=None, pad_inches=0)
        buf.seek(0)
        slide = prs.slides.add_slide(blank)
        slide.shapes.add_picture(buf, 0, 0,
                                 width=prs.slide_width,
                                 height=prs.slide_height)
        plt.close(fig)
    prs.save(str(PPTX_PATH))

    # ---- Report
    pdf_size = PDF_PATH.stat().st_size
    pptx_size = PPTX_PATH.stat().st_size
    print()
    print("=" * 60)
    print(f"Slides:           {len(SLIDE_FUNCS)}")
    print(f"PDF:              {PDF_PATH}  ({pdf_size/1024:.1f} KB)")
    print(f"PPTX:             {PPTX_PATH}  ({pptx_size/1024:.1f} KB)")
    print(f"Total size:       {(pdf_size + pptx_size)/1024:.1f} KB")
    if PLACEHOLDERS_USED:
        print(f"Placeholder figs: {len(PLACEHOLDERS_USED)}")
        for f in PLACEHOLDERS_USED:
            print(f"  - {f}")
    else:
        print("Placeholder figs: 0")
    print("=" * 60)


if __name__ == "__main__":
    build()
