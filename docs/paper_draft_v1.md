# Paper Draft v1 — Outline mapped to repo artifacts

> Counterfactual Conversation Emulation for Medical AI Agents
> Target venue: NeurIPS main / ICLR / NEJM AI

## 1. Introduction

**Bottleneck.** Medical AI agents are updated every few months; prospective
trials take a year and cost $1M+. Static benchmarks (MedQA, USMLE) test
isolated capability, not deployment effect.

**Insight.** Combine target trial emulation (Hernán-Robins) with off-policy
evaluation, but replace the 2010s estimator stack with modern alternatives
designed for high-dimensional NLP action spaces and small-n medical data.

**Contribution.** Four methodological upgrades over the naive
emulation pipeline (the paper's main pitch):

| # | Upgrade | Replaces | Source |
|---|---------|----------|--------|
| U1 | DML cross-fit + TMLE + Conformal OPE CI | DM/IPS/DR + bootstrap | `src/ccema/estimators/{dml_dr,tmle,conformal_ci}.py` |
| U2 | Multi-agent debate judge + 4-axis process reward | Single LLM judge | `src/ccema/judges/debate.py` |
| U3 | Causal Contrastive Embedding + MIPS | Off-the-shelf embedding + classifier DRE | `src/ccema/embeddings/cce.py` |
| U4 | Path-specific decomposition + Marginal Sensitivity Model | Scalar V(π) under no-confounding | `src/ccema/analysis/{decomposition,sensitivity}.py` |

## 2. Problem formulation

See `docs/problem_statement_v2.md`. Key items:

- Estimand: V(π_agent) = E_x [E_{a~π_agent} [Y(x,a)]]
- 4-axis outcome (DDx / Mx safety / Mx practicality / Mx cost)
- 5 identification assumptions: time-zero, consistency, positivity,
  exchangeability, no-anticipation, no-interference

## 3. Method

### 3.1 Outcome scoring (U2)
- Triadic debate over each rubric axis: Anthropic / Google / OpenAI
  (different vendors avoids self-preference)
- Beta posterior output; concentration → outcome uncertainty
- 100-case calibration set selected via active sampling (`src/ccema/analysis/active_sampling.py`)
- Calibration metrics: Spearman ρ ≥ 0.70 + κ ≥ 0.40 + ICC ≥ 0.75 + |bias| ≤ 0.05

### 3.2 Embedding + Density Ratio (U3)
- Backbone: MedCPT or BGE-M3 (HF) wrapped via `src/ccema/embeddings/backbones.py`
- CCE projection trained with InfoNCE on (anchor, positive=same-x-different-a) +
  HSIC penalty against confounders (`src/ccema/embeddings/{cce,losses}.py`)
- Classifier-based DRE on CCE embeddings → density ratio
- Positivity diagnostics: ESS/n, AUC, MMD (`src/ccema/embeddings/diagnostics.py`)

### 3.3 Estimator stack (U1)
- Six estimators: DM, IPS, DR (paper baselines) + MIPS, DML cross-fit DR, TMLE
- Conformal OPE CIs (Taufiq 2022) — finite-sample valid alongside bootstrap
- All in `src/ccema/estimators/`

### 3.4 Decomposition + Sensitivity (U4)
- Path-specific effects per (axis × subpopulation)
  in `src/ccema/analysis/decomposition.py`
- Logistic Marginal Sensitivity Model bounds at Γ ∈ {1, 1.5, 2, 3}
  in `src/ccema/analysis/sensitivity.py`

## 4. Experiments

### 4.1 Synthetic benchmark (Step 8)
- Closed-form V_true via large MC; verifies estimator math
- Empirical RMSE ordering: TMLE < DR < DM < DML-DR < IPS < MIPS

### 4.2 Real data — ACI-Bench prototype + MTS-Dialog scale-up (Steps 5, 7, 9)
- Time-zero cut at intent=assessment
- 4-arm π_agent ladder: Claude Opus 4.7 / GPT-3.5 / MedGemma-27B
  (+ optional Opus + MedAgentBench scaffolded)
- Headline table: 6 estimators × {Bias, RMSE, CI coverage (bootstrap),
  CI coverage (conformal), direction agreement}

### 4.3 Ablations (Step 10)
- Judge mode: single vs ensemble vs debate
- Embedding: raw vs CCE on each backbone
- Estimator: DML vs vanilla DR
- Agent capability ladder
- Subpopulation heterogeneity (axis × ICD chapter)

### 4.4 When to trust emulation (Step 10)
- 8-item upgraded checklist (`docs/when_to_trust.md`)
- Pass/fail per encounter dataset

## 5. Discussion

- Effect-decomposition reveals where agent + clinician differ — directly
  addresses the BIDMC finding (PCPs beat AMIE on practicality and cost,
  not DDx accuracy).
- MSM robustness: Γ = 2 threshold is medical-OPE convention; we report
  Γ-fragility for transparency.
- Limitations: (a) AMIE / Multimodal AMIE not API-accessible, cited as
  upper-bound; (b) CCE training on small n (200-1700) — torch
  fine-tuning required for backbone (current numpy reference for tests
  only); (c) sensitivity bound is conservative per-case form (Yadlowsky
  2018), not the sharp constrained-optimization version.

## 6. Reproducibility

All locked-in choices are recorded:

- `docs/decision_log.md` — methodological decisions D001-D005
- `configs/eval_config_v1.yaml` — every model id, prompt SHA, threshold
- `data/calibration_set.parquet` — frozen 100-case set after Step 3
- Step-by-step scripts in `scripts/01_*.py … scripts/10_*.py`

Random seeds, prompt SHAs, and config hashes are written into every
output JSON.

## 7. Citations

Inherits the home-file Tier 1-3 reading list plus the U1-U4 references in
the workflow plan (NeurIPS / ICLR / Nature / NEJM-AI / medRxiv 2024-2026).
Full DOI list lives in `docs/problem_statement_v2.md`.
