# Research Plan v2 — Canonical 12-Week Plan

> **Counterfactual Conversation Emulation for Off-Policy Evaluation of Medical Agents**
> Author: Oliver Liu · Last updated 2026-05-03
> Status: **canonical** — supersedes the previous "U1–U4 four upgrades" framing.
> Older plans (`docs/phase1_problem_statement.md`, `docs/problem_statement_v2.md`,
> `docs/paper_draft_v1.md`, `docs/decision_log.md` D001–D005) defer to this file
> on any conflict about scope, estimator stack, or paper claims.

---

## 0. The pitch

### 0.1 Current gap

Medical AI agents are evaluated almost exclusively on **static, capability-style
benchmarks** (USMLE-style MCQ, MedQA, PubMedQA, MedAgentBench task completion).
These measure *isolated competence under a frozen prompt*, not *deployment effect
on real encounters*. Existing process-reward / rubric work (AMIE, MedAgentBench,
the BIDMC clinical evaluation) either runs prospective small-n studies or scores
LLM outputs in isolation — they do **not** estimate the counterfactual
"what would patient outcome have been if this agent had replaced (or augmented)
the clinician on this exact historical encounter?".

That counterfactual is the deployment value
$V(\pi_{\mathrm{agent}}) = \mathbb{E}_x[\mathbb{E}_{a\sim\pi_{\mathrm{agent}}}[Y(x,a)]]$
and it is exactly the object of off-policy evaluation (OPE). The medical-NLP
community has not yet adopted modern OPE.

### 0.2 Why nobody has done this

The action space is **natural-language conversation**, which has three properties
that break textbook IPS / DR:

1. **Combinatorial / continuous** — exact action overlap $a^{\mathrm{agent}} = a^{\mathrm{clin}}$ has probability ~0; classical IPS is undefined.
2. **High-cardinality** — even after embedding, the effective action space is huge relative to $n \approx 200$–$2000$, so DR variance explodes.
3. **Logging policy unknown** — the clinician policy $\pi_b$ is implicit; we need behavior-cloning / DRE to estimate it.

The OPE-for-recommender-systems community solved exactly these three problems
between 2022–2024 (Saito MIPS 2022, Saito OffCEM 2023, Cief embedding-learning
2024). **No medical-AI-agent paper has yet imported them.**

### 0.3 Conversation as action

The bridge is to treat the agent's **final assessment + plan** (one block per
encounter) as an action $a \in \mathcal{A}_{\mathrm{text}}$, embed it via
$\phi(a) \in \mathbb{R}^d$, and apply Marginalized IPS / OffCEM in embedding
space. The **embedding becomes the action representation** and the OPE problem
reduces to overlap and outcome modeling in $\phi(\cdot)$-space.

### 0.4 One-line contribution

> *First migration of MIPS and OffCEM from recommender systems to medical
> AI-agent off-policy evaluation, with a KL-controlled DM baseline (Jaques 2019)
> and an embedding-space identification analysis.*

---

## 1. Estimator stack (3 estimators)

| Tag | Estimator | Source | Role |
|-----|-----------|--------|------|
| **DM-KL** | Direct Method with KL penalty toward $\pi_b$ | Jaques et al. 2019 | Robust low-variance baseline, calibration anchor |
| **MIPS** | Marginalized IPS over action embedding $\phi(a)$ | Saito & Joachims 2022 | Reduces variance from $|\mathcal{A}|$-cardinality to $d$-cardinality |
| **OffCEM** | Off-policy evaluation with Cluster-Effect Model | Saito et al. 2023 | **Headline estimator** — handles violations of MIPS's "no direct effect" |

The paper's headline number is **OffCEM** because it strictly dominates MIPS
when the residual (within-cluster) reward effect is non-zero, which we expect
in medical text (two assessments with the same clustered embedding can still
differ in safety detail).

---

## 2. Embedding-space assumptions

MIPS / OffCEM identify $V(\pi_{\mathrm{agent}})$ in $\phi$-space under:

1. **(C1) Common embedding support.** $p_{\pi_b}(\phi \mid x) > 0$ wherever
   $p_{\pi_{\mathrm{agent}}}(\phi \mid x) > 0$. Diagnosed via ESS/$n > 0.10$
   and DRE classifier AUC $< 0.90$ on $\phi(a)$.
2. **(C2) No direct effect of $a$ on $Y$ given $\phi(a)$.** Formally
   $Y \perp\!\!\!\perp A \mid X, \phi(A)$. **Plausibly violated** — two
   different texts with the same embedding can yield different rubric scores
   (e.g. one mentions allergy check, one does not).

OffCEM relaxes (C2) by decomposing
$Y = g(X, \phi(A)) + h(X, A) + \varepsilon$, modeling $h$ as a within-cluster
residual and using DM on $h$ + IPS on $g$ — so even when (C2) fails, the
estimator stays consistent as long as $h$ can be modeled with bounded variance.

---

## 3. Phases (7 phases over 12 weeks)

### Phase 1 — Problem statement & identification (Week 1, **DONE**)

**Deliverables.**
- One-page Phase-1 statement → `docs/phase1_problem_statement.md` (cross-links here).
- Long-form derivation → `docs/problem_statement_v2.md`.
- Identification assumptions A1–A5 (consistency / positivity / exchangeability /
  no-anticipation / no-interference) and embedding assumptions C1–C2.
- Frozen estimand: $V(\pi_{\mathrm{agent}})$ scalar + $\Delta = V - V(\pi_b)$.

### Phase 2 — Data & time-zero (Weeks 2–3)

**Deliverables.**
- ACI-Bench (n=207) prototyping loader, MTS-Dialog (n≈1700) main loader,
  MedDialog-EN noise-robustness loader.
- Layer-1 leakage probe: x-side post-decision content stripped (Assessment,
  Plan, post-encounter dialogue, ICD codes, downstream test results).
- Layer-2 training-corpus decontamination: n-gram + semantic similarity audit
  for each candidate $\pi_{\mathrm{agent}}$ training corpus.
- Frozen `configs/data_config_v1.yaml` with split seeds.

### Phase 3 — Outcome model: rubric judge (Weeks 3–4)

**Deliverables.**
- 4-axis rubric: DDx accuracy / Mx safety / Mx practicality / Mx cost.
- **Single rubric judge** (GPT-5 or Claude Opus 4.7) with Beta-posterior output
  per axis. This is the **core paper outcome model**.
- Calibration set: 100 cases, two physician raters, ICC ≥ 0.75 target.
- *(Multi-agent debate judge is an Extension, not part of core paper — see
  `docs/extensions_and_supplementary.md` Extension 2.)*

### Phase 4 — OPE estimators (Weeks 5–7)

**Deliverables.**
- `src/ccema/estimators/dm.py` (extended with KL-control variant).
- `src/ccema/estimators/mips.py` (already exists — verify against Saito 2022 reference).
- **NEW** `src/ccema/estimators/offcem.py` — implement Saito 2023 cluster-effect
  decomposition. Reuse the existing MIPS/DM scaffolding.
- DRE / behavior-cloning module for $\hat\pi_b$ estimation.
- Synthetic OPE oracle (`tests/test_synthetic_oracle.py`) covering all 3 estimators.

### Phase 5 — Main experiment (Weeks 8–10)

**Deliverables.**
- 4 $\pi_{\mathrm{agent}}$ arms × 2 datasets × 3 estimators table.
- Headline = OffCEM with bootstrap CI; DM-KL and MIPS reported alongside.
- Calibration anchor: agreement between OffCEM and ground-truth rubric on a
  held-out subset.

### Phase 6 — Ablations (Weeks 10–11)

Six ablation dimensions:

1. **Embedding choice.** Off-the-shelf MedCPT / BGE-M3 vs **CCE-trained**
   (`src/ccema/embeddings/cce.py`). Aligned with Cief et al. 2024 finding that
   learned action embeddings beat off-the-shelf for OPE. **This is where the
   real ACI-Bench result raw=AUC 1.0, ESS/n 0.10 → CCE AUC 0.55, ESS/n 0.96
   appears in the paper.**
2. **Outcome aggregation.** Scalar $\bar Y$ vs 4-axis vector decomposition
   (`src/ccema/analysis/decomposition.py`).
3. **Judge robustness.** Single-rubric vs 3-vendor debate judge
   (`src/ccema/judges/debate.py`).
4. **Sensitivity to unmeasured confounding.** MSM bounds at $\Gamma \in \{1, 1.5, 2, 3\}$
   via `src/ccema/analysis/sensitivity.py`.
5. **CI method.** Bootstrap vs split-conformal (`src/ccema/estimators/conformal_ci.py`).
6. **Modern semi-parametric estimators.** TMLE / DML cross-fit DR
   (`src/ccema/estimators/{tmle,dml_dr}.py`) as efficiency upper bound vs
   the headline OffCEM.

### Phase 7 — Writing & release (Weeks 11–12)

**Deliverables.**
- arXiv preprint (target venue: NeurIPS Datasets & Benchmarks or ML4H).
- Public repo, pinned environment (`requirements-lock.txt`), reproducible
  Makefile entrypoints.
- `CITATION.cff` with the published DOI once available.

---

## 4. Risks and fallbacks

### Risk 1 — Embedding overlap collapses (C1 fails)

*Trigger.* Across both datasets, ESS/$n < 0.10$ and DRE AUC $> 0.90$ even after
CCE training.

*Fallback.* Restrict the target population to encounters where the agent's
sampled action is *near-neighbors* of some clinician action under $\phi$;
report restricted-support $V$. This is the OPE analogue of trimming.

### Risk 2 — OffCEM cluster model insufficient (C2 violation too severe)

*Trigger.* On synthetic oracle the within-cluster residual $h(x,a)$ has
variance comparable to the cross-cluster signal; OffCEM RMSE is not better
than MIPS.

*Fallback.* Switch headline to **TMLE + CCE narrative** (Extensions 1 + 3
become primary). The paper's pitch shifts from "MIPS/OffCEM in medicine"
to "modern semi-parametric OPE with learned action embeddings in medicine".
This is acceptable — the existing extension code carries the load.

### Risk 3 — Single rubric judge fails calibration

*Trigger.* Spearman $\rho < 0.6$ with physician raters on the 100-case set.

*Fallback.* Promote the debate judge (Extension 2) into the core paper and
re-derive calibration on a fresh 50-case set. Compute and budget supports
this — it's a 2× judge cost on the calibration set only.

---

## 5. Reading list

### Core OPE for large action spaces

- Saito & Joachims, *Off-Policy Evaluation for Large Action Spaces via
  Embeddings* (MIPS), ICML 2022. https://arxiv.org/abs/2202.06317
- Saito, Udagawa, Kiyohara, *Off-Policy Evaluation for Large Action Spaces
  via Conjunct Effect Modeling* (OffCEM), NeurIPS 2023.
  https://arxiv.org/abs/2305.08062
- Cief, Kveton, Vasile, *Learning Action Embeddings for Off-Policy
  Evaluation*, RecSys 2024. https://arxiv.org/abs/2305.03954

### KL-controlled DM

- Jaques et al., *Way Off-Policy Batch Deep Reinforcement Learning of
  Implicit Human Preferences in Dialog*, 2019.
  https://arxiv.org/abs/1907.00456

### Medical-AI evaluation context

- Tu et al., *Towards Conversational Diagnostic AI* (AMIE), 2024.
  https://arxiv.org/abs/2401.05654
- Brodeur et al., *Towards Conversational Diagnostic AI: a randomized study
  with primary-care physicians and patients* (AMIE-BIDMC), 2026 (preprint).
- MedAgentBench, 2024. https://arxiv.org/abs/2501.14654

### Target trial emulation

- Hernán & Robins, *Using Big Data to Emulate a Target Trial When a
  Randomized Trial Is Not Available*, AJE 2016.
- Hernán & Robins, *Causal Inference: What If*, 2022 textbook. https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/

### Modern semi-parametric OPE (Phase 6 ablation)

- van der Laan & Rose, *Targeted Learning: Causal Inference for Observational
  and Experimental Data*, Springer 2011 (TMLE).
- Chernozhukov et al., *Double/Debiased Machine Learning*, Econometrics
  Journal 2018. https://arxiv.org/abs/1608.00060
- Lei, G'Sell, Rinaldo, Tibshirani, Wasserman, *Distribution-free predictive
  inference for regression*, JASA 2018 (conformal).

### Sensitivity analysis (Phase 6 ablation)

- Tan, *A distributional approach for causal inference using propensity
  scores*, JASA 2006 (Marginal Sensitivity Model).
- Dorn, Guo, Kallus, *Sharp Bounds for Generalized Causal Sensitivity*,
  2024. https://arxiv.org/abs/2308.01281

---

## 6. Relationship to prior framing

The earlier "U1–U4 four upgrades" plan (TMLE / debate-judge / CCE / MSM)
introduced four novel upgrades simultaneously. That framing is **superseded**
by this v2 plan, which:

- Makes the **published-method migration** the headline claim
  (DM-KL / MIPS / OffCEM, all from 2019–2023 references).
- Demotes TMLE / DML / conformal CI to **Phase 6 Ablation 6** (modern
  efficiency upper bound).
- Demotes the multi-agent debate judge to **Phase 6 Ablation 3** (judge
  robustness).
- Demotes CCE to **Phase 6 Ablation 1** (learned vs off-the-shelf
  embedding), aligned with Cief 2024.
- Demotes path-specific decomposition + MSM sensitivity to **Phase 6
  Ablations 2 + 4**.

No code is deleted. Every U1–U4 module remains in the repo as supplementary
material — see `docs/extensions_and_supplementary.md` for the catalog.
