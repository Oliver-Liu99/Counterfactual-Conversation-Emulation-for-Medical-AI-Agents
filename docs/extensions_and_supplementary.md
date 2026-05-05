# Extensions and Supplementary Material

> Catalog of code modules and methodological contributions that exist in this
> repository **beyond the core paper claims** (DM-KL / MIPS / OffCEM, see
> `docs/research_plan_v2.md`). Nothing here is deleted — every extension is a
> supported, tested module that may be promoted into the main paper as needed
> (Risk-2 fallback) or written up as a supplement / follow-up paper.
> Last updated 2026-05-05.

---

## Extension 1 — Modern semi-parametric OPE estimators

**Modules.**
- `src/ccema/estimators/tmle.py` — Targeted Maximum Likelihood Estimator
  (van der Laan & Rose 2011) with one-step targeting on a logistic
  fluctuation submodel.
- `src/ccema/estimators/dml_dr.py` — Double / Debiased Machine Learning
  cross-fit doubly-robust estimator (Chernozhukov et al. 2018).
- `src/ccema/estimators/conformal_ci.py` — split-conformal prediction
  intervals for OPE point estimates (Lei et al. 2018).

**What it adds beyond core paper.** Provides semi-parametric efficient
($\sqrt{n}$-consistent under correct nuisance specification) point estimates
and finite-sample-valid confidence intervals — both stronger than the
bootstrap-CI guarantees on the headline OffCEM.

**Where it could go.** Phase 6 Ablation 6 ("modern efficiency upper bound");
fallback headline if OffCEM cluster-effect modeling underperforms on real
medical data (Risk 2 in `docs/research_plan_v2.md`); supplementary appendix
showing DR-vs-OffCEM RMSE comparison on the synthetic oracle.

---

## Extension 2 — Multi-agent debate judge

**Module.** `src/ccema/judges/debate.py` — three-vendor triadic debate
(Anthropic Claude defender, Google Gemini prosecutor, OpenAI GPT judge),
one round per rubric axis, judge outputs a Beta posterior over $[0, 1]$.

**What it adds beyond core paper.** The core paper uses a **single rubric
judge** (Phase 3 of `docs/research_plan_v2.md`) for simplicity and cost.
Debate reduces systematic biases (self-preference, verbosity, leniency)
that calibration alone cannot remove (Du et al. 2024 ICML, ~30–50%
bias reduction).

**Where it could go.** Phase 6 Ablation 3 ("judge robustness"); fallback
core outcome model if single-judge calibration fails (Risk 3); standalone
methods paper on cross-vendor debate for medical-NLP evaluation.

---

## Extension 3 — Causal Contrastive Embedding (CCE)

**Modules.**
- `src/ccema/embeddings/cce.py` — sklearn / numpy reference implementation
  of the InfoNCE + HSIC training loop on top of MedCPT and BGE-M3 backbones.
- `src/ccema/embeddings/cce_torch.py` — PyTorch implementation with
  GPU-friendly batching, used for the ACI-Bench experiment.

**What it adds beyond core paper.** A **learned action embedding** trained
specifically for OPE — InfoNCE on (same-context, different-action) positives
plus HSIC penalty against demographic confounders, projecting the 768-d
backbone into a 128-d space optimized for embedding-space overlap.

**Highlight result (ACI-Bench, real data).** Off-the-shelf MedCPT yields
DRE classifier AUC = **1.00** with ESS/$n$ = **0.10** — i.e. complete
separation between $\pi_{\mathrm{agent}}$ and $\pi_b$ in embedding space,
catastrophic positivity failure. After CCE training: AUC = **0.55**,
ESS/$n$ = **0.96** — overlap restored, MIPS becomes well-defined.

**Where it could go.** **Phase 6 Ablation 1 ("learned embedding vs
off-the-shelf"), aligned with Cief et al. 2024** RecSys finding that learned
action embeddings strictly improve OPE in large action spaces. Also a strong
fallback headline if OffCEM underperforms (Risk 2): "TMLE + CCE in medicine"
becomes a tight alternative pitch using Extensions 1 + 3.

---

## Extension 4 — Path-specific decomposition + MSM sensitivity

**Modules.**
- `src/ccema/analysis/decomposition.py` — per-axis ($k \in$ {DDx accuracy,
  Mx safety, Mx practicality, Mx cost}) and per-subpopulation ($s$ defined
  by ICD chapter / age band / comorbidity count) contrast estimation
  $\Delta_{k,s}$.
- `src/ccema/analysis/sensitivity.py` — Marginal Sensitivity Model bounds
  (Tan 2006, Dorn et al. 2024) at $\Gamma \in \{1, 1.5, 2, 3\}$.

**What it adds beyond core paper.** Decomposition addresses the AMIE-BIDMC
finding that PCPs beat AMIE on practicality ($p = 0.003$) and cost
($p = 0.004$) — a scalar $V$ averages that signal away. MSM gives a worst-
case bound on $V$ and $\Delta$ under unmeasured confounding (assumption A3
violation), directly addressing reviewer concerns about clinician's
unmeasured social cues.

**Where it could go.** Phase 6 Ablations 2 and 4; supplementary appendix
showing per-axis breakdown and "headline survives $\Gamma = 2$" robustness
claim.

---

## Cross-reference table

| Phase 6 Ablation | Extension | Code |
|---|---|---|
| 1. Embedding choice | Extension 3 | `src/ccema/embeddings/{cce,cce_torch}.py` |
| 2. Outcome aggregation | Extension 4 | `src/ccema/analysis/decomposition.py` |
| 3. Judge robustness | Extension 2 | `src/ccema/judges/debate.py` |
| 4. Sensitivity | Extension 4 | `src/ccema/analysis/sensitivity.py` |
| 5. CI method | Extension 1 | `src/ccema/estimators/conformal_ci.py` |
| 6. Modern semi-parametric | Extension 1 | `src/ccema/estimators/{tmle,dml_dr}.py` |
