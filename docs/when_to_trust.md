# When to Trust Counterfactual Conversation Emulation

> Diagnostic checklist for medical-AI-agent OPE (extends the home-file's
> Phase-6 checklist from 5 → 8 items). All thresholds are locked in
> `configs/eval_config_v1.yaml`. Each item is computed by Step 4-9 and
> aggregated by `scripts/10_failure_modes.py`.

| # | Item | Threshold | Source |
|---|------|-----------|--------|
| 1 | Agent capability gap (strong − weak rubric mean) | > 0.10 | Step 5 capability-gap test |
| 2 | Positivity ESS/n | > 0.10 | Step 6 + Step 9 weight diagnostics |
| 3 | Positivity DRE classifier AUC | < 0.90 | Step 6 |
| 4 | Judge vs human Spearman ρ | > 0.70 | Step 4 calibration |
| 5 | CCE vs raw embedding AUC drop | > 0 | Step 6 ablation |
| 6 | Conformal CI width / \|V̂\| | < 4 | **NEW** — Step 9 conformal CI |
| 7 | Marginal Sensitivity Model robust at Γ = 2 | sign-stable | **NEW** — Step 7 |
| 8 | Per-axis process-reward stdev | > 0.05 | **NEW** — Step 4 process reward |

## Why each item

1. **Capability gap** — without a detectable strong/weak gap, neither the
   rubric nor the agent generation is doing anything; the OPE estimator
   has no signal to recover.
2. **ESS/n** — Kish's effective sample size catches weight collapse (one
   sample dominates the IPS estimate); below 0.10 the variance of any
   IPS-type estimator is uninterpretable.
3. **DRE classifier AUC** — when the classifier nearly perfectly separates
   π_agent samples from π_b samples, positivity is failing; the density
   ratio is unreliable regardless of clipping.
4. **Spearman ρ** — Step 4's main gate; <0.70 means the judge does not
   reliably track human rubric scoring even after debate.
5. **CCE drop** — if CCE doesn't reduce AUC compared with the raw
   off-the-shelf embedding, U3 has no value and should be dropped from
   the paper's pitch (or replaced with a better contrastive objective).
6. **Conformal CI width / |V̂|** — when the CI is more than 4× the point
   estimate, the estimator is too uncertain to support a substantive
   claim. This is the *width* check, complementing the existing CI
   coverage check.
7. **MSM robust at Γ = 2** — medical OPE always has unmeasured
   confounding; if a Γ = 2 sensitivity bound flips the sign of the
   effect, the headline claim is fragile and should be flagged.
8. **Process-reward stdev** — per-axis dispersion below 0.05 means the
   debate judge is collapsing all axes onto a single near-constant
   value (an "agreeableness" failure mode for LLM judges); the 4-axis
   decomposition is then meaningless.

## Decision rule

If items 1–4 pass: paper is publishable.
If items 6–8 also pass: paper is publishable as a *novel* contribution
(the U1/U2/U4 upgrades have measurable benefit).
If item 5 fails: drop the U3 contribution claim; keep the paper as
U1+U2+U4.
If item 7 fails: re-position headline to "fragile under unmeasured
confounding" — still publishable per home-file Risk 3.
