# Decision Log

A running log of methodological choices and their justifications. Each
entry should answer: *what did we choose, why, and what would invalidate it?*

---

## 2026-05-03 — D001: Use multi-agent debate judge instead of single LLM-as-judge

**Choice.** Outcome scoring uses a triadic debate (Claude Opus 4.5 defender /
Gemini 2.5 Pro prosecutor / GPT-5 judge), one round per axis, with the judge
outputting a Beta posterior over $[0,1]$.

**Why.** Single-judge calibration cannot remove systematic biases (self-
preference, verbosity, lenient) — only random noise. Du et al. 2024 ICML
shows multi-agent debate reduces bias 30–50%. Three vendors are required
because $a^{\mathrm{agent}}$ is GPT-generated; an all-GPT panel would
inherit self-preference bias.

**What invalidates this.** If calibration metrics (Spearman $\rho \geq 0.7$,
weighted $\kappa \geq 0.4$, ICC $\geq 0.75$, $|\text{bias}| \leq 0.05$) fail
on the 100-case set even with debate, fall back to a different rubric
design or a gold-label task per R1.

---

## 2026-05-03 — D002: Replace DM/IPS/DR with DML cross-fit, TMLE, and conformal CIs

**Choice.** Primary estimators are MIPS over CCE embedding (U3), DML
cross-fit DR (U1), and TMLE (U1). The paper's original DM / IPS / DR are
kept as baselines.

**Why.** DM/IPS/DR are 2011-2014 methods. In the small-$n$, high-dimensional
NLP regime they exhibit large first-order bias (DR) and exploding variance
(IPS). DML cross-fit removes nuisance overfit bias; TMLE is semi-parametric
efficient; conformal CIs are finite-sample valid where bootstrap CIs
under-cover with heavy-tailed weights.

**What invalidates this.** TMLE failing on $n \approx 200$ (RMSE > DML-DR)
would push DML-DR to be the headline estimator with TMLE in supplement.

---

## 2026-05-03 — D003: Train Causal Contrastive Embedding rather than use off-the-shelf

**Choice.** Train a 768→128 projection head on top of MedCPT and BGE-M3
backbones with InfoNCE on (same-context, different-action) positives plus
HSIC penalty against demographic confounders.

**Why.** The IPS density ratio needs an embedding that separates *same-x
different-a* — generic semantic similarity does not. CCE aligns the
embedding objective with the OPE objective; OffCEM-style experiments
suggest a 30–50% positivity improvement.

**What invalidates this.** If CCE classifier AUC does not drop below the
raw-embedding baseline, fall back to raw MIPS (still a paper contribution
but a weaker one).

---

## 2026-05-03 — D004: Use Claude Opus 4.7 strong / GPT-3.5-turbo-0125 weak as π_agent

**Choice.** Strong frontier = Claude Opus 4.7; weak frontier = GPT-3.5-
turbo-0125. Optional 3rd arm: MedGemma-27B.

**Why.** Opus 4.7 is the current MedAgentBench leader with cheap, deterministic
JSON output; GPT-3.5 is the canonical weak baseline in 2024-25 medical AI
papers. MedGemma-27B disentangles "frontier capability" from "medical fine-
tuning" — a reviewer-anticipated confound.

**What invalidates this.** AMIE / Multimodal AMIE were considered but have
no public API; cited as upper-bound aspiration only.

---

## 2026-05-03 — D005: ACI-Bench prototyping → MTS-Dialog main run

**Choice.** Use ACI-Bench (n=207) for prototyping (Phase 2-4) and MTS-Dialog
(n≈1700) for the main experiment (Phase 5). MedDialog-EN reserved for noise
robustness only.

**Why.** ACI-Bench has 2025 release intent annotations enabling clean
time-zero cuts; MTS-Dialog adds scale without sacrificing structure.
MedDialog-EN is too noisy for primary outcomes.

**What invalidates this.** If MTS-Dialog leaks into MedGemma's training
corpus (Layer-2 audit), drop MedGemma 3rd arm and report only the closed-
source pair on MTS-Dialog.

---

## 2026-05-05 — D006: Refocus paper around DM-KL / MIPS / OffCEM (research_plan_v2)

**Choice.** The core paper's estimator stack is the three published
methods **DM with KL-control (Jaques 2019), MIPS (Saito & Joachims 2022),
and OffCEM (Saito et al. 2023)**, with OffCEM as the headline. The
canonical 12-week plan is `docs/research_plan_v2.md`; older plans defer
to it on conflict. The previous "U1–U4 four upgrades" framing
(TMLE / debate / CCE / MSM) is **not deleted** — it moves to Phase 6
ablations and supplementary material, cataloged in
`docs/extensions_and_supplementary.md`. D002–D005 remain valid as
sub-decisions, now interpreted as Phase 6 ablation choices rather than
core-paper headline choices.

**Why.** Tighter, more defensible pitch as *"first medical-NLP
application of published OPE-for-large-action methods (MIPS / OffCEM)
plus a KL-control DM baseline"* rather than *"we propose 4 novel
upgrades simultaneously"*. The former has a cleaner novelty story
(method migration across domains, à la "first import of X from
recommender systems to medicine") and a smaller surface area for
reviewer attack. Each Phase-6 ablation then becomes a positive
robustness result rather than a load-bearing claim.

**What invalidates this.** If Phase 4 implementation reveals that
OffCEM's cluster-effect decomposition is insufficient on real medical
data — i.e. the within-cluster residual $h(X, A)$ has variance
comparable to the cross-cluster signal $g(X, \phi(A))$ — fall back to
the **TMLE + CCE narrative** (Extensions 1 + 3 from
`docs/extensions_and_supplementary.md` become primary). Specifically:
trigger the fallback if synthetic-oracle OffCEM RMSE $\geq$ MIPS RMSE,
or if real-data OffCEM CIs cover a wider range than DML-DR by more
than 1.5×.
