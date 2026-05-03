# Counterfactual Conversation Emulation for Medical AI Agents

[![tests](https://github.com/Oliver-Liu99/Counterfactual-Conversation-Emulation-for-Medical-AI-Agents/actions/workflows/test.yml/badge.svg)](https://github.com/Oliver-Liu99/Counterfactual-Conversation-Emulation-for-Medical-AI-Agents/actions/workflows/test.yml)

> Estimating the deployment effect of medical AI agents from historical
> clinician–patient dialogue, by combining **target trial emulation** with
> modern **off-policy evaluation** for natural-language action spaces.

## Why

Prospective trials of medical AI agents take months and cost millions, but
agents are updated every few months — by the time a trial finishes, the agent
under evaluation is already outdated. Static benchmarks (MedQA, USMLE-style)
test isolated capability rather than deployment effect. This project closes
that gap: take historical clinician–patient encounters, retrospectively
estimate how a medical AI agent *would have* changed the outcome.

The estimand is

```
V(π_agent) = E_x [ E_{a ~ π_agent(·|x)} [ Y(x, a) ] ]
```

where `Y(x, a)` is a rubric-based outcome score on the encounter.

## Methodological position

Beyond the standard target-trial-emulation + DM/IPS/DR stack, this codebase
implements four upgrades that each independently strengthen the contribution:

| # | Upgrade | Replaces |
|---|---------|----------|
| **U1** | DML cross-fit DR + TMLE + Conformal OPE CIs | DM / IPS / DR + bootstrap CI |
| **U2** | Multi-agent debate judge + 4-axis process reward | Single LLM-as-judge with scalar y |
| **U3** | Causal Contrastive Embedding (CCE) + MIPS | Off-the-shelf embeddings + classifier DRE |
| **U4** | Path-specific decomposition + Marginal Sensitivity Model | Scalar V(π_agent) under no-confounding |

Active calibration sampling (U5) replaces stratified random sampling for the
human-rated calibration set.

## Repository layout

```
configs/                YAML configs (locked model versions, prompts, hparams)
docs/                   Problem statements, decision log, plan files
src/ccema/              Main Python package
  data/                 ACI-Bench / MTS-Dialog loaders, time-zero cut, leakage probes
  judges/               Single + debate LLM-as-judge implementations
  agents/               π_agent system prompts, sampling, JSON schema
  embeddings/           Embedding backbones, CCE training, positivity diagnostics
  estimators/           DM, IPS, DR, MIPS, DML-DR, TMLE, conformal CI
  analysis/             Effect decomposition, sensitivity bounds, active sampling
  utils/                API wrappers, caching, seeding
scripts/                Numbered runnable scripts mirroring the 10-step workflow
tests/                  Unit tests including a synthetic OPE oracle
data/                   Raw + processed data (gitignored)
```

## Workflow (10 steps)

| Step | Script | Output |
|------|--------|--------|
| 1 | `scripts/01_problem_statement.py` | `docs/problem_statement_v2.md` |
| 2 | `scripts/02_data_audit.py` | Processed datasets + leakage report |
| 3 | `scripts/03_active_sampling.py` | `data/calibration_set.parquet` |
| 4 | `scripts/04_run_judges.py` | Debate-judge scores + calibration metrics |
| 5 | `scripts/05_run_agents.py` | π_agent samples (4 arms × n=5) |
| 6 | `scripts/06_train_cce.py` | CCE checkpoints |
| 7 | `scripts/07_ground_truth.py` | V_true per arm + decomposition + sensitivity |
| 8 | `scripts/08_synthetic_benchmark.py` | Estimator unit-test results |
| 9 | `scripts/09_main_experiment.py` | Main results table |
| 10 | `scripts/10_failure_modes.py` | Ablations + paper-ready figures |

See `docs/problem_statement_v2.md` and the planning file in
`/Users/niuniu/.claude/plans/` for the full design.

## Quick start

```bash
git clone https://github.com/Oliver-Liu99/Counterfactual-Conversation-Emulation-for-Medical-AI-Agents.git
cd Counterfactual-Conversation-Emulation-for-Medical-AI-Agents
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # fill in API keys

# Run the toy synthetic oracle (no data / API needed) to verify estimators
pytest tests/test_synthetic_oracle.py -v
```

## Verifying API connectivity

Before running any real-API pipeline step (judges, agents), verify that
each provider's SDK + key is wired up correctly:

```bash
export ANTHROPIC_API_KEY=...   # for the debate defender
export OPENAI_API_KEY=...      # for the debate judge / pi_agent sampling
export GOOGLE_API_KEY=...      # for the debate prosecutor

PYTHONPATH=src python3 scripts/verify_apis.py            # require all 3
PYTHONPATH=src python3 scripts/verify_apis.py --skip-missing  # CI-friendly
```

The script prints a per-provider PASS/FAIL/SKIP table from a 1-token
`health_check()` ping. When all three keys are present it additionally
runs ONE end-to-end debate-judge round (defender=Anthropic,
prosecutor=Google, judge=OpenAI) on a tiny synthetic case to verify the
full chain. Exit code is non-zero if any configured provider fails, so
you can drop the `--skip-missing` invocation directly into CI.

## Hard constraints (built into the code)

1. **Time-zero**: `x` must contain only pre-decision information. Loaders
   strip the SOAP "Assessment & Plan", post-decision dialogue, ICD codes,
   and post-encounter test results. Layer-1 leakage probe enforces this.
2. **Training-set decontamination**: Layer-2 audit checks all candidate
   π_agent training corpora against test data via n-gram exact match and
   semantic similarity.
3. **No GPT-only judge**: a^agent is generated by GPT-family models, so
   the debate judge requires at least one non-GPT vendor (Anthropic or
   Google) to avoid self-preference bias.
4. **Positivity gate**: IPS-type estimators only run when ESS/n > 0.1
   and DRE classifier AUC < 0.9; otherwise the pipeline reports DM + DR
   only and flags the failure mode.

## Reproducibility

This project ships three layers of reproducibility tooling so results can be
re-derived bit-for-bit:

1. **`requirements.txt`** - declares the dependency *ranges* the codebase is
   developed against (`>=` minimums only).
2. **`requirements-lock.txt`** - exact `==` pins of a known-working environment.
   Use this for paper-grade reruns:
   ```bash
   make install-locked
   ```
3. **`Makefile`** - canonical entrypoints used by both humans and CI:
   - `make install` - install with version ranges
   - `make install-locked` - install pinned versions from the lockfile
   - `make test` - run the full test suite (real-data tests skip when
     `data/raw/aci_bench/` is absent)
   - `make synthetic` - run the Step-8 synthetic OPE oracle benchmark
   - `make lint` - non-blocking `ruff` style report
   - `make clean` - remove caches and build artefacts
4. **GitHub Actions CI** (`.github/workflows/test.yml`) runs the full test
   suite on Python 3.10 and 3.11 on every push to `main` and every PR.

## Citation

If you use this code, please cite via the `CITATION.cff` in the repo root.
GitHub renders this automatically as a "Cite this repository" button.
A method paper (arXiv preprint, TBD) will be linked here once posted.
