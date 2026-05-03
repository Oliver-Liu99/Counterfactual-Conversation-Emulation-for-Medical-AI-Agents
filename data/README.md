# `data/` — datasets

Raw datasets are **never committed**. The `.gitignore` at the repo root
excludes `data/raw/`, `data/processed/`, and any top-level `.csv` / `.jsonl`
/ `.parquet` / `.json` files. Only `data/synthetic/` (toy fixtures) and this
README are tracked.

## Layout

```
data/
  README.md            # this file
  synthetic/           # tiny synthetic fixtures shipped with the repo
  raw/                 # GIT-IGNORED — public clinical datasets you fetch
    aci_bench/
    mts_dialog/
  processed/           # GIT-IGNORED — derived caches
```

## ACI-Bench

[ACI-Bench (Yim et al. 2023, Sci Data)](https://www.nature.com/articles/s41597-023-02487-3)
is a public corpus of paired ambient-clinical-intelligence dialogues + clinical
notes. It is hosted as a regular GitHub repo with no extra license barrier —
you simply clone it.

```
git clone https://github.com/wyim/aci-bench data/raw/aci_bench
```

After cloning, the loader at `src/ccema/data/aci_bench.py` reads:

```
data/raw/aci_bench/data/challenge_data/{train,valid,clinicalnlp_taskB_test1,
                                        clinicalnlp_taskC_test2,
                                        clef_taskC_test3}.csv
```

Each CSV has columns `dataset, encounter_id, dialogue, note`. The loader maps
clinical-note headers (CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, PHYSICAL
EXAM, ASSESSMENT AND PLAN, …) into the four canonical SOAP buckets and builds
`Encounter` objects with `case_id = "aci_bench/{dataset}/{encounter_id}"`.

Run the leakage audit end-to-end with:

```
PYTHONPATH=src python3 scripts/02_data_audit.py --dataset aci_bench \
    --root data/raw/aci_bench
```

## MTS-Dialog

[MTS-Dialog (Ben Abacha et al. 2023, EACL)](https://github.com/abachaa/MTS-Dialog)
ships `MTS-Dialog-TrainingSet.csv` and friends. Clone into `data/raw/mts_dialog/`
the same way; see `src/ccema/data/mts_dialog.py` for expected layout.

## Synthetic

`data/synthetic/` and `src/ccema/data/synthetic.py` provide toy encounters used
by the test-suite and as a smoke-test for downstream pipelines. No external
download required; everything runs on a fresh clone.
