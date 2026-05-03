"""Step 2 driver: load a dataset, run dual-layer leakage audit, write report.

Usage examples:
    python scripts/02_data_audit.py --dataset synthetic --n 60
    python scripts/02_data_audit.py --dataset aci_bench --root data/raw/aci_bench
    python scripts/02_data_audit.py --dataset mts_dialog --root data/raw/mts_dialog \\
        --decontamination-corpus data/raw/medqa.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccema.data.holdout import make_leak_probe_holdout
from ccema.data.leakage import (
    audit_dataset_layer1,
    layer2_audit,
    ngrams,
    summarize,
)
from ccema.data.loaders import load_dataset
from ccema.utils.seeding import set_global_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "aci_bench", "mts_dialog"])
    p.add_argument("--root", default=None)
    p.add_argument("--n", type=int, default=60, help="synthetic only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--decontamination-corpus",
        action="append",
        default=[],
        help="path to corpus .txt or .jsonl for layer-2 ngram check (repeat per corpus)",
    )
    p.add_argument("--ngram-n", type=int, default=8)
    p.add_argument("--cosine-threshold", type=float, default=0.90)
    p.add_argument("--output-dir", default="outputs/step2_audit")
    p.add_argument("--holdout-frac", type=float, default=0.10)
    return p.parse_args()


def load_corpus_ngrams(paths: list[str], n: int) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for p in paths:
        text = _read_text_corpus(p)
        out[Path(p).stem] = ngrams(text, n=n)
    return out


def _read_text_corpus(path: str) -> str:
    p = Path(path)
    if p.suffix == ".jsonl":
        chunks: list[str] = []
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                # heuristic — concatenate all string values
                for v in rec.values():
                    if isinstance(v, str):
                        chunks.append(v)
        return "\n".join(chunks)
    return p.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if args.dataset == "synthetic":
        kwargs = {"n_encounters": args.n, "seed": args.seed}
    encounters = load_dataset(args.dataset, root=args.root, **kwargs)
    print(f"Loaded {len(encounters)} encounters from {args.dataset}")

    main_set, leak_probe = make_leak_probe_holdout(encounters, holdout_frac=args.holdout_frac, seed=args.seed)
    print(f"  main: {len(main_set)}   leak_probe (truncated): {len(leak_probe)}")

    layer1 = audit_dataset_layer1(main_set)

    layer2 = []
    if args.decontamination_corpus:
        corpus_ngrams = load_corpus_ngrams(args.decontamination_corpus, n=args.ngram_n)
        print(f"Layer 2: {len(corpus_ngrams)} corpora loaded for n-gram check")
        for enc in main_set:
            layer2.append(
                layer2_audit(
                    enc,
                    corpus_ngrams=corpus_ngrams,
                    n=args.ngram_n,
                    cosine_threshold=args.cosine_threshold,
                )
            )

    summary = summarize(layer1, layer2)

    report = {
        "dataset": args.dataset,
        "n_encounters_main": summary.n_encounters,
        "n_encounters_leak_probe": len(leak_probe),
        "layer1": {
            "n_clean": summary.n_layer1_clean,
            "failures": summary.layer1_failures,
        },
        "layer2": {
            "n_flagged": summary.n_layer2_flagged,
            "corpora_checked": [Path(p).stem for p in args.decontamination_corpus],
        },
    }

    report_path = out_dir / f"audit_{args.dataset}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {report_path.relative_to(REPO_ROOT)}")
    print(json.dumps(report, indent=2))

    if summary.n_layer1_clean < summary.n_encounters:
        print("\n  Layer-1 leakage detected. Review report and re-run with stricter time-zero cuts.")


if __name__ == "__main__":
    main()
