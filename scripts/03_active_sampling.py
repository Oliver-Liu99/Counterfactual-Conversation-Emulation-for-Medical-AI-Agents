"""Step 3 driver: score full D with single judge, select calibration set.

Pipeline:
    1. Load encounters (synthetic by default; --dataset to override)
    2. Synthesize a 'clinician' arm and a placeholder 'agent_strong' arm
       (just a templated rewrite for dry-run; real Step 5 outputs go here)
    3. Run single-judge (mock by default) on (x, a) for each arm
    4. Compute acquisition scores
    5. Select 100 cases stratified by ICD chapter
    6. Write data/calibration_set.parquet (jsonl fallback if pandas missing)

Usage:
    python scripts/03_active_sampling.py --dataset synthetic --n 100 --target 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccema.analysis.active_sampling import (
    CaseRecord,
    SamplingConfig,
    acquisition_score,
    select_calibration_set,
)
from ccema.data.loaders import load_dataset
from ccema.judges.llm_client import MockLLMClient
from ccema.judges.rubric import DEFAULT_RUBRIC
from ccema.judges.single import SingleJudge
from ccema.utils.config import load_config
from ccema.utils.seeding import set_global_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval_config_v1.yaml")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "aci_bench", "mts_dialog"])
    p.add_argument("--root", default=None)
    p.add_argument("--n", type=int, default=200, help="size of source pool (synthetic only)")
    p.add_argument("--target", type=int, default=100, help="calibration set size")
    p.add_argument("--min-chapters", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge-mode", choices=["mock", "single"], default="mock")
    p.add_argument("--output-dir", default="data")
    return p.parse_args()


def make_placeholder_agent(a_clin: str) -> str:
    """Produce a placeholder agent A&P from clinician's. Step 5 will replace this."""
    return a_clin.replace("supportive care", "antibiotic course") + "\n[placeholder agent variant]"


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    cfg = load_config(REPO_ROOT / args.config)

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if args.dataset == "synthetic":
        kwargs = {"n_encounters": args.n, "seed": args.seed}
    encounters = load_dataset(args.dataset, root=args.root, **kwargs)
    print(f"Loaded {len(encounters)} encounters from {args.dataset}")

    # Build judge
    if args.judge_mode == "mock":
        # Heterogeneous mock — vary alpha/beta by encounter index to create
        # meaningful disagreement and uncertainty distributions
        import random as _r

        rng = _r.Random(args.seed)

        class HetMock(MockLLMClient):
            def complete(self, system, user, *, max_tokens=1024, temperature=0.0, json_mode=False):
                # Sample alpha,beta deterministically from user content hash
                h = hash(user) & 0xFFFF
                rng.seed(h)
                # Concentration in [4, 30]
                conc = 4 + rng.random() * 26
                mean = 0.3 + rng.random() * 0.6
                a = mean * conc
                b = (1 - mean) * conc
                return MockLLMClient(return_alpha_beta=(a, b)).complete(
                    system, user, max_tokens=max_tokens, temperature=temperature, json_mode=True
                )

        judge = SingleJudge(client=HetMock())
    else:
        from ccema.judges.llm_client import make_client

        s = cfg["judges"]["single_fallback"]
        judge = SingleJudge(client=make_client("anthropic", s["id"]))

    # Score every encounter on (clinician arm) and (placeholder agent arm)
    case_records: list[CaseRecord] = []
    for i, enc in enumerate(encounters):
        a_clin = enc.a_clinician
        a_agent = make_placeholder_agent(a_clin)
        ms_clin = judge.score_all_axes(enc.x, a_clin, DEFAULT_RUBRIC)
        ms_agent = judge.score_all_axes(enc.x, a_agent, DEFAULT_RUBRIC)
        case_records.append(
            CaseRecord(
                case_id=enc.case_id,
                x=enc.x,
                arm_texts={"clinician": a_clin, "agent_strong": a_agent},
                arm_scores={"clinician": ms_clin, "agent_strong": ms_agent},
                metadata={
                    "icd_chapter": enc.metadata.get("icd_chapter", "_unk"),
                    "age_band": enc.metadata.get("age_band", "_unk"),
                    "source": enc.source,
                },
            )
        )
        if (i + 1) % 50 == 0:
            print(f"  scored {i + 1}/{len(encounters)}")

    # Select
    sampling_cfg = SamplingConfig(
        n_total=args.target,
        min_chapters=args.min_chapters,
        stratify_keys=("icd_chapter",),
        seed=args.seed,
    )
    selected = select_calibration_set(case_records, sampling_cfg)
    print(f"\nSelected {len(selected)} cases")

    chapters = {c.metadata["icd_chapter"] for c in selected}
    print(f"  unique ICD chapters: {len(chapters)} ({sorted(chapters)})")

    # Persist
    out_path = out_dir / "calibration_set.jsonl"
    with open(out_path, "w") as f:
        for c in selected:
            row = {
                "case_id": c.case_id,
                "x": c.x,
                "arm_texts": c.arm_texts,
                "arm_judge_means": {arm: ms.means for arm, ms in c.arm_scores.items()},
                "arm_judge_overall": {arm: ms.overall_mean for arm, ms in c.arm_scores.items()},
                "metadata": c.metadata,
                "acquisition": acquisition_score(c),
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")

    # Summary log
    log_path = out_dir / "active_sampling_log.json"
    log_path.write_text(
        json.dumps(
            {
                "n_pool": len(case_records),
                "n_selected": len(selected),
                "n_chapters": len(chapters),
                "chapters": sorted(chapters),
                "judge_mode": args.judge_mode,
                "min_chapters": args.min_chapters,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    print(f"Wrote {log_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
