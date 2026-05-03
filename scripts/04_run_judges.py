"""Step 4 driver: run debate vs single-judge ablation on a calibration set.

This script expects a calibration parquet (from Step 3) with columns:
    case_id, x, a, arm, human_score   (one row per (case, axis))

If --human-scores-csv is supplied it computes calibration metrics
against the gates locked in eval_config_v1.yaml.

Usage:
    python scripts/04_run_judges.py --mode debate --rubric ccema_v1 \\
        --inputs data/calibration_set.parquet --output outputs/step4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccema.judges.calibration import (
    CalibrationGates,
    calibration_report,
)
from ccema.judges.debate import DebateJudge
from ccema.judges.llm_client import MockLLMClient, make_client
from ccema.judges.rubric import DEFAULT_RUBRIC
from ccema.judges.single import SingleJudge
from ccema.utils.config import config_hash, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "debate", "mock"], default="mock")
    p.add_argument("--config", default="configs/eval_config_v1.yaml")
    p.add_argument("--rubric", default="ccema_v1")
    p.add_argument("--inputs", default=None, help="parquet/csv with case_id,x,a,arm,(human_score)")
    p.add_argument("--output-dir", default="outputs/step4_judges")
    p.add_argument("--dry-run", action="store_true", help="run on 1 synthetic case only")
    return p.parse_args()


def build_judge(mode: str, cfg: dict):
    rubric = DEFAULT_RUBRIC

    if mode == "mock":
        client = MockLLMClient(name="mock-judge", return_alpha_beta=(7.0, 3.0))
        return SingleJudge(client=client), rubric

    if mode == "single":
        s = cfg["judges"]["single_fallback"]
        client = make_client(vendor="anthropic", model=s["id"])
        return SingleJudge(client=client), rubric

    # debate
    j_cfg = cfg["judges"]["debate"]
    defender = make_client(vendor=j_cfg["defender"]["vendor"], model=j_cfg["defender"]["id"])
    prosecutor = make_client(vendor=j_cfg["prosecutor"]["vendor"], model=j_cfg["prosecutor"]["id"])
    judge = make_client(vendor=j_cfg["judge"]["vendor"], model=j_cfg["judge"]["id"])
    prompts = cfg["judges"]["prompts"]
    debate = DebateJudge.from_prompt_files(
        defender=defender,
        prosecutor=prosecutor,
        judge=judge,
        defender_path=REPO_ROOT / prompts["defender"],
        prosecutor_path=REPO_ROOT / prompts["prosecutor"],
        judge_path=REPO_ROOT / prompts["judge"],
    )
    return debate, rubric


def main() -> None:
    args = parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    print(f"Config v1 hash: {config_hash(cfg)}")

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    judge, rubric = build_judge(args.mode, cfg)
    print(f"Judge mode: {args.mode}; rubric: {rubric.name} v{rubric.version}; axes: {rubric.axes}")

    if args.dry_run or args.inputs is None:
        # Single synthetic case to verify wiring end-to-end
        x = "55F with productive cough 5 days, smoker, no SOB at rest. Lungs: rhonchi."
        a = "Assessment: acute bronchitis. Plan: supportive care, follow up 1 week."
        score = judge.score_all_axes(x, a, rubric)
        print("\nDry-run score (mock judge expected mean ≈ 0.7):")
        for axis, beta in score.axis_scores.items():
            print(f"  {axis}: alpha={beta.alpha:.2f} beta={beta.beta:.2f} mean={beta.mean:.3f}")
        return

    # Real run with input file (parquet or CSV)
    print(f"Loading inputs from {args.inputs}")
    rows = _load_inputs(args.inputs)
    print(f"  {len(rows)} rows to score")

    judge_means: list[float] = []
    human_scores: list[float] = []
    per_case = []
    for row in rows:
        ms = judge.score_all_axes(row["x"], row["a"], rubric)
        judge_means.append(ms.overall_mean)
        if "human_score" in row and row["human_score"] is not None:
            human_scores.append(float(row["human_score"]))
        per_case.append({"case_id": row["case_id"], "arm": row.get("arm"), "judge": ms.means})

    (out_dir / "scores.json").write_text(json.dumps(per_case, indent=2))

    if len(human_scores) == len(judge_means) and len(human_scores) >= 2:
        gate_cfg = cfg["calibration"]["metrics"]
        gates = CalibrationGates(
            spearman_min=gate_cfg["spearman_rho"]["gate"],
            kappa_min=gate_cfg["quadratic_weighted_kappa"]["gate"],
            icc_min=gate_cfg["icc_2_1"]["gate"],
            bland_altman_abs_max=gate_cfg["bland_altman_bias_abs"]["gate"],
        )
        rep = calibration_report(judge_means, human_scores, gates=gates)
        (out_dir / "calibration.json").write_text(json.dumps(rep.as_dict(), indent=2))
        print("\nCalibration report:")
        print(json.dumps(rep.as_dict(), indent=2))


def _load_inputs(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if p.suffix == ".csv":
        import csv

        with open(p) as f:
            return list(csv.DictReader(f))
    raise NotImplementedError(f"Unsupported input format: {p.suffix}")


if __name__ == "__main__":
    main()
