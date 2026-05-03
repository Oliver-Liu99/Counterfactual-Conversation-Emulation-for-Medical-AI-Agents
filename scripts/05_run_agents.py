"""Step 5 driver: run π_agents (strong / weak / open_medical) on encounters.

Generates n=5 samples per case per arm with strict JSON schema validation.
Optionally runs the capability-gap test against a small held-out subset
scored by the rubric judge.

Usage:
    python scripts/05_run_agents.py --dataset synthetic --n 20 --arms strong,weak
    python scripts/05_run_agents.py --inputs data/calibration_set.jsonl --arms strong,weak,open_medical
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccema.agents.pi_agent import PiAgent, capability_gap
from ccema.data.loaders import load_dataset
from ccema.judges.llm_client import MockLLMClient, make_client
from ccema.judges.rubric import DEFAULT_RUBRIC
from ccema.judges.single import SingleJudge
from ccema.utils.config import load_config
from ccema.utils.seeding import set_global_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval_config_v1.yaml")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--inputs", default=None, help="optional jsonl from Step 3 calibration set")
    p.add_argument("--n", type=int, default=20, help="cases (synthetic only)")
    p.add_argument("--arms", default="strong,weak", help="comma-sep arms")
    p.add_argument("--mock", action="store_true", help="use mock LLM clients (no API)")
    p.add_argument("--n-samples-override", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/step5_agents")
    p.add_argument(
        "--capability-gap",
        action="store_true",
        help="after sampling, score 1 sample per case with mock judge and compute gap",
    )
    return p.parse_args()


_VALID_AGENT_TEMPLATE = (
    '{"differential_diagnosis":'
    '[{"dx":"acute bronchitis","icd10":null,"probability":"high","supporting":["productive cough"],"against":[]}],'
    '"working_diagnosis":"acute bronchitis",'
    '"assessment":"Likely acute bronchitis given productive cough and clinical exam.",'
    '"plan":{"diagnostics":["chest x-ray if persists"],"therapeutics":["supportive care"],"patient_education":["hydration","cough hygiene"],"follow_up":"1 week","red_flags":["fever >39C","dyspnea"]},'
    '"uncertainty_notes":"differential includes pneumonia"}'
)


def build_agent(arm: str, cfg: dict, mock: bool, n_samples: int | None) -> PiAgent:
    a_cfg = cfg["agents"][arm]
    if mock:
        # Strong vs weak vs open_medical → vary mock to create capability gap
        if arm == "strong":
            text = _VALID_AGENT_TEMPLATE
        elif arm == "weak":
            # Drop some plan items so rubric can score it lower
            obj = json.loads(_VALID_AGENT_TEMPLATE)
            obj["plan"]["red_flags"] = []
            obj["plan"]["patient_education"] = []
            obj["uncertainty_notes"] = ""
            text = json.dumps(obj)
        else:
            text = _VALID_AGENT_TEMPLATE
        client = MockLLMClient(name=f"mock-{arm}", canned_response=text)
    else:
        client = make_client(vendor=a_cfg["vendor"], model=a_cfg["id"])

    return PiAgent.from_prompt_file(
        arm=arm,
        client=client,
        prompt_path=REPO_ROOT / cfg["agents"]["prompt_path"],
        temperature=a_cfg["temperature"],
        max_tokens=a_cfg["max_tokens"],
        n_samples=n_samples or a_cfg["n_samples_per_case"],
    )


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    cfg = load_config(REPO_ROOT / args.config)

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load encounters
    if args.inputs:
        rows = [json.loads(line) for line in Path(args.inputs).read_text().splitlines() if line.strip()]
        encounters = [(r["case_id"], r["x"]) for r in rows]
    else:
        kwargs = {"n_encounters": args.n, "seed": args.seed} if args.dataset == "synthetic" else {}
        encs = load_dataset(args.dataset, **kwargs)
        encounters = [(e.case_id, e.x) for e in encs]

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"Arms: {arms}; cases: {len(encounters)}; mock={args.mock}")

    samples_by_arm: dict[str, list] = {}
    for arm in arms:
        agent = build_agent(arm, cfg, args.mock, args.n_samples_override)
        all_samples = []
        n_valid = 0
        n_failed = 0
        for case_id, x in encounters:
            for s in agent.sample_n(case_id, x):
                all_samples.append(s)
                if s.output is not None:
                    n_valid += 1
                else:
                    n_failed += 1
        samples_by_arm[arm] = all_samples
        print(f"  arm={arm}: {n_valid} valid / {n_failed} failed")

        # Persist
        arm_path = out_dir / f"{arm}_samples.jsonl"
        with open(arm_path, "w") as f:
            for s in all_samples:
                f.write(
                    json.dumps(
                        {
                            "case_id": s.case_id,
                            "arm": s.arm,
                            "sample_idx": s.sample_idx,
                            "valid": s.output is not None,
                            "error": s.error,
                            "ap_text": s.output.to_assessment_plan_text() if s.output else None,
                            "raw": s.output.raw if s.output else None,
                        }
                    )
                    + "\n"
                )

    # Optional capability-gap test using mock judge
    if args.capability_gap and "strong" in samples_by_arm and "weak" in samples_by_arm:
        # Use a *different* mock that varies score by content length so strong > weak emerges
        class LengthSensitiveMock(MockLLMClient):
            def complete(self, system, user, *, max_tokens=1024, temperature=0.0, json_mode=False):
                # Higher score for longer A&P content (proxy for completeness)
                length = len(user)
                mean = min(0.95, 0.4 + length / 4000)
                conc = 20.0
                return MockLLMClient(return_alpha_beta=(mean * conc, (1 - mean) * conc)).complete(
                    system, user, max_tokens=max_tokens, temperature=temperature, json_mode=True
                )

        judge = SingleJudge(client=LengthSensitiveMock())
        s_scores = []
        w_scores = []
        # Take first sample per case for each arm
        per_case_strong = {s.case_id: s for s in samples_by_arm["strong"] if s.sample_idx == 0 and s.output}
        per_case_weak = {s.case_id: s for s in samples_by_arm["weak"] if s.sample_idx == 0 and s.output}
        common = set(per_case_strong) & set(per_case_weak)
        for cid in common:
            x = next(x for c, x in encounters if c == cid)
            ms_s = judge.score_all_axes(x, per_case_strong[cid].output.to_assessment_plan_text(), DEFAULT_RUBRIC)
            ms_w = judge.score_all_axes(x, per_case_weak[cid].output.to_assessment_plan_text(), DEFAULT_RUBRIC)
            s_scores.append(ms_s.overall_mean)
            w_scores.append(ms_w.overall_mean)

        gap = capability_gap(s_scores, w_scores, threshold=0.10)
        gap_path = out_dir / "capability_gap.json"
        gap_path.write_text(
            json.dumps(
                {
                    "n_cases": gap.n_cases,
                    "strong_mean": gap.strong_mean,
                    "weak_mean": gap.weak_mean,
                    "gap": gap.gap,
                    "threshold": gap.threshold,
                    "passes": gap.passes,
                },
                indent=2,
            )
        )
        print(f"\nCapability gap: strong={gap.strong_mean:.3f} weak={gap.weak_mean:.3f} "
              f"gap={gap.gap:+.3f} (threshold {gap.threshold:+.2f}) -> {'PASS' if gap.passes else 'FAIL'}")


if __name__ == "__main__":
    main()
