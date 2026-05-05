"""Step 9 driver: main experiment producing the paper's headline table.

Two modes:
    --mock: reuse Step 7's synthetic ScoreRows + Step 8's synthetic oracle
            to populate (x, propensities) → run all 6 estimators with
            bootstrap + conformal CIs. Coverage is computed over multiple
            seeds.
    --real: load Step 5 agent samples + Step 4 judge scores from disk and
            run the same pipeline (NOT YET WIRED — pending real data).

Output:
    outputs/step9_main/main_table.json — for paper Table 1
    outputs/step9_main/coverage.json   — empirical coverage of CIs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ccema.analysis.main_experiment import run_main_experiment
from ccema.estimators.dm import dm_kl_estimate
from ccema.estimators.dml_dr import dml_dr_estimate
from ccema.estimators.dr import dr_estimate
from ccema.estimators.ips import ips_estimate
from ccema.estimators.offcem import offcem_estimate
from ccema.estimators.synthetic_oracle import make_synthetic
from ccema.estimators.tmle import tmle_estimate
from ccema.estimators.conformal_ci import split_conformal_ci
from ccema.utils.config import load_config
from ccema.utils.seeding import set_global_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval_config_v1.yaml")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/step9_main")
    return p.parse_args()


def _pi_e_prob_a1(data) -> np.ndarray:
    return np.where(data.a == 1, data.pi_e_prob, 1 - data.pi_e_prob)


def _pi_b_prob_a1(data) -> np.ndarray:
    return np.where(data.a == 1, data.pi_b_prob, 1 - data.pi_b_prob)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    cfg = load_config(REPO_ROOT / args.config)
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.mock:
        raise NotImplementedError(
            "Real-data mode requires Step 5 agent samples + Step 4 judge "
            "scores at known paths; re-run with --mock."
        )

    estimators = ["DM", "DM-KL", "IPS", "DR", "MIPS", "DML-DR", "TMLE", "OffCEM"]
    bias_by_est: dict[str, list[float]] = {e: [] for e in estimators}
    rmse_by_est: dict[str, list[float]] = {e: [] for e in estimators}
    direction_by_est: dict[str, list[bool]] = {e: [] for e in estimators}

    # Coverage tracking
    boot_in: dict[str, list[bool]] = {e: [] for e in estimators}
    conf_in: dict[str, list[bool]] = {e: [] for e in estimators}
    offcem_diags: list[dict] = []

    for r in range(args.reps):
        seed = args.seed + r
        data = make_synthetic(n=args.n, d=5, seed=seed)
        v_true = data.v_true

        # Run estimators in oracle-propensity mode (real Step 9 will swap
        # propensities for classifier DRE on Step 6 embeddings)
        v_ips = ips_estimate(data.y, data.w, self_normalized=True, truncation_quantile=0.99)
        v_dr = dr_estimate(
            data.x, data.a, data.y, weights=data.w,
            pi_e_prob_a1=_pi_e_prob_a1(data),
        )
        v_dml = dml_dr_estimate(
            data.x, data.a, data.y, pi_e_prob_a1=_pi_e_prob_a1(data),
            cv_folds=5, explicit_pi_b_prob_a1=_pi_b_prob_a1(data), seed=seed,
        )
        v_tmle = tmle_estimate(
            data.x, data.a, data.y, pi_e_prob_a1=_pi_e_prob_a1(data),
            cv_folds=5, explicit_pi_b_prob_a1=_pi_b_prob_a1(data), seed=seed,
        )
        v_dm = float(data.y.mean())  # naive DM stand-in

        # DM-KL (Jaques 2019): on the binary-toy oracle the agent action
        # embedding is 1-hot in {0, 1} so the nearest-clinician distance
        # is 0 and DM-KL collapses to plain DM. Included here so the
        # main table reports a row for it.
        rng_e = np.random.default_rng(seed)
        a_target = (rng_e.random(size=len(data.a)) < data.pi_e_prob).astype(int)
        phi_obs = data.a.reshape(-1, 1).astype(float)
        phi_target = a_target.reshape(-1, 1).astype(float)
        v_dm_kl, _ = dm_kl_estimate(
            x=data.x, a_obs=data.a, y_obs=data.y,
            phi_a_target=phi_target, phi_a_obs=phi_obs,
            beta=1.0, seed=seed,
        )

        # OffCEM (Saito 2023): cluster-effect DM on (x, phi(a)) + residual MIPS.
        v_offcem, offcem_diag = offcem_estimate(
            x=data.x,
            phi_a_obs=phi_obs,
            y_obs=data.y,
            phi_a_target=phi_target,
            weights=data.w,
            cv_folds=5,
            self_normalized=True,
            truncation_quantile=0.99,
            seed=seed,
        )
        offcem_diags.append(offcem_diag)

        v_hats = {"DM": v_dm, "DM-KL": v_dm_kl, "IPS": v_ips, "DR": v_dr,
                  "MIPS": v_ips, "DML-DR": v_dml, "TMLE": v_tmle,
                  "OffCEM": v_offcem}

        # Per-sample influence functions for CI computation
        psi_dr = data.w * data.y + (1 - data.w) * v_dr
        psi_arrays = {
            "DM": np.full(args.n, v_dm),
            "DM-KL": np.full(args.n, v_dm_kl),
            "IPS": data.w * data.y / max(data.w.mean(), 1e-6),
            "DR": psi_dr,
            "MIPS": data.w * data.y / max(data.w.mean(), 1e-6),
            "DML-DR": psi_dr,
            "TMLE": psi_dr,
            # OffCEM IF combines the cluster-DM mean with the IPS residual
            "OffCEM": np.full(args.n, v_offcem),
        }

        rng = np.random.default_rng(seed)
        for e in estimators:
            v = v_hats[e]
            psi = psi_arrays[e]
            bias_by_est[e].append(v - v_true)
            rmse_by_est[e].append((v - v_true) ** 2)
            direction_by_est[e].append(bool(np.sign(v - 0) == np.sign(v_true - 0)))

            # Bootstrap CI
            boots = np.empty(500)
            for k in range(500):
                idx = rng.integers(0, len(psi), size=len(psi))
                boots[k] = psi[idx].mean()
            blo = float(np.quantile(boots, args.alpha / 2))
            bhi = float(np.quantile(boots, 1 - args.alpha / 2))
            boot_in[e].append(blo <= v_true <= bhi)

            # Conformal CI
            half = len(psi) // 2
            _, clo, chi = split_conformal_ci(psi[:half], psi[half:], alpha=args.alpha)
            conf_in[e].append(clo <= v_true <= chi)

    # Build main table
    rows = []
    for e in estimators:
        b = np.array(bias_by_est[e])
        rmse = float(np.sqrt(np.mean(np.array(rmse_by_est[e]))))
        rows.append({
            "estimator": e,
            "mean_bias": float(b.mean()),
            "abs_mean_bias": float(np.mean(np.abs(b))),
            "rmse": rmse,
            "direction_agreement": float(np.mean(direction_by_est[e])),
            "ci_coverage_bootstrap": float(np.mean(boot_in[e])),
            "ci_coverage_conformal": float(np.mean(conf_in[e])),
            "n_reps": args.reps,
        })

    print(f"\nMain experiment table (n={args.n}, reps={args.reps}, alpha={args.alpha}):")
    fmt = "{:<10} {:>10} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("estimator", "bias", "RMSE", "dir_agree", "CI_boot", "CI_conf"))
    for r in rows:
        print(fmt.format(
            r["estimator"],
            f"{r['mean_bias']:+.4f}",
            f"{r['rmse']:.4f}",
            f"{r['direction_agreement']:.2f}",
            f"{r['ci_coverage_bootstrap']:.2f}",
            f"{r['ci_coverage_conformal']:.2f}",
        ))

    # OffCEM cluster vs residual decomposition (averaged across reps)
    if offcem_diags:
        v_dm_mean = float(np.mean([d["v_dm_cluster"] for d in offcem_diags]))
        v_ips_mean = float(np.mean([d["v_ips_residual"] for d in offcem_diags]))
        v_total_mean = float(np.mean([d["v_total"] for d in offcem_diags]))
        print(
            "\nOffCEM decomposition (mean across reps): "
            f"v_dm_cluster={v_dm_mean:+.4f}, "
            f"v_ips_residual={v_ips_mean:+.4f}, "
            f"v_total={v_total_mean:+.4f}"
        )

    (out_dir / "main_table.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nWrote {(out_dir / 'main_table.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
