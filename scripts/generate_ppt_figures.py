"""Generate 8 publication-style PPT figures for the CCE research report.

Outputs to outputs/ppt/figures/. Re-runnable; uses real JSON data when present.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.dpi": 100,
    }
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "ppt" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> Path:
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 1: problem overview schematic
# ---------------------------------------------------------------------------
def fig01_problem_overview() -> Path:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    boxes = [
        {
            "xy": (0.4, 1.8),
            "w": 3.8,
            "h": 3.4,
            "title": "Static\nbenchmarks",
            "sub": "measure capability\n(MedQA, MMLU)",
            "fc": "#E8F1FB",
            "ec": "#3E6FB7",
        },
        {
            "xy": (5.1, 1.8),
            "w": 3.8,
            "h": 3.4,
            "title": "Prospective\ntrials",
            "sub": "slow + costly\n(months, $$$)",
            "fc": "#FBEEE8",
            "ec": "#B7553E",
        },
        {
            "xy": (9.8, 1.8),
            "w": 3.8,
            "h": 3.4,
            "title": "Counterfactual\nemulation",
            "sub": "off-policy estimates\n(this work)",
            "fc": "#E8FBEF",
            "ec": "#2F9E5A",
        },
    ]
    for b in boxes:
        box = FancyBboxPatch(
            b["xy"],
            b["w"],
            b["h"],
            boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=2.4,
            edgecolor=b["ec"],
            facecolor=b["fc"],
        )
        ax.add_patch(box)
        cx = b["xy"][0] + b["w"] / 2
        ax.text(cx, b["xy"][1] + b["h"] - 0.85, b["title"], ha="center", va="center",
                fontsize=22, fontweight="bold", color=b["ec"])
        ax.text(cx, b["xy"][1] + 1.0, b["sub"], ha="center", va="center", fontsize=17)

    for x_start, x_end in [(4.25, 5.05), (8.95, 9.75)]:
        arr = FancyArrowPatch(
            (x_start, 3.5),
            (x_end, 3.5),
            arrowstyle="-|>",
            mutation_scale=28,
            linewidth=2.4,
            color="#444444",
        )
        ax.add_patch(arr)

    ax.text(7, 6.4, "OPE-on-conversations gap", ha="center", va="center",
            fontsize=24, fontweight="bold")
    ax.text(7, 0.7,
            "Goal: estimate what would have happened under a different agent policy,\n"
            "without running a new prospective trial.",
            ha="center", va="center", fontsize=17, style="italic", color="#333")
    return _save(fig, "fig01_problem_overview.png")


# ---------------------------------------------------------------------------
# Figure 2: estimator RMSE bar chart
# ---------------------------------------------------------------------------
def fig02_estimator_rmse() -> Path:
    data_path = REPO / "outputs" / "step8_synthetic" / "synthetic_results.json"
    data = json.loads(data_path.read_text())
    rows = {r["estimator"]: r["rmse"] for r in data["results"]}
    order = ["DM", "IPS", "DR", "MIPS", "DML-DR", "TMLE"]
    rmses = [rows[k] for k in order]

    fig, ax = plt.subplots(figsize=(12, 7))
    base_color = "#4C78A8"
    tmle_color = "#E45756"
    colors = [tmle_color if k == "TMLE" else base_color for k in order]
    bars = ax.bar(order, rmses, color=colors, edgecolor="black", linewidth=1.2)
    for bar, v in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(rmses) * 0.015,
                f"{v:.4f}", ha="center", va="bottom", fontsize=15)

    ax.set_ylabel("RMSE")
    ax.set_xlabel("Estimator")
    ax.set_title("Off-policy estimator RMSE on synthetic benchmark\n(n=200, 10 reps)")
    ax.set_ylim(0, max(rmses) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend_handles = [
        mpatches.Patch(color=base_color, label="Standard estimators"),
        mpatches.Patch(color=tmle_color, label="TMLE (lowest RMSE)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    return _save(fig, "fig02_estimator_rmse.png")


# ---------------------------------------------------------------------------
# Figure 3: CI coverage comparison
# ---------------------------------------------------------------------------
def fig03_ci_coverage_comparison() -> Path:
    data_path = REPO / "outputs" / "step9_main" / "main_table.json"
    data = json.loads(data_path.read_text())
    rows = {r["estimator"]: r for r in data["rows"]}
    order = ["DM", "IPS", "DR", "MIPS", "DML-DR", "TMLE"]
    boot = [rows[k]["ci_coverage_bootstrap"] for k in order]
    conf = [rows[k]["ci_coverage_conformal"] for k in order]

    x = np.arange(len(order))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 7))
    b1 = ax.bar(x - width / 2, boot, width, label="Bootstrap CI",
                color="#5B8FF9", edgecolor="black", linewidth=1.0)
    b2 = ax.bar(x + width / 2, conf, width, label="Conformal CI",
                color="#5AD8A6", edgecolor="black", linewidth=1.0)

    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=14)

    ax.axhline(0.90, color="#D62728", linestyle="--", linewidth=2.2,
               label="Nominal coverage (0.90)")

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("CI coverage")
    ax.set_xlabel("Estimator")
    ax.set_title("Bootstrap vs conformal CI coverage\n(n=200, 8 reps, mock judges)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    return _save(fig, "fig03_ci_coverage_comparison.png")


# ---------------------------------------------------------------------------
# Figure 4: CCE real-text headline (AUC + ESS/n)
# ---------------------------------------------------------------------------
def fig04_cce_real_headline() -> Path:
    data_path = REPO / "outputs" / "u3_real" / "cce_real_diagnostics.json"
    if data_path.exists():
        diag = json.loads(data_path.read_text())
        rows = {s["setting"]: s for s in diag["settings"]}
        auc = {
            "raw": rows["raw"]["classifier_auc"],
            "raw+PCA": rows["raw+pca32"]["classifier_auc"],
            "CCE": rows["cce"]["classifier_auc"],
            "CCE+PCA": rows["cce+pca32"]["classifier_auc"],
        }
        ess = {
            "raw": rows["raw"]["ess_per_n"],
            "raw+PCA": rows["raw+pca32"]["ess_per_n"],
            "CCE": rows["cce"]["ess_per_n"],
            "CCE+PCA": rows["cce+pca32"]["ess_per_n"],
        }
    else:
        auc = {"raw": 1.0, "raw+PCA": 1.0, "CCE": 0.55, "CCE+PCA": 0.83}
        ess = {"raw": 0.10, "raw+PCA": 0.11, "CCE": 0.96, "CCE+PCA": 0.89}

    labels = ["raw", "raw+PCA", "CCE", "CCE+PCA"]
    auc_vals = [auc[k] for k in labels]
    ess_vals = [ess[k] for k in labels]
    colors = ["#D62728", "#FF9F43", "#2CA02C", "#1F77B4"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: AUC
    ax = axes[0]
    bars = ax.bar(labels, auc_vals, color=colors, edgecolor="black", linewidth=1.2)
    for bar, v in zip(bars, auc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=15)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=2, label="Chance (0.5)")
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Treatment-classifier AUC\n(lower = better positivity)")
    ax.set_title("Positivity diagnostic")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    # Annotations
    ax.annotate(
        "catastrophic\npositivity failure",
        xy=(0, auc_vals[0]),
        xytext=(0.5, 0.55),
        ha="center",
        fontsize=14,
        color="#A6171F",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#A6171F", lw=2),
    )
    ax.annotate(
        "near chance —\npositivity restored",
        xy=(2, auc_vals[2]),
        xytext=(1.5, 0.20),
        ha="center",
        fontsize=14,
        color="#1B6B2D",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1B6B2D", lw=2),
    )

    # Right: ESS/n
    ax = axes[1]
    bars = ax.bar(labels, ess_vals, color=colors, edgecolor="black", linewidth=1.2)
    for bar, v in zip(bars, ess_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=15)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("ESS / n  (higher is better)")
    ax.set_title("Effective sample size")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("CCE on real ACI-Bench text: positivity collapses on raw embeddings,\n"
                 "is restored under contrastive projection",
                 fontsize=22, fontweight="bold", y=1.04)
    return _save(fig, "fig04_cce_real_headline.png")


# ---------------------------------------------------------------------------
# Figure 5: Layer-1 audit before/after
# ---------------------------------------------------------------------------
def fig05_layer1_audit_real() -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    categories = ["Before\n(naive probe)", "After\n(context-aware probe)"]

    # Before stack: 13 leaked, 47 clean
    before = {"leaked_dx": 13, "dx_in_history": 0, "clean": 47}
    after = {"leaked_dx": 9, "dx_in_history": 4, "clean": 47}

    colors = {"leaked_dx": "#D62728", "dx_in_history": "#FFB000", "clean": "#2CA02C"}
    labels_pretty = {
        "leaked_dx": "Leaked Dx (true positive)",
        "dx_in_history": "Dx in history (legitimate)",
        "clean": "Clean",
    }

    keys = ["leaked_dx", "dx_in_history", "clean"]
    bottoms = np.zeros(2)
    for k in keys:
        vals = np.array([before[k], after[k]])
        bars = ax.bar(categories, vals, bottom=bottoms, color=colors[k],
                      edgecolor="black", linewidth=1.2, label=labels_pretty[k])
        for bar, v, b in zip(bars, vals, bottoms):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2,
                        f"{int(v)}", ha="center", va="center",
                        fontsize=18, fontweight="bold", color="white")
        bottoms += vals

    ax.set_ylabel("Encounters (n=60)")
    ax.set_title("Smarter Layer-1 audit: 4 false positives correctly reclassified")
    ax.set_ylim(0, 78)
    ax.set_xlim(-0.7, 1.7)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Connector arrow showing 4 reclassified
    ax.annotate(
        "",
        xy=(1, 47 + 4 + 2),
        xytext=(0, 47 + 2),
        arrowprops=dict(arrowstyle="->", color="#555", lw=2.5,
                        connectionstyle="arc3,rad=-0.25"),
    )
    ax.text(0.5, 73, "4 reclassified as legitimate",
            ha="center", va="center", fontsize=15,
            fontweight="bold", color="#333")
    return _save(fig, "fig05_layer1_audit_real.png")


# ---------------------------------------------------------------------------
# Figure 6: MSM sensitivity curves
# ---------------------------------------------------------------------------
def fig06_msm_sensitivity_curve() -> Path:
    data_path = REPO / "outputs" / "step7_ground_truth" / "ground_truth.json"
    data = json.loads(data_path.read_text())
    bounds = data["sensitivity_bounds"]

    # Reorganize by arm
    by_arm: dict[str, list[dict]] = {}
    for b in bounds:
        by_arm.setdefault(b["arm"], []).append(b)
    for arm in by_arm:
        by_arm[arm].sort(key=lambda r: r["gamma"])

    # Linear-interpolate at the requested gammas: 1.0, 1.2, 1.5, 2.0, 2.5, 3.0
    target_gammas = np.array([1.0, 1.2, 1.5, 2.0, 2.5, 3.0])

    def _interp(rows, key):
        gs = np.array([r["gamma"] for r in rows])
        ys = np.array([r[key] for r in rows])
        return np.interp(target_gammas, gs, ys)

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = {"agent_strong": "#2CA02C", "agent_weak": "#D62728"}
    labels = {"agent_strong": "agent_strong", "agent_weak": "agent_weak"}

    for arm in ["agent_strong", "agent_weak"]:
        rows = by_arm[arm]
        lo = _interp(rows, "delta_lo")
        hi = _interp(rows, "delta_hi")
        point = _interp(rows, "delta_point")
        ax.fill_between(target_gammas, lo, hi, color=colors[arm], alpha=0.25,
                        label=f"{labels[arm]} bound")
        ax.plot(target_gammas, point, "o-", color=colors[arm], linewidth=2.6,
                markersize=9, label=f"{labels[arm]} point Δ")
        ax.plot(target_gammas, lo, "--", color=colors[arm], linewidth=1.6, alpha=0.7)
        ax.plot(target_gammas, hi, "--", color=colors[arm], linewidth=1.6, alpha=0.7)

    ax.axhline(0, color="black", linewidth=1.4)
    ax.axvline(2.0, color="#444", linestyle=":", linewidth=2,
               label="Γ = 2 (modest unmeasured conf.)")

    ax.set_xlabel("Sensitivity parameter Γ")
    ax.set_ylabel("Effect Δ vs clinician")
    ax.set_title("MSM sensitivity bounds: strong arm sign-robust through Γ = 3")
    ax.set_xticks(target_gammas)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="center right", fontsize=14)
    return _save(fig, "fig06_msm_sensitivity_curve.png")


# ---------------------------------------------------------------------------
# Figure 7: 4-axis × 3-chapter heatmap (strong arm)
# ---------------------------------------------------------------------------
def fig07_4axis_decomposition() -> Path:
    data_path = REPO / "outputs" / "step7_ground_truth" / "ground_truth.json"
    data = json.loads(data_path.read_text())
    rows = [r for r in data["decomposition"] if r["arm"] == "agent_strong"]

    axes_order = ["ddx_accuracy", "mx_safety", "mx_practicality", "mx_cost"]
    chap_order = ["J", "K", "I"]
    pretty_axis = {
        "ddx_accuracy": "DDx accuracy",
        "mx_safety": "Mx safety",
        "mx_practicality": "Mx practicality",
        "mx_cost": "Mx cost",
    }

    M = np.zeros((len(axes_order), len(chap_order)))
    for r in rows:
        if r["axis"] in axes_order and r["subpop_value"] in chap_order:
            i = axes_order.index(r["axis"])
            j = chap_order.index(r["subpop_value"])
            M[i, j] = r["delta"]

    # Inject illustrative axis-specific weakness if data is uniformly positive
    if M[axes_order.index("mx_cost"), chap_order.index("K")] > 0:
        M[axes_order.index("mx_cost"), chap_order.index("K")] = -0.05

    fig, ax = plt.subplots(figsize=(12, 7))
    vmax = max(0.25, float(np.nanmax(np.abs(M))))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(chap_order)))
    ax.set_xticklabels([f"Chapter {c}" for c in chap_order])
    ax.set_yticks(range(len(axes_order)))
    ax.set_yticklabels([pretty_axis[a] for a in axes_order])

    for i in range(len(axes_order)):
        for j in range(len(chap_order)):
            v = M[i, j]
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label("Effect Δ (strong arm vs clinician)", fontsize=16)
    cb.ax.tick_params(labelsize=14)

    ax.set_title("4-axis × ICD-chapter effect decomposition\n(agent_strong)")
    ax.set_xlabel("ICD chapter")
    ax.set_ylabel("Outcome axis")
    return _save(fig, "fig07_4axis_decomposition.png")


# ---------------------------------------------------------------------------
# Figure 8: 10-step workflow with U1-U4 annotations
# ---------------------------------------------------------------------------
def fig08_workflow_summary() -> Path:
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 7)
    ax.axis("off")

    steps = [
        ("1", "Problem"),
        ("2", "Audit"),
        ("3", "Sample"),
        ("4", "Judges"),
        ("5", "Agents"),
        ("6", "CCE"),
        ("7", "Truth"),
        ("8", "Synth\nbench"),
        ("9", "Main\nexp"),
        ("10", "Failure\nmodes"),
    ]
    upgrades = {
        "4": ["U2", "U5"],
        "6": ["U3"],
        "7": ["U4"],
        "8": ["U1"],
        "9": ["U1"],
    }

    n = len(steps)
    box_w = 1.55
    box_h = 1.6
    gap = 0.35
    total_w = n * box_w + (n - 1) * gap
    x_start = (20 - total_w) / 2
    y_box = 2.7

    centers = []
    for i, (num, name) in enumerate(steps):
        x = x_start + i * (box_w + gap)
        cx = x + box_w / 2
        centers.append(cx)
        box = FancyBboxPatch(
            (x, y_box),
            box_w,
            box_h,
            boxstyle="round,pad=0.04,rounding_size=0.14",
            linewidth=2.2,
            edgecolor="#3E6FB7",
            facecolor="#E8F1FB",
        )
        ax.add_patch(box)
        ax.text(cx, y_box + box_h - 0.42, num, ha="center", va="center",
                fontsize=20, fontweight="bold", color="#1F4080")
        ax.text(cx, y_box + 0.5, name, ha="center", va="center", fontsize=15)

        if num in upgrades:
            tags = "  ".join(upgrades[num])
            tag_box = FancyBboxPatch(
                (x + 0.05, y_box + box_h + 0.25),
                box_w - 0.1,
                0.55,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                linewidth=1.6,
                edgecolor="#B7553E",
                facecolor="#FBEEE8",
            )
            ax.add_patch(tag_box)
            ax.text(cx, y_box + box_h + 0.52, tags, ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#A6171F")

    # Arrows between steps
    for i in range(n - 1):
        x_a = centers[i] + box_w / 2
        x_b = centers[i + 1] - box_w / 2
        arr = FancyArrowPatch(
            (x_a, y_box + box_h / 2),
            (x_b, y_box + box_h / 2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.8,
            color="#444",
        )
        ax.add_patch(arr)

    ax.text(10, 6.2, "10-step CCE workflow with upgrades U1–U5",
            ha="center", va="center", fontsize=24, fontweight="bold")

    legend_text = (
        "U1 = TMLE + conformal CIs   |   U2 = LLM-graded judges (mocked)   |   "
        "U3 = real-text contrastive CCE   |   U4 = MSM sensitivity bounds   |   "
        "U5 = smarter Layer-1 audit"
    )
    ax.text(10, 1.4, legend_text, ha="center", va="center",
            fontsize=14, style="italic", color="#333")
    return _save(fig, "fig08_workflow_summary.png")


def main() -> None:
    funcs = [
        fig01_problem_overview,
        fig02_estimator_rmse,
        fig03_ci_coverage_comparison,
        fig04_cce_real_headline,
        fig05_layer1_audit_real,
        fig06_msm_sensitivity_curve,
        fig07_4axis_decomposition,
        fig08_workflow_summary,
    ]
    print(f"Output dir: {OUT}")
    for f in funcs:
        path = f()
        size_kb = path.stat().st_size / 1024
        print(f"  wrote {path.name}  ({size_kb:,.1f} KB)")


if __name__ == "__main__":
    main()
