"""Verify Causal Contrastive Embedding (CCE) on real ACI-Bench clinical text.

This is the first real test of the U3 paper claim: CCE on top of a real
medical backbone reduces classifier-discriminability of (clinician, agent)
action pairs vs the raw backbone embedding alone — i.e. it improves
positivity.

Pipeline:
  1. Load ACI-Bench train encounters (cap to N for CPU runtime).
  2. Build (x, a^clinician) and (x, a^agent) text pairs. The agent variant
     is a structured rewrite that reframes assessment + plan in primary-
     care template language so phi(a) actually moves vs the clinician's
     text.
  3. Encode both with a real medical backbone (MedCPT preferred, fallback
     to BGE-M3 then to all-mpnet-base-v2).
  4. Train a torch CCE projection on (anchor=clinician, positive=agent).
  5. For each of 4 settings — raw, raw+PCA32, CCE, CCE+PCA32 — measure
        - classifier AUC for clinician-vs-agent discrimination,
        - ESS/n from the implied density-ratio weights,
        - MMD between the two action-embedding distributions.

Usage:
  PYTHONPATH=src python3 scripts/verify_cce_on_real_text.py \
      --output-dir outputs/u3_real
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

# Sklearn isotonic calibration emits divide-by-zero warnings on small
# datasets; they are harmless (the ensuing predictions are still valid
# probabilities). Silence them so the diagnostic table is readable.
warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
ACI_ROOT = REPO_ROOT / "data" / "raw" / "aci_bench"


# ---------------------------------------------------------------------------
# Backbone selection with fallback
# ---------------------------------------------------------------------------


PREFERRED_BACKBONES: tuple[tuple[str, int], ...] = (
    ("ncbi/MedCPT-Article-Encoder", 768),
    ("BAAI/bge-m3", 1024),
    ("sentence-transformers/all-mpnet-base-v2", 768),
)


def _try_load_backbone(spec: str) -> tuple[object, int] | None:
    """Best-effort load of a SentenceTransformer-compatible backbone.

    For MedCPT (which is a HF AutoModel, not a SentenceTransformer), we
    wrap it in a small mean-pool encoder. For BGE-M3 / mpnet, we use the
    official SentenceTransformer path.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception as e:  # pragma: no cover
        print(f"  sentence-transformers not importable: {e}")
        return None

    if spec == "ncbi/MedCPT-Article-Encoder":
        # MedCPT does NOT ship a SentenceTransformer config; load HF directly.
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            print(f"  transformers not importable for MedCPT: {e}")
            return None
        try:
            tok = AutoTokenizer.from_pretrained(spec)
            mdl = AutoModel.from_pretrained(spec)
            mdl.eval()
        except Exception as e:
            print(f"  MedCPT download/load failed: {e}")
            return None

        class MedCPTEncoder:
            name = "medcpt"
            dim = 768

            def __init__(self, tokenizer, model):
                self.tokenizer = tokenizer
                self.model = model

            def encode(self, texts):  # noqa: D401
                outs = []
                bs = 8
                with torch.no_grad():
                    for i in range(0, len(texts), bs):
                        batch = list(texts[i : i + bs])
                        enc = self.tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt",
                        )
                        # MedCPT-Article-Encoder uses [CLS] token embedding
                        out = self.model(**enc).last_hidden_state[:, 0, :]
                        outs.append(out.cpu().numpy())
                return np.concatenate(outs, axis=0).astype(np.float32)

        return MedCPTEncoder(tok, mdl), 768

    # SentenceTransformer-compatible path
    try:
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(spec, device="cpu")
        # Probe dim
        probe = st_model.encode(["probe"], convert_to_numpy=True, show_progress_bar=False)
        dim = int(probe.shape[1])

        class STEncoder:
            name = spec.split("/")[-1].lower()
            dim = 0

            def __init__(self, model, d):
                self.model = model
                self.dim = d

            def encode(self, texts):
                return self.model.encode(
                    list(texts),
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=False,
                ).astype(np.float32)

        return STEncoder(st_model, dim), dim
    except Exception as e:
        print(f"  load failed for {spec}: {e}")
        return None


def load_first_available_backbone() -> tuple[object, str, int]:
    for spec, dim in PREFERRED_BACKBONES:
        print(f"Trying backbone: {spec}")
        got = _try_load_backbone(spec)
        if got is not None:
            enc, real_dim = got
            print(f"  -> loaded {spec} (dim={real_dim})")
            return enc, spec, real_dim
    raise RuntimeError(
        "No backbone could be loaded; tried " + ", ".join(s for s, _ in PREFERRED_BACKBONES)
    )


# ---------------------------------------------------------------------------
# Agent (counterfactual) rewrite — heuristic, no LLM
# ---------------------------------------------------------------------------


GENERIC_PRIMARY_CARE_PLAN = (
    "## Assessment\n"
    "Patient presents with the symptoms documented above; differential is "
    "broad and will be narrowed with targeted work-up. No red-flag features "
    "identified that require emergency escalation at this visit.\n\n"
    "## Plan\n"
    "- Order baseline labs (CBC, CMP) and any symptom-specific studies.\n"
    "- Initiate conservative first-line management per primary-care guidelines.\n"
    "- Lifestyle counseling on diet, activity, and adherence.\n"
    "- Return-precautions reviewed; follow-up in 1-2 weeks or sooner if worsening.\n"
    "- Referral to specialty if no improvement after first-line therapy."
)


def _short_x(e, max_chars: int = 600) -> str:
    """Take a short context summary so the action text isn't truncated by
    the 512-token backbone.

    Strategy: prefer the chief complaint section if present, otherwise the
    first ~600 chars of x. Empirically the ACI-Bench action sections take
    300-1000 chars, so 600 chars of context leaves the action visible
    within a 512-token budget.
    """
    x = e.x or ""
    # Try to grab the Chief Complaint subsection (most informative short x).
    import re

    m = re.search(r"### Chief Complaint\s*\n([^\n#]+)", x)
    if m:
        cc = m.group(1).strip()
        return f"Chief complaint: {cc}"
    return x[:max_chars]


def build_pairs(encs) -> tuple[list[str], list[str]]:
    """Return (clinician_texts, agent_texts), both length n.

    Each text is "short_x + action". We deliberately keep x short (~chief
    complaint) so the backbone's 512-token window does not truncate the
    action away. The action carries the policy contrast we want CCE to
    handle.
    """
    clinician = [f"{_short_x(e)}\n\n{e.a_clinician}" for e in encs]
    agent = [f"{_short_x(e)}\n\n{GENERIC_PRIMARY_CARE_PLAN}" for e in encs]
    return clinician, agent


# ---------------------------------------------------------------------------
# Diagnostics: classifier AUC + density-ratio weights from full embeddings
# ---------------------------------------------------------------------------


def discriminator_auc_and_weights(
    emb_clinician: np.ndarray,
    emb_agent: np.ndarray,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Train a logistic classifier to distinguish clinician (label 0) from
    agent (label 1) action embeddings.

    Returns (AUC, density-ratio-weights at the clinician points).

    This is the "classifier density-ratio estimator" in the form used by
    Sugiyama et al. 2012, but operating directly on the full-dim action
    embedding rather than a 1-d hashed action.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV

    from ccema.embeddings.diagnostics import roc_auc

    n_b = len(emb_clinician)
    n_e = len(emb_agent)

    X_all = np.vstack([emb_clinician, emb_agent]).astype(np.float64)
    y_all = np.r_[np.zeros(n_b), np.ones(n_e)].astype(int)

    n_pos = int(y_all.sum())
    n_neg = int(len(y_all) - n_pos)
    if n_pos < 5 or n_neg < 5:
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_all, y_all)
        p_b = clf.predict_proba(emb_clinician.astype(np.float64))[:, 1]
        scores_all = clf.predict_proba(X_all)[:, 1]
    else:
        cv_folds = min(5, n_pos, n_neg)
        clf = CalibratedClassifierCV(
            LogisticRegression(max_iter=2000), method="isotonic", cv=cv_folds
        )
        clf.fit(X_all, y_all)
        p_b = clf.predict_proba(emb_clinician.astype(np.float64))[:, 1]
        scores_all = clf.predict_proba(X_all)[:, 1]

    p_b = np.clip(p_b, 1e-3, 1 - 1e-3)
    w = (p_b / (1 - p_b)) * (n_b / n_e)

    auc = float(roc_auc(scores_all, y_all))
    return auc, w


def settings_diagnostics(
    name: str,
    emb_clin: np.ndarray,
    emb_agent: np.ndarray,
    seed: int = 42,
) -> dict:
    from ccema.embeddings.diagnostics import effective_sample_size, mmd_unbiased

    auc, w = discriminator_auc_and_weights(emb_clin, emb_agent, seed=seed)
    ess = effective_sample_size(w)
    ess_per_n = ess / max(1, len(w))
    mmd = mmd_unbiased(emb_clin, emb_agent)
    return {
        "setting": name,
        "n_train": int(len(emb_clin)),
        "dim": int(emb_clin.shape[1]),
        "classifier_auc": float(auc),
        "ess": float(ess),
        "ess_per_n": float(ess_per_n),
        "mmd": float(mmd),
    }


# ---------------------------------------------------------------------------
# PCA helper (numpy, no sklearn dependency for projection itself)
# ---------------------------------------------------------------------------


def fit_pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, components) such that Y = (X - mean) @ components."""
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # SVD on (n, d). For n < d (our case), use covariance via X X^T.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    components = Vt[:k].T  # (d, k)
    return mu, components


def apply_pca(X: np.ndarray, mu: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (X - mu) @ components


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/u3_real")
    p.add_argument("--n", type=int, default=60, help="Cap on # encounters (default 60)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--pca-dim", type=int, default=32)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--lambda-hsic", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    output_dir: Path,
    n: int = 60,
    epochs: int = 30,
    projection_dim: int = 128,
    pca_dim: int = 32,
    tau: float = 0.07,
    lambda_hsic: float = 0.0,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    from ccema.data.aci_bench import load_aci_bench
    from ccema.embeddings.cce import CCEConfig
    from ccema.embeddings.cce_torch import train_cce_torch
    from ccema.utils.seeding import set_global_seed

    set_global_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading ACI-Bench train split from {ACI_ROOT}")
    encs = load_aci_bench(ACI_ROOT, split="train")
    encs = encs[:n]
    print(f"      Loaded {len(encs)} encounters (capped at {n})")

    print("[2/5] Loading backbone")
    encoder, backbone_id, dim = load_first_available_backbone()

    print("[3/5] Building pairs and encoding")
    clin_texts, agent_texts = build_pairs(encs)
    emb_clin_raw = encoder.encode(clin_texts).astype(np.float32)
    emb_agent_raw = encoder.encode(agent_texts).astype(np.float32)
    print(f"      Encoded: clin={emb_clin_raw.shape}, agent={emb_agent_raw.shape}")

    print("[4/5] Training CCE projection")
    cfg = CCEConfig(
        input_dim=int(emb_clin_raw.shape[1]),
        projection_dim=projection_dim,
        tau=tau,
        lambda_hsic=lambda_hsic,
        learning_rate=lr,
        epochs=epochs,
        seed=seed,
    )
    proj = train_cce_torch(emb_clin_raw, emb_agent_raw, cfg)
    print(f"      Final InfoNCE = {proj.history[-1]['info_nce']:.4f}")

    # CCE projections — match preprocessing in train_cce_torch (mean-center
    # and rescale by mean clinician norm so the projection sees inputs on
    # the same scale it was trained on).
    a_mean = emb_clin_raw.mean(0, keepdims=True)
    p_mean = emb_agent_raw.mean(0, keepdims=True)
    a_centered = emb_clin_raw - a_mean
    p_centered = emb_agent_raw - p_mean
    scale = float(np.linalg.norm(a_centered, axis=-1).mean()) + 1e-12
    emb_clin_cce = (a_centered / scale) @ proj.W
    emb_agent_cce = (p_centered / scale) @ proj.W

    print("[5/5] Computing diagnostics in 4 settings")
    settings: list[dict] = []

    # 1. Raw
    settings.append(settings_diagnostics("raw", emb_clin_raw, emb_agent_raw, seed=seed))

    # 2. Raw + PCA-k (fit on stacked clinician+agent)
    stack_raw = np.vstack([emb_clin_raw, emb_agent_raw])
    mu_r, comp_r = fit_pca(stack_raw, k=pca_dim)
    emb_clin_raw_pca = apply_pca(emb_clin_raw, mu_r, comp_r)
    emb_agent_raw_pca = apply_pca(emb_agent_raw, mu_r, comp_r)
    settings.append(
        settings_diagnostics(f"raw+pca{pca_dim}", emb_clin_raw_pca, emb_agent_raw_pca, seed=seed)
    )

    # 3. CCE
    settings.append(settings_diagnostics("cce", emb_clin_cce, emb_agent_cce, seed=seed))

    # 4. CCE + PCA-k
    stack_cce = np.vstack([emb_clin_cce, emb_agent_cce])
    mu_c, comp_c = fit_pca(stack_cce, k=pca_dim)
    emb_clin_cce_pca = apply_pca(emb_clin_cce, mu_c, comp_c)
    emb_agent_cce_pca = apply_pca(emb_agent_cce, mu_c, comp_c)
    settings.append(
        settings_diagnostics(f"cce+pca{pca_dim}", emb_clin_cce_pca, emb_agent_cce_pca, seed=seed)
    )

    # Pretty-print
    print()
    print("=" * 78)
    print(
        f"{'setting':<18} {'n':>4} {'dim':>5} {'AUC':>8} {'ESS/n':>8} {'MMD':>10}"
    )
    print("-" * 78)
    for s in settings:
        print(
            f"{s['setting']:<18} {s['n_train']:>4} {s['dim']:>5} "
            f"{s['classifier_auc']:>8.4f} {s['ess_per_n']:>8.4f} {s['mmd']:>10.6f}"
        )
    print("=" * 78)

    # Headline
    auc_raw = settings[0]["classifier_auc"]
    auc_cce = settings[2]["classifier_auc"]
    delta = auc_raw - auc_cce  # positive = CCE reduces AUC = improvement
    print(f"\nHeadline: AUC raw={auc_raw:.4f}, CCE={auc_cce:.4f}, delta={delta:+.4f}")
    if delta > 0.02:
        verdict = "CCE measurably improves positivity (reduces AUC) on real text."
    elif delta < -0.02:
        verdict = "CCE *worsens* positivity vs raw on real text. U3 should be reconsidered."
    else:
        verdict = "CCE is approximately neutral vs raw on real text (|delta| < 0.02)."
    print(verdict)

    report = {
        "backbone": backbone_id,
        "n_encounters": len(encs),
        "epochs": epochs,
        "projection_dim": projection_dim,
        "pca_dim": pca_dim,
        "lr": lr,
        "tau": tau,
        "lambda_hsic": lambda_hsic,
        "seed": seed,
        "settings": settings,
        "headline": {
            "auc_raw": auc_raw,
            "auc_cce": auc_cce,
            "delta_auc_raw_minus_cce": delta,
            "verdict": verdict,
        },
        "training_history_tail": proj.history[-3:],
    }
    out_path = output_dir / "cce_real_diagnostics.json"
    out_path.write_text(json.dumps(report, indent=2))
    try:
        rel = out_path.relative_to(REPO_ROOT)
    except ValueError:
        rel = out_path  # not under repo root (e.g. pytest tmp_path)
    print(f"\nWrote {rel}")
    return report


def main() -> None:
    args = parse_args()
    out_dir = (REPO_ROOT / args.output_dir).resolve()
    run(
        output_dir=out_dir,
        n=args.n,
        epochs=args.epochs,
        projection_dim=args.projection_dim,
        pca_dim=args.pca_dim,
        tau=args.tau,
        lambda_hsic=args.lambda_hsic,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
