"""Step 6 driver: train Causal Contrastive Embedding head.

Reads (x, a^clinician, a^agent) triplets from Step 5 outputs (or builds a
synthetic triplet set), encodes with a backbone, trains the projection
head with InfoNCE + HSIC, runs the same-x vs different-x sanity check.

Usage:
    python scripts/06_train_cce.py --backbone hash --epochs 30 --n 80
    python scripts/06_train_cce.py --backbone ncbi/MedCPT-Article-Encoder --inputs ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ccema.data.loaders import load_dataset
from ccema.embeddings.backbones import make_encoder
from ccema.embeddings.cce import CCEConfig, cce_sanity_check, train_cce_numpy
from ccema.utils.config import load_config
from ccema.utils.seeding import set_global_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval_config_v1.yaml")
    p.add_argument("--backbone", default="hash", help="encoder spec: hash|ncbi/MedCPT-Article-Encoder|BAAI/bge-m3|...")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--lambda-hsic", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/step6_cce")
    return p.parse_args()


def make_synthetic_triplets(encs, encoder) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (anchor, positive, other) embedding triples.

    anchor   = phi(x_i + a_i^clinician)
    positive = phi(x_i + a_i^agent)         — same x, different a
    other    = phi(x_j + a_j^clinician)     — different x

    The placeholder agent text is a templated rewrite (Step 5 stand-in).
    """
    texts_clin = [f"{e.x}\n\n{e.a_clinician}" for e in encs]
    # Meaningful agent variant — different management framing so the
    # embedding has signal to separate same-x different-a
    texts_agent = [
        f"{e.x}\n\n## Assessment\nLikely {e.metadata.get('icd_chapter', '?')}-chapter problem; agent recommends targeted work-up.\n"
        f"## Plan\n- Order specific diagnostics tailored to chief complaint\n- Empiric therapy if guidelines indicate\n- Close follow-up in 1 week"
        for e in encs
    ]

    emb_clin = encoder.encode(texts_clin)
    emb_agent = encoder.encode(texts_agent)

    rng = np.random.default_rng(0)
    n = len(encs)
    perm = rng.permutation(n)
    # Avoid identity permutation
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    emb_other = emb_clin[perm]

    return emb_clin.astype(np.float32), emb_agent.astype(np.float32), emb_other.astype(np.float32)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    cfg = load_config(REPO_ROOT / args.config)
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    encs = load_dataset(args.dataset, n_encounters=args.n, seed=args.seed)
    print(f"Loaded {len(encs)} encounters")

    encoder = make_encoder(args.backbone)
    print(f"Encoder: {encoder.name} dim={encoder.dim}")

    anchor, positive, other = make_synthetic_triplets(encs, encoder)
    print(f"Embedded: anchor={anchor.shape}, positive={positive.shape}, other={other.shape}")

    cfg_train = CCEConfig(
        input_dim=anchor.shape[1],
        projection_dim=args.projection_dim,
        tau=args.tau,
        lambda_hsic=args.lambda_hsic,
        learning_rate=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )
    proj = train_cce_numpy(anchor, positive, cfg_train)
    print(f"Trained for {args.epochs} epochs; final InfoNCE = {proj.history[-1]['info_nce']:.4f}")

    # Sanity check
    p_anchor = anchor @ proj.W
    p_positive = positive @ proj.W
    p_other = other @ proj.W
    sanity = cce_sanity_check(p_anchor, p_positive, p_other)
    print("Sanity:", sanity)

    # Save projection W
    np.save(out_dir / "projection_W.npy", proj.W)
    (out_dir / "training.json").write_text(
        json.dumps(
            {
                "backbone": args.backbone,
                "input_dim": anchor.shape[1],
                "projection_dim": args.projection_dim,
                "epochs": args.epochs,
                "history": proj.history,
                "sanity": sanity,
            },
            indent=2,
        )
    )
    print(f"\nSaved projection -> {(out_dir / 'projection_W.npy').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
