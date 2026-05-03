"""Synthetic OPE oracle: known density ratio + outcome → exact V_true.

Used by Step 8 unit tests to verify each estimator recovers V_true within
a relative-bias tolerance. Linear logistic policies + linear outcomes
are simple enough to admit a near-exact V_true via Monte Carlo on a
large evaluation pool.

Design:
    x         ~ N(0, I_d)
    pi_b(1|x) = sigmoid(<w_b, x>)
    pi_e(1|x) = sigmoid(<w_e, x>)        (target policy)
    Y(x, a)   = <theta_a, x> + a * shift + noise

Returns:
    train: D_obs = {(x_i, a_i ~ pi_b, y_i = Y(x_i, a_i))}
    eval pool: D_true (large n) used to compute V_true(pi_e) exactly
    propensity / density-ratio oracles for diagnostic

Note: action space is binary; for true high-dim NLP MIPS evaluation,
extend by replacing `a` with action embeddings phi(a). The oracle here
tests the estimator math; downstream pipeline does the NLP lift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SyntheticOPEData:
    x: np.ndarray  # (n, d)
    a: np.ndarray  # (n,)  binary
    y: np.ndarray  # (n,)
    pi_b_prob: np.ndarray  # (n,) prob of selected a under pi_b
    pi_e_prob: np.ndarray  # (n,) prob of selected a under pi_e
    w: np.ndarray  # (n,) density ratio at observed (x_i, a_i)
    v_true: float  # V_true(pi_e)


def _pick_action(prob_a1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.uniform(size=prob_a1.shape) < prob_a1).astype(np.int64)


def make_synthetic(
    n: int = 200,
    d: int = 5,
    seed: int = 42,
    n_eval: int = 50_000,
) -> SyntheticOPEData:
    """Generate train data + compute exact V_true(pi_e) via large Monte Carlo."""
    rng = np.random.default_rng(seed)

    # Fixed policy/outcome parameters — kept small so propensities stay in
    # a moderate range and density-ratio weights don't explode (positivity OK).
    scale = 0.5 / np.sqrt(d)
    w_b = rng.normal(size=d) * scale
    w_e = rng.normal(size=d) * scale * 0.7 + 0.4 * w_b
    theta_0 = rng.normal(size=d) * 0.3
    theta_1 = rng.normal(size=d) * 0.3
    shift = 0.3
    noise_sd = 0.1

    # Train pool — suppress cosmetic numpy 2.x matmul warnings; outputs
    # are finite by construction with the small-scale policy params above.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        x = rng.normal(size=(n, d))
        pb_1 = _sigmoid(x @ w_b)
        pe_1 = _sigmoid(x @ w_e)
        a = _pick_action(pb_1, rng)
        y = (1 - a) * (x @ theta_0) + a * (x @ theta_1 + shift) + rng.normal(0, noise_sd, size=n)

    pi_b_prob = np.where(a == 1, pb_1, 1 - pb_1)
    pi_e_prob = np.where(a == 1, pe_1, 1 - pe_1)
    w = pi_e_prob / np.clip(pi_b_prob, 1e-3, None)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        x_eval = rng.normal(size=(n_eval, d))
        pe_1_eval = _sigmoid(x_eval @ w_e)
        y0 = x_eval @ theta_0
        y1 = x_eval @ theta_1 + shift
        v_true = float(((1 - pe_1_eval) * y0 + pe_1_eval * y1).mean())

    return SyntheticOPEData(
        x=x,
        a=a,
        y=y,
        pi_b_prob=pi_b_prob,
        pi_e_prob=pi_e_prob,
        w=w,
        v_true=v_true,
    )


def relative_bias(v_hat: float, v_true: float) -> float:
    """|v_hat - v_true| / |v_true| (returns abs(v_hat - v_true) when |v_true| < eps)."""
    if abs(v_true) < 1e-6:
        return abs(v_hat - v_true)
    return abs(v_hat - v_true) / abs(v_true)
