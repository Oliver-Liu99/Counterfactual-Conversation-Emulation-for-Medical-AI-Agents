"""Split-conformal CI for OPE point estimates (Taufiq et al. 2022).

Given a point estimator V_hat that produces per-sample influence scores
psi_i, the split-conformal procedure guarantees finite-sample valid
intervals:

    1. Split data into train (1-frac_calib) and calibration (frac_calib).
    2. Fit nuisance + estimator on train.
    3. Compute residual scores R_i = |V_hat_train - psi_i| on calibration.
    4. Quantile of R at level (1-alpha) gives the half-width.

For OPE, the natural psi_i is the per-sample doubly-robust score
    psi_i = f(x_i, a^agent) + w_i (y_i - f(x_i, a_i)).

Mean(psi_i) is V_hat^DR. Bootstrap CI on psi gives asymptotic CI; conformal
gives finite-sample CI without normality assumption.
"""

from __future__ import annotations

import numpy as np


def split_conformal_ci(
    psi_train: np.ndarray,
    psi_calib: np.ndarray,
    alpha: float = 0.10,
) -> tuple[float, float, float]:
    """Returns (point, lower, upper) for split-conformal interval.

    Both psi_train and psi_calib are arrays of per-sample DR scores. The
    point estimate is mean(psi_train); the half-width is the (1-alpha)
    quantile of |psi_calib - mean(psi_train)|.
    """
    if len(psi_train) == 0 or len(psi_calib) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(psi_train.mean())
    residuals = np.abs(psi_calib - point)
    # Conservative: use the (n+1)/n adjusted quantile per Vovk's split conformal
    n = len(residuals)
    q_level = min(1.0, (1 - alpha) * (n + 1) / n)
    half_width = float(np.quantile(residuals, q_level))
    return point, point - half_width, point + half_width


def jackknife_plus_ci(psi: np.ndarray, alpha: float = 0.10) -> tuple[float, float, float]:
    """Jackknife+ CI on a 1d array (alternative to split-conformal).

    For OPE point-estimate CIs at small n where split-conformal wastes data.
    """
    n = len(psi)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(psi.mean())
    # Leave-one-out residuals
    sums = psi.sum()
    lo_means = (sums - psi) / (n - 1)
    residuals = np.abs(psi - lo_means)
    q = float(np.quantile(residuals, (1 - alpha) * (n + 1) / n))
    return point, point - q, point + q


def coverage(values_true: np.ndarray, lowers: np.ndarray, uppers: np.ndarray) -> float:
    """Empirical coverage of CIs over a simulation."""
    return float(((lowers <= values_true) & (values_true <= uppers)).mean())
