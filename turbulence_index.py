"""
Baseline 3 — Turbulence Index (Kritzman & Li, 2010)
======================================================
The simplest possible competitor: a single scalar, the Mahalanobis distance
of today's feature vector from the training-period mean/covariance.

    Delta_t = (x_t - mu_train)^T  Sigma_train^-1  (x_t - mu_train)

High Delta_t -> market is statistically unusual ("turbulent"). That's the
entire model. It carries no notion of regime *identity* — just a single
"how weird is today" number.

Honest 4-class extension for metric comparability:
    M1-M3 are defined over a K=4 posterior simplex, but turbulence alone is
    a 1-D, 2-way signal (turbulent vs calm). To produce a (T, 4) posterior
    without inventing new information, we cross it with the one other
    scalar already in the feature set that carries directional meaning —
    momentum_ratio (feature index 7) — and softmax the two raw z-scores
    against their negations:

        logits = [ +momentum_z, -momentum_z, +turbulence_z, -turbulence_z ]
        posterior = softmax(logits)   over [Trending, MeanRev, Volatile, Quiet]

    This is a deliberately simple, transparent construction — not a hidden
    boost. Direction (for trading) is still calibrated purely from training
    returns, exactly like every other model in this comparison.
"""

from __future__ import annotations
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import calibrate_directions_from_labels, K

MODEL_NAME = "Turbulence Index"
MOMENTUM_COL = 7   # momentum_ratio in the 8-dim feature vector (see IDX in walk_forward.py)

# Fixed bucket->index convention (arbitrary; direction is calibrated, not assumed)
TRENDING_IDX, MEANREV_IDX, VOLATILE_IDX, QUIET_IDX = 0, 1, 2, 3


def _mahalanobis(X: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray) -> np.ndarray:
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, sigma_inv, d)


def run_fold(X: np.ndarray,
             ret_v: np.ndarray,
             train_start: int, train_end: int,
             test_start: int, test_end: int) -> dict:
    """Run the Turbulence Index baseline on one walk-forward fold."""
    X_train = X[train_start:train_end]
    X_test  = X[test_start:test_end]
    ret_train = ret_v[train_start:train_end]
    ret_test  = ret_v[test_start:test_end]
    n_test = len(X_test)

    # ── Fit scaler on TRAIN only ─────────────────────────────────────────────
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # ── Turbulence parameters from TRAIN only ───────────────────────────────
    mu_train    = X_tr_sc.mean(axis=0)
    cov_train   = np.cov(X_tr_sc.T) + 1e-6 * np.eye(X_tr_sc.shape[1])
    sigma_inv   = np.linalg.pinv(cov_train)

    turb_train = _mahalanobis(X_tr_sc, mu_train, sigma_inv)
    turb_test  = _mahalanobis(X_te_sc, mu_train, sigma_inv)

    turb_mu, turb_sd = float(turb_train.mean()), float(turb_train.std() + 1e-8)
    turb_z_train = (turb_train - turb_mu) / turb_sd
    turb_z_test  = (turb_test  - turb_mu) / turb_sd

    mom_z_train = X_tr_sc[:, MOMENTUM_COL]   # already standardised by scaler
    mom_z_test  = X_te_sc[:, MOMENTUM_COL]

    def _bucket(turb_z, mom_z):
        logits = np.stack([mom_z, -mom_z, turb_z, -turb_z], axis=1)
        return np.argmax(logits, axis=1)

    # ── Direction calibration from TRAINING data (same discipline as CRAN) ──
    train_hard = _bucket(turb_z_train, mom_z_train)
    dir_map = calibrate_directions_from_labels(train_hard, ret_train)

    # ── Soft posteriors on TEST data ────────────────────────────────────────
    logits_test = np.stack(
        [mom_z_test, -mom_z_test, turb_z_test, -turb_z_test], axis=1)
    posteriors = softmax(logits_test, axis=1)

    return {
        "posteriors": posteriors,
        "ret_test":   ret_test,
        "dir_map":    dir_map,
        "n_test":     n_test,
        "model_name": MODEL_NAME,
        "mean_turbulence": float(turb_test.mean()),
    }
