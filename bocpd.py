"""
Baseline 2 — Bayesian Online Changepoint Detection (BOCPD)
============================================================
Adams & MacKay (2007). Detects WHEN a regime break has occurred; says
nothing about WHAT the new regime is. This is the key conceptual gap
CRAN is meant to fill — BOCPD is the "detection without characterisation"
competitor.

Implementation (from scratch, no library):
    1. Project the 8-dim standardised feature vector onto its first
       principal component (PCA fit on TRAIN only) — BOCPD classically
       operates on a scalar observation stream.
    2. Run exact online changepoint detection with a Normal-Gamma
       conjugate prior and constant hazard 1/LAMBDA, producing
       P(changepoint at t) at every test timestep.
    3. Segment the test window wherever the MAP run-length estimate
       resets to 0 (a detected changepoint).
    4. Assign each segment a regime identity by nearest GMM centroid
       (GMM fit on TRAIN only — used here purely for regime *labelling*,
       not for detecting *when* to switch, which is BOCPD's job).
    5. Posterior = one-hot(segment regime) softened by the local
       changepoint probability: at high P(changepoint), BOCPD is
       genuinely uncertain about the new regime, so posterior mass
       flattens toward uniform. This is the model's own uncertainty,
       not invented for the comparison.

Direction is calibrated from training data exactly like CRAN.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import calibrate_directions_from_labels, K

RANDOM_SEED = 42
MODEL_NAME = "BOCPD"
HAZARD_LAMBDA = 30.0   # expected run length ~30 days, prior belief on regime persistence


def _bocpd_changepoint_probs(x: np.ndarray,
                              hazard: float = 1.0 / HAZARD_LAMBDA,
                              mu0: float = 0.0, kappa0: float = 1.0,
                              alpha0: float = 1.0, beta0: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact online BOCPD for a 1-D Gaussian stream with unknown mean/variance
    (Normal-Gamma conjugate prior), constant hazard function.

    Returns:
        cp_prob:   (T,) P(changepoint at t | x_1:t)  [= P(run length resets to 0)]
        map_run:   (T,) MAP run-length estimate at each t (0 = just changed)
    """
    T = len(x)
    R = np.array([1.0])  # P(r_0 = 0) = 1
    muT, kappaT, alphaT, betaT = (np.array([mu0]), np.array([kappa0]),
                                   np.array([alpha0]), np.array([beta0]))

    cp_prob = np.zeros(T)
    map_run = np.zeros(T, dtype=int)

    for t in range(T):
        xt = x[t]

        df    = 2.0 * alphaT
        scale = np.sqrt(betaT * (kappaT + 1.0) / (alphaT * kappaT))
        pred  = stats.t.pdf(xt, df, loc=muT, scale=np.clip(scale, 1e-8, None))
        pred  = np.clip(pred, 1e-300, None)

        growth_probs = R * pred * (1.0 - hazard)
        cp_mass      = np.sum(R * pred * hazard)

        new_R = np.empty(len(R) + 1)
        new_R[0]  = cp_mass
        new_R[1:] = growth_probs
        new_R /= new_R.sum()

        cp_prob[t]  = new_R[0]
        map_run[t]  = int(np.argmax(new_R))

        # Posterior parameter update (index 0 = fresh run, indices 1: = grown runs)
        new_mu    = np.empty(len(new_R))
        new_kappa = np.empty(len(new_R))
        new_alpha = np.empty(len(new_R))
        new_beta  = np.empty(len(new_R))

        new_mu[0], new_kappa[0], new_alpha[0], new_beta[0] = mu0, kappa0, alpha0, beta0
        new_kappa[1:] = kappaT + 1.0
        new_mu[1:]    = (kappaT * muT + xt) / new_kappa[1:]
        new_alpha[1:] = alphaT + 0.5
        new_beta[1:]  = betaT + (kappaT * (xt - muT) ** 2) / (2.0 * new_kappa[1:])

        R = new_R
        muT, kappaT, alphaT, betaT = new_mu, new_kappa, new_alpha, new_beta

    return cp_prob, map_run


def run_fold(X: np.ndarray,
             ret_v: np.ndarray,
             train_start: int, train_end: int,
             test_start: int, test_end: int) -> dict:
    """Run the BOCPD baseline on one walk-forward fold."""
    X_train = X[train_start:train_end]
    X_test  = X[test_start:test_end]
    ret_train = ret_v[train_start:train_end]
    ret_test  = ret_v[test_start:test_end]
    n_test = len(X_test)

    # ── Fit scaler + PCA(1) on TRAIN only ───────────────────────────────────
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    pca   = PCA(n_components=1, random_state=RANDOM_SEED)
    pc_tr = pca.fit_transform(X_tr_sc).ravel()
    pc_te = pca.transform(X_te_sc).ravel()

    # Prior for the Normal-Gamma comes from the training PC distribution
    mu0, var0 = float(pc_tr.mean()), float(pc_tr.var() + 1e-6)

    # ── Run BOCPD on the TEST PC stream ─────────────────────────────────────
    cp_prob, _ = _bocpd_changepoint_probs(
        pc_te, hazard=1.0 / HAZARD_LAMBDA,
        mu0=mu0, kappa0=1.0, alpha0=1.0, beta0=var0,
    )

    # ── Segment the test window at detected changepoints ───────────────────
    cp_threshold = 0.5
    is_cp = cp_prob > cp_threshold
    is_cp[0] = True  # first day always starts a new segment
    seg_id = np.cumsum(is_cp) - 1

    # ── Regime identity via GMM fit on TRAIN only (labelling, not detection) ─
    gmm = GaussianMixture(n_components=K, covariance_type="full",
                           n_init=10, random_state=RANDOM_SEED, max_iter=300)
    gmm.fit(X_tr_sc)
    train_hard = gmm.predict(X_tr_sc)
    dir_map = calibrate_directions_from_labels(train_hard, ret_train)

    # Assign each segment the GMM label of its mean feature vector
    posteriors = np.zeros((n_test, K))
    for s in np.unique(seg_id):
        mask = seg_id == s
        seg_mean = X_te_sc[mask].mean(axis=0, keepdims=True)
        seg_post = gmm.predict_proba(seg_mean)[0]   # soft GMM read of the segment
        regime   = int(np.argmax(seg_post))

        onehot = np.zeros(K)
        onehot[regime] = 1.0

        # Soften by local changepoint uncertainty: high cp_prob -> more uniform
        local_cp = cp_prob[mask]
        for i, idx in enumerate(np.where(mask)[0]):
            c = float(local_cp[i])
            posteriors[idx] = (1 - c) * onehot + c * (np.ones(K) / K)

    return {
        "posteriors": posteriors,
        "ret_test":   ret_test,
        "dir_map":    dir_map,
        "n_test":     n_test,
        "model_name": MODEL_NAME,
        "n_changepoints": int(is_cp.sum()),
    }
