"""
Baseline 1 — Hamilton Hidden Markov Model
==========================================
Classic regime-switching model (Hamilton, 1989). 4-state Gaussian HMM with
a learned transition matrix, fit via Baum-Welch EM (hmmlearn).

How it differs from CRAN:
    - Discrete hidden states with a parametric Markov transition matrix,
      not a static mixture of named Gaussians.
    - Posterior P(s_t | x_1:T) comes from the forward-backward algorithm
      (uses the WHOLE test window, including future test observations) —
      this is the standard way Hamilton HMMs are scored and is NOT
      lookahead across the train/test boundary (the model itself is fit
      on training data only).
    - No named priors — state→regime labels are assigned post-hoc by
      centroid inspection, exactly as CRAN does, but trading direction is
      calibrated the same data-driven way (see baselines/common.py) so
      the comparison isn't biased by label-naming choices.

Same walk-forward discipline as CRAN:
    scaler.fit()  → train only
    model.fit()   → train only (Baum-Welch)
    test data     → scaler.transform() + model.predict_proba() only
"""

from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import calibrate_directions_from_labels, K

RANDOM_SEED = 42
MODEL_NAME = "Hamilton HMM"


def run_fold(X: np.ndarray,
             ret_v: np.ndarray,
             train_start: int, train_end: int,
             test_start: int, test_end: int) -> dict:
    """
    Run the Hamilton HMM baseline on one walk-forward fold.

    Returns a dict matching the shape expected by run_comparison.py:
        {"posteriors": (n_test, K), "ret_test": (n_test,), "dir_map": {...}}
    """
    if not HMM_AVAILABLE:
        raise RuntimeError(
            "hmmlearn is not installed. Run: pip install hmmlearn --break-system-packages")

    X_train = X[train_start:train_end]
    X_test  = X[test_start:test_end]
    ret_train = ret_v[train_start:train_end]
    ret_test  = ret_v[test_start:test_end]

    # ── Fit scaler on TRAIN only ─────────────────────────────────────────────
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # ── Fit Gaussian HMM on TRAIN only (Baum-Welch EM) ──────────────────────
    model = GaussianHMM(
        n_components=K,
        covariance_type="full",
        n_iter=200,
        tol=1e-4,
        random_state=RANDOM_SEED,
        init_params="stmc",
    )
    # Guard against rare EM non-convergence / singular covariance on tiny folds
    try:
        model.fit(X_tr_sc)
    except Exception:
        model = GaussianHMM(
            n_components=K, covariance_type="diag", n_iter=200,
            random_state=RANDOM_SEED, init_params="stmc",
        )
        model.fit(X_tr_sc)

    # ── Direction calibration from TRAINING data (Viterbi hard path) ───────
    train_hard = model.predict(X_tr_sc)
    dir_map = calibrate_directions_from_labels(train_hard, ret_train)

    # ── Posteriors on TEST data via forward-backward smoothing ─────────────
    posteriors = model.predict_proba(X_te_sc)

    return {
        "posteriors": posteriors,
        "ret_test":   ret_test,
        "dir_map":    dir_map,
        "n_test":     len(X_test),
        "model_name": MODEL_NAME,
    }
