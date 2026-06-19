"""
Shared utilities for Phase 5/6 baseline + ablation models.

Every baseline model and ablation.py model must:
    1. Fit any scaler/clusterer on TRAIN data only.
    2. Output a (T, K) posterior array on TEST data.
    3. Calibrate trading direction from TRAINING returns only (no lookahead).

This file holds the one piece of logic that must be IDENTICAL across every
model for the comparison to be fair: direction calibration.
"""

import numpy as np

K = 4


def calibrate_directions_from_labels(hard_labels: np.ndarray,
                                      ret_train: np.ndarray,
                                      k: int = K) -> dict:
    """
    Data-driven direction calibration — identical rule used for CRAN in
    Phase 4 (walk_forward.calibrate_directions), generalised to take any
    hard-label array (works for GMM, HMM, K-Means, or any discrete labeller).

    For each component/state j, direction = sign(mean training return
    on days where j was the hard label). This removes any dependency on
    GMM/HMM component-index permutation between folds.

    Args:
        hard_labels: (N_train,) integer labels in [0, k)
        ret_train:   (N_train,) training log-returns
        k:           number of regimes

    Returns:
        dir_map: {component_index: direction in {-1.0, 0.0, +0.5, +1.0}}
    """
    dir_map = {}
    for j in range(k):
        mask = hard_labels == j
        if mask.sum() < 5:
            dir_map[j] = 0.0
            continue
        mean_ret = float(ret_train[mask].mean())
        if mean_ret > 1e-4:
            dir_map[j] = +1.0
        elif mean_ret < -1e-4:
            dir_map[j] = -1.0
        else:
            dir_map[j] = +0.5
    return dir_map
