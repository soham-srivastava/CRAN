"""
CRAN Phase 6 — Ablations
==========================
Three controlled variants of CRAN, each removing exactly one design choice
from the full model, to show that each piece earns its place.

    1. CRAN (no Bayes)     — skip the manual Bayesian posterior update;
                              use the GMM's own predict_proba() directly.
                              Tests: does the explicit Bayesian layer over
                              the GMM likelihoods add anything beyond what
                              sklearn's own responsibility computation gives?

    2. CRAN (hard labels)  — replace the GMM with K-Means; posteriors
                              become one-hot. Tests: does the continuous
                              affinity vector (soft regime membership) beat
                              a discrete-label competitor built the same way?

    3. CRAN (GMM prior)    — keep the full Bayesian update, but replace the
                              uniform prior (1/K) with the GMM's own learned
                              mixing weights pi_k as the prior. Tests:
                              does CRAN's choice of an *uninformative*
                              uniform prior matter, versus letting the
                              model's own training-set regime frequencies
                              bias the posterior?

All three reuse the exact fold infrastructure and direction-calibration
discipline from walk_forward.py — same TRAIN_WIN/TEST_WIN/STEP, same
scaler/GMM-fit-on-train-only rule, same data-driven direction map.

Run:    python ablation.py
Output: results/ablation_results.md
"""

from __future__ import annotations
import os
import sys
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from walk_forward import (
    load_daily, compute_features, generate_folds, fit_gmm,
    K, LOOKBACK, TRAIN_WIN, TEST_WIN, STEP, ANN_FACTOR, TC_BPS, DATA_PATH,
)
from common import calibrate_directions_from_labels
from metrics import compute_all_metrics, m4_sharpe


# Ablation 1 — CRAN (no Bayes): raw GMM predict_proba, no manual Bayesian step


def run_fold_no_bayes(X, ret_v, train_start, train_end, test_start, test_end) -> dict:
    X_train, X_test = X[train_start:train_end], X[test_start:test_end]
    ret_train, ret_test = ret_v[train_start:train_end], ret_v[test_start:test_end]

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    gmm = fit_gmm(X_tr_sc)
    train_hard = gmm.predict(X_tr_sc)
    dir_map = calibrate_directions_from_labels(train_hard, ret_train)

    # The only difference from CRAN: skip the explicit Bayesian update and
    # use the GMM's own posterior responsibilities directly.
    posteriors = gmm.predict_proba(X_te_sc)

    return {"posteriors": posteriors, "ret_test": ret_test, "dir_map": dir_map,
            "n_test": len(X_test), "model_name": "CRAN (no Bayes)"}


# ═════════════════════════════════════════════════════════════════════════════
# Ablation 2 — CRAN (hard labels): K-Means, one-hot posteriors
# ═════════════════════════════════════════════════════════════════════════════

def run_fold_hard_labels(X, ret_v, train_start, train_end, test_start, test_end) -> dict:
    X_train, X_test = X[train_start:train_end], X[test_start:test_end]
    ret_train, ret_test = ret_v[train_start:train_end], ret_v[test_start:test_end]

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    km = KMeans(n_clusters=K, n_init=15, random_state=42)
    km.fit(X_tr_sc)

    train_hard = km.predict(X_tr_sc)
    dir_map = calibrate_directions_from_labels(train_hard, ret_train)

    test_hard = km.predict(X_te_sc)
    posteriors = np.zeros((len(X_test), K))
    posteriors[np.arange(len(X_test)), test_hard] = 1.0

    return {"posteriors": posteriors, "ret_test": ret_test, "dir_map": dir_map,
            "n_test": len(X_test), "model_name": "CRAN (hard labels)"}


# ═════════════════════════════════════════════════════════════════════════════
# Ablation 3 — CRAN (GMM prior): Bayesian update with learned pi_k, not uniform
# ═════════════════════════════════════════════════════════════════════════════

def _compute_posteriors_with_prior(X_test_scaled, means, covs, prior) -> np.ndarray:
    """Identical to walk_forward.compute_posteriors but with a custom prior."""
    n = len(X_test_scaled)
    posteriors = np.zeros((n, K))
    for t in range(n):
        log_liks = np.array([
            stats.multivariate_normal.logpdf(X_test_scaled[t], mean=means[j], cov=covs[j])
            for j in range(K)
        ])
        log_post = np.log(np.clip(prior, 1e-300, None)) + log_liks
        log_post -= logsumexp(log_post)
        posteriors[t] = np.exp(log_post)
    return posteriors


def run_fold_gmm_prior(X, ret_v, train_start, train_end, test_start, test_end) -> dict:
    X_train, X_test = X[train_start:train_end], X[test_start:test_end]
    ret_train, ret_test = ret_v[train_start:train_end], ret_v[test_start:test_end]

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    gmm = fit_gmm(X_tr_sc)
    train_hard = gmm.predict(X_tr_sc)
    dir_map = calibrate_directions_from_labels(train_hard, ret_train)

    # The only difference from CRAN: prior = GMM's learned mixing weights,
    # not uniform 1/K.
    prior = gmm.weights_
    posteriors = _compute_posteriors_with_prior(X_te_sc, gmm.means_, gmm.covariances_, prior)

    return {"posteriors": posteriors, "ret_test": ret_test, "dir_map": dir_map,
            "n_test": len(X_test), "model_name": "CRAN (GMM prior)"}


ABLATIONS = {
    "CRAN (no Bayes)":     run_fold_no_bayes,
    "CRAN (hard labels)":  run_fold_hard_labels,
    "CRAN (GMM prior)":    run_fold_gmm_prior,
}


# ═════════════════════════════════════════════════════════════════════════════
# Runner — shares fold schedule with CRAN/baselines (see run_comparison.py)
# ═════════════════════════════════════════════════════════════════════════════

def run_ablation_on_folds(run_fold_fn, X, ret_v, folds) -> dict:
    """Run one ablation across all folds and aggregate, same shape as CRAN agg."""
    fold_metrics = []
    all_post, all_ret = [], []
    last_dir_map = None

    for i, (ts, te, ss, se) in enumerate(folds):
        out = run_fold_fn(X, ret_v, ts, te, ss, se)
        post, ret_test, dir_map = out["posteriors"], out["ret_test"], out["dir_map"]
        last_dir_map = dir_map

        m = compute_all_metrics(
            posteriors=post, returns=ret_test, direction_map=dir_map,
            ann_factor=ANN_FACTOR, tc_bps=TC_BPS, model_name=out["model_name"],
        )
        fold_metrics.append(m)

        if len(post) > 1:
            all_post.append(post[:-1])
            all_ret.append(ret_test[1:])

    m1s = [m.m1.score for m in fold_metrics]
    m2s = [m.m2.recall for m in fold_metrics if not np.isnan(m.m2.recall)]
    m3s = [m.m3.spearman_r for m in fold_metrics]
    m4s = [m.m4.sharpe for m in fold_metrics]

    if all_post:
        combined_post = np.vstack(all_post)
        combined_ret  = np.concatenate(all_ret)
        combined_m4   = m4_sharpe(combined_post, combined_ret, last_dir_map,
                                   ann_factor=ANN_FACTOR, tc_bps=TC_BPS)
        combined_sharpe = combined_m4.sharpe
    else:
        combined_sharpe = float("nan")

    return {
        "n_folds": len(folds),
        "m1_mean": float(np.mean(m1s)), "m1_std": float(np.std(m1s)),
        "m2_mean": float(np.mean(m2s)) if m2s else float("nan"),
        "m3_mean": float(np.mean(m3s)), "m3_std": float(np.std(m3s)),
        "m4_mean": float(np.mean(m4s)), "m4_std": float(np.std(m4s)),
        "combined_sharpe": combined_sharpe,
    }


def main():
    print("CRAN Phase 6 — Ablations")
    print("=" * 60)

    daily   = load_daily(DATA_PATH)
    returns = daily["log_ret"].fillna(0).values
    features = compute_features(returns, lookback=LOOKBACK)
    valid  = ~np.any(np.isnan(features), axis=1)
    X, ret_v = features[valid], returns[valid]
    n = len(X)

    folds = generate_folds(n, TRAIN_WIN, TEST_WIN, STEP)
    print(f"Folds: {len(folds)}  (TRAIN={TRAIN_WIN} TEST={TEST_WIN} STEP={STEP})\n")

    results = {}
    for name, fn in ABLATIONS.items():
        print(f"Running {name} ...")
        agg = run_ablation_on_folds(fn, X, ret_v, folds)
        results[name] = agg
        print(f"  M1={agg['m1_mean']:.4f}  M2={agg['m2_mean']:.4f}  "
              f"M3={agg['m3_mean']:.4f}  M4(mean)={agg['m4_mean']:.3f}  "
              f"Combined Sharpe={agg['combined_sharpe']:.3f}\n")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_results.md")
    lines = ["# CRAN Phase 6 — Ablation Results", "",
             "| Model | M1 | M2 | M3 | M4 (mean) | Combined OOS Sharpe |",
             "|-------|----|----|----|-----------|----------------------|"]
    for name, agg in results.items():
        lines.append(
            f"| {name} | {agg['m1_mean']:.4f} | {agg['m2_mean']:.4f} | "
            f"{agg['m3_mean']:.4f} | {agg['m4_mean']:.3f} | {agg['combined_sharpe']:.3f} |"
        )
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Output -> {out_path}")
    return results


if __name__ == "__main__":
    main()
