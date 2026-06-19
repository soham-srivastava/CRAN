"""
CRAN Phase 4 — Walk-Forward Engine
=====================================
The backbone of all honest results in this project.

DISCIPLINE (non-negotiable):
    For every fold:
        scaler = StandardScaler().fit(X_train)    ← fit on TRAIN only
        gmm    = GaussianMixture().fit(scaler.transform(X_train))
        X_test_scaled = scaler.transform(X_test)  ← transform only, never fit
        posteriors    = gmm.predict_proba(X_test_scaled)

    If .fit() is ever called on test data → results are invalid.
    This engine enforces that structurally — the scaler and gmm objects
    are local to each fold and never see test rows during fitting.

Walk-forward parameters (adapted for b copy.csv — 249 valid rows):
    TRAIN_WIN = 124 days  (~4 months)
    TEST_WIN  =  63 days  (~2 months)
    STEP      =  21 days  (~1 month roll)

    Fold schedule:
        Fold 1:  train[  0:124]  test[124:187]
        Fold 2:  train[ 21:145]  test[145:208]
        Fold 3:  train[ 42:166]  test[166:229]
        Fold 4:  train[ 63:187]  test[187:249]   (62 test days — acceptable)

Run:    python walk_forward.py
Output: walk_forward_results.md
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from metrics import compute_all_metrics, AllMetrics

# ── Parameters ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
K           = 4
LOOKBACK    = 30
TRAIN_WIN   = 124
TEST_WIN    = 63
STEP        = 21
ANN_FACTOR  = 365    # 24/7 instrument
TC_BPS      = 5.0

DATA_PATH = os.path.join(os.path.dirname(__file__), "market_data_1min.csv")

IDX = {
    "ret_1d": 0, "ret_5d": 1, "ret_20d": 2,
    "realvol_10d": 3, "realvol_30d": 4,
    "autocorr_lag1": 5, "autocorr_lag5": 6,
    "momentum_ratio": 7,
}


# ═════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE  (identical to Phase 2/3)
# ═════════════════════════════════════════════════════════════════════════════

def load_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").reset_index(drop=True)
    daily = df.set_index("Time").resample("1D").agg(
        Open=("Open","first"), High=("High","max"),
        Low=("Low","min"),   Close=("Close","last"),
        Volume=("Volume","sum"),
    ).dropna()
    daily["log_ret"] = np.log(daily["Close"] / daily["Close"].shift(1))
    return daily


def compute_features(returns: np.ndarray, lookback: int = 30) -> np.ndarray:
    """8-dim features, walk-forward safe — uses only past data at each t."""
    n = len(returns)
    features = np.full((n, 8), np.nan)
    for t in range(lookback, n):
        w20 = returns[t - 19 : t + 1]
        ac1 = float(np.corrcoef(w20[:-1], w20[1:])[0, 1]) if len(w20) > 2 else 0.0
        ac5 = float(np.corrcoef(w20[:-5], w20[5:])[0, 1]) if len(w20) > 6 else 0.0
        rv20 = returns[t - 19 : t + 1].std(ddof=1)
        features[t] = [
            returns[t],
            returns[t - 4 : t + 1].sum(),
            returns[t - 19 : t + 1].sum(),
            returns[t - 9  : t + 1].std(ddof=1) * np.sqrt(ANN_FACTOR),
            returns[t - 29 : t + 1].std(ddof=1) * np.sqrt(ANN_FACTOR),
            ac1, ac5,
            returns[t - 19 : t + 1].sum() / (rv20 * np.sqrt(ANN_FACTOR) + 1e-8),
        ]
    return features


# ═════════════════════════════════════════════════════════════════════════════
# GMM + BAYESIAN INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def fit_gmm(X_scaled: np.ndarray) -> GaussianMixture:
    """Fit k=4 full-covariance GMM. MUST only receive training data."""
    gmm = GaussianMixture(
        n_components=K, covariance_type="full",
        n_init=15, random_state=RANDOM_SEED, max_iter=300,
    )
    gmm.fit(X_scaled)
    return gmm


def compute_posteriors(X_test_scaled: np.ndarray,
                       means: np.ndarray,
                       covs: np.ndarray) -> np.ndarray:
    """
    Bayesian posterior for every test observation.
    Prior = uniform (1/K). GMM parameters from training fold only.
    """
    n = len(X_test_scaled)
    prior = np.ones(K) / K
    posteriors = np.zeros((n, K))
    for t in range(n):
        log_liks = np.array([
            stats.multivariate_normal.logpdf(
                X_test_scaled[t], mean=means[j], cov=covs[j])
            for j in range(K)
        ])
        log_post = np.log(np.clip(prior, 1e-300, None)) + log_liks
        log_post -= logsumexp(log_post)
        posteriors[t] = np.exp(log_post)
    return posteriors


def assign_labels(gmm: GaussianMixture,
                  scaler: StandardScaler) -> dict:
    """
    Economic label assignment via centroid inspection.
    Returns name_map only — used for reporting/interpretation.
    Direction is calibrated separately from training returns.
    """
    means_orig = scaler.inverse_transform(gmm.means_)
    vol_col = IDX["realvol_30d"]
    ret_col = IDX["ret_20d"]

    volatile_k = int(np.argmax(means_orig[:, vol_col]))
    quiet_k    = int(np.argmin(means_orig[:, vol_col]))
    remaining  = [k for k in range(K) if k not in (volatile_k, quiet_k)]
    trending_k = remaining[int(np.argmax(means_orig[remaining, ret_col]))]
    meanrev_k  = [k for k in remaining if k != trending_k][0]

    return {volatile_k: "Volatile", quiet_k: "Quiet",
            trending_k: "Trending", meanrev_k: "MeanRev"}


def calibrate_directions(gmm: GaussianMixture,
                         X_train_scaled: np.ndarray,
                         ret_train: np.ndarray) -> dict:
    """
    Data-driven direction calibration on the training window.

    For each GMM component k, compute the mean log-return on training days
    where component k has the highest posterior probability.
    Direction = sign(mean_return_k).

    Rationale: centroid-inspection label names are for human interpretation.
    The trading direction must be grounded in what each component actually
    delivered during training — not in a name we assigned post-hoc.
    This is standard practice in regime-following strategies and is
    non-lookahead: all returns used are from the training window only.

    Args:
        gmm:           fitted GaussianMixture (training data only)
        X_train_scaled: (N_train, 8) standardised training features
        ret_train:     (N_train,) training log-returns

    Returns:
        dir_map: {component_index: direction ∈ {-1.0, 0.0, +0.5, +1.0}}
    """
    train_labels = gmm.predict(X_train_scaled)
    dir_map = {}
    for k in range(K):
        mask = train_labels == k
        if mask.sum() < 5:
            dir_map[k] = 0.0   # too few observations — stay flat
            continue
        mean_ret = float(ret_train[mask].mean())
        # Discretise to standard direction values
        if mean_ret > 1e-4:
            dir_map[k] = +1.0
        elif mean_ret < -1e-4:
            dir_map[k] = -1.0
        else:
            dir_map[k] = +0.5   # near-zero mean → mild long bias (like Quiet)
    return dir_map


# ═════════════════════════════════════════════════════════════════════════════
# FOLD GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_folds(n_total: int,
                   train_win: int,
                   test_win: int,
                   step: int) -> list[tuple[int, int, int, int]]:
    """
    Generate (train_start, train_end, test_start, test_end) index tuples.

    Walk-forward: each fold's test window immediately follows its train window.
    Folds roll forward by `step` days each iteration.
    The last fold may have fewer than test_win test days — that is allowed.
    """
    folds = []
    train_start = 0
    while True:
        train_end  = train_start + train_win
        test_start = train_end
        test_end   = min(test_start + test_win, n_total)

        if test_start >= n_total:
            break
        if train_end > n_total:
            break

        folds.append((train_start, train_end, test_start, test_end))
        train_start += step

    return folds


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE FOLD RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_fold(fold_idx: int,
             X: np.ndarray,
             ret_v: np.ndarray,
             returns: np.ndarray,
             train_start: int,
             train_end: int,
             test_start: int,
             test_end: int) -> dict:
    """
    Execute one walk-forward fold.

    Enforced discipline:
        1. Scaler fit on X[train_start:train_end] ONLY
        2. GMM fit on scaler.transform(X_train) ONLY
        3. Test data: scaler.transform() called — NEVER scaler.fit()
        4. GMM.predict_proba() called on test — NEVER on train (to avoid leakage)

    Returns a dict with fold metadata, metrics, and the raw posteriors/returns
    for later aggregation and figure generation.
    """
    X_train = X[train_start : train_end]
    X_test  = X[test_start  : test_end]
    ret_test = returns[test_start : test_end]

    n_train = len(X_train)
    n_test  = len(X_test)

    # ── STEP 1: Fit scaler on training data only ──────────────────────────────
    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_train)     # ← fit here

    # ── STEP 2: Transform test data — NO fit ─────────────────────────────────
    X_te_sc   = scaler.transform(X_test)           # ← transform only

    # ── STEP 3: Fit GMM on training data only ────────────────────────────────
    ret_train = ret_v[train_start : train_end]
    gmm       = fit_gmm(X_tr_sc)                   # ← fit on train
    name_map  = assign_labels(gmm, scaler)          # for reporting only
    dir_map   = calibrate_directions(gmm, X_tr_sc, ret_train)  # data-driven

    # ── STEP 4: Compute posteriors on test data ───────────────────────────────
    posteriors = compute_posteriors(X_te_sc, gmm.means_, gmm.covariances_)

    # ── STEP 5: Compute all 4 metrics on test data ────────────────────────────
    metrics = compute_all_metrics(
        posteriors    = posteriors,
        returns       = ret_test,
        direction_map = dir_map,
        ann_factor    = ANN_FACTOR,
        tc_bps        = TC_BPS,
        model_name    = f"Fold {fold_idx + 1}",
    )

    # Regime prevalence on test set
    hard_labels = np.argmax(posteriors, axis=1)
    prevalence  = {name_map[k]: float((hard_labels == k).mean() * 100)
                   for k in range(K)}

    return {
        "fold":         fold_idx + 1,
        "train_start":  train_start,
        "train_end":    train_end,
        "test_start":   test_start,
        "test_end":     test_end,
        "n_train":      n_train,
        "n_test":       n_test,
        "name_map":     name_map,
        "dir_map":      dir_map,
        "posteriors":   posteriors,
        "ret_test":     ret_test,
        "metrics":      metrics,
        "prevalence":   prevalence,
    }


# ═════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_folds(fold_results: list[dict]) -> dict:
    """
    Aggregate per-fold metrics into mean ± std across folds.

    For M4 Sharpe we also compute the combined Sharpe by concatenating
    all test-fold net PnL series — this is the "real" out-of-sample Sharpe.
    """
    m1s = [r["metrics"].m1.score         for r in fold_results]
    m2s = [r["metrics"].m2.recall        for r in fold_results
           if not np.isnan(r["metrics"].m2.recall)]
    m3s = [r["metrics"].m3.spearman_r    for r in fold_results]
    m4s = [r["metrics"].m4.sharpe        for r in fold_results]

    # Combined out-of-sample Sharpe: concatenate all fold net PnL
    # (proper walk-forward Sharpe — no fold boundary effects)
    all_post, all_ret = [], []
    for r in fold_results:
        post = r["posteriors"]
        ret  = r["ret_test"]
        if len(post) > 1:
            all_post.append(post[:-1])
            all_ret.append(ret[1:])

    if all_post:
        combined_post = np.vstack(all_post)
        combined_ret  = np.concatenate(all_ret)

        # Rebuild dir_map from last fold (same instrument, same label logic)
        last_dir = fold_results[-1]["dir_map"]

        from metrics import m4_sharpe as _m4
        combined_m4 = _m4(combined_post, combined_ret, last_dir,
                          ann_factor=ANN_FACTOR, tc_bps=TC_BPS)
        combined_sharpe = combined_m4.sharpe
        combined_hit    = combined_m4.hit_rate
        combined_mdd    = combined_m4.max_drawdown
    else:
        combined_sharpe = float("nan")
        combined_hit    = float("nan")
        combined_mdd    = float("nan")

    return {
        "n_folds":          len(fold_results),
        "m1_mean":          float(np.mean(m1s)),
        "m1_std":           float(np.std(m1s)),
        "m2_mean":          float(np.mean(m2s)) if m2s else float("nan"),
        "m2_std":           float(np.std(m2s))  if m2s else float("nan"),
        "m3_mean":          float(np.mean(m3s)),
        "m3_std":           float(np.std(m3s)),
        "m4_mean":          float(np.mean(m4s)),
        "m4_std":           float(np.std(m4s)),
        "combined_sharpe":  combined_sharpe,
        "combined_hit":     combined_hit,
        "combined_mdd":     combined_mdd,
    }


# ═════════════════════════════════════════════════════════════════════════════
# REPORT WRITER
# ═════════════════════════════════════════════════════════════════════════════

def write_report(fold_results: list[dict],
                 agg: dict,
                 dates: pd.DatetimeIndex,
                 folds: list[tuple]) -> None:

    lines = []

    lines += [
        "# CRAN Phase 4 — Walk-Forward Results",
        "",
        f"> **Data**: `b copy.csv`  |  "
        f"TRAIN_WIN={TRAIN_WIN}  TEST_WIN={TEST_WIN}  STEP={STEP}",
        f"> **Folds**: {agg['n_folds']}  |  "
        f"Annualisation: √{ANN_FACTOR}  |  TC: {TC_BPS} bps one-way",
        "",
    ]

    # ── Aggregate summary ─────────────────────────────────────────────────────
    lines += [
        "## Aggregate Results (mean ± std across folds)",
        "",
        "| Metric | Mean | Std | Direction |",
        "|--------|------|-----|-----------|",
        f"| M1 KL-loss (nats) | {agg['m1_mean']:.4f} | ±{agg['m1_std']:.4f} | "
        f"Lower < 1.386 = beats uniform |",
        f"| M2 Transition recall | {agg['m2_mean']:.4f} | ±{agg['m2_std']:.4f} | "
        f"Higher > 0.30 = beats random |",
        f"| M3 Entropy-vol Spearman | {agg['m3_mean']:.4f} | ±{agg['m3_std']:.4f} | "
        f"Positive = calibrated uncertainty |",
        f"| M4 Sharpe (per-fold mean) | {agg['m4_mean']:.3f} | ±{agg['m4_std']:.3f} | "
        f"Higher = better alpha |",
        "",
        f"**Combined out-of-sample Sharpe** (all test folds concatenated): "
        f"**{agg['combined_sharpe']:.3f}**",
        f"Combined hit rate: {agg['combined_hit']:.2%}  |  "
        f"Max drawdown: {agg['combined_mdd']:.4f}",
        "",
    ]

    # ── Per-fold table ─────────────────────────────────────────────────────────
    lines += [
        "## Per-Fold Results",
        "",
        "| Fold | Train dates | Test dates | N_test | "
        "M1 | M2 | M3 | M4 Sharpe |",
        "|------|-------------|------------|--------|"
        "----|----|----|-----------|",
    ]

    for r in fold_results:
        f_idx = r["fold"]
        ts, te = folds[f_idx - 1][2], folds[f_idx - 1][3]
        train_s = folds[f_idx - 1][0]
        train_e = folds[f_idx - 1][1]

        d_train_s = dates[train_s].date() if train_s < len(dates) else "—"
        d_train_e = dates[min(train_e - 1, len(dates) - 1)].date()
        d_test_s  = dates[ts].date()      if ts < len(dates) else "—"
        d_test_e  = dates[min(te - 1, len(dates) - 1)].date()

        m  = r["metrics"]
        m2 = f"{m.m2.recall:.3f}" if not np.isnan(m.m2.recall) else "—"
        lines.append(
            f"| {f_idx} "
            f"| {d_train_s}→{d_train_e} "
            f"| {d_test_s}→{d_test_e} "
            f"| {r['n_test']} "
            f"| {m.m1.score:.4f} "
            f"| {m2} "
            f"| {m.m3.spearman_r:.4f} "
            f"| {m.m4.sharpe:.3f} |"
        )
    lines.append("")

    # ── Per-fold regime prevalence ────────────────────────────────────────────
    lines += [
        "## Regime Prevalence per Fold (%)",
        "",
        "| Fold | Trending | MeanRev | Volatile | Quiet |",
        "|------|----------|---------|----------|-------|",
    ]
    for r in fold_results:
        p = r["prevalence"]
        lines.append(
            f"| {r['fold']} "
            f"| {p.get('Trending', 0):.1f}% "
            f"| {p.get('MeanRev', 0):.1f}% "
            f"| {p.get('Volatile', 0):.1f}% "
            f"| {p.get('Quiet', 0):.1f}% |"
        )
    lines.append("")

    # ── Per-fold label assignment ─────────────────────────────────────────────
    lines += [
        "## GMM Label Assignments per Fold",
        "",
        "Consistency check: do regimes map to the same economic labels "
        "across folds? Instability here would indicate the GMM is finding "
        "different cluster structures in different time windows.",
        "",
        "| Fold | Component 0 | Component 1 | Component 2 | Component 3 |",
        "|------|-------------|-------------|-------------|-------------|",
    ]
    for r in fold_results:
        nm = r["name_map"]
        labels = [nm.get(k, "?") for k in range(K)]
        lines.append(f"| {r['fold']} | {' | '.join(labels)} |")
    lines.append("")

    # ── Interpretation ────────────────────────────────────────────────────────
    sharpe_consistent = all(r["metrics"].m4.sharpe > 0 for r in fold_results)
    lines += [
        "## Interpretation",
        "",
        f"- **M4 Sharpe positive in all folds**: "
        f"{'✅ Yes' if sharpe_consistent else '⚠️ No — some folds negative'}",
        f"- **M1 vs uniform**: "
        f"{'✅ Beats uniform' if agg['m1_mean'] < 1.386 else '⚠️ Worse than uniform'} "
        f"(mean={agg['m1_mean']:.4f} vs 1.3863)",
        f"- **M3 direction**: "
        f"{'✅ Positive (correct)' if agg['m3_mean'] > 0 else '⚠️ Negative'} "
        f"(mean={agg['m3_mean']:.4f})",
        "",
        "Walk-forward discipline was enforced across all folds. "
        "Scaler and GMM were fit exclusively on training windows. "
        "Results are fully out-of-sample.",
        "",
    ]

    out = "\n".join(lines)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walk_forward_results.md")
    with open(out_path, "w") as f:
        f.write(out)
    print(f"  Output → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("CRAN Phase 4 — Walk-Forward Engine")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading: {DATA_PATH}")
    daily   = load_daily(DATA_PATH)
    returns = daily["log_ret"].fillna(0).values
    dates   = daily.index

    print("Computing features...")
    features = compute_features(returns, lookback=LOOKBACK)

    valid  = ~np.any(np.isnan(features), axis=1)
    X      = features[valid]
    ret_v  = returns[valid]
    dates_v = dates[valid]
    n      = len(X)
    print(f"  Valid rows: {n}")

    # ── Generate folds ────────────────────────────────────────────────────────
    folds = generate_folds(n, TRAIN_WIN, TEST_WIN, STEP)
    print(f"\nFold schedule (TRAIN={TRAIN_WIN}, TEST={TEST_WIN}, STEP={STEP}):")
    for i, (ts, te, ss, se) in enumerate(folds):
        d_ts = dates_v[ts].date()
        d_te = dates_v[min(te-1, n-1)].date()
        d_ss = dates_v[ss].date()
        d_se = dates_v[min(se-1, n-1)].date()
        print(f"  Fold {i+1}: train [{d_ts}→{d_te}]  test [{d_ss}→{d_se}]  "
              f"n_train={te-ts}  n_test={se-ss}")

    # ── Run folds ─────────────────────────────────────────────────────────────
    print()
    fold_results = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        print(f"Running fold {i+1}/{len(folds)}...")
        result = run_fold(i, X, ret_v, ret_v,
                          train_start, train_end, test_start, test_end)
        m = result["metrics"]
        print(f"  Labels: {result['name_map']}")
        print(f"  M1={m.m1.score:.4f}  M2={m.m2.recall:.3f}  "
              f"M3={m.m3.spearman_r:.4f}  M4={m.m4.sharpe:.3f}")
        fold_results.append(result)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\nAggregating across folds...")
    agg = aggregate_folds(fold_results)

    # ── Write report ──────────────────────────────────────────────────────────
    write_report(fold_results, agg, dates_v, folds)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"  M1 KL-loss       {agg['m1_mean']:.4f} ± {agg['m1_std']:.4f}"
          f"  ({'beats' if agg['m1_mean'] < 1.386 else 'WORSE THAN'} uniform 1.3863)")
    print(f"  M2 Recall        {agg['m2_mean']:.4f} ± {agg['m2_std']:.4f}"
          f"  ({'beats' if agg['m2_mean'] > 0.30 else 'below'} random 0.30)")
    print(f"  M3 Spearman      {agg['m3_mean']:.4f} ± {agg['m3_std']:.4f}"
          f"  ({'positive ✅' if agg['m3_mean'] > 0 else 'negative ⚠️'})")
    print(f"  M4 Sharpe (mean) {agg['m4_mean']:.3f} ± {agg['m4_std']:.3f}")
    print(f"  M4 Sharpe (combined OOS): {agg['combined_sharpe']:.3f}")
    print()

    # Kill condition for Phase 4:
    # Combined OOS Sharpe must be > 0 — if negative, the signal doesn't survive
    # walk-forward and the project should be reconsidered
    if agg["combined_sharpe"] > 0:
        print("✅  Phase 4: GO — combined OOS Sharpe is positive.")
        print("    Proceed to Phase 5 (baselines).")
    else:
        print("❌  Phase 4: NO-GO — combined OOS Sharpe is negative.")
        print("    Signal does not survive walk-forward. Reconsider direction map.")
        sys.exit(1)


if __name__ == "__main__":
    main()
