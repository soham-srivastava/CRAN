"""
CRAN Phase 7 — Generate Paper Figures
=======================================
Wires real Phase 4/5/6 walk-forward results into evaluation/figures.py.

The rolling folds overlap (TEST_WIN=63, STEP=21), which is fine for the
Combined-Sharpe number in Table 1 (each fold's own test window is still
genuinely out-of-sample for that fold's fit) but not fine for a date-indexed
plot, where the same calendar day would get multiple posterior values from
different folds.

So for figures we STITCH a single non-overlapping OOS path: walking forward
through the folds in order, each fold only contributes the portion of its
test window that lies beyond the coverage frontier of all earlier folds.
This reconstructs one continuous, walk-forward-consistent series spanning
exactly [test_start(fold 1), test_end(fold last)) with no duplicate days.

For CRAN specifically, each fold's GMM is refit independently, so "component
index 2" in fold 1 is not necessarily the same regime as "component index 2"
in fold 3. Before stitching, CRAN's posterior columns are permuted into a
fixed canonical order (Trending, MeanRev, Volatile, Quiet) using that fold's
own name_map, so the stitched series has one consistent meaning per column.
Baselines/ablations don't carry named regimes, so this reordering only
applies to CRAN; their PnL math is unaffected since direction values are
just scalars in {-1, -0.5, 0, +0.5, +1}, not regime identities.

Run:    python generate_figures.py
Output: *.png (flat in cran/)
"""

from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from walk_forward import (
    load_daily, compute_features, generate_folds, run_fold as cran_run_fold,
    LOOKBACK, TRAIN_WIN, TEST_WIN, STEP, ANN_FACTOR, TC_BPS, DATA_PATH, K,
)
from ablation import ABLATIONS
import hamilton_hmm, bocpd, turbulence_index
from metrics import m3_entropy_vol_calibration, _entropy_nats
import figures as figs

CANONICAL = ["Trending", "MeanRev", "Volatile", "Quiet"]


def _cran_wrapper(X, ret_v, train_start, train_end, test_start, test_end) -> dict:
    out = cran_run_fold(0, X, ret_v, ret_v, train_start, train_end, test_start, test_end)
    return {
        "posteriors": out["posteriors"], "ret_test": out["ret_test"],
        "dir_map":    out["dir_map"],    "n_test":   out["n_test"],
        "model_name": "CRAN",            "name_map": out["name_map"],
    }


MODELS = {
    "CRAN":               _cran_wrapper,
    "Hamilton HMM":       hamilton_hmm.run_fold,
    "BOCPD":              bocpd.run_fold,
    "Turbulence Index":   turbulence_index.run_fold,
    **ABLATIONS,
}


def stitch_oos(run_fold_fn, X, ret_v, folds, canonicalize: bool = False):
    """Stitch one non-overlapping OOS path across rolling folds.

    If canonicalize, each fold's posterior columns are permuted via that
    fold's own name_map into the fixed CANONICAL regime order before being
    written into the output array — required for CRAN so column k means the
    same regime across the whole stitched series.
    """
    n = len(X)
    post_full = np.full((n, K), np.nan)
    dir_map_last = None
    frontier = None

    for (ts, te, ss, se) in folds:
        out = run_fold_fn(X, ret_v, ts, te, ss, se)
        post = out["posteriors"]
        dm = out["dir_map"]

        if canonicalize and out.get("name_map"):
            nm = out["name_map"]
            perm = [next(k for k, v in nm.items() if v == name) for name in CANONICAL]
            post = post[:, perm]
            dm = {i: dm[orig_k] for i, orig_k in enumerate(perm)}

        dir_map_last = dm
        start = ss if frontier is None else max(ss, frontier)
        if start < se:
            post_full[start:se] = post[start - ss:]
        frontier = se

    name_map_out = {i: name for i, name in enumerate(CANONICAL)} if canonicalize else None
    return post_full, dir_map_last, name_map_out


def net_pnl_series(posteriors: np.ndarray, ret_v: np.ndarray, dir_map: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reproduces evaluation/metrics.py::m4_sharpe's signal/PnL construction
    but returns the raw per-day net PnL array (for cumulative-alpha plotting)
    instead of collapsing straight to the aggregate Sharpe scalar."""
    valid = ~np.any(np.isnan(posteriors), axis=1)
    post  = posteriors[valid]
    ret   = ret_v[valid]
    idx   = np.where(valid)[0]

    post_t  = post[:-1]
    ret_tp1 = ret[1:]
    idx_t   = idx[:-1]

    log_k      = np.log(K)
    regimes    = np.argmax(post_t, axis=1)
    directions = np.array([dir_map.get(int(r), 0.0) for r in regimes])
    H          = _entropy_nats(post_t)
    conviction = 1.0 - H / log_k
    signals    = directions * conviction
    pnl        = signals * ret_tp1
    tc         = (TC_BPS / 10_000) * np.abs(np.diff(signals, prepend=signals[0]))
    net_pnl    = pnl - tc
    return idx_t, net_pnl


def main():
    print("CRAN Phase 7 — Generating Paper Figures")
    print("=" * 60)

    daily    = load_daily(DATA_PATH)
    returns  = daily["log_ret"].fillna(0).values
    dates    = daily.index
    features = compute_features(returns, lookback=LOOKBACK)
    valid    = ~np.any(np.isnan(features), axis=1)
    X, ret_v = features[valid], returns[valid]
    dates_v  = dates[valid]
    n        = len(X)

    folds = generate_folds(n, TRAIN_WIN, TEST_WIN, STEP)
    print(f"Folds: {len(folds)}\n")

    test_region_start = folds[0][2]
    test_region_end   = folds[-1][3]
    sl = slice(test_region_start, test_region_end)

    # ── Stitch CRAN's canonical, non-overlapping OOS path ───────────────────
    print("Stitching CRAN OOS path (canonical regime order)...")
    cran_post, cran_dir_map, cran_name_map = stitch_oos(_cran_wrapper, X, ret_v, folds, canonicalize=True)

    fig_dates = dates_v[sl]
    fig_post  = cran_post[sl]
    fig_ret   = ret_v[sl]

    m3 = m3_entropy_vol_calibration(fig_post[:-1], np.abs(fig_ret[1:]))
    print(f"  CRAN stitched M3: r={m3.spearman_r:.4f}  p={m3.p_value:.4f}  n={m3.n}")

    # ── Stitch + compute self-consistent PnL/Sharpe for every model ─────────
    print("\nStitching OOS paths + computing PnL for all models...")
    pnl_series, sharpes = {}, {}
    for name, fn in MODELS.items():
        try:
            post, dir_map, _ = stitch_oos(fn, X, ret_v, folds, canonicalize=False)
            idx_t, net_pnl = net_pnl_series(post, ret_v, dir_map)

            full = np.full(n, np.nan)
            full[idx_t] = net_pnl
            series = full[sl]
            series = np.nan_to_num(series, nan=0.0)
            pnl_series[name] = series

            valid_pnl = net_pnl[~np.isnan(net_pnl)]
            sr = (valid_pnl.mean() / (valid_pnl.std(ddof=1) + 1e-12)) * np.sqrt(ANN_FACTOR)
            sharpes[name] = float(sr)
            print(f"  {name:22s} n_days={len(valid_pnl):4d}  stitched Sharpe={sr:.3f}")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    # ── Generate all 4 figures ───────────────────────────────────────────────
    print()
    paths = figs.generate_all_figures(
        dates=fig_dates,
        posteriors=fig_post,
        returns=fig_ret,
        name_map=cran_name_map,
        m3_result=m3,
        pnl_series=pnl_series,
        sharpes=sharpes,
        instrument="b copy.csv",
    )
    print("\nFigures written:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
