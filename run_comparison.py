"""
CRAN Phase 5+6 — Master Comparison Runner
============================================
Runs CRAN, all 3 baselines, and all 3 ablations on the IDENTICAL walk-forward
folds (same TRAIN_WIN/TEST_WIN/STEP, same data, same TC), and produces the
paper's Table 1.

This is the single source of truth for "does CRAN actually win" — every
model here is held to the same walk-forward discipline (scaler/model fit on
train only) and the same direction-calibration rule (data-driven from
training returns, never from post-hoc label inspection).

Run:    python run_comparison.py
Output: comparison_table.md
"""

from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from walk_forward import (
    load_daily, compute_features, generate_folds, run_fold as cran_run_fold,
    LOOKBACK, TRAIN_WIN, TEST_WIN, STEP, ANN_FACTOR, TC_BPS, DATA_PATH,
)
from ablation import ABLATIONS, run_ablation_on_folds
import hamilton_hmm, bocpd, turbulence_index


# ── Wrap CRAN's run_fold (different signature/return shape) to match the
#    common interface used by run_ablation_on_folds for every other model ──
def _cran_wrapper(X, ret_v, train_start, train_end, test_start, test_end) -> dict:
    out = cran_run_fold(0, X, ret_v, ret_v, train_start, train_end, test_start, test_end)
    return {
        "posteriors": out["posteriors"],
        "ret_test":   out["ret_test"],
        "dir_map":    out["dir_map"],
        "n_test":     out["n_test"],
        "model_name": "CRAN",
    }


MODELS = {
    "CRAN":                _cran_wrapper,
    "Hamilton HMM":         hamilton_hmm.run_fold,
    "BOCPD":                bocpd.run_fold,
    "Turbulence Index":     turbulence_index.run_fold,
    **ABLATIONS,   # CRAN (no Bayes), CRAN (hard labels), CRAN (GMM prior)
}


def main():
    print("CRAN Phase 5+6 — Full Comparison (Baselines + Ablations)")
    print("=" * 70)

    daily   = load_daily(DATA_PATH)
    returns = daily["log_ret"].fillna(0).values
    features = compute_features(returns, lookback=LOOKBACK)
    valid  = ~np.any(np.isnan(features), axis=1)
    X, ret_v = features[valid], returns[valid]
    n = len(X)

    folds = generate_folds(n, TRAIN_WIN, TEST_WIN, STEP)
    print(f"Folds: {len(folds)}  (TRAIN={TRAIN_WIN} TEST={TEST_WIN} STEP={STEP})\n")

    results = {}
    for name, fn in MODELS.items():
        print(f"Running {name} ...")
        try:
            agg = run_ablation_on_folds(fn, X, ret_v, folds)
            results[name] = agg
            print(f"  M1={agg['m1_mean']:.4f}  M2={agg['m2_mean']:.4f}  "
                  f"M3={agg['m3_mean']:.4f}  M4(mean)={agg['m4_mean']:.3f}  "
                  f"Combined Sharpe={agg['combined_sharpe']:.3f}\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")
            results[name] = None

    # ── Build Table 1 ─────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_table.md")

    lines = [
        "# CRAN Paper — Table 1: Full Comparison",
        "",
        f"> Data: `b copy.csv`  |  Folds: {len(folds)}  |  "
        f"TRAIN_WIN={TRAIN_WIN}  TEST_WIN={TEST_WIN}  STEP={STEP}  |  "
        f"TC={TC_BPS}bps  |  Annualisation: sqrt({ANN_FACTOR})",
        "",
        "| Model | M1 KL-loss | M2 Recall | M3 Spearman | M4 Sharpe (mean) | "
        "**Combined OOS Sharpe** |",
        "|-------|-----------:|----------:|------------:|-----------------:|"
        "------------------------:|",
    ]

    order = ["CRAN", "Hamilton HMM", "BOCPD", "Turbulence Index",
             "CRAN (no Bayes)", "CRAN (hard labels)", "CRAN (GMM prior)"]
    for name in order:
        agg = results.get(name)
        if agg is None:
            lines.append(f"| {name} | FAILED | — | — | — | — |")
            continue
        bold = "**" if name == "CRAN" else ""
        lines.append(
            f"| {bold}{name}{bold} | {agg['m1_mean']:.4f} | {agg['m2_mean']:.4f} | "
            f"{agg['m3_mean']:.4f} | {agg['m4_mean']:.3f} | "
            f"{bold}{agg['combined_sharpe']:.3f}{bold} |"
        )
    lines.append("")

    lines += [
        "## Reference points",
        "",
        "- M1 uniform-prior baseline: 1.3863 nats (log 4) — lower is better",
        "- M2 random-transition baseline: 0.30 — higher is better",
        "- M3: positive = entropy correctly anticipates next-day volatility",
        "- M4 / Combined Sharpe: net of 5bps one-way transaction costs, annualised by sqrt(365)",
        "",
        "## Reading this table",
        "",
        "Baselines (Hamilton HMM, BOCPD, Turbulence Index) and ablations "
        "(CRAN variants with one design choice removed) are run on the "
        "identical walk-forward folds as CRAN, with identical direction "
        "calibration discipline. Any Sharpe advantage for CRAN over its own "
        "ablations isolates the contribution of that specific design choice "
        "(Bayesian update, soft posterior, uniform prior). Any advantage over "
        "the baselines reflects the value of continuous, named-prior regime "
        "affinity versus the respective discrete/scalar/changepoint approach.",
        "",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nOutput -> {out_path}")

    # ── Console summary, sorted by combined Sharpe ──────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — sorted by Combined OOS Sharpe")
    print("=" * 70)
    ranked = sorted(
        [(name, results[name]) for name in order if results.get(name) is not None],
        key=lambda kv: kv[1]["combined_sharpe"], reverse=True,
    )
    for name, agg in ranked:
        marker = " <-- CRAN" if name == "CRAN" else ""
        print(f"  {name:25s}  Sharpe={agg['combined_sharpe']:.3f}{marker}")


if __name__ == "__main__":
    main()
