# CRAN Phase 4 — Walk-Forward Results

> **Data**: `b copy.csv`  |  TRAIN_WIN=124  TEST_WIN=63  STEP=21
> **Folds**: 6  |  Annualisation: √365  |  TC: 5.0 bps one-way

## Aggregate Results (mean ± std across folds)

| Metric | Mean | Std | Direction |
|--------|------|-----|-----------|
| M1 KL-loss (nats) | 2.4074 | ±1.6050 | Lower < 1.386 = beats uniform |
| M2 Transition recall | 0.0592 | ±0.0936 | Higher > 0.30 = beats random |
| M3 Entropy-vol Spearman | -0.0935 | ±0.1892 | Positive = calibrated uncertainty |
| M4 Sharpe (per-fold mean) | -0.642 | ±2.222 | Higher = better alpha |

**Combined out-of-sample Sharpe** (all test folds concatenated): **1.643**
Combined hit rate: 51.63%  |  Max drawdown: -0.3856

## Per-Fold Results

| Fold | Train dates | Test dates | N_test | M1 | M2 | M3 | M4 Sharpe |
|------|-------------|------------|--------|----|----|----|-----------|
| 1 | 2025-07-15→2025-11-15 | 2025-11-16→2026-01-17 | 63 | 0.8582 | 0.250 | 0.0662 | -1.797 |
| 2 | 2025-08-05→2025-12-06 | 2025-12-07→2026-02-07 | 63 | 4.1838 | 0.105 | -0.0728 | 1.922 |
| 3 | 2025-08-26→2025-12-27 | 2025-12-28→2026-02-28 | 63 | 3.0349 | 0.000 | 0.0767 | 2.477 |
| 4 | 2025-09-16→2026-01-17 | 2026-01-18→2026-03-20 | 62 | 0.0661 | 0.000 | 0.0697 | -2.950 |
| 5 | 2025-10-07→2026-02-07 | 2026-02-08→2026-03-20 | 41 | 1.9424 | 0.000 | -0.3253 | -0.304 |
| 6 | 2025-10-28→2026-02-28 | 2026-03-01→2026-03-20 | 20 | 4.3591 | 0.000 | -0.3754 | -3.200 |

## Regime Prevalence per Fold (%)

| Fold | Trending | MeanRev | Volatile | Quiet |
|------|----------|---------|----------|-------|
| 1 | 14.3% | 79.4% | 6.3% | 0.0% |
| 2 | 33.3% | 31.7% | 0.0% | 34.9% |
| 3 | 58.7% | 6.3% | 0.0% | 34.9% |
| 4 | 82.3% | 0.0% | 0.0% | 17.7% |
| 5 | 43.9% | 4.9% | 51.2% | 0.0% |
| 6 | 10.0% | 0.0% | 20.0% | 70.0% |

## GMM Label Assignments per Fold

Consistency check: do regimes map to the same economic labels across folds? Instability here would indicate the GMM is finding different cluster structures in different time windows.

| Fold | Component 0 | Component 1 | Component 2 | Component 3 |
|------|-------------|-------------|-------------|-------------|
| 1 | Quiet | Volatile | Trending | MeanRev |
| 2 | Trending | MeanRev | Volatile | Quiet |
| 3 | Volatile | Quiet | Trending | MeanRev |
| 4 | Quiet | Trending | MeanRev | Volatile |
| 5 | Trending | MeanRev | Volatile | Quiet |
| 6 | Volatile | Quiet | MeanRev | Trending |

## Interpretation

- **M4 Sharpe positive in all folds**: ⚠️ No — some folds negative
- **M1 vs uniform**: ⚠️ Worse than uniform (mean=2.4074 vs 1.3863)
- **M3 direction**: ⚠️ Negative (mean=-0.0935)

Walk-forward discipline was enforced across all folds. Scaler and GMM were fit exclusively on training windows. Results are fully out-of-sample.
