# CRAN Phase 3 — Metrics Verification Report

> **Real data**: `b copy.csv`  |  2025-06-15 → 2026-03-20
> Train/test split: 124 / 125 days  |  Annualisation: √365 (24/7 instrument)
> Verification goal: confirm each metric function is correct and returns values in expected ranges on real market data.

## Verification Summary

| Check | Result |
|-------|--------|
| Sanity bounds (all metrics finite and in range) | ✅ PASS |
| Edge case tests (5/5) | ✅ PASS |
| M1 < log(4)=1.386 (beats uniform) | ⚠️ note |
| M2 recall > 0.25 (beats random) | ⚠️ note |
| M3 spearman > 0 (positive correlation) | ✅ |
| M4 |Sharpe| > 0 (some signal) | ✅ |

**Phase 3 verdict: ✅ GO — metrics infrastructure validated.**

---

## M1: KL Calibration (Log-Loss)

| Quantity | Value |
|----------|-------|
| Score (mean −log p(k_{t+1})) | 1.4113 nats |
| Uniform baseline (log 4) | 1.3863 nats |
| Improvement vs uniform | -0.0250 nats |

Lower = better. A positive improvement vs uniform means the posteriors carry genuine predictive information about the next-day regime.

---

## M2: Regime Transition Recall

| Quantity | Value |
|----------|-------|
| Recall | 0.1304 |
| Transitions detected | 23 |
| Transitions anticipated | 3 |
| Daily transition rate | 0.1855 |
| Random baseline | ≈ 0.30 |

Higher = better. Recall above the random baseline means the model's soft affinities provide early warning before hard regime switches.

---

## M3: Entropy–Volatility Calibration

| Quantity | Value |
|----------|-------|
| Spearman r | 0.0796 |
| p-value | 0.37926 |
| Significant (p < 0.05) | No ❌ |
| N pairs | 124 |

Positive and significant Spearman r confirms that high-entropy posteriors precede more volatile days — uncertainty quantification working as intended.

---

## M4: Sharpe Ratio

| Quantity | Value |
|----------|-------|
| Annualised Sharpe | 1.093 |
| Hit rate | 43.55% |
| Max drawdown | -0.2940 |
| Calmar ratio | 3.717 |
| N days | 124 |
| TC (bps) | 5.0 |

---

## Edge Case Tests

| Test | Expected | Result | Pass? |
|------|----------|--------|-------|
| M1 perfect posterior | ≈ 0.0 | -0.000000 | ✅ |
| M1 uniform posterior | = log(4) = 1.3863 | 1.386294 | ✅ |
| M2 no transitions | n_trans = 0 | 0 | ✅ |
| M3 random data | |r| < 0.15 | 0.0337 | ✅ |
| M4 zero signal | Sharpe ≈ 0 | 0.000000 | ✅ |

---

## Sanity Bounds

| Metric | Range | Verified value | Pass? |
|--------|-------|----------------|-------|
| M1 score | [0, 2.773] | 1.4113 | ✅ |
| M2 recall | [0, 1] | 0.1304 | ✅ |
| M3 Spearman r | [-1, 1] | 0.0796 | ✅ |
| M4 Sharpe | [-20, 20] | 1.093 | ✅ |
| M4 hit rate | [0, 1] | 0.4355 | ✅ |
