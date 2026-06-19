# CRAN Phase 2 — Real Data EDA Report

> **Source file**: `b copy.csv`  |  Instrument: 24/7 high-volatility asset (annualised vol 72%, consistent with crypto/commodity futures)
> **Date range**: 2025-06-15 → 2026-03-20  |  Daily rows: 279  |  Valid feature rows: 249

## Kill Condition Summary

| # | Condition | Detail | Status |
|---|-----------|--------|--------|
| ✅ | KC-A features non-degenerate | degenerate: none | ✅ PASS |
| ✅ | KC-B no ghost regimes | ghost: none | ✅ PASS |
| ✅ | KC-C vol ordering correct | Volatile rv30=0.9450, Quiet rv30=0.5606 | ✅ PASS |
| ✅ | KC-D min Bhattacharyya > 0.10 | min B=1.9848 | ✅ PASS |

**Phase 2 verdict: ✅ GO — real data passes all kill conditions.**

### ⚠️ Data Sufficiency Advisory (not a kill condition)

Architecture spec requires TRAIN_WIN = 504 days. Available valid feature rows = **249** (shortfall: **255 days**).

**Recommendation for Phase 4 (walk-forward):** reduce TRAIN_WIN to **124 days** (~4 months). This gives at least 1 test fold. Alternatively, obtain additional historical data to reach 504+ days.

---

## 1. Data Quality Audit

| Metric | Value |
|--------|-------|
| Instrument type | 24/7 (no market hours) — crypto or commodity futures |
| Calendar gaps > 2 days | 0 |
| Days with \|ret\| > 10% | 6 |
| OHLC consistency failures | 0 |
| Daily ret mean | -0.0627% |
| Daily ret std | 3.7443% |
| Annualised vol (×√365) | **71.5%** |
| Skewness | -0.056 |
| Excess kurtosis | 2.186 |

### Extreme Daily Moves (|ret| > 10%)

| Date | Close | Log-ret |
|------|-------|---------|
| 2025-08-22 | 4832.07 | +13.42% |
| 2025-10-10 | 3829.72 | -13.15% |
| 2025-10-12 | 4152.29 | +10.28% |
| 2026-02-05 | 1826.83 | -16.21% |
| 2026-02-06 | 2063.38 | +12.18% |
| 2026-02-25 | 2057.48 | +10.52% |

_Note: extreme moves are real market events. The Volatile regime should capture these days. No imputation applied — outliers are informative for regime detection._

---

## 2. Feature Distributions (real data)

| Feature | Mean | Std | p5 | p95 |
|---------|------|-----|----|-----|
| ret_1d | -0.00137 | 0.03799 | -0.05895 | 0.06148 |
| ret_5d | -0.00575 | 0.08174 | -0.12548 | 0.13355 |
| ret_20d | -0.01850 | 0.18322 | -0.36635 | 0.33526 |
| realvol_10d | 0.67177 | 0.26960 | 0.34204 | 1.19077 |
| realvol_30d | 0.69856 | 0.16575 | 0.44881 | 1.01075 |
| autocorr_lag1 | 0.02679 | 0.18685 | -0.32517 | 0.31385 |
| autocorr_lag5 | -0.03911 | 0.18630 | -0.38233 | 0.23752 |
| momentum_ratio | 0.00290 | 0.26548 | -0.37294 | 0.59581 |

All 8 features show non-degenerate distributions (std >> 0, no excess NaN). Feature magnitudes are consistent with a high-volatility instrument: realvol figures will be ~3–5× higher than for Nifty equity.

---

## 3. GMM Regime Centroids (real data, exploratory fit)

> GMM fit on ALL available data — NOT walk-forward. This is an exploratory sanity check only. Walk-forward GMM is Phase 4.

### Centroid Heatmap (unscaled feature values)

| Regime | ret_1d | ret_5d | ret_20d | realvol_10d | realvol_30d | autocorr_lag1 | autocorr_lag5 | momentum_ratio |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| Trending | 0.00283 | 0.00081 | -0.03448 | 0.63016 | 0.69080 | 0.03583 | -0.00078 | -0.05414 |
| MeanRev | -0.04782 | -0.15470 | -0.28625 | 0.75922 | 0.71729 | 0.06762 | -0.08148 | -0.37798 |
| Volatile | -0.00414 | -0.02191 | -0.15155 | 0.89173 | 0.94498 | -0.22943 | -0.06564 | -0.13247 |
| Quiet | 0.00646 | 0.03731 | 0.17542 | 0.59458 | 0.56063 | 0.14956 | -0.08513 | 0.31369 |

---

## 4. Regime-Conditional Return Distributions

| Regime | Ann. Mean | Ann. Vol | Skew | Prevalence | N days |
|--------|-----------|----------|------|------------|--------|
| Trending | +110.13% | 59.13% | +0.523 | 51.0% | 127 |
| MeanRev | -1749.15% | 95.07% | -0.501 | 8.0% | 20 |
| Volatile | -164.19% | 65.98% | +0.840 | 15.7% | 39 |
| Quiet | +236.58% | 75.27% | +0.521 | 25.3% | 63 |

Economic alignment check:
- Volatile has highest realvol_30d centroid: 0.9450  ✅
- Quiet has lowest realvol_30d centroid:     0.5606  ✅

---

## 5. Regime Separation on Real Data

Kill threshold for real data: min B > 0.10 (weaker than Phase 1's 0.30 because real regimes overlap more than simulated ones).

| Pair | B-distance | Pass? |
|------|-----------|-------|
| Volatile vs MeanRev | 16.8991 | ✅ |
| Quiet vs Volatile | 10.8986 | ✅ |
| Quiet vs MeanRev | 6.6039 | ✅ |
| Volatile vs Trending | 4.3324 | ✅ |
| Trending vs MeanRev | 3.2872 | ✅ |
| Quiet vs Trending | 1.9848 | ✅ |

---

## 6. Bayesian Sharpening on Real Data

| Quantity | Value |
|----------|-------|
| H(uniform) | 1.3863 nats |
| Mean H(posterior) | 0.0862 nats |
| Entropy reduction | 93.8% |
| Median H(posterior) | 0.0001 nats |

Posterior sharpening on real data is expected to be weaker than on the clean simulated data (Phase 1). As long as the entropy reduction is positive, the named priors carry information.

---

## 7. Phase 4 Design Implications

1. **Annualisation factor**: use √365 (not √252) — instrument trades 24/7.
2. **TRAIN_WIN**: reduce from 504 to 124 days for walk-forward    given only 249 valid feature rows.    Or fetch more history to meet the 504-day spec.
3. **Extreme move handling**: 6 days with |ret| > 10%.    No clipping applied here. Monitor whether Volatile regime    captures them correctly.
4. **Transaction costs**: 5bps assumed in Phase 1.    For a 24/7 crypto/commodity instrument, verify actual spread +    funding costs — may be higher.
