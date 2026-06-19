# CRAN Paper — Table 1: Full Comparison

> Data: `b copy.csv`  |  Folds: 6  |  TRAIN_WIN=124  TEST_WIN=63  STEP=21  |  TC=5.0bps  |  Annualisation: sqrt(365)

| Model | M1 KL-loss | M2 Recall | M3 Spearman | M4 Sharpe (mean) | **Combined OOS Sharpe** |
|-------|-----------:|----------:|------------:|-----------------:|------------------------:|
| **CRAN** | 2.4074 | 0.0592 | -0.0935 | -0.642 | **1.643** |
| Hamilton HMM | 4.2238 | 0.1000 | -0.0282 | 0.920 | -0.775 |
| BOCPD | 0.0253 | nan | 0.1237 | -0.163 | 0.407 |
| Turbulence Index | 0.5428 | 0.3028 | -0.0596 | 0.807 | 0.719 |
| CRAN (no Bayes) | 2.3884 | 0.0588 | -0.0956 | -0.745 | 1.613 |
| CRAN (hard labels) | 129.4339 | 0.0000 | nan | 1.385 | 1.751 |
| CRAN (GMM prior) | 2.3884 | 0.0588 | -0.0956 | -0.745 | 1.613 |

## Reference points

- M1 uniform-prior baseline: 1.3863 nats (log 4) — lower is better
- M2 random-transition baseline: 0.30 — higher is better
- M3: positive = entropy correctly anticipates next-day volatility
- M4 / Combined Sharpe: net of 5bps one-way transaction costs, annualised by sqrt(365)

## Reading this table

Baselines (Hamilton HMM, BOCPD, Turbulence Index) and ablations (CRAN variants with one design choice removed) are run on the identical walk-forward folds as CRAN, with identical direction calibration discipline. Any Sharpe advantage for CRAN over its own ablations isolates the contribution of that specific design choice (Bayesian update, soft posterior, uniform prior). Any advantage over the baselines reflects the value of continuous, named-prior regime affinity versus the respective discrete/scalar/changepoint approach.

