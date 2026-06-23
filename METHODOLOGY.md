# CRAN — Methodology & Math

This file walks through exactly what happens at each step of CRAN, what gets
fit vs. frozen, and the formula behind it. Each step links to the actual
function in `walk_forward.py` / `metrics.py` that implements it.

## 1. Pipeline diagram

```
                    ┌─────────────────────────────────────────────┐
                    │   RAW DATA: 1-min OHLCV → resampled to daily │
                    │   load_daily()                               │
                    └───────────────────┬───────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │  8 FEATURES PER DAY (rolling, past-only)     │
                    │  compute_features()                          │
                    │  ret_1d, ret_5d, ret_20d,                    │
                    │  realvol_10d, realvol_30d,                   │
                    │  autocorr_lag1, autocorr_lag5,                │
                    │  momentum_ratio                               │
                    └───────────────────┬───────────────────────────┘
                                        │
                       split into per-fold TRAIN / TEST windows
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        ▼  (TRAIN window only)                                          ▼  (TEST window only)
┌──────────────────────────────┐                          ┌──────────────────────────────┐
│ FIT  StandardScaler           │   scaler params frozen   │ TRANSFORM only               │
│ scaler.fit_transform(X_train) │ ────────────────────────▶ │ scaler.transform(X_test)     │
└──────────────┬────────────────┘                          └──────────────┬───────────────┘
               ▼                                                          │
┌──────────────────────────────┐                                         │
│ FIT  Gaussian Mixture Model    │   GMM means/covariances frozen          │
│ fit_gmm(X_train_scaled)       │ ────────────────────────┐               │
│ → 4 cluster means + covariances│                        │               │
└──────────────┬────────────────┘                        │               │
               │                                          │               │
     ┌─────────┴──────────┐                               │               │
     ▼                    ▼                               ▼               ▼
┌─────────────┐   ┌────────────────────┐      ┌─────────────────────────────────┐
│ assign_labels│   │ calibrate_directions│      │ compute_posteriors(X_test_scaled,│
│ (centroid →  │   │ (mean training      │      │   means, covs)                  │
│  name)       │   │  return per cluster) │      │ Bayes' rule → affinity vector   │
│ → name_map   │   │ → dir_map            │      │ per test day                    │
└─────────────┘   └────────────────────┘      └─────────────────┬───────────────┘
                                                                  ▼
                                                  ┌─────────────────────────────────┐
                                                  │ SIGNAL                          │
                                                  │ conviction = 1 - H/log(K)        │
                                                  │ signal = direction × conviction  │
                                                  └─────────────────┬───────────────┘
                                                                    ▼
                                                  ┌─────────────────────────────────┐
                                                  │ PnL = signal_t × return_{t+1}    │
                                                  │   minus transaction cost          │
                                                  └─────────────────┬───────────────┘
                                                                    ▼
                                                  ┌─────────────────────────────────┐
                                                  │ M1–M4 metrics computed on        │
                                                  │ this fold's TEST output only     │
                                                  └─────────────────────────────────┘

This entire right-hand path repeats once per walk-forward fold, rolling
forward by STEP=21 days each time. Nothing fit on TRAIN ever sees TEST rows,
and nothing fit in fold N carries over into fold N+1 — every fold starts
fresh (`scaler`, `gmm` are local variables created inside `run_fold()`).
```

## 2. Step-by-step: what is updated, and which formula

### Step A — Features (`compute_features`, walk_forward.py:81)
No fitting happens here — it's pure arithmetic on a rolling window of past
returns (last 30 days), so it's automatically walk-forward safe. Output:
an 8-number vector per day.

### Step B — Standardize (`StandardScaler`, sklearn)
**Fit only on TRAIN.** For each of the 8 features, computes the mean `μ`
and standard deviation `σ` over the training window:

```
x_scaled = (x - μ) / σ
```

`μ` and `σ` are then frozen and reused to transform the TEST window —
TEST days are never used to compute them. This is what "fit only on
training data" means operationally.

### Step C — Fit the Gaussian Mixture Model (`fit_gmm`, walk_forward.py:106)
**Fit only on TRAIN.** A GMM models the 8-dimensional training data as a
mixture of `K=4` multivariate Gaussian clusters. Fitting estimates, per
cluster `j`:

```
mean_j         (8,)     — the cluster's centroid in feature space
covariance_j   (8, 8)   — the cluster's shape/spread
mixture_weight_j         — how much of the data belongs to cluster j
```

via the EM (Expectation-Maximization) algorithm, with 15 random
restarts (`n_init=15`) to avoid a bad local optimum. These four
`(mean_j, covariance_j)` pairs are the **named priors** — frozen for
the rest of this fold.

### Step D — Name the clusters (`assign_labels`, walk_forward.py:138)
No fitting — just inspection of the frozen cluster means:

```
Volatile  = argmax_j ( mean_j[realvol_30d] )
Quiet     = argmin_j ( mean_j[realvol_30d] )
Trending  = argmax over remaining 2 clusters of mean_j[ret_20d]
MeanRev   = the other remaining cluster
```

This is reporting/interpretation only — it has no effect on the math
below; it just lets a human read "Trending" instead of "cluster 2."

### Step E — Calibrate trading direction (`calibrate_directions`, walk_forward.py:159)
**Uses TRAIN returns only.** For each cluster `k`, look at training
days where cluster `k` was the most likely cluster, and take the mean
realized return on those days:

```
direction_k =  +1.0   if mean_return_k >  1e-4
              -1.0   if mean_return_k < -1e-4
              +0.5   otherwise (near-zero → mild long bias)
```

This is deliberately *not* tied to the human-readable name from Step D —
it's grounded directly in what that cluster's days actually returned
during training, so a relabeling quirk can't silently flip a trading
direction.

### Step F — Daily posterior / affinity vector (`compute_posteriors`, walk_forward.py:116)
**Applied to TEST data, using the frozen Step C parameters.** This is
Bayes' rule. Start from a uniform prior (no reason to favor any mood
before seeing today's data):

```
prior_j = 1 / K                                  for each of the 4 moods

likelihood_j(x_t) = N(x_t ; mean_j, covariance_j)     ← Gaussian density,
                                                          how plausible today's
                                                          8 features are under
                                                          cluster j

posterior_j(x_t) =      prior_j · likelihood_j(x_t)
                   ───────────────────────────────────────
                    Σ_k  prior_k · likelihood_k(x_t)
```

(Implemented in log-space with `scipy.special.logsumexp` for numerical
stability, but the math is exactly the normalized prior-times-likelihood
above.) The output is a 4-number vector that sums to 1 — the affinity
vector for that single day.

### Step G — Convert affinity into a trade signal (`m4_sharpe`, metrics.py:376)
```
H_t          = - Σ_j  posterior_j(x_t) · log( posterior_j(x_t) )     (Shannon entropy, nats)
conviction_t = 1 - H_t / log(K)                                       ∈ [0, 1]
regime_t     = argmax_j  posterior_j(x_t)
signal_t     = direction_{regime_t}  ×  conviction_t                  ∈ [-1, +1]
```

Entropy is maximal (`log K`) when the posterior is perfectly uniform
(total uncertainty) and zero when it's a one-hot spike (total certainty).
`conviction_t` is therefore 0 when the model has no idea which mood
today is, and 1 when it's completely sure — this is the mechanism that
makes CRAN size down automatically under uncertainty, instead of always
betting full size.

### Step H — PnL, net of costs (`m4_sharpe`, metrics.py:376)
```
pnl_t = signal_t × return_{t+1}
tc_t  = (TC_BPS / 10,000) × | signal_t − signal_{t-1} |
net_pnl_t = pnl_t − tc_t
Sharpe = mean(net_pnl) / std(net_pnl)  ×  sqrt(ANN_FACTOR)
```

`return_{t+1}` (not `return_t`) is used deliberately — the signal is
formed using only information available up to and including day `t`,
and is only allowed to earn the *next* day's return, so there's no
lookahead.

### Step I — Calibration check (`m1_kl_calibration`, metrics.py:176)
A separate question from Sharpe: *is the posterior's confidence
trustworthy?* For each day, look at the probability the posterior
assigned to whatever regime actually showed up the next day:

```
M1 = mean_t [ -log( posterior_t[ regime_{t+1} ] ) ]
```

This is a cross-entropy / log-loss. A perfect model scores 0; a model
that knows nothing (uniform posterior every day) scores `log(4) ≈ 1.386`.
This is the metric that exposes the project's central finding: a
hard-label variant can score a *higher* Sharpe (1.75) while scoring
catastrophically on M1 (129.4) — because forcing one-hot labels gives
it no way to express "I'm not sure," so on a wrong day it's
maximally, confidently wrong.

## 3. One fold, start to finish

```
Fold k:
  1. X_train, X_test  ← slice this fold's date range
  2. scaler ← StandardScaler().fit(X_train)              [Step B]
  3. gmm    ← fit_gmm(scaler.transform(X_train))          [Step C]
  4. name_map ← assign_labels(gmm, scaler)                [Step D]
  5. dir_map  ← calibrate_directions(gmm, X_train, ret_train)  [Step E]
  6. posteriors ← compute_posteriors(scaler.transform(X_test), gmm.means_, gmm.covariances_)  [Step F]
  7. metrics  ← compute_all_metrics(posteriors, ret_test, dir_map)  [Steps G, H, I]
  8. discard scaler, gmm, name_map, dir_map — fold k+1 starts from scratch
```

Repeated across all rolling folds (`generate_folds`, walk_forward.py:205),
then every fold's TEST-only metrics are pooled (`aggregate_folds`) into
the headline numbers reported in `comparison_table.md` and `README.md`.
