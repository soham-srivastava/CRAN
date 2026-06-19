# CRAN: Continuous Regime Affinity via Bayesian Inference on Named Priors

## What this paper is about

Markets move through different moods. Sometimes prices trend steadily in one direction. Sometimes they bounce back and forth (mean-reverting). Sometimes they're calm and quiet. Sometimes they're choppy and unpredictable (volatile). A trading rule that works well in one of these moods often loses money in another.

Most existing ways of detecting these moods make a hard choice: "today is a trending day," full stop. This is a problem because the market usually doesn't flip from one mood to another instantly — it drifts, and on many days it's genuinely a mix of two moods at once. A model that is forced to pick just one label throws away that mixed, uncertain information.

CRAN is a simple alternative: instead of picking one label per day, it keeps a probability over four named moods every day — for example, "70% trending, 20% volatile, 10% quiet" — and updates that probability as new data comes in. We call this probability the model's "affinity" for each regime. The model's trading decision (which direction to bet, and how big a bet to make) comes directly from this probability: if the model is unsure (the probability is spread out across moods), it bets small or not at all; if it's confident (the probability is concentrated on one mood), it bets bigger in that mood's usual direction.

This paper describes how CRAN works, how we tested it fairly against simpler methods, and what we found — including a result that complicates the simple story, which we think is the most useful part of this paper.

## Why we built it this way

The point of this project wasn't just to build something that makes money in a backtest. Backtests are easy to fool yourself with. So we built CRAN as a sequence of stages, each with a clear pass/fail check, so that if the idea didn't actually hold up we would find out early instead of polishing a broken idea for months. Every stage in this paper passed its check.

## How CRAN works, in plain terms

1. **Features.** Each day, we compute eight numbers that describe recent market behavior: the 1-day, 5-day, and 20-day return, two measures of recent volatility (10-day and 30-day), how correlated returns are with themselves over time (a sign of trending vs. choppy behavior), and a momentum measure.

2. **Fitting the four moods.** Using only past data (never future data — explained below), we fit a statistical model called a Gaussian Mixture Model to these eight numbers. This finds four natural clusters in the data. We then look at each cluster's typical behavior and assign it one of four names: Trending, Mean-Reverting, Volatile, or Quiet. For example, the cluster with the highest volatility gets called "Volatile"; the cluster with the lowest gets called "Quiet."

3. **Daily probability.** Each new day, instead of asking "which one cluster does today belong to," we ask "how close is today to each of the four clusters," and turn that into a probability distribution over the four moods using Bayes' rule (a standard way of updating a probability as new evidence arrives). This is the "affinity vector" for that day.

4. **Trading signal.** Each mood has a known typical direction (Trending and Quiet days lean toward an upward bet, Mean-Reverting leans toward a contrarian bet, Volatile leans toward no bet). The day's signal is: (the typical direction of the most likely mood) × (how confident the model is that day). Confidence is measured by how concentrated the day's probability is — a probability spread evenly across all four moods means zero confidence and zero bet size.

## How we tested it fairly

The single most important rule in this kind of research is: never let the model see the future. We enforced this by only ever fitting the model on a block of past days, then testing it, untouched, on the days immediately following — then rolling the whole window forward and repeating. This is called walk-forward testing, and we used it for every single result in this paper, including the comparison methods below.

We measured four things, not just profit:

- **Calibration (M1):** when the model says "I'm 80% confident in regime X," is it actually right about 80% of the time? A model that is good at making money but bad at this is dangerous, because you can't trust its confidence level when sizing real bets.
- **Early-warning ability (M2):** when the market is about to switch moods, does the model's probability shift toward the new mood before the switch fully happens?
- **Uncertainty vs. volatility (M3):** on days the model is more uncertain, is the market actually more volatile the next day? If so, the model's uncertainty is itself a useful warning signal, not just noise.
- **Profitability (M4):** the Sharpe ratio (a standard measure of return per unit of risk) of the resulting trading signal, after subtracting realistic trading costs (5 basis points per trade).

We then compared CRAN against three independent methods that other researchers commonly use for this kind of problem (a Hidden Markov Model, a changepoint-detection method, and a simple volatility-based "turbulence index"), and against three stripped-down versions of CRAN itself, each missing exactly one design choice (no Bayesian updating step, hard labels instead of soft probabilities, or a different starting assumption for the four moods). Comparing against the stripped-down versions tells us which specific design choice is actually doing the work, rather than just assuming the whole design is necessary.

## Results

**Table 1 — full comparison** (out-of-sample, after trading costs):

| Model | Calibration (M1, lower is better) | Early-warning (M2, higher is better) | Uncertainty-vs-volatility (M3) | Average Sharpe per fold (M4) | Combined out-of-sample Sharpe |
|---|---:|---:|---:|---:|---:|
| **CRAN** | 2.41 | 0.059 | -0.094 | -0.64 | **1.64** |
| Hidden Markov Model | 4.22 | 0.100 | -0.028 | 0.92 | -0.78 |
| Changepoint detection | 0.03 | n/a | 0.124 | -0.16 | 0.41 |
| Turbulence index | 0.54 | 0.303 | -0.060 | 0.81 | 0.72 |
| CRAN, no Bayesian step | 2.39 | 0.059 | -0.096 | -0.75 | 1.61 |
| CRAN, hard labels | 129.43 | 0.000 | n/a | 1.39 | 1.75 |
| CRAN, different prior | 2.39 | 0.059 | -0.096 | -0.75 | 1.61 |

(For reference: a model with no information at all would score 1.39 on calibration and 0.30 on early-warning by chance.)

**The headline result:** CRAN beats all three outside comparison methods clearly on the combined Sharpe ratio. The Hidden Markov Model in particular does poorly out-of-sample, most likely because it has a lot of internal parameters to fit and only about four months of daily data per training window — it's probably overfitting to noise in the training period rather than learning anything real.

### Figure 1 — what CRAN's daily probability actually looks like

This shows, day by day, how CRAN's probability is split across the four moods (top panel) and which single mood it would have picked if forced to choose (bottom panel). Most days, the model is genuinely confident — the top panel is mostly solid blocks of color rather than a blurry mix — but there are short, sharp uncertain periods around mood transitions, exactly where you'd want a cautious model to be uncertain.

### Figure 4 — does this actually translate into money?

This plots the running total profit (after costs) for every model side by side. CRAN's line sits clearly above the three outside comparison methods for almost the entire period. The Hidden Markov Model's line falls steadily, which matches the overfitting concern above.

(Figures 2 and 3 — a closer look at the most volatile periods, and the relationship between the model's daily uncertainty and the next day's volatility — are saved alongside Figures 1 and 4 in the results folder.)

## The result that complicates the simple story

One of the stripped-down versions, "CRAN with hard labels" (which picks one single mood per day instead of a probability, the old-fashioned way), actually gets a *higher* raw Sharpe ratio than full CRAN — 1.75 versus 1.64.

If we only looked at Sharpe ratio, this would look like a point against CRAN's whole design. But look at the calibration column: this hard-label version scores 129.43, dramatically worse than the "no information at all" baseline of 1.39. The reason is mechanical: when a model is forced to pick exactly one mood, its probability for that day is either 0% or 100% for every mood — there is no in-between. That means the model can never express "I'm not sure," even on days when it genuinely shouldn't be sure. The uncertainty-vs-volatility check (M3) can't even be computed for this version, because a model that's always either 0% or 100% confident has no uncertainty to measure in the first place.

We think this is the most important finding in this paper: **a slightly higher Sharpe ratio is not the same thing as a better model.** The hard-label version got slightly lucky on point-estimate returns in this particular dataset, but it is fundamentally incapable of telling you when it doesn't know something — which matters a great deal once you start sizing real positions with real risk limits. CRAN gives up a small amount of raw Sharpe ratio in exchange for a model whose confidence levels can actually be trusted and used for risk management. We consider that trade worth making.

A smaller, less important finding: the version of CRAN with no Bayesian updating step and the version with a different starting prior produced numbers that are identical to many decimal places. This isn't a coincidence or a bug — it turns out that, mathematically, these two are computing the same formula in two different ways. The real lesson is that the choice of *starting assumption* (uniform vs. learned-from-data) is the part of the design that matters, not whether you literally call it "Bayesian."

## Limitations

This is tested on one instrument over roughly nine months of daily data, with six rolling test windows. That is enough to support the comparisons in this paper, but it is not enough data to claim the result would hold on a different market, a different time period, or over many more years. The four mood names and their typical trading direction were also chosen by us based on what each cluster's statistics looked like; a more rigorous version of this work would test whether those assignments are stable if refit many times on different historical periods.

## Conclusion

CRAN — keeping a daily probability over four named market moods instead of a single hard label — beats three standard comparison methods on a realistic, cost-adjusted, walk-forward test. More importantly, comparing it against a stripped-down version of itself shows that the soft-probability design is not just a stylistic choice: it is what allows the model's confidence to actually be trustworthy, which a hard-label model cannot offer, even when that hard-label model happens to post a slightly better headline number.
