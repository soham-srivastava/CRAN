"""
CRAN Evaluation Metrics — M1 through M4
=========================================
All four metrics are defined here before any model results are computed.
This is non-negotiable: metrics written after seeing results are p-hacked.

The same four functions are applied identically to:
    - CRAN (full model)
    - Hamilton HMM baseline
    - BOCPD baseline
    - Turbulence Index baseline
    - CRAN ablations (3 variants)

Usage
-----
    from evaluation.metrics import compute_all_metrics

    scores = compute_all_metrics(
        posteriors=posteriors,   # (T, K) float — posterior over K regimes
        returns=returns,         # (T,)   float — log-returns aligned with posteriors
        direction_map=dir_map,   # dict   int → float — regime index → {-1, 0, +0.5, +1}
        ann_factor=365,          # int    252 for daily equity, 365 for 24/7
        tc_bps=5,                # float  one-way transaction cost in basis points
    )

Metric reference
----------------
M1 : KL calibration — log-loss of posterior_t vs next-day actual regime
     Lower = better calibrated. Uniform baseline = log(K) ≈ 1.386 nats.
     Interpretation: how many nats of surprise does the model experience
     when it sees which regime actually materialises tomorrow?

M2 : Regime transition recall — fraction of transitions that were anticipated
     Anticipated = new regime had > ANTICIPATION_THRESHOLD probability in
     yesterday's posterior.
     Higher = better. Random baseline ≈ 1/K = 0.25.

M3 : Entropy-volatility calibration — Spearman(H(z_t), |ret_{t+1}|)
     Positive and significant → uncertain days are followed by more volatile days.
     Expected range: 0.05–0.30 on real data.

M4 : Sharpe ratio of downstream alpha — net of tc_bps one-way cost
     Signal: regime_direction × conviction_score
     Annualised by sqrt(ann_factor). Walk-forward discipline is the caller's job.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ANTICIPATION_THRESHOLD: float = 0.30   # M2: new regime must have had > this prob
LOG_FLOOR: float = 1e-300              # numerical safety for log operations
UNIFORM_LOG_LOSS: float = np.log(4)   # M1 baseline for K=4 uniform model


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class M1Result:
    """KL calibration (log-loss) result."""
    score: float          # mean -log p(k_{t+1}) — lower is better
    uniform_baseline: float = UNIFORM_LOG_LOSS
    improvement_vs_uniform: float = 0.0   # positive = better than uniform

    def __post_init__(self):
        self.improvement_vs_uniform = self.uniform_baseline - self.score

    def __str__(self):
        sign = "+" if self.improvement_vs_uniform >= 0 else ""
        return (f"M1={self.score:.4f} nats  "
                f"(uniform={self.uniform_baseline:.4f}, "
                f"Δ={sign}{self.improvement_vs_uniform:.4f})")


@dataclass
class M2Result:
    """Regime transition recall result."""
    recall: float          # fraction of transitions anticipated
    n_transitions: int     # total transitions detected
    n_anticipated: int     # transitions where new regime had > threshold prior mass
    transition_rate: float # fraction of days with a regime switch
    threshold: float = ANTICIPATION_THRESHOLD
    random_baseline: float = 0.0

    def __post_init__(self):
        self.random_baseline = self.threshold   # P(random anticipation) ≈ threshold

    def __str__(self):
        return (f"M2={self.recall:.4f}  "
                f"({self.n_anticipated}/{self.n_transitions} transitions anticipated, "
                f"rate={self.transition_rate:.3f}/day, "
                f"random≈{self.random_baseline:.2f})")


@dataclass
class M3Result:
    """Entropy–volatility calibration result."""
    spearman_r: float     # Spearman correlation
    p_value: float        # two-sided p-value
    significant: bool     # p < 0.05
    n: int                # number of (entropy, vol) pairs used

    def __str__(self):
        sig = "✅ sig." if self.significant else "❌ not sig."
        return (f"M3={self.spearman_r:.4f}  "
                f"(p={self.p_value:.4f}, n={self.n}, {sig})")


@dataclass
class M4Result:
    """Downstream alpha Sharpe result."""
    sharpe: float          # annualised Sharpe net of transaction costs
    hit_rate: float        # fraction of days with positive net PnL
    max_drawdown: float    # peak-to-trough drawdown in PnL units
    calmar: float          # |Sharpe| / |max_drawdown| (risk-adjusted)
    mean_pnl: float        # mean daily net PnL
    vol_pnl: float         # std of daily net PnL
    n_days: int
    tc_bps: float
    ann_factor: int

    def __str__(self):
        return (f"M4={self.sharpe:.3f} SR  "
                f"(hit={self.hit_rate:.2%}, MDD={self.max_drawdown:.4f}, "
                f"calmar={self.calmar:.3f})")


@dataclass
class AllMetrics:
    """Aggregated result from compute_all_metrics()."""
    m1: M1Result
    m2: M2Result
    m3: M3Result
    m4: M4Result
    model_name: str = "unnamed"

    def summary_row(self) -> dict:
        """Return a flat dict suitable for a results table row."""
        return {
            "model":            self.model_name,
            "M1_kl_loss":       round(self.m1.score, 4),
            "M1_vs_uniform":    round(self.m1.improvement_vs_uniform, 4),
            "M2_recall":        round(self.m2.recall, 4),
            "M2_n_trans":       self.m2.n_transitions,
            "M3_spearman":      round(self.m3.spearman_r, 4),
            "M3_pvalue":        round(self.m3.p_value, 4),
            "M4_sharpe":        round(self.m4.sharpe, 3),
            "M4_hit_rate":      round(self.m4.hit_rate, 4),
            "M4_max_drawdown":  round(self.m4.max_drawdown, 4),
            "M4_calmar":        round(self.m4.calmar, 3),
        }

    def __str__(self):
        return "\n".join([
            f"=== {self.model_name} ===",
            str(self.m1),
            str(self.m2),
            str(self.m3),
            str(self.m4),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# M1 — KL Calibration
# ─────────────────────────────────────────────────────────────────────────────

def m1_kl_calibration(
    posteriors: np.ndarray,
    next_labels: np.ndarray,
) -> M1Result:
    """
    M1: KL calibration — log-loss of posterior vs next-day actual regime.

    Computes the mean negative log-probability that today's posterior assigns
    to the regime that actually materialises tomorrow:

        M1 = mean_t [ -log( posterior_t[ argmax(z_{t+1}) ] ) ]

    This is the cross-entropy of the posterior w.r.t. the one-hot distribution
    of the next-day regime. Equivalent to KL(one_hot_{k_{t+1}} || posterior_t)
    up to a constant.

    Interpretation:
        - Perfect (posterior = one_hot on correct next regime): M1 = 0.0
        - Uniform model (K=4): M1 = log(4) ≈ 1.386 nats
        - M1 < 1.386 → beats uniform → model has predictive power
        - M1 > 1.386 → worse than random → model is miscalibrated

    Args:
        posteriors:  (T, K) float — posterior probability vectors, each row ∈ Δ^K
        next_labels: (T,)   int   — argmax(z_{t+1}) for t = 0..T-1
                                    next_labels[t] is the regime on day t+1
                                    (caller is responsible for alignment and
                                     walk-forward discipline)

    Returns:
        M1Result with score, uniform baseline, and improvement delta
    """
    posteriors  = np.asarray(posteriors,  dtype=float)
    next_labels = np.asarray(next_labels, dtype=int)

    if posteriors.shape[0] != next_labels.shape[0]:
        raise ValueError(
            f"posteriors has {posteriors.shape[0]} rows but "
            f"next_labels has {next_labels.shape[0]} elements. "
            "Ensure alignment: next_labels[t] = regime of day t+1."
        )

    T = len(next_labels)
    # Extract p(k_{t+1}) from each posterior row
    probs = posteriors[np.arange(T), next_labels]
    probs = np.clip(probs, LOG_FLOOR, 1.0)

    score = float(-np.mean(np.log(probs)))
    return M1Result(score=score)


# ─────────────────────────────────────────────────────────────────────────────
# M2 — Regime Transition Recall
# ─────────────────────────────────────────────────────────────────────────────

def m2_transition_recall(
    posteriors: np.ndarray,
    threshold: float = ANTICIPATION_THRESHOLD,
) -> M2Result:
    """
    M2: Fraction of regime transitions that were anticipated by the prior-day posterior.

    A regime transition occurs at time t when:
        argmax(posterior_t) ≠ argmax(posterior_{t-1})

    The transition is "anticipated" if the posterior at t-1 already assigned
    more than `threshold` probability mass to the new regime:
        posterior_{t-1}[ argmax(posterior_t) ] > threshold

    Intuition: a model with continuous affinities can "pre-position" before
    a hard regime switch by gradually increasing mass on the incoming regime.
    This metric rewards that behaviour and penalises abrupt, uninformed switches.

    Random baseline: if posteriors were independently drawn from Dirichlet(1),
    the probability of any component exceeding threshold ≈ threshold (rough).
    So random baseline recall ≈ ANTICIPATION_THRESHOLD = 0.30.

    Args:
        posteriors: (T, K) float — posterior sequence (walk-forward ordered)
        threshold:  float         — minimum prior mass required to call "anticipated"

    Returns:
        M2Result with recall, transition counts, and transition rate
    """
    posteriors = np.asarray(posteriors, dtype=float)
    T = len(posteriors)

    hard_labels = np.argmax(posteriors, axis=1)          # (T,)
    transitions = np.where(np.diff(hard_labels) != 0)[0] # indices t where switch at t→t+1

    n_trans = len(transitions)
    if n_trans == 0:
        return M2Result(
            recall=float("nan"),
            n_transitions=0,
            n_anticipated=0,
            transition_rate=0.0,
        )

    # For each transition at t, check if posteriors[t][new_regime] > threshold
    anticipated = 0
    for t in transitions:
        new_regime  = hard_labels[t + 1]
        prior_prob  = posteriors[t, new_regime]
        if prior_prob > threshold:
            anticipated += 1

    recall = anticipated / n_trans
    return M2Result(
        recall=float(recall),
        n_transitions=int(n_trans),
        n_anticipated=int(anticipated),
        transition_rate=float(n_trans / (T - 1)),
        threshold=threshold,
    )


# ─────────────────────────────────────────────────────────────────────────────
# M3 — Entropy-Volatility Calibration
# ─────────────────────────────────────────────────────────────────────────────

def m3_entropy_vol_calibration(
    posteriors: np.ndarray,
    next_returns: np.ndarray,
) -> M3Result:
    """
    M3: Spearman correlation between today's posterior entropy and tomorrow's |return|.

    Hypothesis: a high-entropy posterior (diffuse, uncertain about regime)
    should precede a more volatile day. A model that correctly captures
    regime uncertainty should have a positive and significant Spearman correlation.

        M3 = Spearman( H(posterior_t), |ret_{t+1}| )

    where H(p) = -Σ_k p_k log(p_k) in nats.

    Expected range on real data: 0.05 to 0.30.
    Significance threshold: p < 0.05.

    Note: we use next-day |return| (not rolling vol) to keep the metric
    purely forward-looking and avoid any smoothing that could inflate the
    correlation artificially.

    Args:
        posteriors:   (T, K) float — posterior sequence
        next_returns: (T,)   float — |ret_{t+1}| aligned with posteriors
                                     next_returns[t] = |log-return on day t+1|
                                     (caller responsible for walk-forward alignment)

    Returns:
        M3Result with Spearman r, p-value, significance flag, and n
    """
    posteriors   = np.asarray(posteriors,   dtype=float)
    next_returns = np.asarray(next_returns, dtype=float)

    if len(posteriors) != len(next_returns):
        raise ValueError(
            f"posteriors ({len(posteriors)}) and next_returns ({len(next_returns)}) "
            "must have the same length."
        )

    # Compute entropy for each row
    p     = np.clip(posteriors, LOG_FLOOR, 1.0)
    H     = -np.sum(p * np.log(p), axis=1)     # (T,)
    abs_r = np.abs(next_returns)

    # Drop any NaN in either series
    mask = ~(np.isnan(H) | np.isnan(abs_r))
    H_v, r_v = H[mask], abs_r[mask]

    if len(H_v) < 5:
        return M3Result(spearman_r=float("nan"), p_value=float("nan"),
                        significant=False, n=len(H_v))

    rho, pval = stats.spearmanr(H_v, r_v)
    return M3Result(
        spearman_r=float(rho),
        p_value=float(pval),
        significant=bool(pval < 0.05),
        n=int(len(H_v)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# M4 — Sharpe Ratio of Downstream Alpha
# ─────────────────────────────────────────────────────────────────────────────

def _entropy_nats(p: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy in nats. Input shape (T, K), output shape (T,)."""
    p = np.clip(p, LOG_FLOOR, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _max_drawdown(cum_pnl: np.ndarray) -> float:
    """Peak-to-trough maximum drawdown of a cumulative PnL series."""
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown    = cum_pnl - running_max
    return float(drawdown.min())


def m4_sharpe(
    posteriors:   np.ndarray,
    next_returns: np.ndarray,
    direction_map: dict[int, float],
    ann_factor: int   = 252,
    tc_bps:     float = 5.0,
) -> M4Result:
    """
    M4: Annualised Sharpe ratio of the downstream alpha signal, net of transaction costs.

    Signal construction:
        regime_t      = argmax( posterior_t )
        direction_t   = direction_map[ regime_t ]
        conviction_t  = 1 - H(posterior_t) / log(K)     ∈ [0, 1]
        signal_t      = direction_t × conviction_t       ∈ [-1, +1]

    PnL construction (walk-forward, no lookahead):
        pnl_t         = signal_t   × ret_{t+1}
        tc_t          = tc_bps/10000 × |signal_t - signal_{t-1}|
        net_pnl_t     = pnl_t - tc_t

    Sharpe:
        Sharpe        = mean(net_pnl) / std(net_pnl) × sqrt(ann_factor)

    Walk-forward discipline: the caller must ensure that posteriors[t] was
    computed using a GMM that was fit only on data up to t. This function
    does NOT enforce walk-forward — it takes posteriors as-is and computes PnL.

    Args:
        posteriors:   (T, K) float — posterior sequence
        next_returns: (T,)   float — log-returns, ret[t] = log-return of day t+1
                                     Alignment: pnl_t = signal_t × next_returns[t]
        direction_map: dict  int → float — regime index → trading direction
                             Typical: {trending: +1.0, meanrev: -1.0,
                                       volatile: 0.0,  quiet: +0.5}
        ann_factor:   int    annualisation denominator
                             252 for daily equity (NSE), 365 for 24/7 (crypto)
        tc_bps:       float  one-way transaction cost in basis points

    Returns:
        M4Result with Sharpe, hit rate, max drawdown, Calmar, and diagnostics
    """
    posteriors   = np.asarray(posteriors,   dtype=float)
    next_returns = np.asarray(next_returns, dtype=float)
    T, K         = posteriors.shape

    if len(next_returns) != T:
        raise ValueError(
            f"posteriors ({T} rows) and next_returns ({len(next_returns)}) "
            "must align. next_returns[t] is the return AFTER day t."
        )

    log_k = np.log(K)

    # Signal
    regimes     = np.argmax(posteriors, axis=1)                    # (T,)
    directions  = np.array([direction_map.get(int(r), 0.0)
                             for r in regimes], dtype=float)       # (T,)
    H           = _entropy_nats(posteriors)                        # (T,)
    conviction  = 1.0 - H / log_k                                  # (T,) ∈ [0,1]
    signals     = directions * conviction                          # (T,)

    # PnL (t indexes the signal day; return is on day t+1)
    pnl         = signals * next_returns                           # (T,)
    tc          = (tc_bps / 10_000) * np.abs(np.diff(signals,
                   prepend=signals[0]))                            # (T,)
    net_pnl     = pnl - tc                                         # (T,)

    # Remove NaN
    mask        = ~np.isnan(net_pnl)
    net_pnl     = net_pnl[mask]
    n           = len(net_pnl)

    if n < 5:
        return M4Result(sharpe=float("nan"), hit_rate=float("nan"),
                        max_drawdown=float("nan"), calmar=float("nan"),
                        mean_pnl=float("nan"), vol_pnl=float("nan"),
                        n_days=n, tc_bps=tc_bps, ann_factor=ann_factor)

    mean_p   = float(net_pnl.mean())
    std_p    = float(net_pnl.std(ddof=1))
    sharpe   = mean_p / (std_p + 1e-12) * np.sqrt(ann_factor)
    hit_rate = float((net_pnl > 0).mean())

    cum_pnl  = np.cumsum(net_pnl)
    mdd      = _max_drawdown(cum_pnl)
    calmar   = abs(sharpe) / (abs(mdd) + 1e-12)

    return M4Result(
        sharpe=float(sharpe),
        hit_rate=hit_rate,
        max_drawdown=mdd,
        calmar=float(calmar),
        mean_pnl=mean_p,
        vol_pnl=std_p,
        n_days=n,
        tc_bps=tc_bps,
        ann_factor=ann_factor,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Master function — compute all 4 metrics at once
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    posteriors:   np.ndarray,
    returns:      np.ndarray,
    direction_map: dict[int, float],
    ann_factor:   int   = 252,
    tc_bps:       float = 5.0,
    model_name:   str   = "unnamed",
) -> AllMetrics:
    """
    Compute M1–M4 for a given posterior sequence and return series.

    This is the single entry point used by all models and baselines.
    Pass the same arguments to every model to ensure fair comparison.

    Args:
        posteriors:   (T, K) float — posterior sequence over K regimes
        returns:      (T,)   float — log-returns. returns[t] is the return
                                     *entering* day t (i.e., the return that
                                     happens after the signal on day t-1).
                                     For metrics, we shift internally:
                                       - M1 uses labels of day t+1
                                       - M3 uses |ret_{t+1}|
                                       - M4 uses signal_t × ret_{t+1}
                                     So alignment is: posteriors[t] and returns[t]
                                     are on the same day; returns[t+1] is the
                                     next-day return.
        direction_map: dict  int → float — regime trading direction
        ann_factor:   int    252 (equity) or 365 (24/7)
        tc_bps:       float  one-way transaction cost in basis points
        model_name:   str    label for the results table

    Returns:
        AllMetrics container with M1, M2, M3, M4 results
    """
    posteriors = np.asarray(posteriors, dtype=float)
    returns    = np.asarray(returns,    dtype=float)
    T          = len(posteriors)

    # ── Alignment: align posteriors[0..T-2] with returns[1..T-1] ─────────────
    # posterior at t predicts regime at t+1, and we trade from t to t+1
    post_t  = posteriors[:-1]   # (T-1, K) — signal day
    ret_tp1 = returns[1:]       # (T-1,)   — next-day return (what we trade into)

    # Hard labels for next day (used in M1 and M2)
    hard_labels     = np.argmax(posteriors, axis=1)      # (T,)
    next_day_labels = hard_labels[1:]                    # (T-1,) labels at t+1

    # M1: log-loss on next-day regime
    m1 = m1_kl_calibration(post_t, next_day_labels)

    # M2: transition recall (uses full posterior sequence for transition detection)
    m2 = m2_transition_recall(posteriors)

    # M3: entropy–vol calibration
    m3 = m3_entropy_vol_calibration(post_t, ret_tp1)

    # M4: downstream Sharpe
    m4 = m4_sharpe(post_t, ret_tp1, direction_map,
                   ann_factor=ann_factor, tc_bps=tc_bps)

    return AllMetrics(m1=m1, m2=m2, m3=m3, m4=m4, model_name=model_name)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity bounds — used in verify_metrics.py
# ─────────────────────────────────────────────────────────────────────────────

SANITY_BOUNDS = {
    "M1_score":        (0.0,  UNIFORM_LOG_LOSS * 2,  "should be finite, ideally < log(4)"),
    "M2_recall":       (0.0,  1.0,                   "recall is a proportion ∈ [0, 1]"),
    "M3_spearman_r":   (-1.0, 1.0,                   "Spearman r ∈ [-1, 1]"),
    "M4_sharpe":       (-20., 20.,                    "Sharpe should be finite"),
    "M4_hit_rate":     (0.0,  1.0,                    "hit rate is a proportion"),
}


def check_sanity(result: AllMetrics) -> list[str]:
    """
    Return a list of failed sanity checks for an AllMetrics result.
    An empty list means all checks passed.
    """
    failures = []
    checks = [
        ("M1_score",      result.m1.score),
        ("M2_recall",     result.m2.recall if not np.isnan(result.m2.recall) else 0.5),
        ("M3_spearman_r", result.m3.spearman_r if not np.isnan(result.m3.spearman_r) else 0.0),
        ("M4_sharpe",     result.m4.sharpe if not np.isnan(result.m4.sharpe) else 0.0),
        ("M4_hit_rate",   result.m4.hit_rate if not np.isnan(result.m4.hit_rate) else 0.5),
    ]
    for name, value in checks:
        lo, hi, desc = SANITY_BOUNDS[name]
        if not (lo <= value <= hi):
            failures.append(f"{name}={value:.4f} outside [{lo}, {hi}] — {desc}")
    return failures
