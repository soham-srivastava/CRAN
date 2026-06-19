"""
CRAN Paper Figures — Scaffold
==============================
Four figures for the CRAN paper results section.
Functions are fully documented and structured; data is wired in after Phase 5-6
when walk-forward results exist.

Figure 1: Affinity timeline — z_t stacked area over the full test period
Figure 2: Event-period zoom — posterior on crash/rally windows
Figure 3: Entropy vs realised vol scatter (M3 calibration visualised)
Figure 4: Cumulative alpha — CRAN vs all baselines and ablations

All figures output flat into cran/ as 300dpi PNG.
Colour scheme follows the CRAN regime palette.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

# ── Colour palette (one colour per regime, colour-blind safe) ──────────────
REGIME_COLOURS = {
    "Trending": "#2196F3",   # blue
    "MeanRev":  "#FF9800",   # orange
    "Volatile": "#F44336",   # red
    "Quiet":    "#4CAF50",   # green
}

FIGURE_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_fig_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Affinity Timeline
# ─────────────────────────────────────────────────────────────────────────────

def figure1_affinity_timeline(
    dates:       pd.DatetimeIndex,
    posteriors:  np.ndarray,
    name_map:    dict[int, str],
    instrument:  str = "Instrument",
    out_prefix:  str = "fig1_affinity",
) -> str:
    """
    Figure 1: Stacked area chart of z_t over the full test period.

    Each band represents the posterior probability mass on one regime.
    The bands sum to 1 at every point (probability simplex constraint).
    Visually reveals:
        - Persistent regimes (wide flat bands)
        - Transitions (band boundary crossings)
        - Ambiguous periods (no single regime dominates → mixed colours)

    Args:
        dates:      DatetimeIndex of length T (test set dates)
        posteriors: (T, K) posterior sequence
        name_map:   {component_index: regime_name}
        instrument: label for the plot title
        out_prefix: filename prefix for PNG/PDF output

    Returns:
        Path to saved PNG file
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — skipping Figure 1")
        return ""

    _ensure_fig_dir()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                              gridspec_kw={"height_ratios": [3, 1]})

    ax_affinity, ax_price = axes

    # Reorder regimes: Trending, MeanRev, Volatile, Quiet
    ordered = ["Trending", "MeanRev", "Volatile", "Quiet"]
    k_order  = [k for name in ordered for k, n in name_map.items() if n == name]
    cols     = [REGIME_COLOURS[name_map[k]] for k in k_order]
    labels_o = [name_map[k] for k in k_order]

    Z = posteriors[:, k_order]
    ax_affinity.stackplot(dates, Z.T, colors=cols, labels=labels_o, alpha=0.85)
    ax_affinity.set_ylim(0, 1)
    ax_affinity.set_ylabel("Posterior probability")
    ax_affinity.set_title(f"CRAN Affinity Vectors — {instrument}")
    ax_affinity.legend(loc="upper left", ncol=4, fontsize=9)
    ax_affinity.tick_params(labelbottom=False)

    # Entropy overlay (secondary y-axis)
    p     = np.clip(posteriors, 1e-300, 1.0)
    H     = -np.sum(p * np.log(p), axis=1)
    ax2   = ax_affinity.twinx()
    ax2.plot(dates, H, color="black", linewidth=0.6, alpha=0.4, label="Entropy H(z_t)")
    ax2.set_ylabel("H(z_t) nats", fontsize=8)
    ax2.set_ylim(0, np.log(4) * 1.1)
    ax2.axhline(np.log(4), color="black", linestyle="--", linewidth=0.5, alpha=0.3)

    # Hard label raster on bottom panel
    hard = np.argmax(posteriors, axis=1)
    for i, k in enumerate(k_order):
        mask = hard == k
        ax_price.fill_between(dates, 0, 1,
                              where=mask, color=cols[i], alpha=0.7, linewidth=0)
    ax_price.set_ylabel("Hard label")
    ax_price.set_yticks([])
    ax_price.set_xlabel("Date")

    plt.tight_layout()
    out_path = os.path.join(FIGURE_DIR, f"{out_prefix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 1 saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Event-Period Zoom
# ─────────────────────────────────────────────────────────────────────────────

def figure2_event_zoom(
    dates:        pd.DatetimeIndex,
    posteriors:   np.ndarray,
    returns:      np.ndarray,
    name_map:     dict[int, str],
    event_windows: list[tuple[str, str, str]] | None = None,
    out_prefix:   str = "fig2_event_zoom",
) -> str:
    """
    Figure 2: Posterior evolution during key market events.

    Shows how the posterior z_t responds during stress periods — the model
    should rapidly concentrate mass on the Volatile regime.

    Args:
        dates:         DatetimeIndex of length T
        posteriors:    (T, K) posterior sequence
        returns:       (T,)   log-return series
        name_map:      {component_index: regime_name}
        event_windows: list of (label, start_date, end_date) strings
                       If None, picks the 3 highest-volatility 30-day windows.
        out_prefix:    filename prefix

    Returns:
        Path to saved PNG file
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping Figure 2")
        return ""

    _ensure_fig_dir()

    # Auto-detect high-vol windows if none provided
    if event_windows is None:
        roll_vol = pd.Series(np.abs(returns), index=dates).rolling(30).std()
        top3_ends = roll_vol.nlargest(3).index
        event_windows = [
            (f"High-vol window {i+1}",
             str((end - pd.Timedelta(days=60)).date()),
             str(end.date()))
            for i, end in enumerate(top3_ends)
        ]

    n_events = len(event_windows)
    fig, axes = plt.subplots(n_events, 2, figsize=(14, 4 * n_events))
    if n_events == 1:
        axes = [axes]

    ordered = ["Trending", "MeanRev", "Volatile", "Quiet"]
    k_order = [k for name in ordered for k, n in name_map.items() if n == name]
    cols    = [REGIME_COLOURS[name_map[k]] for k in k_order]

    for idx, (label, start, end) in enumerate(event_windows):
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        d_ev = dates[mask]
        p_ev = posteriors[mask]
        r_ev = returns[mask]
        Z_ev = p_ev[:, k_order]

        ax_post, ax_ret = axes[idx]

        ax_post.stackplot(d_ev, Z_ev.T, colors=cols, alpha=0.85,
                          labels=[name_map[k] for k in k_order])
        ax_post.set_ylim(0, 1)
        ax_post.set_title(f"{label}  [{start} → {end}]")
        ax_post.set_ylabel("Posterior")
        if idx == 0:
            ax_post.legend(loc="upper left", ncol=4, fontsize=8)

        cum_ret = np.cumsum(r_ev)
        ax_ret.fill_between(d_ev, cum_ret, 0,
                            where=cum_ret >= 0, color="#4CAF50", alpha=0.5)
        ax_ret.fill_between(d_ev, cum_ret, 0,
                            where=cum_ret < 0,  color="#F44336", alpha=0.5)
        ax_ret.axhline(0, color="black", linewidth=0.5)
        ax_ret.set_ylabel("Cumulative log-return")

    plt.tight_layout()
    out_path = os.path.join(FIGURE_DIR, f"{out_prefix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 2 saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Entropy vs Realised Vol Scatter (M3 Visualised)
# ─────────────────────────────────────────────────────────────────────────────

def figure3_entropy_vol_scatter(
    posteriors:   np.ndarray,
    next_returns: np.ndarray,
    m3_spearman:  float,
    m3_pvalue:    float,
    out_prefix:   str = "fig3_entropy_vol",
) -> str:
    """
    Figure 3: Scatter plot of H(z_t) vs |ret_{t+1}|, with Spearman r overlay.

    This is the visual companion to the M3 metric. A positive upward trend
    in the scatter confirms that the model's entropy correctly signals
    upcoming volatility.

    Also plots a quantile-mean line (binned by entropy decile) to show the
    trend more clearly over the noisy individual-day scatter.

    Args:
        posteriors:   (T, K) posterior sequence
        next_returns: (T,)   |log-return| of next day
        m3_spearman:  float  pre-computed Spearman r (from M3 result)
        m3_pvalue:    float  pre-computed p-value
        out_prefix:   filename prefix

    Returns:
        Path to saved PNG
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping Figure 3")
        return ""

    _ensure_fig_dir()

    p   = np.clip(posteriors, 1e-300, 1.0)
    H   = -np.sum(p * np.log(p), axis=1)
    r   = np.abs(next_returns)
    sig = "p<0.001" if m3_pvalue < 0.001 else f"p={m3_pvalue:.3f}"

    # Bin by entropy decile and compute mean |ret|
    deciles   = np.percentile(H, np.linspace(0, 100, 11))
    bin_cents = [(deciles[i] + deciles[i+1]) / 2 for i in range(10)]
    bin_means = [r[(H >= deciles[i]) & (H < deciles[i+1])].mean()
                 for i in range(10)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(H, r, alpha=0.2, s=8, color="#2196F3", label="Individual days")
    ax.plot(bin_cents, bin_means, "o-", color="#F44336", linewidth=2,
            markersize=6, label="Decile-mean trend")
    ax.set_xlabel("H(z_t)  — posterior entropy (nats)")
    ax.set_ylabel("|ret_{t+1}|  — next-day absolute return")
    ax.set_title(
        f"M3: Entropy–Volatility Calibration\n"
        f"Spearman r = {m3_spearman:.4f}  ({sig})"
    )
    ax.axvline(np.log(4), color="gray", linestyle="--", linewidth=0.8,
               label="H(uniform) = log(4)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURE_DIR, f"{out_prefix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 3 saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Cumulative Alpha: CRAN vs Baselines
# ─────────────────────────────────────────────────────────────────────────────

def figure4_cumulative_alpha(
    dates:         pd.DatetimeIndex,
    pnl_series:    dict[str, np.ndarray],
    sharpes:       dict[str, float],
    out_prefix:    str = "fig4_cumulative_alpha",
) -> str:
    """
    Figure 4: Cumulative net PnL for CRAN vs all baselines and ablations.

    This is the key "show don't tell" figure — it visually demonstrates that
    CRAN's continuous affinity and Bayesian update translate to real alpha
    improvement over discrete-label competitors.

    Args:
        dates:      DatetimeIndex of length T (aligned with PnL series)
        pnl_series: dict of {model_name: net_pnl_array (T,)}
                    The dict MUST include at least "CRAN" and one baseline.
        sharpes:    dict of {model_name: annualised_sharpe} for legend labels
        out_prefix: filename prefix

    Returns:
        Path to saved PNG
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping Figure 4")
        return ""

    _ensure_fig_dir()

    # Style: CRAN is thick blue, baselines are thin muted, ablations dashed
    STYLES = {
        "CRAN":               dict(color="#2196F3", lw=2.5, ls="-",  zorder=5),
        "Hamilton HMM":       dict(color="#9E9E9E", lw=1.2, ls="-",  zorder=2),
        "BOCPD":              dict(color="#795548", lw=1.2, ls="-",  zorder=2),
        "Turbulence Index":   dict(color="#607D8B", lw=1.2, ls="-",  zorder=2),
        "CRAN (no Bayes)":    dict(color="#03A9F4", lw=1.2, ls="--", zorder=3),
        "CRAN (hard labels)": dict(color="#4FC3F7", lw=1.2, ls="--", zorder=3),
        "CRAN (uniform prior)":dict(color="#81D4FA",lw=1.2, ls="--", zorder=3),
    }
    DEFAULT_STYLE = dict(color="#BDBDBD", lw=1.0, ls="-.", zorder=1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for model, pnl in pnl_series.items():
        cum = np.cumsum(pnl)
        sr  = sharpes.get(model, float("nan"))
        label = f"{model}  (SR={sr:.2f})"
        style = STYLES.get(model, DEFAULT_STYLE)
        ax.plot(dates[:len(cum)], cum, label=label, **style)

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net log-return (net of transaction costs)")
    ax.set_title("Figure 4: Cumulative Alpha — CRAN vs Baselines & Ablations")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(FIGURE_DIR, f"{out_prefix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 4 saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate all figures at once
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(
    dates:          pd.DatetimeIndex,
    posteriors:     np.ndarray,
    returns:        np.ndarray,
    name_map:       dict[int, str],
    m3_result,
    pnl_series:     dict[str, np.ndarray],
    sharpes:        dict[str, float],
    instrument:     str = "Instrument",
    event_windows:  list | None = None,
) -> dict[str, str]:
    """
    Generate all 4 paper figures and return a dict of {fig_name: file_path}.

    Call this after walk-forward results are available (Phase 5+).
    """
    print("Generating paper figures...")
    paths = {}
    paths["fig1"] = figure1_affinity_timeline(
        dates, posteriors, name_map, instrument)
    paths["fig2"] = figure2_event_zoom(
        dates, posteriors, returns, name_map, event_windows)
    paths["fig3"] = figure3_entropy_vol_scatter(
        posteriors[:-1], np.abs(returns[1:]),
        m3_result.spearman_r, m3_result.p_value)
    paths["fig4"] = figure4_cumulative_alpha(
        dates, pnl_series, sharpes)
    print("All figures generated.")
    return paths
