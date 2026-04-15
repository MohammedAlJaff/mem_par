"""
Visualization utilities for all paper figures.

This module produces the main figures for the paper. Each function
generates one figure type that appears in the results section.

Color scheme (consistent across all figures):
    x-prediction:  blue    (#1f77b4)
    ε-prediction:  red     (#d62728)
    v-prediction:  orange  (#ff7f0e)

    This color assignment is non-arbitrary: x is the "clean" target
    (blue = cool/stable), ε is noise (red = hot/noisy), v is
    intermediate (orange = between).

Style:
    - matplotlib with seaborn styling
    - Font size appropriate for NeurIPS single-column
    - All figures saved as PDF (vector) for the paper and PNG for quick inspection
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for cluster/headless
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict


# =============================================================================
# Global style
# =============================================================================

PRED_COLORS = {
    "x": "#1f77b4",  # blue
    "eps": "#d62728",  # red
    "v": "#ff7f0e",  # orange
}

PRED_LABELS = {
    "x": r"$\mathbf{x}$-prediction",
    "eps": r"$\boldsymbol{\varepsilon}$-prediction",
    "v": r"$v$-prediction",
}

PRED_MARKERS = {
    "x": "o",
    "eps": "s",
    "v": "^",
}


def setup_style():
    """Set consistent plot style for all figures."""
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


# =============================================================================
# Figure 1: f_mem vs n (phase boundary curves)
# =============================================================================


def plot_f_mem_vs_n(
    results_df: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None,
    show_sigmoid_fit: bool = True,
):
    """
    Plot memorization fraction vs dataset size, one curve per parameterization.

    This is the MAIN RESULT FIGURE. Shows the phase boundary shifting
    between parameterizations (or not, for the null control).

    Args:
        results_df: DataFrame with columns:
            n, pred_space, f_mem_mean, f_mem_std
            (aggregated over seeds — use aggregate_results.py to produce this)
        output_path: where to save (e.g., 'figures/f_mem_vs_n_d4.pdf')
        title: optional figure title
        show_sigmoid_fit: overlay fitted sigmoid curves
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for pred in ["x", "eps", "v"]:
        mask = results_df["pred_space"] == pred
        df = results_df[mask].sort_values("n")

        if df.empty:
            continue

        ax.errorbar(
            df["n"],
            df["f_mem_mean"],
            yerr=df["f_mem_std"],
            color=PRED_COLORS[pred],
            marker=PRED_MARKERS[pred],
            markersize=5,
            label=PRED_LABELS[pred],
            capsize=3,
            linewidth=1.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Dataset size $n$")
    ax.set_ylabel(r"Memorization fraction $f_{\mathrm{mem}}$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, fancybox=False, edgecolor="gray")

    if title:
        ax.set_title(title)

    # Reference lines
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    _save_figure(fig, output_path)


# =============================================================================
# Figure 2: n* vs resolution (the money figure)
# =============================================================================


def plot_n_star_vs_resolution(
    boundary_results: Dict[int, Dict[str, dict]],
    output_path: str,
    title: Optional[str] = None,
):
    """
    Plot critical dataset size n* as a function of image resolution.

    This is the UNIQUE CONTRIBUTION figure. Shows n* flat for x-pred
    but increasing steeply for ε/v-pred as D grows.

    Args:
        boundary_results: nested dict:
            {resolution: {pred_space: {n_star, n_star_ci}}}
            e.g., {64: {"x": {"n_star": 500, "n_star_ci": (300, 700)}, ...}, ...}
        output_path: save path
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))

    resolutions = sorted(boundary_results.keys())

    for pred in ["x", "eps", "v"]:
        n_stars = []
        ci_low = []
        ci_high = []

        for res in resolutions:
            if pred in boundary_results[res]:
                ns = boundary_results[res][pred]["n_star"]
                ci = boundary_results[res][pred].get("n_star_ci", (ns, ns))
                n_stars.append(ns)
                ci_low.append(ns - ci[0])
                ci_high.append(ci[1] - ns)
            else:
                n_stars.append(np.nan)
                ci_low.append(0)
                ci_high.append(0)

        n_stars = np.array(n_stars)
        valid = np.isfinite(n_stars)

        if valid.any():
            res_valid = np.array(resolutions)[valid]
            ns_valid = n_stars[valid]

            ax.errorbar(
                res_valid,
                ns_valid,
                yerr=[np.array(ci_low)[valid], np.array(ci_high)[valid]],
                color=PRED_COLORS[pred],
                marker=PRED_MARKERS[pred],
                markersize=7,
                label=PRED_LABELS[pred],
                capsize=4,
                linewidth=2,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Image resolution")
    ax.set_ylabel(r"$n^*_{\mathrm{mem}}$ (critical dataset size)")
    ax.set_xticks(resolutions)
    ax.set_xticklabels([str(r) for r in resolutions])
    ax.legend(frameon=True, fancybox=False, edgecolor="gray")

    if title:
        ax.set_title(title)

    _save_figure(fig, output_path)


# =============================================================================
# Figure 3: τ_mem vs n (memorization timescale scaling)
# =============================================================================


def plot_tau_mem_vs_n(
    results_df: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None,
):
    """
    Plot memorization timescale vs dataset size (log-log).

    Bonnaire predicts τ_mem ∝ n (slope = 1 in log-log).
    We test whether the slope or intercept changes with parameterization.

    Args:
        results_df: DataFrame with columns:
            n, pred_space, tau_mem_mean, tau_mem_std
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for pred in ["x", "eps", "v"]:
        mask = results_df["pred_space"] == pred
        df = results_df[mask].sort_values("n")
        df = df[df["tau_mem_mean"].notna() & (df["tau_mem_mean"] > 0)]

        if df.empty:
            continue

        ax.errorbar(
            df["n"],
            df["tau_mem_mean"],
            yerr=df["tau_mem_std"],
            color=PRED_COLORS[pred],
            marker=PRED_MARKERS[pred],
            markersize=5,
            label=PRED_LABELS[pred],
            capsize=3,
            linewidth=1.5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Dataset size $n$")
    ax.set_ylabel(r"$\tau_{\mathrm{mem}}$ (memorization onset)")
    ax.legend(frameon=True, fancybox=False, edgecolor="gray")

    # Reference line: slope 1 (Bonnaire prediction)
    n_range = np.array([results_df["n"].min(), results_df["n"].max()])
    ax.plot(
        n_range,
        n_range * 0.1,
        "--",
        color="gray",
        linewidth=0.8,
        alpha=0.5,
        label=r"$\propto n$ (Bonnaire)",
    )

    if title:
        ax.set_title(title)

    _save_figure(fig, output_path)


# =============================================================================
# Figure 4: n* vs ψ = D/d (synthetic, controlled geometry)
# =============================================================================


def plot_n_star_vs_psi(
    boundary_results: Dict[float, Dict[str, dict]],
    output_path: str,
    title: Optional[str] = None,
):
    """
    Plot n* vs dimension ratio ψ = D/d (synthetic experiments).

    This is the theoretical prediction figure. The manifold-gap hypothesis
    predicts: n* is constant for x-pred (depends only on d) but increases
    with ψ for ε-pred (depends on D = ψ·d).

    Args:
        boundary_results: {psi: {pred_space: {n_star, n_star_ci}}}
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))

    psi_values = sorted(boundary_results.keys())

    for pred in ["x", "eps", "v"]:
        n_stars = []
        for psi in psi_values:
            if pred in boundary_results[psi]:
                n_stars.append(boundary_results[psi][pred]["n_star"])
            else:
                n_stars.append(np.nan)

        n_stars = np.array(n_stars)
        valid = np.isfinite(n_stars)

        if valid.any():
            ax.plot(
                np.array(psi_values)[valid],
                n_stars[valid],
                color=PRED_COLORS[pred],
                marker=PRED_MARKERS[pred],
                markersize=7,
                label=PRED_LABELS[pred],
                linewidth=2,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Dimension ratio $\psi = D/d$")
    ax.set_ylabel(r"$n^*_{\mathrm{mem}}$")
    ax.legend(frameon=True, fancybox=False, edgecolor="gray")

    if title:
        ax.set_title(title)

    _save_figure(fig, output_path)


# =============================================================================
# Scatter plots (synthetic data, for sanity checks and appendix)
# =============================================================================


def plot_scatter_2d(
    training_data: np.ndarray,
    generated_data: np.ndarray,
    output_path: str,
    title: Optional[str] = None,
):
    """
    Scatter plot of training vs generated data in 2D.

    For high-dimensional data, pass PCA-projected coordinates.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(
        training_data[:, 0],
        training_data[:, 1],
        c="black",
        s=10,
        alpha=0.5,
        label="Training",
        zorder=2,
    )
    ax.scatter(
        generated_data[:, 0],
        generated_data[:, 1],
        c=PRED_COLORS["x"],
        s=5,
        alpha=0.3,
        label="Generated",
        zorder=1,
    )

    ax.set_aspect("equal")
    ax.legend(markerscale=2)
    if title:
        ax.set_title(title)

    _save_figure(fig, output_path)


# =============================================================================
# NN ratio histogram (diagnostic)
# =============================================================================


def plot_nn_ratio_histogram(
    r_generated: np.ndarray,
    r_holdout: Optional[np.ndarray] = None,
    output_path: str = "nn_ratio_hist.pdf",
    threshold: float = 0.5,
    title: Optional[str] = None,
):
    """
    Histogram of NN ratios for generated samples vs holdout baseline.

    The key comparison: if the generated histogram matches the holdout
    histogram, the model is generalizing. If the generated histogram
    has excess mass near 0, it's memorizing.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 3))

    bins = np.linspace(0, 1.5, 50)
    ax.hist(
        r_generated,
        bins=bins,
        density=True,
        alpha=0.6,
        color=PRED_COLORS["x"],
        label="Generated",
    )

    if r_holdout is not None:
        ax.hist(
            r_holdout,
            bins=bins,
            density=True,
            alpha=0.4,
            color="gray",
            label="Holdout (baseline)",
        )

    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Threshold ({threshold})",
    )

    ax.set_xlabel(r"NN ratio $r = d_1 / d_2$")
    ax.set_ylabel("Density")
    ax.legend()
    if title:
        ax.set_title(title)

    _save_figure(fig, output_path)


# =============================================================================
# Loss curve
# =============================================================================


def plot_loss_curve(
    loss_history: list,
    output_path: str,
    title: Optional[str] = None,
):
    """Plot training loss over steps/epochs."""
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 3))

    steps = [entry["step"] for entry in loss_history]
    losses = [entry["loss"] for entry in loss_history]

    # Smooth with running average for readability
    window = max(1, len(losses) // 100)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(steps[: len(smoothed)], smoothed, linewidth=1)
    else:
        ax.plot(steps, losses, linewidth=0.5, alpha=0.7)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    if title:
        ax.set_title(title)

    _save_figure(fig, output_path)


# =============================================================================
# Helpers
# =============================================================================


def _save_figure(fig, path: str):
    """Save as both PDF (paper) and PNG (quick look)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path)
    # Also save PNG version
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
