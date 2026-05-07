#!/usr/bin/env python
"""N-train tau-scale analysis from checkpoint FID and F_mem trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = Path("outputs/evals/mem_boundary_20260427_all_done_quality_20260501/all_checkpoint_metrics.csv")
DEFAULT_OUT_DIR = Path("prototyping/figures/n_train_tau_scales")
FIGURE_NAME = "N01_n_train_tau_gen_mem_delta_by_param"
OVERLAY_FIGURE_NAME = "N02_n_train_tau_gen_mem_delta_overlay_params"
GEN_THRESHOLD_SENSITIVITY_FIGURE_NAME = "N03_n_train_tau_gen_threshold_sensitivity_overlay_params"
GEN_THRESHOLD_SENSITIVITY_FMEM001_FIGURE_NAME = "N04_n_train_tau_gen_threshold_sensitivity_fmem_0p01_overlay_params"
BONNAIRE_OVERLAY_FIGURE_NAME = "N05_bonnaire_tau_argmin_fmem_onset_overlay_params"
BONNAIRE_SCALING_FIGURE_NAME = "N06_bonnaire_tau_scaling_guides_overlay_params"
FIGURE_EXTENSIONS = (".pdf", ".png")

FID_COL = "fid_10k"
FMEM_COL = "f_mem"
STEP_COL = "step"
N_TRAIN_LABEL = r"$N_{\mathrm{train}}$"

PARAM_ORDER = ["eps_eps", "v_v", "x_v", "x_x"]
PARAM_LABELS = {
    "eps_eps": r"$\epsilon/\epsilon$",
    "v_v": r"$v/v$",
    "x_v": r"$x/v$",
    "x_x": r"$x/x$",
}
PARAM_COLORS = {
    "eps_eps": "#1f77b4",
    "v_v": "#ff7f0e",
    "x_v": "#2ca02c",
    "x_x": "#d62728",
}
GEN_THRESHOLD_SENSITIVITY = (1.05, 1.10, 1.25)
FMEM_THRESHOLD_SENSITIVITY = 0.01


def resolve_under_repo(repo: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo / path


def sem(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def first_step_where(rows: pd.DataFrame, mask: pd.Series) -> float:
    hits = rows.loc[mask].sort_values(STEP_COL)
    if hits.empty:
        return float("nan")
    return float(hits[STEP_COL].iloc[0])


def compute_run_tau_scales(metrics: pd.DataFrame, fid_rel: float, fmem_threshold: float) -> pd.DataFrame:
    required = ["n_train", "seed", "param", STEP_COL, FID_COL, FMEM_COL]
    missing = [column for column in required if column not in metrics.columns]
    if missing:
        raise RuntimeError(f"Missing required checkpoint columns: {missing}")

    df = metrics.copy()
    for column in ["n_train", "seed", STEP_COL, FID_COL, FMEM_COL]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["n_train", "seed", "param", STEP_COL, FID_COL, FMEM_COL])
    if df.empty:
        raise RuntimeError("No finite checkpoint rows after numeric cleanup.")

    if "run_id" in df.columns:
        group_cols = ["run_id"]
    elif "manifest_index" in df.columns:
        group_cols = ["manifest_index", "n_train", "param", "seed"]
    else:
        group_cols = ["n_train", "param", "seed"]

    records: list[dict[str, object]] = []
    for keys, rows in df.sort_values(STEP_COL).groupby(group_cols, dropna=False):
        rows = rows.sort_values(STEP_COL).copy()
        meta = rows.iloc[0]
        best_fid = float(rows[FID_COL].min())
        if not np.isfinite(best_fid):
            continue

        tau_gen = first_step_where(rows, rows[FID_COL] <= fid_rel * best_fid)
        if not np.isfinite(tau_gen):
            raise RuntimeError(f"No finite tau_gen crossing for run {meta.get('run_id', keys)}")

        tau_mem_observed = first_step_where(rows, rows[FMEM_COL] >= fmem_threshold)
        tau_mem_censored = not np.isfinite(tau_mem_observed)
        final_step = float(rows[STEP_COL].max())
        tau_mem = final_step if tau_mem_censored else tau_mem_observed
        delta_tau = tau_mem - tau_gen

        records.append(
            {
                "run_key": str(meta.get("run_id", keys)),
                "manifest_index": meta.get("manifest_index", np.nan),
                "run_id": meta.get("run_id", ""),
                "n_train": int(meta["n_train"]),
                "seed": int(meta["seed"]),
                "param": str(meta["param"]),
                "tau_gen": tau_gen,
                "tau_mem": tau_mem,
                "delta_tau": delta_tau,
                "tau_mem_observed": tau_mem_observed,
                "tau_mem_censored": bool(tau_mem_censored),
                "delta_tau_lower_bound": bool(tau_mem_censored),
                "final_step": final_step,
                "best_fid": best_fid,
                "fid_rel_threshold": fid_rel,
                "fmem_threshold": fmem_threshold,
            }
        )

    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError("No run-level tau scales were computed.")
    return out.sort_values(["n_train", "param", "seed", "run_key"])


def compute_bonnaire_run_tau_scales(metrics: pd.DataFrame, fmem_baseline: float = 0.0) -> pd.DataFrame:
    required = ["n_train", "seed", "param", STEP_COL, FID_COL, FMEM_COL]
    missing = [column for column in required if column not in metrics.columns]
    if missing:
        raise RuntimeError(f"Missing required checkpoint columns: {missing}")

    df = metrics.copy()
    for column in ["n_train", "seed", STEP_COL, FID_COL, FMEM_COL]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["n_train", "seed", "param", STEP_COL, FID_COL, FMEM_COL])
    if df.empty:
        raise RuntimeError("No finite checkpoint rows after numeric cleanup.")

    if "run_id" in df.columns:
        group_cols = ["run_id"]
    elif "manifest_index" in df.columns:
        group_cols = ["manifest_index", "n_train", "param", "seed"]
    else:
        group_cols = ["n_train", "param", "seed"]

    records: list[dict[str, object]] = []
    for keys, rows in df.sort_values(STEP_COL).groupby(group_cols, dropna=False):
        rows = rows.sort_values(STEP_COL).copy()
        meta = rows.iloc[0]
        valid_fid = rows[np.isfinite(rows[FID_COL])].copy()
        if valid_fid.empty:
            continue
        best = valid_fid.sort_values([FID_COL, STEP_COL]).iloc[0]
        tau_gen = float(best[STEP_COL])

        onset = rows.loc[rows[FMEM_COL] > fmem_baseline].sort_values(STEP_COL)
        tau_mem_censored = onset.empty
        final_step = float(rows[STEP_COL].max())
        tau_mem_observed = float("nan") if tau_mem_censored else float(onset[STEP_COL].iloc[0])
        tau_mem = final_step if tau_mem_censored else tau_mem_observed
        delta_tau = tau_mem - tau_gen

        records.append(
            {
                "run_key": str(meta.get("run_id", keys)),
                "manifest_index": meta.get("manifest_index", np.nan),
                "run_id": meta.get("run_id", ""),
                "n_train": int(meta["n_train"]),
                "seed": int(meta["seed"]),
                "param": str(meta["param"]),
                "tau_gen": tau_gen,
                "tau_mem": tau_mem,
                "delta_tau": delta_tau,
                "tau_mem_observed": tau_mem_observed,
                "tau_mem_censored": bool(tau_mem_censored),
                "delta_tau_lower_bound": bool(tau_mem_censored),
                "final_step": final_step,
                "best_fid": float(best[FID_COL]),
                "fid_rel_threshold": np.nan,
                "fmem_threshold": fmem_baseline,
                "tau_gen_definition": "first_argmin_fid",
                "tau_mem_definition": "first_f_mem_above_baseline",
            }
        )

    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError("No Bonnaire-style run-level tau scales were computed.")
    return out.sort_values(["n_train", "param", "seed", "run_key"])


def aggregate_tau_scales(run_tau: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        run_tau.groupby(["n_train", "param"], dropna=False)
        .agg(
            n_runs=("run_key", "nunique"),
            n_seeds=("seed", "nunique"),
            fid_rel_threshold=("fid_rel_threshold", "first"),
            fmem_threshold=("fmem_threshold", "first"),
            tau_gen_mean=("tau_gen", "mean"),
            tau_gen_sem=("tau_gen", sem),
            tau_mem_mean=("tau_mem", "mean"),
            tau_mem_sem=("tau_mem", sem),
            delta_tau_mean=("delta_tau", "mean"),
            delta_tau_sem=("delta_tau", sem),
            n_tau_mem_censored=("tau_mem_censored", "sum"),
            tau_mem_censored_frac=("tau_mem_censored", "mean"),
            n_delta_tau_lower_bound=("delta_tau_lower_bound", "sum"),
        )
        .reset_index()
        .sort_values(["n_train", "param"])
    )
    return grouped


def save_table(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[table] {path}")


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in FIGURE_EXTENSIONS:
        path = out_dir / f"{name}{ext}"
        if ext == ".png":
            fig.savefig(path, dpi=180, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        print(f"[figure] {path}")


def y_limits_for(summary: pd.DataFrame, mean_cols: list[str], sem_cols: list[str], include_zero: bool = False) -> tuple[float, float]:
    lower: list[float] = []
    upper: list[float] = []
    for mean_col, sem_col in zip(mean_cols, sem_cols):
        means = pd.to_numeric(summary[mean_col], errors="coerce").to_numpy(dtype=float)
        sems = pd.to_numeric(summary[sem_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        finite = np.isfinite(means) & np.isfinite(sems)
        if finite.any():
            lower.extend((means[finite] - sems[finite]).tolist())
            upper.extend((means[finite] + sems[finite]).tolist())
    if include_zero:
        lower.append(0.0)
        upper.append(0.0)
    if not lower or not upper:
        return (0.0, 1.0)
    ymin = min(lower)
    ymax = max(upper)
    pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.1 * max(abs(ymax), 1.0)
    return (ymin - pad, ymax + pad)


def tau_definition_text(summary: pd.DataFrame) -> str:
    fid_rel = float(summary["fid_rel_threshold"].iloc[0]) if "fid_rel_threshold" in summary.columns else 1.05
    fmem_threshold = float(summary["fmem_threshold"].iloc[0]) if "fmem_threshold" in summary.columns else 0.05
    return (
        rf"$\tau_{{\mathrm{{gen}}}}$: first step with FID $\leq {fid_rel:.2f}\times$ run-best FID; "
        rf"$\tau_{{\mathrm{{mem}}}}$: first step with $F_{{\mathrm{{mem}}}}\geq {fmem_threshold:.2f}$; "
        r"censored points use final checkpoint"
    )


def plot_tau_scales(summary: pd.DataFrame, out_dir: Path) -> None:
    params = [param for param in PARAM_ORDER if param in set(summary["param"])]
    params.extend(param for param in sorted(summary["param"].unique()) if param not in params)
    if not params:
        raise RuntimeError("No parameterizations available for plotting.")

    fig, axes = plt.subplots(2, len(params), figsize=(15.0, 5.2), sharex="col", sharey=False)
    axes = np.asarray(axes).reshape(2, len(params))

    tau_ylim = y_limits_for(summary, ["tau_gen_mean", "tau_mem_mean"], ["tau_gen_sem", "tau_mem_sem"])
    delta_ylim = y_limits_for(summary, ["delta_tau_mean"], ["delta_tau_sem"], include_zero=True)

    gen_color = "#1f77b4"
    mem_color = "#ff7f0e"
    delta_color = "#2ca02c"
    n_train_values = sorted(summary["n_train"].unique())
    n_train_labels = [str(int(v)) for v in n_train_values]

    for col_idx, param in enumerate(params):
        sub = summary[summary["param"] == param].sort_values("n_train").copy()
        x = sub["n_train"].to_numpy(dtype=float)
        title = PARAM_LABELS.get(param, str(param).replace("_", "/"))

        top = axes[0, col_idx]
        top.set_title(title, fontsize=15, fontweight="bold")
        top.errorbar(
            x,
            sub["tau_gen_mean"],
            yerr=sub["tau_gen_sem"],
            color=gen_color,
            marker="o",
            linestyle="-",
            linewidth=1.9,
            capsize=3,
            label=r"$\tau_{\mathrm{gen}}$",
        )
        top.errorbar(
            x,
            sub["tau_mem_mean"],
            yerr=sub["tau_mem_sem"],
            color=mem_color,
            marker="s",
            linestyle="--",
            linewidth=1.9,
            capsize=3,
            label=r"$\tau_{\mathrm{mem}}$",
        )
        censored = sub["n_tau_mem_censored"] > 0
        if censored.any():
            top.scatter(
                sub.loc[censored, "n_train"],
                sub.loc[censored, "tau_mem_mean"],
                marker="^",
                s=48,
                facecolors="none",
                edgecolors=mem_color,
                linewidths=1.4,
                zorder=5,
            )

        bottom = axes[1, col_idx]
        bottom.axhline(0.0, color="0.25", linewidth=0.9, linestyle=":")
        bottom.errorbar(
            x,
            sub["delta_tau_mean"],
            yerr=sub["delta_tau_sem"],
            color=delta_color,
            marker="o",
            linestyle="-",
            linewidth=1.9,
            capsize=3,
            label=r"$\Delta\tau$",
        )
        if censored.any():
            bottom.scatter(
                sub.loc[censored, "n_train"],
                sub.loc[censored, "delta_tau_mean"],
                marker="^",
                s=48,
                facecolors="none",
                edgecolors=delta_color,
                linewidths=1.4,
                zorder=5,
            )

        for ax in (top, bottom):
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(mticker.FixedLocator(n_train_values))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter(n_train_labels))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.tick_params(axis="both", labelsize=11)
            ax.grid(True, alpha=0.3)
        top.set_ylim(*tau_ylim)
        bottom.set_ylim(*delta_ylim)
        top.tick_params(axis="x", which="major", bottom=True, labelbottom=True, length=5)
        bottom.tick_params(axis="x", which="major", bottom=True, labelbottom=True, length=5)
        top.set_xlabel(N_TRAIN_LABEL, fontweight="bold", fontsize=14)
        bottom.set_xlabel(N_TRAIN_LABEL, fontweight="bold", fontsize=14)
        for ax in (top, bottom):
            for label in ax.get_xticklabels():
                label.set_rotation(35)
                label.set_horizontalalignment("right")

    axes[0, 0].set_ylabel("training step", fontweight="bold", fontsize=15)
    axes[1, 0].set_ylabel(r"$\Delta\tau=\tau_{\mathrm{mem}}-\tau_{\mathrm{gen}}$", fontweight="bold", fontsize=15)

    legend_handles = [
        Line2D([0], [0], color=gen_color, marker="o", linestyle="-", linewidth=1.9, label=r"$\tau_{\mathrm{gen}}$"),
        Line2D([0], [0], color=mem_color, marker="s", linestyle="--", linewidth=1.9, label=r"$\tau_{\mathrm{mem}}$"),
        Line2D([0], [0], color=delta_color, marker="o", linestyle="-", linewidth=1.9, label=r"$\Delta\tau$"),
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="^",
            linestyle="None",
            markerfacecolor="none",
            markersize=7,
            label="censored/lower bound",
        ),
    ]
    fig.legend(handles=legend_handles, fontsize=12, loc="center right", bbox_to_anchor=(0.995, 0.5))
    fig.text(0.07, 0.035, tau_definition_text(summary), fontsize=10, ha="left", va="bottom")
    fig.subplots_adjust(left=0.07, right=0.8, bottom=0.2, top=0.9, wspace=0.3, hspace=0.48)
    save_figure(fig, out_dir, FIGURE_NAME)
    plt.close(fig)


def ordered_params(summary: pd.DataFrame) -> list[str]:
    params = [param for param in PARAM_ORDER if param in set(summary["param"])]
    params.extend(param for param in sorted(summary["param"].unique()) if param not in params)
    return params


def style_n_train_axis(ax: plt.Axes, n_train_values: list[int], n_train_labels: list[str]) -> None:
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(mticker.FixedLocator(n_train_values))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(n_train_labels))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="both", labelsize=11)
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_horizontalalignment("right")
    ax.set_xlabel(N_TRAIN_LABEL, fontweight="bold", fontsize=14)
    ax.grid(True, alpha=0.3)


def plot_tau_scales_overlay(summary: pd.DataFrame, out_dir: Path) -> None:
    params = ordered_params(summary)
    if not params:
        raise RuntimeError("No parameterizations available for overlay plotting.")

    n_train_values = [int(v) for v in sorted(summary["n_train"].unique())]
    n_train_labels = [str(v) for v in n_train_values]
    metric_specs = [
        ("tau_gen", r"$\tau_{\mathrm{gen}}$", "training step", False, False),
        ("tau_mem", r"$\tau_{\mathrm{mem}}$", "training step", True, False),
        ("tau_mem", r"$\tau_{\mathrm{mem}}$ (log y)", "training step", True, True),
        ("delta_tau", r"$\Delta\tau=\tau_{\mathrm{mem}}-\tau_{\mathrm{gen}}$", r"$\Delta\tau$", True, False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 3.9), sharex=False, sharey=False)
    axes = np.asarray(axes).reshape(-1)

    y_limits = {
        "tau_gen": y_limits_for(summary, ["tau_gen_mean"], ["tau_gen_sem"]),
        "tau_mem": y_limits_for(summary, ["tau_mem_mean"], ["tau_mem_sem"]),
        "tau_linear": y_limits_for(
            summary,
            ["tau_gen_mean", "tau_mem_mean"],
            ["tau_gen_sem", "tau_mem_sem"],
        ),
        "delta_tau": y_limits_for(summary, ["delta_tau_mean"], ["delta_tau_sem"], include_zero=True),
    }

    for ax, (metric, title, ylabel, show_censored, log_y) in zip(axes, metric_specs):
        if metric == "delta_tau":
            ax.axhline(0.0, color="0.25", linewidth=0.9, linestyle=":")

        for param in params:
            sub = summary[summary["param"] == param].sort_values("n_train")
            color = PARAM_COLORS.get(param, None)
            ax.errorbar(
                sub["n_train"],
                sub[f"{metric}_mean"],
                yerr=sub[f"{metric}_sem"],
                color=color,
                marker="o",
                linestyle="-",
                linewidth=2.0,
                capsize=3,
                markersize=5.2,
                label=PARAM_LABELS.get(param, str(param).replace("_", "/")),
            )
            if show_censored:
                censored = sub["n_tau_mem_censored"] > 0
                if censored.any():
                    ax.scatter(
                        sub.loc[censored, "n_train"],
                        sub.loc[censored, f"{metric}_mean"],
                        marker="^",
                        s=64,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=1.5,
                        zorder=5,
                    )

        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=14)
        if log_y:
            ax.set_yscale("log")
            positive = summary.loc[summary[f"{metric}_mean"] > 0, f"{metric}_mean"]
            ymin = max(float(positive.min()) * 0.7, 1.0) if not positive.empty else 1.0
            ymax = y_limits[metric][1]
            ax.set_ylim(ymin, ymax)
        elif metric in {"tau_gen", "tau_mem"}:
            ax.set_ylim(*y_limits["tau_linear"])
            formatter = mticker.ScalarFormatter(useMathText=False)
            formatter.set_powerlimits((6, 6))
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.get_offset_text().set_fontsize(11)
        else:
            ax.set_ylim(*y_limits[metric])
        style_n_train_axis(ax, n_train_values, n_train_labels)

    param_handles = [
        Line2D(
            [0],
            [0],
            color=PARAM_COLORS.get(param, "0.2"),
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=5.2,
            label=PARAM_LABELS.get(param, str(param).replace("_", "/")),
        )
        for param in params
    ]
    censor_handle = Line2D(
        [0],
        [0],
        color="0.25",
        marker="^",
        linestyle="None",
        markerfacecolor="none",
        markersize=7,
        label="censored/lower bound",
    )
    fig.legend(
        handles=param_handles + [censor_handle],
        fontsize=10,
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
    )
    fig.text(0.055, 0.025, tau_definition_text(summary), fontsize=10, ha="left", va="bottom")
    fig.subplots_adjust(left=0.055, right=0.85, bottom=0.33, top=0.84, wspace=0.38)
    save_figure(fig, out_dir, OVERLAY_FIGURE_NAME)
    plt.close(fig)


def plot_tau_gen_threshold_sensitivity(summary: pd.DataFrame, out_dir: Path, figure_name: str) -> None:
    params = ordered_params(summary)
    if not params:
        raise RuntimeError("No parameterizations available for tau_gen threshold sensitivity plotting.")

    thresholds = sorted(float(v) for v in summary["fid_rel_threshold"].unique())
    n_train_values = [int(v) for v in sorted(summary["n_train"].unique())]
    n_train_labels = [str(v) for v in n_train_values]
    fmem_threshold = float(summary["fmem_threshold"].iloc[0])
    metric_specs = [
        ("tau_gen", r"$\tau_{\mathrm{gen}}$", "training step", False, False),
        ("tau_mem", r"$\tau_{\mathrm{mem}}$", "training step", True, False),
        ("tau_mem", r"$\tau_{\mathrm{mem}}$ (log y)", "training step", True, True),
        ("delta_tau", r"$\Delta\tau=\tau_{\mathrm{mem}}-\tau_{\mathrm{gen}}$", r"$\Delta\tau$", True, False),
    ]

    fig, axes = plt.subplots(len(thresholds), 4, figsize=(17.5, 3.15 * len(thresholds)), sharex=False, sharey=False)
    axes = np.asarray(axes).reshape(len(thresholds), 4)
    y_limits = {
        "tau_linear": y_limits_for(
            summary,
            ["tau_gen_mean", "tau_mem_mean"],
            ["tau_gen_sem", "tau_mem_sem"],
        ),
        "tau_mem": y_limits_for(summary, ["tau_mem_mean"], ["tau_mem_sem"]),
        "delta_tau": y_limits_for(summary, ["delta_tau_mean"], ["delta_tau_sem"], include_zero=True),
    }

    for row_idx, threshold in enumerate(thresholds):
        row_summary = summary[np.isclose(summary["fid_rel_threshold"], threshold)].copy()
        for col_idx, (metric, title, ylabel, show_censored, log_y) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(title, fontsize=15, fontweight="bold")
            if metric == "delta_tau":
                ax.axhline(0.0, color="0.25", linewidth=0.9, linestyle=":")

            for param in params:
                sub = row_summary[row_summary["param"] == param].sort_values("n_train")
                color = PARAM_COLORS.get(param, None)
                ax.errorbar(
                    sub["n_train"],
                    sub[f"{metric}_mean"],
                    yerr=sub[f"{metric}_sem"],
                    color=color,
                    marker="o",
                    linestyle="-",
                    linewidth=1.9,
                    capsize=3,
                    markersize=4.8,
                    label=PARAM_LABELS.get(param, str(param).replace("_", "/")),
                )
                if show_censored:
                    censored = sub["n_tau_mem_censored"] > 0
                    if censored.any():
                        ax.scatter(
                            sub.loc[censored, "n_train"],
                            sub.loc[censored, f"{metric}_mean"],
                            marker="^",
                            s=56,
                            facecolors="none",
                            edgecolors=color,
                            linewidths=1.4,
                            zorder=5,
                        )

            if log_y:
                ax.set_yscale("log")
                positive = summary.loc[summary[f"{metric}_mean"] > 0, f"{metric}_mean"]
                ymin = max(float(positive.min()) * 0.7, 1.0) if not positive.empty else 1.0
                ax.set_ylim(ymin, y_limits[metric][1])
            elif metric in {"tau_gen", "tau_mem"}:
                ax.set_ylim(*y_limits["tau_linear"])
                formatter = mticker.ScalarFormatter(useMathText=False)
                formatter.set_powerlimits((6, 6))
                ax.yaxis.set_major_formatter(formatter)
                ax.yaxis.get_offset_text().set_fontsize(10)
            else:
                ax.set_ylim(*y_limits[metric])

            if col_idx == 0:
                ax.set_ylabel(
                    f"FID <= {threshold:.2f}x best\n{ylabel}",
                    fontweight="bold",
                    fontsize=12,
                )
            elif row_idx == len(thresholds) - 1:
                ax.set_ylabel(ylabel, fontweight="bold", fontsize=12)
            style_n_train_axis(ax, n_train_values, n_train_labels)

    param_handles = [
        Line2D(
            [0],
            [0],
            color=PARAM_COLORS.get(param, "0.2"),
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=5.2,
            label=PARAM_LABELS.get(param, str(param).replace("_", "/")),
        )
        for param in params
    ]
    censor_handle = Line2D(
        [0],
        [0],
        color="0.25",
        marker="^",
        linestyle="None",
        markerfacecolor="none",
        markersize=7,
        label="censored/lower bound",
    )
    fig.legend(
        handles=param_handles + [censor_handle],
        fontsize=10,
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
    )
    fig.text(
        0.055,
        0.025,
        rf"Rows vary $\tau_{{\mathrm{{gen}}}}$ threshold: first step with FID $\leq \alpha\times$ run-best FID; "
        rf"$\tau_{{\mathrm{{mem}}}}$: first step with $F_{{\mathrm{{mem}}}}\geq {fmem_threshold:.2f}$; "
        r"censored points use final checkpoint",
        fontsize=10,
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.075, right=0.85, bottom=0.12, top=0.92, wspace=0.38, hspace=0.52)
    save_figure(fig, out_dir, figure_name)
    plt.close(fig)


def plot_bonnaire_overlay(summary: pd.DataFrame, out_dir: Path) -> None:
    params = ordered_params(summary)
    if not params:
        raise RuntimeError("No parameterizations available for Bonnaire overlay plotting.")

    n_train_values = [int(v) for v in sorted(summary["n_train"].unique())]
    n_train_labels = [str(v) for v in n_train_values]
    metric_specs = [
        ("tau_gen", r"$\tau_{\mathrm{gen}}$ (FID argmin)", "training step", False, False),
        ("tau_mem", r"$\tau_{\mathrm{mem}}$ ($F_{\mathrm{mem}}>0$)", "training step", True, False),
        ("tau_mem", r"$\tau_{\mathrm{mem}}$ (log y)", "training step", True, True),
        ("delta_tau", r"$\Delta\tau=\tau_{\mathrm{mem}}-\tau_{\mathrm{gen}}$", r"$\Delta\tau$", True, False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 3.9), sharex=False, sharey=False)
    axes = np.asarray(axes).reshape(-1)
    y_limits = {
        "tau_linear": y_limits_for(
            summary,
            ["tau_gen_mean", "tau_mem_mean"],
            ["tau_gen_sem", "tau_mem_sem"],
        ),
        "tau_mem": y_limits_for(summary, ["tau_mem_mean"], ["tau_mem_sem"]),
        "delta_tau": y_limits_for(summary, ["delta_tau_mean"], ["delta_tau_sem"], include_zero=True),
    }

    for ax, (metric, title, ylabel, show_censored, log_y) in zip(axes, metric_specs):
        if metric == "delta_tau":
            ax.axhline(0.0, color="0.25", linewidth=0.9, linestyle=":")

        for param in params:
            sub = summary[summary["param"] == param].sort_values("n_train")
            color = PARAM_COLORS.get(param, None)
            ax.errorbar(
                sub["n_train"],
                sub[f"{metric}_mean"],
                yerr=sub[f"{metric}_sem"],
                color=color,
                marker="o",
                linestyle="-",
                linewidth=2.0,
                capsize=3,
                markersize=5.2,
                label=PARAM_LABELS.get(param, str(param).replace("_", "/")),
            )
            if show_censored:
                censored = sub["n_tau_mem_censored"] > 0
                if censored.any():
                    ax.scatter(
                        sub.loc[censored, "n_train"],
                        sub.loc[censored, f"{metric}_mean"],
                        marker="^",
                        s=64,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=1.5,
                        zorder=5,
                    )

        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=14)
        if log_y:
            ax.set_yscale("log")
            positive = summary.loc[summary[f"{metric}_mean"] > 0, f"{metric}_mean"]
            ymin = max(float(positive.min()) * 0.7, 1.0) if not positive.empty else 1.0
            ax.set_ylim(ymin, y_limits[metric][1])
        elif metric in {"tau_gen", "tau_mem"}:
            ax.set_ylim(*y_limits["tau_linear"])
            formatter = mticker.ScalarFormatter(useMathText=False)
            formatter.set_powerlimits((6, 6))
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.get_offset_text().set_fontsize(11)
        else:
            ax.set_ylim(*y_limits[metric])
        style_n_train_axis(ax, n_train_values, n_train_labels)

    param_handles = [
        Line2D(
            [0],
            [0],
            color=PARAM_COLORS.get(param, "0.2"),
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=5.2,
            label=PARAM_LABELS.get(param, str(param).replace("_", "/")),
        )
        for param in params
    ]
    censor_handle = Line2D(
        [0],
        [0],
        color="0.25",
        marker="^",
        linestyle="None",
        markerfacecolor="none",
        markersize=7,
        label="censored/lower bound",
    )
    fig.legend(
        handles=param_handles + [censor_handle],
        fontsize=10,
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
    )
    fig.text(
        0.055,
        0.025,
        r"Bonnaire-style definitions: "
        r"$\tau_{\mathrm{gen}}$ is the first checkpoint attaining run-minimum FID; "
        r"$\tau_{\mathrm{mem}}$ is the first checkpoint with $F_{\mathrm{mem}}>0$; "
        r"censored points use final checkpoint",
        fontsize=10,
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.055, right=0.85, bottom=0.33, top=0.84, wspace=0.38)
    save_figure(fig, out_dir, BONNAIRE_OVERLAY_FIGURE_NAME)
    plt.close(fig)


def plot_bonnaire_scaling_guides(summary: pd.DataFrame, out_dir: Path) -> None:
    params = ordered_params(summary)
    if not params:
        raise RuntimeError("No parameterizations available for Bonnaire scaling-guide plotting.")

    n_train_values = [int(v) for v in sorted(summary["n_train"].unique())]
    n_train_labels = [str(v) for v in n_train_values]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), sharex=False, sharey=False)
    tau_gen_ax, tau_mem_ax = axes

    for param in params:
        sub = summary[summary["param"] == param].sort_values("n_train")
        color = PARAM_COLORS.get(param, None)
        label = PARAM_LABELS.get(param, str(param).replace("_", "/"))
        tau_gen_ax.errorbar(
            sub["n_train"],
            sub["tau_gen_mean"],
            yerr=sub["tau_gen_sem"],
            color=color,
            marker="o",
            linestyle="-",
            linewidth=2.0,
            capsize=3,
            markersize=5.2,
            label=label,
        )
        tau_mem_ax.errorbar(
            sub["n_train"],
            sub["tau_mem_mean"],
            yerr=sub["tau_mem_sem"],
            color=color,
            marker="o",
            linestyle="-",
            linewidth=2.0,
            capsize=3,
            markersize=5.2,
            label=label,
        )
        censored = sub["n_tau_mem_censored"] > 0
        if censored.any():
            tau_mem_ax.scatter(
                sub.loc[censored, "n_train"],
                sub.loc[censored, "tau_mem_mean"],
                marker="^",
                s=64,
                facecolors="none",
                edgecolors=color,
                linewidths=1.5,
                zorder=5,
            )

    tau_gen_guide = float(summary["tau_gen_mean"].median())
    tau_gen_ax.axhline(tau_gen_guide, color="0.2", linestyle="--", linewidth=1.2, label="constant guide")

    positive_mem = summary[summary["tau_mem_mean"] > 0].sort_values("n_train")
    if not positive_mem.empty:
        anchor_n = float(positive_mem["n_train"].min())
        anchor_tau = float(positive_mem.loc[positive_mem["n_train"].eq(anchor_n), "tau_mem_mean"].median())
        guide_x = np.asarray(n_train_values, dtype=float)
        tau_mem_ax.plot(
            guide_x,
            anchor_tau * guide_x / anchor_n,
            color="0.2",
            linestyle="--",
            linewidth=1.2,
            label=r"$\propto N_{\mathrm{train}}$ guide",
        )

    tau_gen_ax.set_title(r"$\tau_{\mathrm{gen}}$: first min-FID checkpoint", fontsize=14, fontweight="bold")
    tau_mem_ax.set_title(r"$\tau_{\mathrm{mem}}$: first $F_{\mathrm{mem}}>0$", fontsize=14, fontweight="bold")
    tau_gen_ax.set_ylabel("training step", fontsize=14, fontweight="bold")
    tau_mem_ax.set_ylabel("training step", fontsize=14, fontweight="bold")
    tau_mem_ax.set_yscale("log")

    for ax in axes:
        style_n_train_axis(ax, n_train_values, n_train_labels)

    handles, labels = tau_mem_ax.get_legend_handles_labels()
    tau_mem_ax.legend(handles, labels, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.text(
        0.08,
        0.04,
        r"Reference guides show Bonnaire-style expectations: $\tau_{\mathrm{gen}}\propto N^0$ and "
        r"$\tau_{\mathrm{mem}}\propto N_{\mathrm{train}}$ at fixed width.",
        fontsize=10,
        ha="left",
    )
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.25, top=0.82, wspace=0.35)
    save_figure(fig, out_dir, BONNAIRE_SCALING_FIGURE_NAME)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fid-rel", type=float, default=1.05)
    parser.add_argument("--fmem-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = resolve_under_repo(args.repo, args.metrics)
    out_dir = resolve_under_repo(args.repo, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(metrics_path)
    run_tau = compute_run_tau_scales(metrics, fid_rel=float(args.fid_rel), fmem_threshold=float(args.fmem_threshold))
    summary = aggregate_tau_scales(run_tau)

    if run_tau["tau_gen"].isna().any():
        raise RuntimeError("tau_gen should be finite for all runs, but NaNs were found.")
    if not (run_tau.loc[run_tau["tau_mem_censored"], "tau_mem"] == run_tau.loc[run_tau["tau_mem_censored"], "final_step"]).all():
        raise RuntimeError("Censored tau_mem rows must use the final checkpoint step.")

    save_table(run_tau, out_dir, "N01_run_tau_scales")
    save_table(summary, out_dir, "N01_seed_average_tau_scales")
    plot_tau_scales(summary, out_dir)
    plot_tau_scales_overlay(summary, out_dir)

    sensitivity_runs = [
        compute_run_tau_scales(metrics, fid_rel=fid_rel, fmem_threshold=float(args.fmem_threshold))
        for fid_rel in GEN_THRESHOLD_SENSITIVITY
    ]
    sensitivity_run_tau = pd.concat(sensitivity_runs, ignore_index=True)
    sensitivity_summary = pd.concat(
        [aggregate_tau_scales(run_frame) for run_frame in sensitivity_runs],
        ignore_index=True,
    )
    save_table(sensitivity_run_tau, out_dir, "N03_run_tau_scales_by_tau_gen_threshold")
    save_table(sensitivity_summary, out_dir, "N03_seed_average_tau_scales_by_tau_gen_threshold")
    plot_tau_gen_threshold_sensitivity(sensitivity_summary, out_dir, GEN_THRESHOLD_SENSITIVITY_FIGURE_NAME)

    fmem001_runs = [
        compute_run_tau_scales(metrics, fid_rel=fid_rel, fmem_threshold=FMEM_THRESHOLD_SENSITIVITY)
        for fid_rel in GEN_THRESHOLD_SENSITIVITY
    ]
    fmem001_run_tau = pd.concat(fmem001_runs, ignore_index=True)
    fmem001_summary = pd.concat(
        [aggregate_tau_scales(run_frame) for run_frame in fmem001_runs],
        ignore_index=True,
    )
    save_table(fmem001_run_tau, out_dir, "N04_run_tau_scales_by_tau_gen_threshold_fmem_0p01")
    save_table(fmem001_summary, out_dir, "N04_seed_average_tau_scales_by_tau_gen_threshold_fmem_0p01")
    plot_tau_gen_threshold_sensitivity(fmem001_summary, out_dir, GEN_THRESHOLD_SENSITIVITY_FMEM001_FIGURE_NAME)

    bonnaire_run_tau = compute_bonnaire_run_tau_scales(metrics)
    bonnaire_summary = aggregate_tau_scales(bonnaire_run_tau)
    if bonnaire_run_tau["tau_gen"].isna().any():
        raise RuntimeError("Bonnaire tau_gen should be finite for all runs, but NaNs were found.")
    if not (
        bonnaire_run_tau.loc[bonnaire_run_tau["tau_mem_censored"], "tau_mem"]
        == bonnaire_run_tau.loc[bonnaire_run_tau["tau_mem_censored"], "final_step"]
    ).all():
        raise RuntimeError("Bonnaire censored tau_mem rows must use the final checkpoint step.")
    save_table(bonnaire_run_tau, out_dir, "N05_bonnaire_run_tau_scales")
    save_table(bonnaire_summary, out_dir, "N05_bonnaire_seed_average_tau_scales")
    plot_bonnaire_overlay(bonnaire_summary, out_dir)
    plot_bonnaire_scaling_guides(bonnaire_summary, out_dir)

    print(
        "[done] "
        f"runs={len(run_tau)} censored_tau_mem={int(run_tau['tau_mem_censored'].sum())} "
        f"bonnaire_censored_tau_mem={int(bonnaire_run_tau['tau_mem_censored'].sum())} "
        f"out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
