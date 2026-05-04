#!/usr/bin/env python
"""Prototype figures for mem-boundary quality/memorization evals.

Experiment/job set:
    mem_gen_boundary_20260427

Eval output:
    mem_boundary_20260427_all_done_quality_20260501

Paper objective this script targets:
    The target/parameterization that is optimal for sample quality is not
    necessarily optimal for memorization. For this legacy job set we do not yet
    have the new fixed-k sweep, so "k" is replaced by whatever legacy
    target/parameterization metadata is available in the eval manifest or CSVs.

Main quantities:
    - FID(tau)
    - F_mem(tau)
    - focused n_train sweep trajectories for F_mem, test loss, and FID
    - best FID per run
    - F_mem at the best-FID checkpoint
    - parameterization trajectories at fixed seed and n_train
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ============================================================
# 00. Configuration
# ============================================================

DEFAULT_REPO = Path("/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par")
EXPERIMENT_NAME = "mem_gen_boundary_20260427"
EVAL_NAME = "mem_boundary_20260427_all_done_quality_20260501"

DEFAULT_EVAL_ROOT = Path("outputs/evals") / EVAL_NAME
DEFAULT_JOB_SET_ROOT = Path("outputs/job_sets") / EXPERIMENT_NAME
DEFAULT_OUT_DIR = Path("prototyping/figures") / f"{EVAL_NAME}_figures"

DEFAULT_REQUESTED_STEPS = [0, 100_000, 300_000, 595_000, 1_000_000]
DEFAULT_N_TRAIN_SWEEP_VALUES = [512, 2_048, 8_192]
DEFAULT_N_TRAIN_SWEEP_PARAM = "x_x"

FID_COL = "fid_10k"
FMEM_COL = "f_mem"
TAU_LABEL = r"$\tau$"

DISPLAY_LABELS = {
    FID_COL: "FID",
    FMEM_COL: r"$F_{\mathrm{mem}}$",
    "fid_mean": "FID mean",
    "fmem_mean": r"$F_{\mathrm{mem}}$ mean",
    "test_random_t_loss_10k": "test loss mean",
    "test_fixed_t_loss_mean_10k": "test loss mean",
    "test_loss_0_5_10k": r"test loss ($t=0.5$)",
    "test_loss_0_75_10k": r"test loss ($t=0.75$)",
    "f_mem_rho0.3": r"$F_{\mathrm{mem}}$ ($\rho=0.3$)",
    "f_mem_rho0.5": r"$F_{\mathrm{mem}}$ ($\rho=0.5$)",
    "f_mem_rho0.7": r"$F_{\mathrm{mem}}$ ($\rho=0.7$)",
}

LOSS_COLS = [
    "test_random_t_loss_10k",
    "test_fixed_t_loss_mean_10k",
    "test_loss_0_5_10k",
    "test_loss_0_75_10k",
]

FMEM_RHO_COLS = [
    ("rho=0.3", "f_mem_rho0.3"),
    ("rho=0.5", "f_mem_rho0.5"),
    ("rho=0.7", "f_mem_rho0.7"),
]


# ============================================================
# 01. Small helpers
# ============================================================

def resolve_under_repo(repo: Path, path: Path) -> Path:
    """Resolve relative paths against the repo root."""
    if path.is_absolute():
        return path
    return repo / path


def ensure_dir(path: Path) -> Path:
    """Create an output directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first candidate column that exists."""
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def save_table(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    """Save a dataframe as CSV with a numbered, named file."""
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[table] {path}")
    return path


def save_figure(fig: plt.Figure, out_dir: Path, name: str, show: bool = False) -> Path:
    """Save a matplotlib figure with a numbered, named file."""
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"[figure] {path}")
    if show:
        plt.show()
    plt.close(fig)
    return path


def safe_display_columns(df: pd.DataFrame, preferred: list[str]) -> list[str]:
    """Keep only preferred columns that exist."""
    return [c for c in preferred if c in df.columns]


def choose_step_col(df: pd.DataFrame) -> str:
    """Pick the training-step column used for checkpoint trajectories."""
    if "step" in df.columns:
        return "step"
    if "checkpoint_step" in df.columns:
        return "checkpoint_step"
    raise RuntimeError("No step or checkpoint_step column found.")


def choose_tau_plot_col(df: pd.DataFrame, step_col: str) -> str:
    """Use normalized tau for trajectory plots when eval outputs provide it."""
    if "tau" in df.columns:
        return "tau"
    return step_col


def choose_seed_col(df: pd.DataFrame) -> str | None:
    """Pick the seed column, if present."""
    return first_existing(df.columns, ["seed", "run_seed", "data_seed"])


def display_label(name: str) -> str:
    """Return consistent human-facing labels for plots."""
    return DISPLAY_LABELS.get(name, name)


def n_train_ticks_from(df: pd.DataFrame) -> list[int]:
    """Return exact n_train ticks for log-scaled dataset-size axes."""
    if "n_train" not in df.columns:
        return []
    ticks = []
    for value in sorted(df["n_train"].dropna().unique()):
        try:
            ticks.append(int(value))
        except (TypeError, ValueError):
            continue
    return ticks


def apply_n_train_log_ticks(ax: plt.Axes, ticks: list[int]) -> None:
    """Use exact powers/multiples from the sweep as log-axis ticks."""
    if not ticks:
        return
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])


def nearest_n_train_values(df: pd.DataFrame, requested: list[int]) -> list[int]:
    """Map requested dataset sizes to the nearest visible n_train values."""
    if "n_train" not in df.columns:
        return []

    available = []
    for value in sorted(df["n_train"].dropna().unique()):
        try:
            available.append(int(value))
        except (TypeError, ValueError):
            continue

    picked: list[int] = []
    for target in requested:
        if not available:
            break
        nearest = min(available, key=lambda value: (abs(value - int(target)), value))
        if nearest not in picked:
            picked.append(nearest)

    return picked


def normalize_param_token(value: object) -> str:
    """Normalize raw and plot-ready parameterization labels for filtering."""
    text = str(value).lower().strip()
    for old, new in [
        ("\\", ""),
        ("$", ""),
        ("{", ""),
        ("}", ""),
        ("mathrm", ""),
        ("epsilon", "eps"),
        ("/", "_"),
        ("-", "_"),
        (" ", "_"),
    ]:
        text = text.replace(old, new)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def filter_by_param(metrics: pd.DataFrame, param_filter: str | None) -> tuple[pd.DataFrame, str]:
    """Filter by parameterization while accepting raw or pretty labels."""
    if param_filter is None or str(param_filter).strip().lower() in {"", "all", "*"}:
        return metrics, "all parameterizations"

    normalized_filter = normalize_param_token(param_filter)
    mask = pd.Series(False, index=metrics.index)
    for column in ["param", "param_label_raw", "param_label"]:
        if column not in metrics.columns:
            continue
        mask = mask | metrics[column].map(normalize_param_token).eq(normalized_filter)

    if not mask.any():
        print(f"[warn] no rows matched --n-train-sweep-param={param_filter!r}; using all parameterizations")
        return metrics, "all parameterizations"

    return metrics[mask].copy(), str(param_filter)


def format_tau_tick(value: float) -> str:
    """Compact tick labels for normalized tau."""
    if np.isclose(value, 0.0):
        return "0"
    if value < 0.001:
        return f"{value:.2g}"
    if value < 0.1:
        return f"{value:g}"
    return f"{value:g}"


def apply_tau_axis_for_triptych(ax: plt.Axes, tau_values: pd.Series) -> None:
    """Use a log tau axis for positive normalized tau checkpoints."""
    positive = pd.to_numeric(tau_values, errors="coerce")
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.empty:
        return

    min_positive = float(positive.min())
    tick_candidates = [min_positive, 0.001, 0.01, 0.1, 1.0]
    max_tau = float(positive.max())
    ticks = []
    for tick in tick_candidates:
        if tick < 0.0 or tick > max_tau:
            continue
        if any(np.isclose(tick, existing, rtol=0.0, atol=min_positive * 0.05) for existing in ticks):
            continue
        ticks.append(tick)

    ax.set_xscale("log", base=10)
    ax.set_xlim(left=min_positive, right=max(1.0, max_tau))
    ax.set_xticks(ticks)
    ax.set_xticklabels([format_tau_tick(tick) for tick in ticks])


# ============================================================
# 02. Loading eval metrics and manifest metadata
# ============================================================

def load_checkpoint_metrics(repo: Path, eval_root: Path) -> pd.DataFrame:
    """Load all currently visible per-attempt checkpoint_metrics.csv files.

    The eval array writes one CSV per eval attempt. This function gives a live
    snapshot: if the Slurm array is still running, rerun the script later to
    pick up newly finished attempts.
    """
    combined_metrics = eval_root / "all_checkpoint_metrics.csv"
    if combined_metrics.exists():
        metrics = pd.read_csv(combined_metrics)
        metrics["metrics_file"] = str(combined_metrics.relative_to(repo))
        print(f"loaded combined checkpoint metrics: {combined_metrics}")
        print(f"loaded checkpoint rows: {len(metrics)}")
        return metrics

    metric_files = sorted(eval_root.rglob("checkpoint_metrics.csv"))

    # Fallback for notebook/prototype confusion: search all evals if the
    # expected eval root is empty.
    if not metric_files:
        broader_root = repo / "outputs/evals"
        metric_files = sorted(broader_root.rglob("checkpoint_metrics.csv"))

    rows: list[pd.DataFrame] = []
    for path in metric_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["metrics_file"] = str(path.relative_to(repo))
        rows.append(df)

    metrics = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    print(f"checkpoint_metrics.csv files: {len(metric_files)}")
    print(f"loaded checkpoint rows: {len(metrics)}")
    return metrics


def load_manifest_candidates(eval_root: Path, job_set_root: Path) -> list[pd.DataFrame]:
    """Load manifest-like CSVs that may contain parameterization metadata."""
    candidates: list[pd.DataFrame] = []
    roots = [eval_root, job_set_root]

    for root in roots:
        if not root.exists():
            continue

        for path in sorted(root.rglob("*.csv")):
            lower = path.name.lower()
            if "manifest" not in lower:
                continue
            if "checkpoint_metrics" in lower:
                continue

            try:
                df = pd.read_csv(path)
            except Exception as exc:  # pragma: no cover - prototype guard
                print(f"[warn] could not read manifest candidate {path}: {exc}")
                continue

            if df.empty:
                continue

            df["manifest_file"] = str(path)
            candidates.append(df)

    print(f"manifest candidates loaded: {len(candidates)}")
    return candidates


def merge_manifest_metadata(metrics: pd.DataFrame, manifests: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge missing metadata from the first manifest keyed by manifest_index.

    This keeps the eval CSVs as the source of metric values and only borrows
    columns that are absent from the metrics table.
    """
    if metrics.empty or "manifest_index" not in metrics.columns:
        return metrics

    out = metrics.copy()

    for manifest in manifests:
        if "manifest_index" not in manifest.columns:
            continue

        meta_cols = [c for c in manifest.columns if c not in out.columns]
        if not meta_cols:
            continue

        meta = manifest[["manifest_index"] + meta_cols].drop_duplicates("manifest_index")
        out = out.merge(meta, on="manifest_index", how="left")
        print(f"merged manifest metadata columns: {meta_cols}")

    return out


# ============================================================
# 03. Parameterization label inference
# ============================================================

def add_param_label(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Add a standard param_label column for plots/tables.

    For the future k sweep, this should become target_mode/k_value. For the
    current legacy sweep, we infer from available manifest/eval metadata.
    """
    out = df.copy()

    # Prefer explicit k metadata if this script is reused for the future sweep.
    if "target_mode" in out.columns and "k_value" in out.columns:
        out["param_label"] = (
            out["target_mode"].astype(str)
            + ":k="
            + out["k_value"].map(lambda x: "nan" if pd.isna(x) else f"{float(x):.3g}")
        )
        return out, "param_label"

    # Common single-column target/parameterization labels.
    single_candidates = [
        "param",
        "param_pair",
        "target_pair",
        "parameterization",
        "prediction_type",
        "target_type",
        "target",
        "model_target",
        "objective",
        "loss_space",
    ]

    for col in single_candidates:
        if col in out.columns and out[col].notna().any():
            out["param_label"] = out[col].astype(str)
            return out, "param_label"

    # Common paired metadata. These names are intentionally broad because this
    # is a prototype over older job sets.
    paired_candidates = [
        ("input_param", "target_param"),
        ("input_parameterization", "target_parameterization"),
        ("source_param", "target_param"),
        ("source_target", "prediction_target"),
        ("x_param", "eps_param"),
    ]

    for left, right in paired_candidates:
        if left in out.columns and right in out.columns:
            out["param_label"] = out[left].astype(str) + "/" + out[right].astype(str)
            return out, "param_label"

    # Last fallback: try to infer from path text.
    known_tokens = ["x_x", "eps_eps", "u_u", "v_v", "velocity", "x", "eps"]
    if "metrics_file" in out.columns:
        def infer_from_path(text: str) -> str:
            parts = Path(str(text)).parts
            for part in parts:
                for token in known_tokens:
                    if part == token or part.startswith(token + "_"):
                        return token
            return "unknown_param"

        out["param_label"] = out["metrics_file"].map(infer_from_path)
        return out, "param_label"

    out["param_label"] = "unknown_param"
    return out, "param_label"


def pretty_param_label(value: object) -> str:
    """Convert legacy/raw parameterization labels into plot-ready labels.

    The paper framing uses target/parameterization comparisons. For this
    legacy job set we want the labels to read as the canonical baselines:
        x/x, v/v, epsilon/epsilon.
    """
    text = str(value)
    norm = text.lower().strip()
    norm = norm.replace("\\", "/").replace("-", "_").replace("/", "_").replace(" ", "_")

    if norm in {"x", "x_x", "clean", "clean_clean", "data", "data_data"} or "x_x" in norm:
        return r"$x/x$"

    if norm in {"x_v", "clean_v", "data_v"} or "x_v" in norm:
        return r"$x/v$"

    if norm in {"v", "v_v", "u", "u_u", "vel", "vel_vel"} or "v_v" in norm or "u_u" in norm:
        return r"$v/v$"

    if (
        norm in {"eps", "eps_eps", "epsilon", "epsilon_epsilon", "noise", "noise_noise"}
        or "eps_eps" in norm
        or "epsilon_epsilon" in norm
    ):
        return r"$\epsilon/\epsilon$"

    return text


def apply_pretty_param_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep raw labels but use pretty labels everywhere downstream."""
    out = df.copy()
    if "param_label" not in out.columns:
        return out

    out["param_label_raw"] = out["param_label"]
    out["param_label"] = out["param_label"].map(pretty_param_label)
    return out


# ============================================================
# 04. Derived run summaries: final and best-FID rows
# ============================================================

def final_rows(metrics: pd.DataFrame, step_col: str) -> pd.DataFrame:
    """One row per run at the latest available checkpoint."""
    return (
        metrics
        .sort_values(["manifest_index", step_col])
        .groupby("manifest_index", as_index=False)
        .tail(1)
        .copy()
    )


def best_fid_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per run at the checkpoint with best FID.

    This is the core paper-objective quantity:
        F_mem@FID_best = F_mem at argmin_tau FID(tau)
    """
    required = ["manifest_index", FID_COL, FMEM_COL]
    missing = [c for c in required if c not in metrics.columns]
    if missing:
        raise RuntimeError(f"Missing required best-FID columns: {missing}")

    return (
        metrics
        .dropna(subset=[FID_COL, FMEM_COL])
        .sort_values(["manifest_index", FID_COL])
        .groupby("manifest_index", as_index=False)
        .head(1)
        .copy()
    )


# ============================================================
# 05. Tables
# ============================================================

def write_01_coverage_tables(metrics: pd.DataFrame, out_dir: Path, step_col: str) -> None:
    """Coverage tables: what has finished and what is visible."""
    id_cols = safe_display_columns(
        metrics,
        ["manifest_index", "n_train", "seed", "param_label", "metrics_file"],
    )

    visible_runs = metrics[id_cols].drop_duplicates() if id_cols else pd.DataFrame()
    save_table(visible_runs, out_dir, "01_visible_runs")

    group_cols = safe_display_columns(metrics, ["n_train", "param_label"])
    if group_cols:
        coverage = (
            metrics
            .groupby(group_cols, dropna=False)
            .agg(
                visible_runs=("manifest_index", "nunique"),
                checkpoint_rows=("manifest_index", "size"),
                min_step=(step_col, "min"),
                max_step=(step_col, "max"),
            )
            .reset_index()
        )
        save_table(coverage, out_dir, "02_coverage_by_n_train_and_parameterization")


def write_02_single_run_stage_table(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    requested_steps: list[int],
    manifest_index: int | None,
) -> int:
    """Table for one run at selected training stages."""
    if manifest_index is None:
        manifest_index = int(metrics["manifest_index"].dropna().sort_values().iloc[0])

    run = (
        metrics[metrics["manifest_index"] == manifest_index]
        .sort_values(step_col)
        .copy()
    )

    picked = []
    for requested in requested_steps:
        idx = (run[step_col] - requested).abs().idxmin()
        row = run.loc[idx].copy()
        row["requested_step"] = requested
        picked.append(row)

    stage_df = pd.DataFrame(picked)
    cols = safe_display_columns(
        stage_df,
        [
            "requested_step",
            "manifest_index",
            "n_train",
            "seed",
            "param_label",
            step_col,
            FID_COL,
            FMEM_COL,
            *LOSS_COLS,
            "metrics_file",
        ],
    )
    save_table(stage_df[cols], out_dir, "03_single_run_selected_stages")
    return manifest_index


def write_03_seed_average_tables(metrics: pd.DataFrame, out_dir: Path, step_col: str) -> None:
    """Seed/run averages at each checkpoint, plus final and best-FID summaries."""
    metric_cols = [c for c in [FID_COL, FMEM_COL, *LOSS_COLS] if c in metrics.columns]
    group_cols = safe_display_columns(metrics, ["n_train", "param_label", step_col])

    if group_cols and metric_cols:
        checkpoint_avg = (
            metrics
            .groupby(group_cols, dropna=False)
            .agg(
                **{f"{m}_mean": (m, "mean") for m in metric_cols},
                **{f"{m}_std": (m, "std") for m in metric_cols},
                n_runs=("manifest_index", "nunique"),
            )
            .reset_index()
        )
        save_table(checkpoint_avg, out_dir, "04_seed_run_average_by_checkpoint")

    summary_group_cols = safe_display_columns(metrics, ["n_train", "param_label"])

    for name, df in [
        ("05_final_checkpoint_summary", final_rows(metrics, step_col)),
        ("06_best_fid_summary_fmem_at_fid_best", best_fid_rows(metrics)),
    ]:
        metric_cols_here = [c for c in [FID_COL, FMEM_COL, *LOSS_COLS] if c in df.columns]
        if not summary_group_cols or not metric_cols_here:
            continue
        summary = (
            df
            .groupby(summary_group_cols, dropna=False)
            .agg(
                **{f"{m}_mean": (m, "mean") for m in metric_cols_here},
                **{f"{m}_std": (m, "std") for m in metric_cols_here},
                n_runs=("manifest_index", "nunique"),
            )
            .reset_index()
        )
        save_table(summary, out_dir, name)


# ============================================================
# 06. Figure: single-run metric trajectories
# ============================================================

def fig_01_single_run_trajectories(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    manifest_index: int,
    show: bool,
) -> None:
    """One run over checkpoints: FID, F_mem, and test losses."""
    run = (
        metrics[metrics["manifest_index"] == manifest_index]
        .sort_values(step_col)
        .copy()
    )

    plot_metrics = [c for c in [FID_COL, FMEM_COL, *LOSS_COLS] if c in run.columns]
    if not plot_metrics:
        return

    fig, axes = plt.subplots(
        len(plot_metrics),
        1,
        figsize=(9, 2.7 * len(plot_metrics)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, plot_metrics):
        ax.plot(run[step_col], run[metric], marker="o", linewidth=1.7)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(TAU_LABEL)
    label = run["param_label"].iloc[0] if "param_label" in run.columns else "unknown_param"
    fig.suptitle(f"01 single-run trajectories | run={manifest_index} | {label}", y=1.01)
    save_figure(fig, out_dir, "01_single_run_metric_trajectories", show=show)


# ============================================================
# 07. Figure: metrics along training, curves by n_train
# ============================================================

def fig_02_metrics_by_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    show: bool,
) -> None:
    """Notebook Cell 5 as a script figure: one subplot per metric."""
    plot_metrics = [c for c in [FID_COL, FMEM_COL, *LOSS_COLS] if c in metrics.columns]
    if "n_train" not in metrics.columns or not plot_metrics:
        return

    curve_df = (
        metrics
        .groupby(["n_train", step_col], dropna=False)
        .agg(
            **{f"{metric}_mean": (metric, "mean") for metric in plot_metrics},
            **{f"{metric}_std": (metric, "std") for metric in plot_metrics},
            n_runs=("manifest_index", "nunique"),
        )
        .reset_index()
    )

    n_trains = sorted(curve_df["n_train"].dropna().unique())
    fig, axes = plt.subplots(
        len(plot_metrics),
        1,
        figsize=(10, 2.8 * len(plot_metrics)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, plot_metrics):
        for n_train in n_trains:
            sub = curve_df[curve_df["n_train"] == n_train].sort_values(step_col)
            mean_col = f"{metric}_mean"

            # Step 0 and nonpositive metrics cannot appear on log axes.
            sub = sub[(sub[step_col] > 0) & (sub[mean_col] > 0)]
            if sub.empty:
                continue

            mean = sub[mean_col]

            ax.plot(sub[step_col], mean, marker="o", linewidth=1.5, label=f"N={n_train}")

        ax.set_ylabel(metric)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(title="n_train", fontsize=7)

    axes[-1].set_xlabel(TAU_LABEL)
    fig.suptitle("02 metrics along training by n_train | seed/run averaged", y=1.01)
    save_figure(fig, out_dir, "02_metrics_along_training_by_n_train", show=show)


# ============================================================
# 07a. Figure: focused n_train sweep triptych
# ============================================================

def fig_02b_n_train_sweep_tau_triptych(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    requested_n_trains: list[int],
    param_filter: str | None,
    show: bool,
) -> None:
    """Focused n_train sweep: F_mem, test loss mean, and FID vs tau.

    Curves are one selected dataset size each. By default this filters to a
    single parameterization before averaging, so the mean is over seeds rather
    than over heterogeneous targets.
    """
    if "n_train" not in metrics.columns:
        return

    loss_col = first_existing(
        metrics.columns,
        [
            "test_fixed_t_loss_mean_10k",
            "test_random_t_loss_10k",
            "test_loss_0_5_10k",
            "test_loss_0_75_10k",
        ],
    )
    plot_specs = [
        (FMEM_COL, display_label(FMEM_COL), False),
        (loss_col, "test loss mean", True),
        (FID_COL, display_label(FID_COL), True),
    ]
    plot_specs = [(metric, ylabel, log_y) for metric, ylabel, log_y in plot_specs if metric is not None]
    missing = [metric for metric, _, _ in plot_specs if metric not in metrics.columns]
    if missing or len(plot_specs) < 3:
        print(f"[warn] focused n_train sweep skipped; missing columns: {missing}")
        return

    filtered, param_label = filter_by_param(metrics, param_filter)
    n_trains = nearest_n_train_values(filtered, requested_n_trains)
    if not n_trains:
        print("[warn] focused n_train sweep skipped; no visible n_train values")
        return

    requested_text = ",".join(str(v) for v in requested_n_trains)
    picked_text = ",".join(str(v) for v in n_trains)
    if [int(v) for v in requested_n_trains] != n_trains:
        print(f"[info] focused n_train sweep requested [{requested_text}], using nearest visible [{picked_text}]")

    x_col = choose_tau_plot_col(filtered, step_col)
    seed_col = choose_seed_col(filtered)
    run_col = first_existing(filtered.columns, ["manifest_index", "run_id", "metrics_file"])

    sub = filtered.copy()
    sub["_n_train_plot"] = pd.to_numeric(sub["n_train"], errors="coerce")
    sub = sub[sub["_n_train_plot"].isin(n_trains)].copy()
    sub["_n_train_plot"] = sub["_n_train_plot"].astype(int)

    numeric_cols = [x_col] + [metric for metric, _, _ in plot_specs]
    for column in numeric_cols:
        sub[column] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub.dropna(subset=["_n_train_plot", x_col])
    if sub.empty:
        print("[warn] focused n_train sweep skipped; selected data has no finite x values")
        return

    agg_spec = {
        f"{metric}_mean": (metric, "mean")
        for metric, _, _ in plot_specs
    }
    agg_spec.update({
        f"{metric}_std": (metric, "std")
        for metric, _, _ in plot_specs
    })
    if run_col is not None:
        agg_spec["n_runs"] = (run_col, "nunique")
    else:
        agg_spec["n_rows"] = (x_col, "size")
    if seed_col is not None:
        agg_spec["n_seeds"] = (seed_col, "nunique")

    summary = (
        sub
        .groupby(["_n_train_plot", x_col], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .rename(columns={"_n_train_plot": "n_train"})
        .sort_values(["n_train", x_col])
    )
    summary["param_filter"] = param_label
    save_table(summary, out_dir, "09_n_train_sweep_tau_triptych_seed_average")

    colors = {
        n_train: plt.rcParams["axes.prop_cycle"].by_key()["color"][idx % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])]
        for idx, n_train in enumerate(n_trains)
    }

    def metric_group(metric: str, log_y: bool, n_train: int) -> pd.DataFrame:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        group = summary[summary["n_train"] == n_train].sort_values(x_col).copy()
        group = group[group[x_col] > 0]
        if log_y:
            group = group[group[mean_col] > 0]
        return group.dropna(subset=[x_col, mean_col])

    def plot_metric_curve(
        ax: plt.Axes,
        group: pd.DataFrame,
        metric: str,
        n_train: int,
        label: str,
        linestyle: str = "-",
        marker: str = "o",
        band: bool = True,
    ) -> None:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        x = group[x_col].to_numpy(dtype=float)
        y = group[mean_col].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            color=colors[n_train],
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=4.0,
            label=label,
        )

        if not band:
            return

        y_std = group[std_col].fillna(0.0).to_numpy(dtype=float)
        if not np.any(y_std > 0):
            return

        lower = y - y_std
        upper = y + y_std
        if ax.get_yscale() == "log":
            lower = np.clip(lower, np.finfo(float).tiny, None)
            upper = np.clip(upper, np.finfo(float).tiny, None)
        ax.fill_between(x, lower, upper, color=colors[n_train], alpha=0.14)

    def style_x_axis(ax: plt.Axes) -> None:
        ax.set_xlabel(TAU_LABEL if x_col == "tau" else "training step")
        if x_col == "tau":
            apply_tau_axis_for_triptych(ax, summary[x_col])
        else:
            ax.set_xscale("log")

    def style_y_axis(ax: plt.Axes, ylabel: str, log_y: bool) -> None:
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")

    fig, axes = plt.subplots(2, 3, figsize=(18.8, 9.6), sharex=False)
    top_axes = axes[0]
    pair_axes = axes[1]

    for ax, (metric, ylabel, log_y) in zip(top_axes, plot_specs):
        style_y_axis(ax, ylabel, log_y)
        for n_train in n_trains:
            group = metric_group(metric, log_y, n_train)
            if group.empty:
                continue

            seed_text = ""
            if "n_seeds" in group.columns:
                seed_text = f", seeds={int(group['n_seeds'].max())}"
            plot_metric_curve(ax, group, metric, n_train, f"N={int(n_train)}{seed_text}")

        style_x_axis(ax)
        ax.grid(True, alpha=0.3)
        ax.legend(title="n_train", fontsize=7, loc="best")

    top_axes[0].set_title(r"$F_{\mathrm{mem}}(\tau)$")
    top_axes[1].set_title("test loss mean")
    top_axes[2].set_title("FID")

    metric_pairs = [
        (plot_specs[0], plot_specs[1]),
        (plot_specs[0], plot_specs[2]),
        (plot_specs[1], plot_specs[2]),
    ]
    for ax_left, (left_spec, right_spec) in zip(pair_axes, metric_pairs):
        left_metric, left_ylabel, left_log_y = left_spec
        right_metric, right_ylabel, right_log_y = right_spec
        if left_metric == FID_COL:
            left_log_y = False
        if right_metric == FID_COL:
            right_log_y = False
        ax_right = ax_left.twinx()

        style_y_axis(ax_left, left_ylabel, left_log_y)
        style_y_axis(ax_right, right_ylabel, right_log_y)
        ax_left.yaxis.labelpad = 8
        ax_right.yaxis.labelpad = 12
        ax_left.tick_params(axis="y", which="both", left=True, labelleft=True)
        ax_right.tick_params(axis="y", which="both", right=True, labelright=True)

        for n_train in n_trains:
            left_group = metric_group(left_metric, left_log_y, n_train)
            if not left_group.empty:
                plot_metric_curve(
                    ax_left,
                    left_group,
                    left_metric,
                    n_train,
                    "_nolegend_",
                    linestyle="-",
                    marker="o",
                    band=False,
                )

            right_group = metric_group(right_metric, right_log_y, n_train)
            if not right_group.empty:
                plot_metric_curve(
                    ax_right,
                    right_group,
                    right_metric,
                    n_train,
                    "_nolegend_",
                    linestyle="--",
                    marker="s",
                    band=False,
                )

        style_x_axis(ax_left)
        ax_left.grid(True, alpha=0.3)
        ax_left.set_title(f"{left_ylabel} + {right_ylabel}")

        n_train_handles = [
            Line2D([0], [0], color=colors[n_train], marker="o", linewidth=2.0, label=f"N={int(n_train)}")
            for n_train in n_trains
        ]
        metric_handles = [
            Line2D([0], [0], color="black", marker="o", linewidth=2.0, linestyle="-", label=f"{left_ylabel} (left)"),
            Line2D([0], [0], color="black", marker="s", linewidth=2.0, linestyle="--", label=f"{right_ylabel} (right)"),
        ]

        n_train_legend = ax_left.legend(
            handles=n_train_handles,
            title="n_train",
            fontsize=6,
            title_fontsize=7,
            loc="upper left",
        )
        ax_left.add_artist(n_train_legend)
        ax_left.legend(
            handles=metric_handles,
            fontsize=6,
            loc="upper right",
        )

    fig.suptitle(f"09 focused n_train sweep | param={param_label} | seed averaged", y=1.01)
    fig.subplots_adjust(hspace=0.42, wspace=0.52)
    save_figure(fig, out_dir, "09_n_train_sweep_tau_triptych_fmem_testloss_fid", show=show)


def fig_02c_n_train_sweep_all_parameterizations(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    requested_n_trains: list[int],
    show: bool,
) -> None:
    """Plot-09 duplicate with parameterization emphasized.

    Color encodes parameterization. Marker and line style encode n_train.
    Curves are seed-averaged for each parameterization/n_train/tau cell.
    """
    if "n_train" not in metrics.columns or "param_label" not in metrics.columns:
        return

    loss_col = first_existing(
        metrics.columns,
        [
            "test_fixed_t_loss_mean_10k",
            "test_random_t_loss_10k",
            "test_loss_0_5_10k",
            "test_loss_0_75_10k",
        ],
    )
    plot_specs = [
        (FMEM_COL, display_label(FMEM_COL), False),
        (loss_col, "test loss mean", True),
        (FID_COL, display_label(FID_COL), True),
    ]
    plot_specs = [(metric, ylabel, log_y) for metric, ylabel, log_y in plot_specs if metric is not None]
    missing = [metric for metric, _, _ in plot_specs if metric not in metrics.columns]
    if missing or len(plot_specs) < 3:
        print(f"[warn] all-parameterization n_train sweep skipped; missing columns: {missing}")
        return

    n_trains = nearest_n_train_values(metrics, requested_n_trains)
    if not n_trains:
        print("[warn] all-parameterization n_train sweep skipped; no visible n_train values")
        return

    x_col = choose_tau_plot_col(metrics, step_col)
    seed_col = choose_seed_col(metrics)
    run_col = first_existing(metrics.columns, ["manifest_index", "run_id", "metrics_file"])

    sub = metrics.copy()
    sub["_n_train_plot"] = pd.to_numeric(sub["n_train"], errors="coerce")
    sub = sub[sub["_n_train_plot"].isin(n_trains)].copy()
    sub["_n_train_plot"] = sub["_n_train_plot"].astype(int)

    numeric_cols = [x_col] + [metric for metric, _, _ in plot_specs]
    for column in numeric_cols:
        sub[column] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub.dropna(subset=["_n_train_plot", "param_label", x_col])
    if sub.empty:
        print("[warn] all-parameterization n_train sweep skipped; selected data has no finite x values")
        return

    agg_spec = {
        f"{metric}_mean": (metric, "mean")
        for metric, _, _ in plot_specs
    }
    agg_spec.update({
        f"{metric}_std": (metric, "std")
        for metric, _, _ in plot_specs
    })
    if run_col is not None:
        agg_spec["n_runs"] = (run_col, "nunique")
    else:
        agg_spec["n_rows"] = (x_col, "size")
    if seed_col is not None:
        agg_spec["n_seeds"] = (seed_col, "nunique")

    summary = (
        sub
        .groupby(["param_label", "_n_train_plot", x_col], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .rename(columns={"_n_train_plot": "n_train"})
        .sort_values(["param_label", "n_train", x_col])
    )
    save_table(summary, out_dir, "10_n_train_sweep_all_parameterizations_seed_average")

    params = sorted(summary["param_label"].dropna().unique(), key=str)
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    param_colors = {
        param: base_colors[idx % len(base_colors)]
        for idx, param in enumerate(params)
    }

    def metric_shade(param: object, side: str) -> tuple[float, float, float]:
        base = np.asarray(to_rgb(param_colors[param]))
        if side == "left":
            return tuple(0.72 * base)
        if side == "right":
            return tuple(base + (1.0 - base) * 0.42)
        return tuple(base)

    markers = ["o", "s", "^", "D", "P", "X"]
    n_markers = {
        n_train: markers[idx % len(markers)]
        for idx, n_train in enumerate(n_trains)
    }
    linestyles = ["-", "--", ":", "-."]
    n_linestyles = {
        n_train: linestyles[idx % len(linestyles)]
        for idx, n_train in enumerate(n_trains)
    }

    def metric_group(metric: str, log_y: bool, param: object, n_train: int) -> pd.DataFrame:
        mean_col = f"{metric}_mean"
        group = summary[
            (summary["param_label"] == param)
            & (summary["n_train"] == n_train)
        ].sort_values(x_col).copy()
        group = group[group[x_col] > 0]
        if log_y:
            group = group[group[mean_col] > 0]
        return group.dropna(subset=[x_col, mean_col])

    def plot_curve(
        ax: plt.Axes,
        group: pd.DataFrame,
        metric: str,
        param: object,
        n_train: int,
        marker_fill: str = "filled",
    ) -> None:
        mean_col = f"{metric}_mean"
        markerfacecolor = param_colors[param] if marker_fill == "filled" else "none"
        ax.plot(
            group[x_col].to_numpy(dtype=float),
            group[mean_col].to_numpy(dtype=float),
            color=param_colors[param],
            marker=n_markers[n_train],
            linestyle=n_linestyles[n_train],
            linewidth=1.8,
            markersize=4.0,
            markerfacecolor=markerfacecolor,
            markeredgecolor=param_colors[param],
            markeredgewidth=1.1,
            alpha=0.95,
            label="_nolegend_",
        )

    def style_x_axis(ax: plt.Axes) -> None:
        ax.set_xlabel(TAU_LABEL if x_col == "tau" else "training step")
        if x_col == "tau":
            apply_tau_axis_for_triptych(ax, summary[x_col])
        else:
            ax.set_xscale("log")

    def style_y_axis(ax: plt.Axes, ylabel: str, log_y: bool) -> None:
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")

    def add_factor_legends(
        ax: plt.Axes,
        include_metric_styles: tuple[str, str] | None = None,
    ) -> None:
        param_handles = [
            Line2D([0], [0], color=param_colors[param], marker="o", linewidth=2.0, label=str(param))
            for param in params
        ]
        n_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                marker=n_markers[n_train],
                linestyle=n_linestyles[n_train],
                linewidth=1.8,
                label=f"N={int(n_train)}",
            )
            for n_train in n_trains
        ]

        param_legend = ax.legend(
            handles=param_handles,
            title="param",
            fontsize=6,
            title_fontsize=7,
            loc="upper left",
        )
        ax.add_artist(param_legend)

        n_legend = ax.legend(
            handles=n_handles,
            title="n_train",
            fontsize=6,
            title_fontsize=7,
            loc="lower left",
        )
        ax.add_artist(n_legend)

        if include_metric_styles is not None:
            left_ylabel, right_ylabel = include_metric_styles
            metric_handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="o",
                    markerfacecolor="black",
                    linewidth=1.8,
                    linestyle="-",
                    label=f"{left_ylabel} (left)",
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markeredgewidth=1.1,
                    linewidth=1.8,
                    linestyle="-",
                    label=f"{right_ylabel} (right)",
                ),
            ]
            ax.legend(handles=metric_handles, fontsize=6, loc="upper right")

    fig, axes = plt.subplots(2, 3, figsize=(19.2, 9.8), sharex=False)
    top_axes = axes[0]
    pair_axes = axes[1]

    for ax, (metric, ylabel, log_y) in zip(top_axes, plot_specs):
        style_y_axis(ax, ylabel, log_y)
        for param in params:
            for n_train in n_trains:
                group = metric_group(metric, log_y, param, n_train)
                if group.empty:
                    continue
                plot_curve(ax, group, metric, param, n_train)

        style_x_axis(ax)
        ax.grid(True, alpha=0.3)
        add_factor_legends(ax)

    top_axes[0].set_title(r"$F_{\mathrm{mem}}(\tau)$")
    top_axes[1].set_title("test loss mean")
    top_axes[2].set_title("FID")

    metric_pairs = [
        (plot_specs[0], plot_specs[1]),
        (plot_specs[0], plot_specs[2]),
        (plot_specs[1], plot_specs[2]),
    ]
    for ax_left, (left_spec, right_spec) in zip(pair_axes, metric_pairs):
        left_metric, left_ylabel, left_log_y = left_spec
        right_metric, right_ylabel, right_log_y = right_spec
        if left_metric == FID_COL:
            left_log_y = False
        if right_metric == FID_COL:
            right_log_y = False

        ax_right = ax_left.twinx()
        style_y_axis(ax_left, left_ylabel, left_log_y)
        style_y_axis(ax_right, right_ylabel, right_log_y)
        ax_left.yaxis.labelpad = 8
        ax_right.yaxis.labelpad = 12

        for param in params:
            for n_train in n_trains:
                left_group = metric_group(left_metric, left_log_y, param, n_train)
                if not left_group.empty:
                    plot_curve(
                        ax_left,
                        left_group,
                        left_metric,
                        param,
                        n_train,
                        marker_fill="filled",
                    )

                right_group = metric_group(right_metric, right_log_y, param, n_train)
                if not right_group.empty:
                    plot_curve(
                        ax_right,
                        right_group,
                        right_metric,
                        param,
                        n_train,
                        marker_fill="hollow",
                    )

        style_x_axis(ax_left)
        ax_left.grid(True, alpha=0.3)
        ax_left.set_title(f"{left_ylabel} + {right_ylabel}")
        add_factor_legends(ax_left, include_metric_styles=(left_ylabel, right_ylabel))

    fig.suptitle("10 focused n_train sweep | all parameterizations | seed averaged", y=1.01)
    fig.subplots_adjust(hspace=0.42, wspace=0.52)
    save_figure(fig, out_dir, "10_n_train_sweep_all_parameterizations_fmem_testloss_fid", show=show)


def fig_02d_parameterization_pair_grid_by_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    requested_n_trains: list[int],
    show: bool,
) -> None:
    """Pairwise metric overlays, faceted by n_train rows.

    This is a less cluttered parameterization-focused duplicate of plot 10:
    rows are all visible n_train values, columns are the three metric pairs
    from plot 09's bottom row, and curves inside each panel are
    parameterizations. When FID appears in a pair it is always placed on the
    left y-axis, matching the Bonnaire-style convention.
    """
    if "n_train" not in metrics.columns or "param_label" not in metrics.columns:
        return

    loss_col = first_existing(
        metrics.columns,
        [
            "test_fixed_t_loss_mean_10k",
            "test_random_t_loss_10k",
            "test_loss_0_5_10k",
            "test_loss_0_75_10k",
        ],
    )
    plot_specs = [
        (FMEM_COL, display_label(FMEM_COL), False),
        (loss_col, "test loss mean", True),
        (FID_COL, display_label(FID_COL), True),
    ]
    plot_specs = [(metric, ylabel, log_y) for metric, ylabel, log_y in plot_specs if metric is not None]
    missing = [metric for metric, _, _ in plot_specs if metric not in metrics.columns]
    if missing or len(plot_specs) < 3:
        print(f"[warn] n_train-row pair grid skipped; missing columns: {missing}")
        return

    _ = requested_n_trains
    n_trains = n_train_ticks_from(metrics)
    if not n_trains:
        print("[warn] n_train-row pair grid skipped; no visible n_train values")
        return

    x_col = choose_tau_plot_col(metrics, step_col)
    seed_col = choose_seed_col(metrics)
    run_col = first_existing(metrics.columns, ["manifest_index", "run_id", "metrics_file"])

    sub = metrics.copy()
    sub["_n_train_plot"] = pd.to_numeric(sub["n_train"], errors="coerce")
    sub = sub[sub["_n_train_plot"].isin(n_trains)].copy()
    sub["_n_train_plot"] = sub["_n_train_plot"].astype(int)

    numeric_cols = [x_col] + [metric for metric, _, _ in plot_specs]
    for column in numeric_cols:
        sub[column] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub.dropna(subset=["_n_train_plot", "param_label", x_col])
    if sub.empty:
        print("[warn] n_train-row pair grid skipped; selected data has no finite x values")
        return

    agg_spec = {
        f"{metric}_mean": (metric, "mean")
        for metric, _, _ in plot_specs
    }
    agg_spec.update({
        f"{metric}_std": (metric, "std")
        for metric, _, _ in plot_specs
    })
    if run_col is not None:
        agg_spec["n_runs"] = (run_col, "nunique")
    else:
        agg_spec["n_rows"] = (x_col, "size")
    if seed_col is not None:
        agg_spec["n_seeds"] = (seed_col, "nunique")

    summary = (
        sub
        .groupby(["param_label", "_n_train_plot", x_col], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .rename(columns={"_n_train_plot": "n_train"})
        .sort_values(["n_train", "param_label", x_col])
    )
    save_table(summary, out_dir, "11_parameterization_pair_grid_by_n_train_seed_average")

    params = sorted(summary["param_label"].dropna().unique(), key=str)
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    param_colors = {
        param: base_colors[idx % len(base_colors)]
        for idx, param in enumerate(params)
    }

    def metric_shade(param: object, side: str) -> tuple[float, float, float]:
        base = np.asarray(to_rgb(param_colors[param]))
        if side == "left":
            return tuple(0.72 * base)
        if side == "right":
            return tuple(base + (1.0 - base) * 0.42)
        return tuple(base)

    def fid_on_left(left_spec: tuple[str, str, bool], right_spec: tuple[str, str, bool]) -> tuple[
        tuple[str, str, bool],
        tuple[str, str, bool],
    ]:
        if right_spec[0] == FID_COL and left_spec[0] != FID_COL:
            return right_spec, left_spec
        return left_spec, right_spec

    metric_pairs = [
        fid_on_left(plot_specs[0], plot_specs[1]),
        fid_on_left(plot_specs[0], plot_specs[2]),
        fid_on_left(plot_specs[1], plot_specs[2]),
    ]

    def metric_group(metric: str, log_y: bool, param: object, n_train: int) -> pd.DataFrame:
        mean_col = f"{metric}_mean"
        group = summary[
            (summary["param_label"] == param)
            & (summary["n_train"] == n_train)
        ].sort_values(x_col).copy()
        group = group[group[x_col] > 0]
        if log_y:
            group = group[group[mean_col] > 0]
        return group.dropna(subset=[x_col, mean_col])

    def plot_curve(
        ax: plt.Axes,
        group: pd.DataFrame,
        metric: str,
        param: object,
        linestyle: str,
        marker_fill: str,
        side: str,
    ) -> None:
        mean_col = f"{metric}_mean"
        line_color = metric_shade(param, side)
        markerfacecolor = line_color if marker_fill == "filled" else "none"
        ax.plot(
            group[x_col].to_numpy(dtype=float),
            group[mean_col].to_numpy(dtype=float),
            color=line_color,
            marker="o",
            linestyle=linestyle,
            linewidth=1.9,
            markersize=4.2,
            markerfacecolor=markerfacecolor,
            markeredgecolor=line_color,
            markeredgewidth=1.1,
            alpha=0.95,
            label="_nolegend_",
        )

    def style_x_axis(ax: plt.Axes) -> None:
        ax.set_xlabel(TAU_LABEL if x_col == "tau" else "training step")
        if x_col == "tau":
            apply_tau_axis_for_triptych(ax, summary[x_col])
        else:
            ax.set_xscale("log")

    def style_y_axis(ax: plt.Axes, ylabel: str, log_y: bool) -> None:
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")

    param_handles = [
        Line2D([0], [0], color=param_colors[param], marker="o", linewidth=2.0, label=str(param))
        for param in params
    ]

    nrows = len(n_trains)
    ncols = len(metric_pairs)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5 * ncols, 3.9 * nrows),
        sharex=False,
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    for row_idx, n_train in enumerate(n_trains):
        for col_idx, (left_spec, right_spec) in enumerate(metric_pairs):
            ax_left = axes[row_idx, col_idx]
            left_metric, left_ylabel, left_log_y = left_spec
            right_metric, right_ylabel, right_log_y = right_spec
            if left_metric == FID_COL:
                left_log_y = False
            if right_metric == FID_COL:
                right_log_y = False

            ax_right = ax_left.twinx()
            ax_left.spines["right"].set_visible(False)
            ax_right.spines["left"].set_visible(False)
            style_y_axis(ax_left, left_ylabel, left_log_y)
            style_y_axis(ax_right, right_ylabel, right_log_y)
            ax_left.yaxis.labelpad = 8
            ax_right.yaxis.labelpad = 12

            for param in params:
                left_group = metric_group(left_metric, left_log_y, param, n_train)
                if not left_group.empty:
                    plot_curve(
                        ax_left,
                        left_group,
                        left_metric,
                        param,
                        linestyle=":",
                        marker_fill="filled",
                        side="left",
                    )

                right_group = metric_group(right_metric, right_log_y, param, n_train)
                if not right_group.empty:
                    plot_curve(
                        ax_right,
                        right_group,
                        right_metric,
                        param,
                        linestyle="-",
                        marker_fill="hollow",
                        side="right",
                    )

            style_x_axis(ax_left)
            ax_left.grid(True, alpha=0.3)

            if row_idx == 0:
                ax_left.set_title(f"left: {left_ylabel} | right: {right_ylabel}")
            if col_idx == 0:
                ax_left.text(
                    -0.24,
                    0.5,
                    f"N={int(n_train)}",
                    transform=ax_left.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )

    fig.legend(
        handles=param_handles,
        title="param color",
        fontsize=8,
        title_fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.982),
        ncol=max(1, len(param_handles)),
        frameon=False,
    )
    fig.suptitle("11 parameterization metric-pair overlays by n_train | seed averaged", y=0.997)
    fig.subplots_adjust(top=0.955, hspace=0.42, wspace=0.52)
    save_figure(fig, out_dir, "11_parameterization_pair_grid_by_n_train", show=show)


# ============================================================
# 07b. Figure: F_mem threshold sweep along training by n_train
# ============================================================

def fig_07b_fmem_thresholds_by_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    show: bool,
) -> None:
    """Figure-02-style overview for F_mem thresholds.

    Each subplot is one nearest-neighbor ratio threshold. Curves are
    seed/run-averaged over currently visible eval outputs for each n_train.
    Zero-valued points are omitted because these panels use log-y scale.
    """
    if "n_train" not in metrics.columns:
        return

    rho_cols = [(label, col) for label, col in FMEM_RHO_COLS if col in metrics.columns]
    if not rho_cols:
        return

    agg = (
        metrics
        .groupby(["n_train", step_col], dropna=False)
        .agg(
            **{f"{col}_mean": (col, "mean") for _, col in rho_cols},
            n_runs=("manifest_index", "nunique"),
        )
        .reset_index()
    )

    n_trains = sorted(agg["n_train"].dropna().unique())
    fig, axes = plt.subplots(1, len(rho_cols), figsize=(5.4 * len(rho_cols), 4.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (rho_label, col) in zip(axes, rho_cols):
        mean_col = f"{col}_mean"
        for n_train in n_trains:
            sub = agg[agg["n_train"] == n_train].sort_values(step_col)
            sub = sub[(sub[step_col] > 0) & (sub[mean_col] > 0)]
            if sub.empty:
                continue
            ax.plot(sub[step_col], sub[mean_col], marker="o", linewidth=1.6, label=f"N={int(n_train)}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(TAU_LABEL)
        ax.set_ylabel(display_label(col))
        ax.set_title(rho_label)
        ax.grid(True, alpha=0.3)
        ax.legend(title="n_train", fontsize=8)

    fig.suptitle(r"07b $F_{\mathrm{mem}}$ threshold sweep along training by n_train", y=1.03)
    save_figure(fig, out_dir, "07b_fmem_thresholds_along_training_by_n_train", show=show)


# ============================================================
# 08. Figure: final FID vs final F_mem, faceted by n_train
# ============================================================

def fig_03_final_fid_vs_fmem_faceted_by_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    show: bool,
) -> None:
    """Final checkpoint diagnostic.

    This is useful for debugging overtraining, but the paper objective prefers
    F_mem@FID_best.
    """
    if "n_train" not in metrics.columns:
        return

    final_df = final_rows(metrics, step_col)
    n_trains = sorted(final_df["n_train"].dropna().unique())
    if not n_trains:
        return

    ncols = min(3, len(n_trains))
    nrows = math.ceil(len(n_trains) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.7 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    seed_col = choose_seed_col(final_df)

    for ax, n_train in zip(axes, n_trains):
        sub = final_df[final_df["n_train"] == n_train]
        if seed_col is not None:
            ax.scatter(sub[FID_COL], sub[FMEM_COL], c=sub[seed_col], cmap="tab10", s=55, alpha=0.85)
        else:
            ax.scatter(sub[FID_COL], sub[FMEM_COL], s=55, alpha=0.85)
        ax.set_title(f"N={n_train} | runs={len(sub)}")
        ax.set_xlabel(f"final {FID_COL}")
        ax.set_ylabel(r"final $F_{\mathrm{mem}}$")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(n_trains):]:
        ax.axis("off")

    fig.suptitle(r"03 final checkpoint FID vs $F_{\mathrm{mem}}$ by n_train", y=1.01)
    save_figure(fig, out_dir, "03_final_fid_vs_fmem_by_n_train", show=show)


# ============================================================
# 09. Figure: F_mem@FID_best vs n_train, split by parameterization
# ============================================================

def fig_04_fmem_at_best_fid_vs_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    show: bool,
) -> None:
    """Main N-sweep paper-objective figure for current legacy data."""
    if "n_train" not in metrics.columns:
        return

    best = best_fid_rows(metrics)
    n_train_ticks = n_train_ticks_from(best)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for label, sub in best.groupby("param_label", dropna=False):
        # Light points show individual seeds/runs.
        ax.scatter(sub["n_train"], sub[FMEM_COL], s=55, alpha=0.45, label=f"{label} runs")

        # Heavy line shows seed/run mean at each n_train.
        mean = (
            sub
            .groupby("n_train", as_index=False)
            .agg(fmem_mean=(FMEM_COL, "mean"), fmem_std=(FMEM_COL, "std"), n_runs=("manifest_index", "nunique"))
        )
        ax.errorbar(
            mean["n_train"],
            mean["fmem_mean"],
            yerr=mean["fmem_std"],
            marker="o",
            linewidth=2.0,
            capsize=3,
            label=f"{label} mean",
        )

    apply_n_train_log_ticks(ax, n_train_ticks)
    ax.set_xlabel("n_train")
    ax.set_ylabel(r"$F_{\mathrm{mem}}$ at best FID")
    ax.set_title(r"04 quality-controlled memorization: $F_{\mathrm{mem}}$@FID_best vs n_train")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, out_dir, "04_fmem_at_best_fid_vs_n_train_by_parameterization", show=show)


# ============================================================
# 10. Figure: Pareto diagnostic, best FID vs F_mem@FID_best
# ============================================================

def fig_05_pareto_best_fid_vs_fmem(
    metrics: pd.DataFrame,
    out_dir: Path,
    show: bool,
) -> None:
    """Pareto-style diagnostic: lower-left is better."""
    if "n_train" not in metrics.columns:
        return

    best = best_fid_rows(metrics)
    n_train_ticks = n_train_ticks_from(best)

    fig, ax = plt.subplots(figsize=(7.3, 5.8))
    scatter = ax.scatter(
        best[FID_COL],
        best[FMEM_COL],
        c=best["n_train"],
        cmap="viridis",
        s=70,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.35,
    )

    ax.set_xlabel("best FID 10k")
    ax.set_ylabel(r"$F_{\mathrm{mem}}$ at best FID")
    ax.set_title(r"05 Pareto diagnostic: best quality vs $F_{\mathrm{mem}}$ at best quality")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("n_train")
    save_figure(fig, out_dir, "05_pareto_best_fid_vs_fmem_at_best_fid", show=show)


# ============================================================
# 11. Figure: fixed seed/n_train parameterization trajectories
# ============================================================

def pick_fixed_seed_n_train(metrics: pd.DataFrame, seed: int | None, n_train: int | None) -> tuple[int | None, int]:
    """Choose a fixed seed and n_train for parameterization trajectory comparison."""
    seed_col = choose_seed_col(metrics)

    if n_train is None:
        # Prefer the n_train with the most parameterization coverage.
        coverage = (
            metrics
            .groupby("n_train")
            .agg(n_params=("param_label", "nunique"), n_runs=("manifest_index", "nunique"))
            .reset_index()
            .sort_values(["n_params", "n_runs", "n_train"], ascending=[False, False, True])
        )
        n_train = int(coverage["n_train"].iloc[0])

    if seed is None and seed_col is not None:
        sub = metrics[metrics["n_train"] == n_train]
        coverage = (
            sub
            .groupby(seed_col)
            .agg(n_params=("param_label", "nunique"), n_runs=("manifest_index", "nunique"))
            .reset_index()
            .sort_values(["n_params", "n_runs", seed_col], ascending=[False, False, True])
        )
        seed = int(coverage[seed_col].iloc[0])

    return seed, n_train


def fig_06_parameterization_trajectories_fixed_seed_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    seed: int | None,
    n_train: int | None,
    show: bool,
) -> None:
    """Fixed seed and n_train: each trajectory is one parameterization.

    This directly targets the missing comparison:
        Does parameterization change the FID(tau), F_mem(tau), and
        FID-vs-F_mem trajectory?
    """
    if "n_train" not in metrics.columns:
        return

    seed_col = choose_seed_col(metrics)
    seed, n_train = pick_fixed_seed_n_train(metrics, seed, n_train)

    sub = metrics[metrics["n_train"] == n_train].copy()
    if seed_col is not None and seed is not None:
        sub = sub[sub[seed_col] == seed].copy()

    if sub.empty:
        print("[warn] no data for requested fixed seed/n_train trajectory figure")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    for label, group in sub.groupby("param_label", dropna=False):
        group = group.sort_values(step_col)
        axes[0].plot(group[step_col], group[FID_COL], marker="o", linewidth=1.8, label=str(label))
        axes[1].plot(group[step_col], group[FMEM_COL], marker="o", linewidth=1.8, label=str(label))
        axes[2].plot(group[FID_COL], group[FMEM_COL], marker="o", linewidth=1.8, label=str(label))

        # Mark the best-FID checkpoint on the phase plot.
        best_idx = group[FID_COL].idxmin()
        best_row = group.loc[best_idx]
        axes[2].scatter(best_row[FID_COL], best_row[FMEM_COL], s=120, edgecolor="black", linewidth=1.0)

    axes[0].set_xlabel(TAU_LABEL)
    axes[0].set_ylabel(FID_COL)
    axes[0].set_title("FID(tau)")
    axes[0].set_xscale("log")

    axes[1].set_xlabel(TAU_LABEL)
    axes[1].set_ylabel(display_label(FMEM_COL))
    axes[1].set_title(r"$F_{\mathrm{mem}}(\tau)$")
    axes[1].set_xscale("log")

    axes[2].set_xlabel(FID_COL)
    axes[2].set_ylabel(display_label(FMEM_COL))
    axes[2].set_title(r"FID vs $F_{\mathrm{mem}}$ trajectory")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    title = f"06 parameterization trajectories | n_train={n_train}"
    if seed_col is not None and seed is not None:
        title += f" | {seed_col}={seed}"
    fig.suptitle(title, y=1.02)
    save_figure(fig, out_dir, "06_parameterization_trajectories_fixed_seed_n_train", show=show)


# ============================================================
# 12. Figure: seed-averaged parameterization trajectories at fixed n_train
# ============================================================

def pick_n_train_for_param_coverage(metrics: pd.DataFrame, n_train: int | None) -> int:
    """Choose an n_train with strong parameterization coverage."""
    if n_train is not None:
        return int(n_train)

    coverage = (
        metrics
        .groupby("n_train")
        .agg(n_params=("param_label", "nunique"), n_runs=("manifest_index", "nunique"))
        .reset_index()
        .sort_values(["n_params", "n_runs", "n_train"], ascending=[False, False, True])
    )
    return int(coverage["n_train"].iloc[0])


def fig_07_seed_averaged_parameterization_trajectories_fixed_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    n_train: int | None,
    show: bool,
) -> None:
    """Fixed n_train dashboard: parameterization trajectories averaged over seeds/runs.

    This is the main paper-objective prototype figure for a single dataset
    size. It asks whether parameterization changes not only sample quality, but
    the full generation-to-memorization dynamics:
        - FID(tau)
        - F_mem(tau)
        - FID-vs-F_mem trajectory, linear axes
        - FID-vs-F_mem trajectory, log-log axes
        - held-out test-loss trajectories, where available
    """
    if "n_train" not in metrics.columns:
        return

    n_train = pick_n_train_for_param_coverage(metrics, n_train)
    sub = metrics[metrics["n_train"] == n_train].copy()
    if sub.empty:
        return

    loss_cols = [c for c in LOSS_COLS if c in sub.columns]

    summary = (
        sub
        .groupby(["param_label", step_col], dropna=False)
        .agg(
            fid_mean=(FID_COL, "mean"),
            fid_std=(FID_COL, "std"),
            fmem_mean=(FMEM_COL, "mean"),
            fmem_std=(FMEM_COL, "std"),
            **{f"{col}_mean": (col, "mean") for col in loss_cols},
            **{f"{col}_std": (col, "std") for col in loss_cols},
            n_runs=("manifest_index", "nunique"),
        )
        .reset_index()
        .sort_values(["param_label", step_col])
    )

    save_table(
        summary,
        out_dir,
        "07_seed_averaged_parameterization_trajectories_fixed_n_train",
    )

    base_panels = 4
    panel_count = base_panels + len(loss_cols)
    ncols = 4
    nrows = math.ceil(panel_count / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 4.8 * nrows),
    )
    axes = np.array(axes).reshape(-1)

    def plot_with_band(
        ax: plt.Axes,
        group: pd.DataFrame,
        x_col: str,
        mean_col: str,
        std_col: str,
        label: object,
        log_x: bool,
        log_y: bool,
    ) -> None:
        """Plot one mean trajectory with a std band."""
        plot_group = group[[x_col, mean_col, std_col]].copy()

        if log_x:
            plot_group = plot_group[plot_group[x_col] > 0]
        if log_y:
            plot_group = plot_group[plot_group[mean_col] > 0]

        if plot_group.empty:
            return

        y = plot_group[mean_col]
        yerr = plot_group[std_col].fillna(0.0)
        lower = y - yerr
        upper = y + yerr

        # Log-y bands cannot include nonpositive values.
        if log_y:
            lower = lower.clip(lower=np.finfo(float).tiny)
            upper = upper.clip(lower=np.finfo(float).tiny)

        ax.plot(plot_group[x_col], y, marker="o", linewidth=2.0, label=str(label))
        ax.fill_between(plot_group[x_col], lower, upper, alpha=0.15)

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

    def plot_directed_phase(ax: plt.Axes, group: pd.DataFrame, label: object, log_axes: bool) -> None:
        """Plot F_mem-vs-FID trajectory and add arrows for training direction."""
        phase = group[["fid_mean", "fmem_mean", step_col]].copy()

        if log_axes:
            phase = phase[(phase["fid_mean"] > 0) & (phase["fmem_mean"] > 0)]

        if phase.empty:
            return

        # Draw a faint connecting line first. The returned color anchors the
        # per-parameterization light-to-dark time gradient.
        line = ax.plot(
            phase["fmem_mean"],
            phase["fid_mean"],
            linewidth=1.5,
            alpha=0.35,
            label=str(label),
        )[0]
        base_rgb = np.asarray(to_rgb(line.get_color()))
        light_rgb = 0.82 + 0.18 * base_rgb
        dark_rgb = 0.35 * base_rgb
        weights = np.linspace(0.0, 1.0, len(phase))
        point_colors = [
            tuple((1.0 - weight) * light_rgb + weight * dark_rgb)
            for weight in weights
        ]

        # Dots encode passage of training time: light early, dark late.
        ax.scatter(
            phase["fmem_mean"],
            phase["fid_mean"],
            s=58,
            color=point_colors,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

        # Add sparse arrows from earlier checkpoints to later checkpoints.
        # Arrows are drawn after dots so directionality sits on top.
        arrow_stride = max(1, len(phase) // 6)
        for i in range(0, len(phase) - 1, arrow_stride):
            start = phase.iloc[i]
            end = phase.iloc[i + 1]
            ax.annotate(
                "",
                xy=(end["fmem_mean"], end["fid_mean"]),
                xytext=(start["fmem_mean"], start["fid_mean"]),
                arrowprops=dict(arrowstyle="->", lw=1.2, alpha=0.85, color=dark_rgb),
                zorder=4,
            )

        # Mark the mean best-FID checkpoint on the phase trajectory.
        best_idx = phase["fid_mean"].idxmin()
        best_row = phase.loc[best_idx]
        ax.scatter(
            best_row["fmem_mean"],
            best_row["fid_mean"],
            s=140,
            facecolor="none",
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )

        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")

    for label, group in summary.groupby("param_label", dropna=False):
        group = group.sort_values(step_col)

        plot_with_band(
            axes[0],
            group,
            step_col,
            "fid_mean",
            "fid_std",
            label,
            log_x=True,
            log_y=False,
        )
        plot_with_band(
            axes[1],
            group,
            step_col,
            "fmem_mean",
            "fmem_std",
            label,
            log_x=True,
            log_y=False,
        )

        plot_directed_phase(axes[2], group, label, log_axes=False)
        plot_directed_phase(axes[3], group, label, log_axes=True)

        for loss_idx, loss_col in enumerate(loss_cols):
            plot_with_band(
                axes[base_panels + loss_idx],
                group,
                step_col,
                f"{loss_col}_mean",
                f"{loss_col}_std",
                label,
                log_x=True,
                log_y=True,
            )

    axes[0].set_xlabel(TAU_LABEL)
    axes[0].set_ylabel(f"{FID_COL} mean")
    axes[0].set_title("Seed-averaged FID(tau)")
    axes[0].set_xscale("log")

    axes[1].set_xlabel(TAU_LABEL)
    axes[1].set_ylabel(display_label("fmem_mean"))
    axes[1].set_title(r"Seed-averaged $F_{\mathrm{mem}}(\tau)$")
    axes[1].set_xscale("log")

    axes[2].set_xlabel(display_label("fmem_mean"))
    axes[2].set_ylabel(f"{FID_COL} mean")
    axes[2].set_title(r"Seed-averaged $F_{\mathrm{mem}}$ vs FID trajectory")

    axes[3].set_xlabel(display_label("fmem_mean"))
    axes[3].set_ylabel(f"{FID_COL} mean")
    axes[3].set_title(r"Seed-averaged $F_{\mathrm{mem}}$ vs FID trajectory, log-log")

    for loss_idx, loss_col in enumerate(loss_cols):
        ax = axes[base_panels + loss_idx]
        ax.set_xlabel(TAU_LABEL)
        ax.set_ylabel(f"{loss_col} mean")
        ax.set_title(loss_col)

    for ax in axes[:panel_count]:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[panel_count:]:
        ax.axis("off")

    fig.suptitle(f"07 seed-averaged parameterization trajectories | n_train={n_train}", y=1.02)
    fig.subplots_adjust(hspace=0.55, wspace=0.32)
    save_figure(
        fig,
        out_dir,
        "07_seed_averaged_parameterization_trajectories_fixed_n_train",
        show=show,
    )


# ============================================================
# 12b. Figure: parameterization trajectories by F_mem threshold
# ============================================================

def fig_07c_parameterization_fmem_thresholds_fixed_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    n_train: int | None,
    show: bool,
) -> None:
    """Figure-07-style parameterization comparison for F_mem thresholds.

    Fixed n_train. Each subplot is one rho threshold. Each curve is one
    parameterization averaged over visible seeds/runs.
    """
    if "n_train" not in metrics.columns:
        return

    rho_cols = [(label, col) for label, col in FMEM_RHO_COLS if col in metrics.columns]
    if not rho_cols:
        return

    n_train = pick_n_train_for_param_coverage(metrics, n_train)
    sub = metrics[metrics["n_train"] == n_train].copy()
    if sub.empty:
        return

    summary = (
        sub
        .groupby(["param_label", step_col], dropna=False)
        .agg(
            **{f"{col}_mean": (col, "mean") for _, col in rho_cols},
            n_runs=("manifest_index", "nunique"),
        )
        .reset_index()
        .sort_values(["param_label", step_col])
    )

    save_table(
        summary,
        out_dir,
        "07c_seed_averaged_parameterization_fmem_thresholds_fixed_n_train",
    )

    fig, axes = plt.subplots(1, len(rho_cols), figsize=(5.6 * len(rho_cols), 4.7), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (rho_label, col) in zip(axes, rho_cols):
        mean_col = f"{col}_mean"
        for label, group in summary.groupby("param_label", dropna=False):
            group = group.sort_values(step_col)
            group = group[(group[step_col] > 0) & (group[mean_col] > 0)]
            if group.empty:
                continue
            ax.plot(group[step_col], group[mean_col], marker="o", linewidth=2.0, label=str(label))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(TAU_LABEL)
        ax.set_ylabel(display_label(col))
        ax.set_title(rho_label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(rf"07c parameterization $F_{{\mathrm{{mem}}}}$ threshold sweep | n_train={n_train}", y=1.03)
    save_figure(
        fig,
        out_dir,
        "07c_parameterization_fmem_thresholds_fixed_n_train",
        show=show,
    )


# ============================================================
# 13. Figure: parameterization summary at best FID
# ============================================================

def fig_08_parameterization_best_fid_summary(
    metrics: pd.DataFrame,
    out_dir: Path,
    show: bool,
) -> None:
    """Compare parameterizations using best FID and F_mem@FID_best."""
    if "n_train" not in metrics.columns:
        return

    best = best_fid_rows(metrics)
    n_train_ticks = n_train_ticks_from(best)
    summary = (
        best
        .groupby(["param_label", "n_train"], dropna=False)
        .agg(
            best_fid_mean=(FID_COL, "mean"),
            best_fid_std=(FID_COL, "std"),
            fmem_at_best_fid_mean=(FMEM_COL, "mean"),
            fmem_at_best_fid_std=(FMEM_COL, "std"),
            n_runs=("manifest_index", "nunique"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True)

    for label, sub in summary.groupby("param_label", dropna=False):
        sub = sub.sort_values("n_train")
        axes[0].errorbar(
            sub["n_train"],
            sub["best_fid_mean"],
            yerr=sub["best_fid_std"],
            marker="o",
            linewidth=2.0,
            capsize=3,
            label=str(label),
        )
        axes[1].errorbar(
            sub["n_train"],
            sub["fmem_at_best_fid_mean"],
            yerr=sub["fmem_at_best_fid_std"],
            marker="o",
            linewidth=2.0,
            capsize=3,
            label=str(label),
        )

    axes[0].set_ylabel("best FID")
    axes[0].set_title("Best sample quality")
    axes[1].set_ylabel(r"$F_{\mathrm{mem}}$ at best FID")
    axes[1].set_title("Memorization at best sample quality")

    for ax in axes:
        apply_n_train_log_ticks(ax, n_train_ticks)
        ax.set_xlabel("n_train")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(r"08 parameterization summary: quality optimum vs $F_{\mathrm{mem}}$ at quality optimum", y=1.02)
    save_figure(fig, out_dir, "08_parameterization_best_fid_summary", show=show)


# ============================================================
# 14. CLI and main
# ============================================================

def parse_requested_steps(text: str) -> list[int]:
    """Parse comma-separated checkpoint requests."""
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--job-set-root", type=Path, default=DEFAULT_JOB_SET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--requested-steps", type=str, default=",".join(map(str, DEFAULT_REQUESTED_STEPS)))
    parser.add_argument(
        "--n-train-sweep-values",
        type=str,
        default=",".join(map(str, DEFAULT_N_TRAIN_SWEEP_VALUES)),
        help="Comma-separated n_train targets for the focused tau triptych; nearest visible values are used.",
    )
    parser.add_argument(
        "--n-train-sweep-param",
        type=str,
        default=DEFAULT_N_TRAIN_SWEEP_PARAM,
        help="Parameterization for the focused n_train sweep. Use 'all' to average all visible parameterizations.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo = args.repo
    eval_root = resolve_under_repo(repo, args.eval_root)
    job_set_root = resolve_under_repo(repo, args.job_set_root)
    out_dir = ensure_dir(resolve_under_repo(repo, args.out_dir))
    requested_steps = parse_requested_steps(args.requested_steps)
    requested_n_train_sweep_values = parse_requested_steps(args.n_train_sweep_values)

    print(f"repo: {repo}")
    print(f"eval_root: {eval_root}")
    print(f"job_set_root: {job_set_root}")
    print(f"out_dir: {out_dir}")

    metrics = load_checkpoint_metrics(repo, eval_root)
    if metrics.empty:
        raise RuntimeError("No checkpoint metrics loaded. Eval outputs may not be written yet.")

    manifests = load_manifest_candidates(eval_root, job_set_root)
    metrics = merge_manifest_metadata(metrics, manifests)
    metrics, param_col = add_param_label(metrics)
    metrics = apply_pretty_param_labels(metrics)

    step_col = choose_step_col(metrics)

    print(f"step_col: {step_col}")
    print(f"parameterization column for plots: {param_col}")
    print(f"visible runs: {metrics['manifest_index'].nunique() if 'manifest_index' in metrics.columns else 'unknown'}")
    print(f"visible n_train values: {sorted(metrics['n_train'].dropna().unique()) if 'n_train' in metrics.columns else 'missing'}")
    print(f"visible parameterizations: {sorted(metrics['param_label'].dropna().unique())}")

    write_01_coverage_tables(metrics, out_dir, step_col)
    manifest_index = write_02_single_run_stage_table(
        metrics,
        out_dir,
        step_col,
        requested_steps,
        args.manifest_index,
    )
    write_03_seed_average_tables(metrics, out_dir, step_col)

    fig_01_single_run_trajectories(metrics, out_dir, step_col, manifest_index, show=args.show)
    fig_02_metrics_by_n_train(metrics, out_dir, step_col, show=args.show)
    fig_02b_n_train_sweep_tau_triptych(
        metrics,
        out_dir,
        step_col,
        requested_n_trains=requested_n_train_sweep_values,
        param_filter=args.n_train_sweep_param,
        show=args.show,
    )
    fig_02c_n_train_sweep_all_parameterizations(
        metrics,
        out_dir,
        step_col,
        requested_n_trains=requested_n_train_sweep_values,
        show=args.show,
    )
    fig_02d_parameterization_pair_grid_by_n_train(
        metrics,
        out_dir,
        step_col,
        requested_n_trains=requested_n_train_sweep_values,
        show=args.show,
    )
    fig_07b_fmem_thresholds_by_n_train(metrics, out_dir, step_col, show=args.show)
    fig_03_final_fid_vs_fmem_faceted_by_n_train(metrics, out_dir, step_col, show=args.show)
    fig_04_fmem_at_best_fid_vs_n_train(metrics, out_dir, show=args.show)
    fig_05_pareto_best_fid_vs_fmem(metrics, out_dir, show=args.show)
    fig_06_parameterization_trajectories_fixed_seed_n_train(
        metrics,
        out_dir,
        step_col,
        seed=args.seed,
        n_train=args.n_train,
        show=args.show,
    )
    fig_07_seed_averaged_parameterization_trajectories_fixed_n_train(
        metrics,
        out_dir,
        step_col,
        n_train=args.n_train,
        show=args.show,
    )
    fig_07c_parameterization_fmem_thresholds_fixed_n_train(
        metrics,
        out_dir,
        step_col,
        n_train=args.n_train,
        show=args.show,
    )
    fig_08_parameterization_best_fid_summary(metrics, out_dir, show=args.show)

    print("done")


if __name__ == "__main__":
    main()
