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
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ============================================================
# 00. Configuration
# ============================================================

DEFAULT_REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "mem_gen_boundary_20260427"
EVAL_NAME = "mem_boundary_20260427_all_done_quality_20260501"

DEFAULT_EVAL_ROOT = Path("outputs/evals") / EVAL_NAME
DEFAULT_JOB_SET_ROOT = Path("outputs/job_sets") / EXPERIMENT_NAME
DEFAULT_OUT_DIR = Path("prototyping/figures") / f"{EVAL_NAME}_figures"
FIGURE_EXTENSIONS = (".pdf", ".png")
GENERATED_FIGURES: list[str] = []

PAPER_FIGURE_WIDTH = 4.4
PAPER_FIGURE_HEIGHT = 3.2
GRID_CELL_WIDTH = 1.65
GRID_CELL_HEIGHT = 1.45
LINE_MARKER_SIZE = 2.2
LEGEND_MARKER_SIZE = 2.8
SCATTER_SIZE_SMALL = 18
SCATTER_SIZE_MEDIUM = 24
SCATTER_SIZE_LARGE = 34
SCATTER_SIZE_HIGHLIGHT = 48
FIG_08_TEXT_COLOR = "#111111"
FIG_08_PANEL_SIZE = (2.85, 2.08)
FIG_08_LAYOUT = {
    "left": 0.10,
    "right": 0.82,
    "bottom": 0.12,
    "top": 0.86,
    "wspace": 0.24,
    "hspace": 0.45,
}
FIG_08_N_TRAIN_LABEL = r"$\mathbf{N}_{\mathbf{train}}$"
FIG_12C_PANEL_SIZE = (2.46, 1.35)
FIG_12C_LAYOUT = {
    "left": 0.075,
    "right": 0.97,
    "bottom": 0.23,
    "top": 0.86,
    "wspace": 0.10,
    "hspace": 0.18,
}
FIG_12C_TEXT_COLOR = "#111111"
FIG_12C_TICK_WEIGHT = "semibold"
FIG_12C_ROW_BACKGROUNDS = ("#f5f8ff", "#fff8ef")
FIG_12C_FMEM_LABEL = r"$\mathbf{F}_{\mathbf{mem}}$"


def pub_figsize(width_units: float = 1.0, height_units: float = 1.0) -> tuple[float, float]:
    """Return a compact publication-style figsize."""
    return (PAPER_FIGURE_WIDTH * width_units, PAPER_FIGURE_HEIGHT * height_units)


def grid_figsize(ncols: int, nrows: int) -> tuple[float, float]:
    """Return a denser figsize for faceted figure grids."""
    return (GRID_CELL_WIDTH * ncols, GRID_CELL_HEIGHT * nrows)


def figsize_from_panel_size(
    panel_size: tuple[float, float],
    ncols: int,
    nrows: int,
    layout: dict[str, float],
) -> tuple[float, float]:
    """Compute total figure size from explicit subplot panel dimensions."""
    panel_width, panel_height = panel_size
    wspace = layout.get("wspace", 0.0)
    hspace = layout.get("hspace", 0.0)
    axes_width = panel_width * (ncols + (ncols - 1) * wspace)
    axes_height = panel_height * (nrows + (nrows - 1) * hspace)
    fig_width = axes_width / (layout["right"] - layout["left"])
    fig_height = axes_height / (layout["top"] - layout["bottom"])
    return fig_width, fig_height

DEFAULT_REQUESTED_STEPS = [0, 100_000, 300_000, 595_000, 1_000_000]
DEFAULT_N_TRAIN_SWEEP_VALUES = [512, 2_048, 8_192]
DEFAULT_N_TRAIN_SWEEP_PARAM = "x_x"
DEFAULT_FIG_12B_N_TRAIN_VALUES = [512, 1_024, 2_048, 4_096, 8_192]

FID_COL = "fid_10k"
FMEM_COL = "f_mem"
TAU_LABEL = r"$\tau$"
N_TRAIN_LABEL = r"$N_{\mathrm{train}}$"
DIAGONAL_PARAM_LABELS = [r"$\epsilon/\epsilon$", r"$v/v$", r"$x/x$"]
HIDDEN_PARAM_LABELS = {r"$x/v$"}
CANONICAL_PARAM_COLORS = {
    r"$\epsilon/\epsilon$": "#1f77b4",
    r"$v/v$": "#2ca02c",
    r"$x/v$": "#ff7f0e",
    r"$x/x$": "#d62728",
    "eps_eps": "#1f77b4",
    "eps/eps": "#1f77b4",
    "epsilon_epsilon": "#1f77b4",
    "epsilon/epsilon": "#1f77b4",
    "v_v": "#2ca02c",
    "v/v": "#2ca02c",
    "u_u": "#2ca02c",
    "x_v": "#ff7f0e",
    "x/v": "#ff7f0e",
    "x_x": "#d62728",
    "x/x": "#d62728",
}


def format_n_train_value(value: int) -> str:
    """Return a math-text label for a concrete N_train value."""
    return rf"$N_{{\mathrm{{train}}}}={int(value)}$"

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
    (r"$\rho=0.3$", "f_mem_rho0.3"),
    (r"$\rho=0.5$", "f_mem_rho0.5"),
    (r"$\rho=0.7$", "f_mem_rho0.7"),
]

FIGURE_CAPTIONS = {
    "01_metric_trajectories_by_n_train": (
        "Figure 01 (metric trajectories by $N_{\\mathrm{train}}$): Checkpoint trajectories for the main quality, "
        "memorization, and loss metrics, grouped by training-set size. Each panel aggregates visible runs over "
        "checkpoints; color/line encodings compare parameterizations within each $N_{\\mathrm{train}}$ condition."
    ),
    "02_metric_trajectories_by_param": (
        "Figure 02 (metric trajectories by parameterization): Checkpoint trajectories for the same metrics as "
        "Figure 01, reorganized by parameterization. Curves compare $N_{\\mathrm{train}}$ values, making it easier "
        "to see how data scale changes each parameterization's quality and memorization dynamics."
    ),
    "03_final_fid_vs_fmem_by_n_train": (
        "Figure 03 (final FID versus final $F_{\\mathrm{mem}}$): Final-checkpoint trade-off between sample quality "
        "and memorization, faceted by $N_{\\mathrm{train}}$. Points correspond to visible runs/parameterizations "
        "and diagnose whether late training improves FID at the cost of higher memorization."
    ),
    "05_pareto_best_fid_vs_fmem_at_best_fid": (
        "Figure 05 (best-FID trade-off): For each run, the checkpoint with the lowest FID is selected and plotted "
        "against the corresponding $F_{\\mathrm{mem}}$. This isolates the memorization level at each run's best "
        "sample-quality checkpoint rather than at the final checkpoint."
    ),
    "06_parameterization_trajectories_fixed_seed_n_train": (
        "Figure 06 (parameterization trajectories at a representative seed and $N_{\\mathrm{train}}$): Training "
        "trajectories compare parameterizations for a fixed seed/data-size slice. The figure shows how FID, "
        "$F_{\\mathrm{mem}}$, and their phase relationship evolve over checkpoints."
    ),
    "07_seed_averaged_parameterization_trajectories_fixed_n_train": (
        "Figure 07 (seed-averaged parameterization trajectories): Parameterization trajectories averaged over "
        "visible seeds/runs and arranged by $N_{\\mathrm{train}}$. Panels include FID, $F_{\\mathrm{mem}}$, "
        "$F_{\\mathrm{mem}}$-versus-FID phase plots, and the available test-loss trajectory."
    ),
    "07d_all_run_phase_trajectories_by_n_train": (
        "Figure 07d (all-run phase trajectories): Individual run trajectories in the FID--$F_{\\mathrm{mem}}$ "
        "phase plane, arranged by $N_{\\mathrm{train}}$. The two columns reproduce the phase-view columns from "
        "Figure 07, using a linear FID view and a log-FID view with linear $F_{\\mathrm{mem}}$ axes; each trajectory is one run over checkpoints, and color "
        "encodes the diagonal parameterization."
    ),
    "07b_fmem_thresholds_along_training_by_n_train": (
        "Figure 07b ($F_{\\mathrm{mem}}$ threshold sweep): Log-scale $F_{\\mathrm{mem}}$ trajectories for "
        "nearest-neighbor ratio thresholds $\\rho=0.3$, $\\rho=0.5$, and $\\rho=0.7$. Rows show the diagonal "
        "parameterizations $\\epsilon/\\epsilon$, $v/v$, and $x/x$; colors encode $N_{\\mathrm{train}}$ with a "
        "single shared legend."
    ),
    "07c_parameterization_fmem_thresholds_fixed_n_train": (
        "Figure 07c (parameterization threshold sweep): Seed-averaged $F_{\\mathrm{mem}}$ trajectories across "
        "ratio thresholds, arranged by $N_{\\mathrm{train}}$. Curves compare parameterizations at each threshold "
        "and show how memorization timing changes across the sweep."
    ),
    "08_parameterization_best_fid_summary": (
        "Figure 08 (parameterization summary at best FID): Summary of sample-quality optima by parameterization "
        "and $N_{\\mathrm{train}}$. The top panel spans both columns and shows best achieved FID; the bottom "
        "panels show the memorization measured at the same best-FID checkpoint on linear and log y-scales, "
        "highlighting cases where quality and memorization optima diverge."
    ),
    "09_n_train_sweep_tau_triptych_fmem_testloss_fid": (
        "Figure 09 (focused $N_{\\mathrm{train}}$ sweep): Seed-averaged trajectories for selected training-set "
        "sizes under the focused parameterization. The triptych tracks $F_{\\mathrm{mem}}$, test loss, and FID "
        "along training time/$\\tau$."
    ),
    "10_n_train_sweep_all_parameterizations_fmem_testloss_fid": (
        "Figure 10 (focused $N_{\\mathrm{train}}$ sweep across parameterizations): Seed-averaged trajectories for "
        "the selected $N_{\\mathrm{train}}$ values, with all visible parameterizations included. The panels compare "
        "$F_{\\mathrm{mem}}$, test loss, and FID over training time/$\\tau$."
    ),
    "11_parameterization_pair_grid_by_n_train": (
        "Figure 11 (parameterization metric-pair overlays): Seed-averaged metric-pair trajectories arranged by "
        "$N_{\\mathrm{train}}$. Each panel overlays parameterizations to compare paired quality, memorization, "
        "and loss dynamics along the training trajectory."
    ),
    "12_fid_fmem_tau_grid_by_param": (
        "Figure 12 (FID and $F_{\\mathrm{mem}}$ along $\\tau$): Seed-averaged FID and $F_{\\mathrm{mem}}$ trajectories "
        "arranged by $N_{\\mathrm{train}}$ rows and parameterization columns. Dotted lines show FID on the left "
        "axis, solid lines show $F_{\\mathrm{mem}}$ on the right axis, and pastel row shading groups each "
        "$N_{\\mathrm{train}}$ condition."
    ),
    "12b_fid_fmem_tau_grid_by_param_n_train_512_to_8192": (
        "Figure 12b (mid-range FID and $F_{\\mathrm{mem}}$ along $\\tau$): Same layout and encodings as Figure 12, "
        "restricted to $512 <= N_{\\mathrm{train}} <= 8192$. Rows use pastel background bands and larger "
        "$N_{\\mathrm{train}}$ labels to emphasize the selected data-scale range."
    ),
    "12c_fid_fmem_tau_grid_by_param_merged_n_train_512_to_8192": (
        "Figure 12c (merged mid-range $N_{\\mathrm{train}}$ trajectories): Seed-averaged log-FID and "
        "$F_{\\mathrm{mem}}$ trajectories for $512 <= N_{\\mathrm{train}} <= 8192$, arranged as log-FID and "
        "$F_{\\mathrm{mem}}$ rows crossed with diagonal-parameterization columns. Columns use the canonical parameterization colors "
        "($\\epsilon/\\epsilon$ blue, $v/v$ green, $x/x$ red), while curve shade and line style encode "
        "$N_{\\mathrm{train}}$ within each column, dark-to-light as $N_{\\mathrm{train}}$ increases; $x/v$ is omitted. "
        "The horizontal axis uses $\\tau=\\mathrm{step}/10^6$."
    ),
}


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
    first_path = None
    for ext in FIGURE_EXTENSIONS:
        path = out_dir / f"{name}{ext}"
        if ext == ".png":
            fig.savefig(path, dpi=180, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        print(f"[figure] {path}")
        if first_path is None:
            first_path = path
    if name not in GENERATED_FIGURES:
        GENERATED_FIGURES.append(name)
    if show:
        plt.show()
    plt.close(fig)
    return first_path if first_path is not None else out_dir / f"{name}.pdf"


def caption_for_figure(name: str) -> str:
    """Return a paste-ready caption for a generated figure."""
    return FIGURE_CAPTIONS.get(
        name,
        f"{name}: Caption not registered yet. Add this output stem to FIGURE_CAPTIONS.",
    )


def print_figure_captions() -> None:
    """Print captions in the same order figures were generated."""
    if not GENERATED_FIGURES:
        return

    print("===== FIGURE CAPTIONS =====")
    for name in GENERATED_FIGURES:
        print(caption_for_figure(name))
        print()
    print("===== END FIGURE CAPTIONS =====")


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


def build_trajectory_summary(
    metrics: pd.DataFrame,
    x_col: str,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    """Aggregate checkpoint metrics for faceted trajectory plots."""
    work = metrics.copy()
    numeric_cols = [x_col, *metric_cols]
    for column in numeric_cols:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    subset_cols = [column for column in [*group_cols, x_col] if column in work.columns]
    work = work.dropna(subset=subset_cols)

    agg_spec: dict[str, tuple[str, str]] = {
        f"{metric}_mean": (metric, "mean")
        for metric in metric_cols
        if metric in work.columns
    }
    agg_spec.update(
        {
            f"{metric}_std": (metric, "std")
            for metric in metric_cols
            if metric in work.columns
        }
    )
    if "manifest_index" in work.columns:
        agg_spec["n_runs"] = ("manifest_index", "nunique")

    summary = (
        work
        .groupby([*group_cols, x_col], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values([*group_cols, x_col])
    )
    return summary


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


def hide_excluded_parameterizations(df: pd.DataFrame) -> pd.DataFrame:
    """Drop parameterizations that should not appear in generated figures/tables."""
    if "param_label" not in df.columns:
        return df
    return df[~df["param_label"].isin(HIDDEN_PARAM_LABELS)].copy()


def canonical_param_color(value: object, fallback_index: int | None = None) -> str:
    """Return the canonical color for known parameterizations."""
    text = str(value)
    if text in CANONICAL_PARAM_COLORS:
        return CANONICAL_PARAM_COLORS[text]

    pretty = pretty_param_label(value)
    if pretty in CANONICAL_PARAM_COLORS:
        return CANONICAL_PARAM_COLORS[pretty]

    if fallback_index is not None:
        base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return base_colors[fallback_index % len(base_colors)]
    return "0.2"


def canonical_param_color_map(params: Iterable[object]) -> dict[object, str]:
    """Build a stable parameterization color map with canonical known colors."""
    return {
        param: canonical_param_color(param, idx)
        for idx, param in enumerate(params)
    }


def mix_color_with_white(color: object, amount: float) -> tuple[float, float, float]:
    """Lighten a color by mixing it with white."""
    base = np.asarray(to_rgb(color))
    return tuple(base + (1.0 - base) * amount)


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

    sort_cols = ["manifest_index", FID_COL]
    step_col = first_existing(metrics.columns, ["step", "checkpoint_step"])
    if step_col is not None:
        sort_cols.append(step_col)

    best = (
        metrics
        .dropna(subset=[FID_COL])
        .sort_values(sort_cols)
        .groupby("manifest_index", as_index=False)
        .head(1)
        .copy()
    )
    missing_fmem = best[best[FMEM_COL].isna()]
    if not missing_fmem.empty:
        run_ids = missing_fmem["manifest_index"].astype(str).tolist()
        raise RuntimeError(
            "Cannot compute F_mem@FID_best because F_mem is missing at the "
            f"best-FID checkpoint for manifest_index values: {run_ids}"
        )

    return best


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
# 06. Figure: faceted metric trajectories by n_train
# ============================================================

def fig_01_metric_trajectories_by_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    show: bool,
) -> None:
    """Facet trajectories by n_train with parameterization as line series."""
    if "n_train" not in metrics.columns or "param_label" not in metrics.columns:
        return

    plot_metrics = [c for c in [FID_COL, FMEM_COL, *LOSS_COLS] if c in metrics.columns]
    if not plot_metrics:
        return

    x_col = choose_tau_plot_col(metrics, step_col)
    summary = build_trajectory_summary(metrics, x_col, ["n_train", "param_label"], plot_metrics)
    n_trains = sorted(summary["n_train"].dropna().unique())
    params = sorted(summary["param_label"].dropna().unique(), key=str)
    if not n_trains or not params:
        return

    param_colors = canonical_param_color_map(params)

    nrows = len(plot_metrics)
    ncols = len(n_trains)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=grid_figsize(ncols, nrows),
        sharex=True,
        sharey="row",
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    for row_idx, metric in enumerate(plot_metrics):
        mean_col = f"{metric}_mean"
        log_y = metric != FMEM_COL
        for col_idx, n_train in enumerate(n_trains):
            ax = axes[row_idx, col_idx]
            for param in params:
                group = summary[
                    (summary["n_train"] == n_train)
                    & (summary["param_label"] == param)
                ].sort_values(x_col)
                group = group[group[x_col] > 0]
                if log_y:
                    group = group[group[mean_col] > 0]
                if group.empty:
                    continue

                ax.plot(
                    group[x_col],
                    group[mean_col],
                    color=param_colors[param],
                    marker="o",
                    markersize=LINE_MARKER_SIZE,
                    linewidth=1.35,
                    alpha=0.95,
                    label="_nolegend_",
                )

            if log_y:
                ax.set_yscale("log")
            if x_col == "tau":
                ax.set_xscale("log")
            else:
                ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.tick_params(
                axis="both",
                which="both",
                left=True,
                bottom=True,
                labelleft=True,
                labelbottom=True,
                labelsize=6,
            )

            if row_idx == 0:
                ax.set_title(format_n_train_value(n_train))
            if col_idx == 0:
                ax.set_ylabel(display_label(metric))

    for ax in axes[-1, :]:
        ax.set_xlabel(TAU_LABEL if x_col == "tau" else "training step")

    param_handles = [
        Line2D(
            [0],
            [0],
            color=param_colors[param],
            marker="o",
            linewidth=1.35,
            markersize=LEGEND_MARKER_SIZE,
            label=str(param),
        )
        for param in params
    ]
    fig.legend(
        handles=param_handles,
        title="param",
        fontsize=7,
        title_fontsize=8,
        loc="lower center",
        ncol=min(4, len(param_handles)),
        frameon=False,
    )
    fig.suptitle(r"01 metric trajectories by $N_{\mathrm{train}}$ | parameterization lines", y=0.995)
    fig.subplots_adjust(top=0.90, bottom=0.10, wspace=0.15, hspace=0.20)
    save_figure(fig, out_dir, "01_metric_trajectories_by_n_train", show=show)


# ============================================================
# 07. Figure: faceted metric trajectories by parameterization
# ============================================================

def fig_02_metric_trajectories_by_param(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    show: bool,
) -> None:
    """Facet trajectories by parameterization with n_train as line series."""
    if "n_train" not in metrics.columns or "param_label" not in metrics.columns:
        return

    plot_metrics = [c for c in [FID_COL, FMEM_COL, *LOSS_COLS] if c in metrics.columns]
    if not plot_metrics:
        return

    x_col = choose_tau_plot_col(metrics, step_col)
    summary = build_trajectory_summary(metrics, x_col, ["param_label", "n_train"], plot_metrics)
    params = sorted(summary["param_label"].dropna().unique(), key=str)
    n_trains = sorted(summary["n_train"].dropna().unique())
    if not n_trains or not params:
        return

    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_train_colors = {n_train: base_colors[idx % len(base_colors)] for idx, n_train in enumerate(n_trains)}

    nrows = len(plot_metrics)
    ncols = len(params)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=grid_figsize(ncols, nrows),
        sharex=True,
        sharey="row",
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    for row_idx, metric in enumerate(plot_metrics):
        mean_col = f"{metric}_mean"
        log_y = metric != FMEM_COL
        for col_idx, param in enumerate(params):
            ax = axes[row_idx, col_idx]
            for n_train in n_trains:
                group = summary[
                    (summary["param_label"] == param)
                    & (summary["n_train"] == n_train)
                ].sort_values(x_col)
                group = group[group[x_col] > 0]
                if log_y:
                    group = group[group[mean_col] > 0]
                if group.empty:
                    continue

                ax.plot(
                    group[x_col],
                    group[mean_col],
                    color=n_train_colors[n_train],
                    marker="o",
                    markersize=LINE_MARKER_SIZE,
                    linewidth=1.35,
                    alpha=0.95,
                    label="_nolegend_",
                )

            if log_y:
                ax.set_yscale("log")
            if x_col == "tau":
                ax.set_xscale("log")
            else:
                ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.tick_params(
                axis="both",
                which="both",
                left=True,
                bottom=True,
                labelleft=True,
                labelbottom=True,
                labelsize=6,
            )

            if row_idx == 0:
                ax.set_title(str(param))
            if col_idx == 0:
                ax.set_ylabel(display_label(metric))

    for ax in axes[-1, :]:
        ax.set_xlabel(TAU_LABEL if x_col == "tau" else "training step")

    n_train_handles = [
        Line2D(
            [0],
            [0],
            color=n_train_colors[n_train],
            marker="o",
            linewidth=1.35,
            markersize=LEGEND_MARKER_SIZE,
            label=format_n_train_value(n_train),
        )
        for n_train in n_trains
    ]
    fig.legend(
        handles=n_train_handles,
        title=N_TRAIN_LABEL,
        fontsize=7,
        title_fontsize=8,
        loc="lower center",
        ncol=min(4, len(n_train_handles)),
        frameon=False,
    )
    fig.suptitle(r"02 metric trajectories by parameterization | $N_{\mathrm{train}}$ lines", y=0.995)
    fig.subplots_adjust(top=0.90, bottom=0.10, wspace=0.15, hspace=0.20)
    save_figure(fig, out_dir, "02_metric_trajectories_by_param", show=show)


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
            markersize=LINE_MARKER_SIZE,
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

    fig, axes = plt.subplots(2, 3, figsize=pub_figsize(3.0, 2.0), sharex=False)
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
            plot_metric_curve(ax, group, metric, n_train, f"{format_n_train_value(n_train)}{seed_text}")

        style_x_axis(ax)
        ax.grid(True, alpha=0.3)
        ax.legend(title=N_TRAIN_LABEL, fontsize=7, loc="best")

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
            Line2D([0], [0], color=colors[n_train], marker="o", linewidth=2.0, markersize=LEGEND_MARKER_SIZE, label=format_n_train_value(n_train))
            for n_train in n_trains
        ]
        metric_handles = [
            Line2D([0], [0], color="black", marker="o", linewidth=2.0, linestyle="-", markersize=LEGEND_MARKER_SIZE, label=f"{left_ylabel} (left)"),
            Line2D([0], [0], color="black", marker="s", linewidth=2.0, linestyle="--", markersize=LEGEND_MARKER_SIZE, label=f"{right_ylabel} (right)"),
        ]

        n_train_legend = ax_left.legend(
            handles=n_train_handles,
            title=N_TRAIN_LABEL,
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

    fig.suptitle(rf"09 focused $N_{{\mathrm{{train}}}}$ sweep | param={param_label} | seed averaged", y=1.01)
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
    param_colors = canonical_param_color_map(params)

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
            markersize=LINE_MARKER_SIZE,
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
            Line2D([0], [0], color=param_colors[param], marker="o", linewidth=2.0, markersize=LEGEND_MARKER_SIZE, label=str(param))
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
                markersize=LEGEND_MARKER_SIZE,
                label=format_n_train_value(n_train),
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
            title=N_TRAIN_LABEL,
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
                    markersize=LEGEND_MARKER_SIZE,
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
                    markersize=LEGEND_MARKER_SIZE,
                    label=f"{right_ylabel} (right)",
                ),
            ]
            ax.legend(handles=metric_handles, fontsize=6, loc="upper right")

    fig, axes = plt.subplots(2, 3, figsize=pub_figsize(3.0, 2.0), sharex=False)
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

    fig.suptitle(r"10 focused $N_{\mathrm{train}}$ sweep | all parameterizations | seed averaged", y=1.01)
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
    param_colors = canonical_param_color_map(params)

    def metric_shade(param: object, side: str) -> tuple[float, float, float]:
        base = np.asarray(to_rgb(param_colors[param]))
        if side == "left":
            return tuple(0.86 * base)
        if side == "right":
            return tuple(base + (1.0 - base) * 0.18)
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
        linestyle: object,
        marker_fill: str,
        side: str,
    ) -> None:
        mean_col = f"{metric}_mean"
        line_color = metric_shade(param, side)
        ax.plot(
            group[x_col].to_numpy(dtype=float),
            group[mean_col].to_numpy(dtype=float),
            color=line_color,
            linestyle=linestyle,
            linewidth=1.9,
            alpha=0.95,
            label="_nolegend_",
        )

    def style_x_axis(ax: plt.Axes) -> None:
        ax.set_xlabel(TAU_LABEL if x_col == "tau" else "training step")
        if x_col == "tau":
            apply_tau_axis_for_triptych(ax, summary[x_col])
        else:
            ax.set_xscale("log")

    def angle_x_tick_labels(ax: plt.Axes) -> None:
        ax.tick_params(axis="x", labelsize=6, pad=1)
        for tick_label in ax.get_xticklabels(which="both"):
            tick_label.set_rotation(35)
            tick_label.set_horizontalalignment("right")
            tick_label.set_rotation_mode("anchor")

    def style_y_axis(ax: plt.Axes, ylabel: str, log_y: bool) -> None:
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")

    param_handles = [
        Line2D([0], [0], color=param_colors[param], linewidth=2.0, label=str(param))
        for param in params
    ]

    nrows = len(n_trains)
    ncols = len(metric_pairs)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.0 * nrows),
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
            ax_right.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False, labeltop=False)
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
                        linestyle=(0, (5.5, 2.5)),
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
            angle_x_tick_labels(ax_left)
            ax_left.grid(True, alpha=0.3)

            if row_idx == 0:
                ax_left.set_title(f"left: {left_ylabel} | right: {right_ylabel}")
            ax_left.legend(
                handles=param_handles,
                fontsize=5,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.985),
                ncol=len(param_handles),
                frameon=True,
                framealpha=0.82,
                borderpad=0.2,
                handlelength=1.0,
                handletextpad=0.25,
                columnspacing=0.55,
            )

    fig.suptitle(r"11 parameterization metric-pair overlays by $N_{\mathrm{train}}$ | seed averaged", y=0.997)
    fig.subplots_adjust(left=0.12, top=0.955, hspace=0.50, wspace=0.52)
    for ax in axes.ravel():
        angle_x_tick_labels(ax)
    for row_idx, n_train in enumerate(n_trains):
        row_bbox = axes[row_idx, 0].get_position()
        fig.text(
            0.035,
            0.5 * (row_bbox.y0 + row_bbox.y1),
            format_n_train_value(n_train),
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
        )
    save_figure(fig, out_dir, "11_parameterization_pair_grid_by_n_train", show=show)


# ============================================================
# 07e. Figure: FID/F_mem tau trajectories by parameterization
# ============================================================

def fig_02e_fid_fmem_tau_grid_by_param(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    show: bool,
    n_train_values: list[int] | None = None,
    table_name: str = "12_fid_fmem_tau_grid_by_param_seed_average",
    figure_name: str = "12_fid_fmem_tau_grid_by_param",
    title: str = r"12 FID and $F_{\mathrm{mem}}$ along $\tau$ by parameterization and $N_{\mathrm{train}}$",
    x_axis_bottom_only: bool = False,
    subplot_hspace: float = 0.50,
) -> None:
    """Figure-11 middle-panel duplicate, split by parameterization columns."""
    if "n_train" not in metrics.columns or "param_label" not in metrics.columns:
        return
    if FID_COL not in metrics.columns or FMEM_COL not in metrics.columns:
        return

    visible_n_trains = n_train_ticks_from(metrics)
    if n_train_values is None:
        n_trains = visible_n_trains
    else:
        requested = {int(value) for value in n_train_values}
        n_trains = [n_train for n_train in visible_n_trains if n_train in requested]
    if not n_trains:
        print("[warn] FID/F_mem by-param grid skipped; no visible n_train values")
        return

    x_col = choose_tau_plot_col(metrics, step_col)
    seed_col = choose_seed_col(metrics)
    run_col = first_existing(metrics.columns, ["manifest_index", "run_id", "metrics_file"])

    sub = metrics.copy()
    sub["_n_train_plot"] = pd.to_numeric(sub["n_train"], errors="coerce")
    sub = sub[sub["_n_train_plot"].isin(n_trains)].copy()
    sub["_n_train_plot"] = sub["_n_train_plot"].astype(int)

    for column in [x_col, FID_COL, FMEM_COL]:
        sub[column] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub.dropna(subset=["_n_train_plot", "param_label", x_col, FID_COL, FMEM_COL])
    if sub.empty:
        print("[warn] FID/F_mem by-param grid skipped; selected data has no finite values")
        return

    agg_spec = {
        f"{FID_COL}_mean": (FID_COL, "mean"),
        f"{FMEM_COL}_mean": (FMEM_COL, "mean"),
    }
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
    save_table(summary, out_dir, table_name)

    params = sorted(summary["param_label"].dropna().unique(), key=str)
    if not params:
        return

    param_colors = canonical_param_color_map(params)

    def metric_colors(param: object) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        base = np.asarray(to_rgb(param_colors[param]))
        fid_color = tuple(0.86 * base)
        fmem_color = tuple(base + (1.0 - base) * 0.18)
        return fid_color, fmem_color

    def style_x_axis(ax: plt.Axes) -> None:
        ax.set_xlabel(TAU_LABEL if x_col == "tau" else "training step")
        if x_col == "tau":
            apply_tau_axis_for_triptych(ax, summary[x_col])
        else:
            ax.set_xscale("log")

    def angle_x_tick_labels(ax: plt.Axes) -> None:
        ax.tick_params(axis="x", labelsize=6, pad=1)
        for tick_label in ax.get_xticklabels(which="both"):
            tick_label.set_rotation(35)
            tick_label.set_horizontalalignment("right")
            tick_label.set_rotation_mode("anchor")

    row_fid_limits: dict[int, tuple[float, float]] = {}
    row_fmem_limits: dict[int, tuple[float, float]] = {}
    for n_train in n_trains:
        fid_values = (
            pd.to_numeric(summary.loc[summary["n_train"] == n_train, f"{FID_COL}_mean"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if not fid_values.empty:
            lower = min(0.0, float(fid_values.min()))
            upper = float(fid_values.max())
            if upper <= lower:
                upper = lower + max(abs(lower) * 0.05, 1.0)
            pad = 0.05 * (upper - lower)
            row_fid_limits[n_train] = (lower, upper + pad)

        values = (
            pd.to_numeric(summary.loc[summary["n_train"] == n_train, f"{FMEM_COL}_mean"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if values.empty:
            continue
        lower = min(0.0, float(values.min()))
        upper = float(values.max())
        if upper <= lower:
            upper = lower + max(abs(lower) * 0.05, 1.0)
        pad = 0.05 * (upper - lower)
        row_fmem_limits[n_train] = (lower, upper + pad)

    nrows = len(n_trains)
    ncols = len(params)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.35 * ncols, 2.0 * nrows),
        sharex=False,
    )
    axes = np.asarray(axes).reshape(nrows, ncols)
    row_backgrounds = ["#f4f8ff", "#fff7ed", "#f4fbf6", "#fff5fb", "#f7f5ff"]

    for row_idx, n_train in enumerate(n_trains):
        row_facecolor = row_backgrounds[row_idx % len(row_backgrounds)]
        for col_idx, param in enumerate(params):
            ax_left = axes[row_idx, col_idx]
            ax_right = ax_left.twinx()
            ax_left.spines["right"].set_visible(False)
            ax_right.spines["left"].set_visible(False)
            ax_left.set_facecolor(row_facecolor)
            ax_right.patch.set_alpha(0.0)
            ax_right.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False, labeltop=False)

            group = summary[
                (summary["n_train"] == n_train)
                & (summary["param_label"] == param)
            ].sort_values(x_col)
            group = group[group[x_col] > 0]
            if group.empty:
                continue

            fid_color, fmem_color = metric_colors(param)
            ax_left.plot(
                group[x_col].to_numpy(dtype=float),
                group[f"{FID_COL}_mean"].to_numpy(dtype=float),
                color=fid_color,
                linestyle=":",
                linewidth=1.9,
                alpha=0.95,
            )
            ax_right.plot(
                group[x_col].to_numpy(dtype=float),
                group[f"{FMEM_COL}_mean"].to_numpy(dtype=float),
                color=fmem_color,
                linestyle="-",
                linewidth=1.9,
                alpha=0.95,
            )
            if n_train in row_fmem_limits:
                ax_right.set_ylim(row_fmem_limits[n_train])
            if n_train in row_fid_limits:
                ax_left.set_ylim(row_fid_limits[n_train])

            ax_left.set_ylabel("FID")
            ax_right.set_ylabel(display_label(FMEM_COL))
            ax_left.tick_params(axis="y", labelsize=7)
            ax_right.tick_params(axis="y", labelsize=7)
            ax_left.grid(True, alpha=0.3)
            style_x_axis(ax_left)
            angle_x_tick_labels(ax_left)
            if x_axis_bottom_only and row_idx != nrows - 1:
                ax_left.set_xlabel("")
                ax_left.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

            if row_idx == 0:
                ax_left.set_title(str(param))

            legend_handles = [
                Line2D([0], [0], color=fid_color, linewidth=2.0, linestyle=":", label="FID"),
                Line2D([0], [0], color=fmem_color, linewidth=2.0, linestyle="-", label=display_label(FMEM_COL)),
            ]
            ax_left.legend(
                handles=legend_handles,
                fontsize=6,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.985),
                ncol=2,
                frameon=True,
                framealpha=0.82,
                borderpad=0.2,
                handlelength=1.3,
                handletextpad=0.35,
                columnspacing=0.7,
            )

    fig.suptitle(title, y=0.997)
    fig.subplots_adjust(left=0.10, top=0.955, hspace=subplot_hspace, wspace=0.62)
    for row_idx in range(nrows):
        for ax in axes[row_idx, :]:
            angle_x_tick_labels(ax)
            if x_axis_bottom_only and row_idx != nrows - 1:
                ax.set_xlabel("")
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for row_idx, n_train in enumerate(n_trains):
        row_bbox = axes[row_idx, 0].get_position()
        row_right = axes[row_idx, -1].get_position().x1
        row_facecolor = row_backgrounds[row_idx % len(row_backgrounds)]
        band = plt.Rectangle(
            (row_bbox.x0 - 0.035, row_bbox.y0 - 0.012),
            row_right - row_bbox.x0 + 0.07,
            row_bbox.y1 - row_bbox.y0 + 0.024,
            transform=fig.transFigure,
            facecolor=row_facecolor,
            edgecolor="none",
            alpha=0.42,
            zorder=-1,
        )
        fig.patches.append(band)
        fig.text(
            0.055,
            0.5 * (row_bbox.y0 + row_bbox.y1),
            format_n_train_value(n_train),
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="semibold",
        )
    save_figure(fig, out_dir, figure_name, show=show)


def fig_12b_fid_fmem_tau_grid_by_param_mid_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    show: bool,
) -> None:
    """Figure 12 restricted to mid-range N_train rows."""
    fig_02e_fid_fmem_tau_grid_by_param(
        metrics,
        out_dir,
        step_col,
        show=show,
        n_train_values=DEFAULT_FIG_12B_N_TRAIN_VALUES,
        table_name="12b_fid_fmem_tau_grid_by_param_n_train_512_to_8192_seed_average",
        figure_name="12b_fid_fmem_tau_grid_by_param_n_train_512_to_8192",
        title=r"12b FID and $F_{\mathrm{mem}}$ along $\tau$ by parameterization | $512 \leq N_{\mathrm{train}} \leq 8192$",
        x_axis_bottom_only=True,
        subplot_hspace=0.28,
    )


def fig_12c_fid_fmem_tau_grid_by_param_merged_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    show: bool,
) -> None:
    """Merged mid-range N_train trajectories in metric rows and parameter columns."""
    if "n_train" not in metrics.columns or "param_label" not in metrics.columns:
        return
    if FID_COL not in metrics.columns or FMEM_COL not in metrics.columns:
        return

    visible_n_trains = n_train_ticks_from(metrics)
    requested = set(DEFAULT_FIG_12B_N_TRAIN_VALUES)
    n_trains = [n_train for n_train in visible_n_trains if n_train in requested]
    if not n_trains:
        print("[warn] merged FID/F_mem grid skipped; no selected mid-range n_train values")
        return

    x_col = choose_tau_plot_col(metrics, step_col)
    seed_col = choose_seed_col(metrics)
    run_col = first_existing(metrics.columns, ["manifest_index", "run_id", "metrics_file"])

    sub = metrics.copy()
    sub["_n_train_plot"] = pd.to_numeric(sub["n_train"], errors="coerce")
    sub = sub[
        sub["_n_train_plot"].isin(n_trains)
        & sub["param_label"].isin(DIAGONAL_PARAM_LABELS)
    ].copy()
    if sub.empty:
        print("[warn] merged FID/F_mem grid skipped; selected parameterizations are absent")
        return

    sub["_n_train_plot"] = sub["_n_train_plot"].astype(int)
    for column in [x_col, FID_COL, FMEM_COL]:
        sub[column] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub.dropna(subset=["_n_train_plot", "param_label", x_col, FID_COL, FMEM_COL])
    if sub.empty:
        print("[warn] merged FID/F_mem grid skipped; selected data has no finite values")
        return

    agg_spec = {
        f"{FID_COL}_mean": (FID_COL, "mean"),
        f"{FID_COL}_std": (FID_COL, "std"),
        f"{FMEM_COL}_mean": (FMEM_COL, "mean"),
        f"{FMEM_COL}_std": (FMEM_COL, "std"),
    }
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
    save_table(
        summary,
        out_dir,
        "12c_fid_fmem_tau_grid_by_param_merged_n_train_512_to_8192_seed_average",
    )

    params = [param for param in DIAGONAL_PARAM_LABELS if param in set(summary["param_label"].dropna())]
    if not params:
        return

    x_axis_label = r"$\tau$ (training steps / $10^6$)" if x_col == "tau" else "training step"

    def shared_limits(column: str, positive_only: bool = False) -> tuple[float, float] | None:
        values = (
            pd.to_numeric(summary[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if positive_only:
            values = values[values > 0]
        if values.empty:
            return None
        lower = float(values.min()) if positive_only else min(0.0, float(values.min()))
        upper = float(values.max())
        if upper <= lower:
            upper = lower + max(abs(lower) * 0.05, 1.0)
        pad = 0.06 * (upper - lower)
        if positive_only:
            lower = max(lower / 1.15, np.finfo(float).tiny)
            return lower, upper * 1.15
        return lower, upper + pad

    fid_log_limits = shared_limits(f"{FID_COL}_mean", positive_only=True)
    fmem_limits = shared_limits(f"{FMEM_COL}_mean")
    linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]
    n_train_linestyles = {
        n_train: linestyles[idx % len(linestyles)]
        for idx, n_train in enumerate(n_trains)
    }
    shade_amounts = {
        n_train: float(amount)
        for n_train, amount in zip(n_trains, np.linspace(0.0, 0.62, len(n_trains)))
    }

    def apply_12c_x_axis(ax: plt.Axes) -> None:
        if x_col == "tau":
            apply_tau_axis_for_triptych(ax, summary[x_col])
            ticks = list(ax.get_xticks())
            labels = [tick.get_text() for tick in ax.get_xticklabels()]
            if len(ticks) > 1:
                ax.set_xticks(ticks[1:])
                ax.set_xticklabels(labels[1:])
        else:
            ax.set_xscale("log")

    nrows = 2
    ncols = len(params)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize_from_panel_size(FIG_12C_PANEL_SIZE, ncols, nrows, FIG_12C_LAYOUT),
        sharex=True,
        sharey="row",
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    for col_idx, param in enumerate(params):
        base_color = canonical_param_color(param)
        axes[0, col_idx].set_title(
            str(param),
            fontsize=10,
            fontweight="bold",
            color=FIG_12C_TEXT_COLOR,
        )
        for n_train in n_trains:
            group = summary[
                (summary["param_label"] == param)
                & (summary["n_train"] == n_train)
            ].sort_values(x_col)
            group = group[group[x_col] > 0]
            if group.empty:
                continue

            color = mix_color_with_white(base_color, shade_amounts[n_train])
            linestyle = n_train_linestyles[n_train]
            log_group = group[group[f"{FID_COL}_mean"] > 0]
            if not log_group.empty:
                axes[0, col_idx].plot(
                    log_group[x_col].to_numpy(dtype=float),
                    log_group[f"{FID_COL}_mean"].to_numpy(dtype=float),
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.3,
                    alpha=0.96,
                )
            axes[1, col_idx].plot(
                group[x_col].to_numpy(dtype=float),
                group[f"{FMEM_COL}_mean"].to_numpy(dtype=float),
                color=color,
                linestyle=linestyle,
                linewidth=2.3,
                alpha=0.96,
            )

        for row_idx, ax in enumerate(axes[:, col_idx]):
            apply_12c_x_axis(ax)
            ax.set_facecolor(FIG_12C_ROW_BACKGROUNDS[row_idx % len(FIG_12C_ROW_BACKGROUNDS)])
            if row_idx == 0:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=8, colors=FIG_12C_TEXT_COLOR, labelcolor=FIG_12C_TEXT_COLOR)
            if row_idx != nrows - 1:
                ax.tick_params(axis="x", which="both", labelbottom=False)
            else:
                ax.set_xlabel(x_axis_label, fontsize=10, fontweight="bold", color=FIG_12C_TEXT_COLOR)
                for tick_label in ax.get_xticklabels(which="both"):
                    tick_label.set_rotation(30)
                    tick_label.set_horizontalalignment("right")
                    tick_label.set_rotation_mode("anchor")
            for tick_label in [*ax.get_xticklabels(which="both"), *ax.get_yticklabels(which="both")]:
                tick_label.set_color(FIG_12C_TEXT_COLOR)
                tick_label.set_fontweight(FIG_12C_TICK_WEIGHT)

    if fid_log_limits is not None:
        axes[0, 0].set_ylim(fid_log_limits)
    if fmem_limits is not None:
        axes[1, 0].set_ylim(fmem_limits)
    axes[0, 0].set_ylabel("FID (log)", fontsize=10, fontweight="bold", color=FIG_12C_TEXT_COLOR)
    axes[1, 0].set_ylabel(FIG_12C_FMEM_LABEL, fontsize=10, fontweight="bold", color=FIG_12C_TEXT_COLOR)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=mix_color_with_white("0.15", shade_amounts[n_train]),
            linestyle=n_train_linestyles[n_train],
            linewidth=2.3,
            label=format_n_train_value(n_train),
        )
        for n_train in n_trains
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8,
        handlelength=2.8,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    for text in legend.get_texts():
        text.set_fontweight(FIG_12C_TICK_WEIGHT)
        text.set_color(FIG_12C_TEXT_COLOR)

    fig.suptitle(
        rf"12c log-FID and {FIG_12C_FMEM_LABEL}: shade + line style | $\tau=\mathrm{{step}}/10^6$",
        y=0.985,
        fontsize=12,
        fontweight="bold",
        color=FIG_12C_TEXT_COLOR,
    )
    fig.subplots_adjust(**FIG_12C_LAYOUT)
    for row_idx in range(nrows):
        row_bbox = axes[row_idx, 0].get_position()
        row_right = axes[row_idx, -1].get_position().x1
        band = plt.Rectangle(
            (row_bbox.x0 - 0.02, row_bbox.y0 - 0.012),
            row_right - row_bbox.x0 + 0.04,
            row_bbox.y1 - row_bbox.y0 + 0.024,
            transform=fig.transFigure,
            facecolor=FIG_12C_ROW_BACKGROUNDS[row_idx % len(FIG_12C_ROW_BACKGROUNDS)],
            edgecolor="none",
            alpha=0.38,
            zorder=-1,
        )
        fig.patches.append(band)
    save_figure(
        fig,
        out_dir,
        "12c_fid_fmem_tau_grid_by_param_merged_n_train_512_to_8192",
        show=show,
    )


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
    if "n_train" not in metrics.columns or "param_label" not in metrics.columns:
        return

    rho_cols = [(label, col) for label, col in FMEM_RHO_COLS if col in metrics.columns]
    if not rho_cols:
        return

    diagonal_params = [r"$\epsilon/\epsilon$", r"$v/v$", r"$x/x$"]
    plot_metrics = metrics[metrics["param_label"].isin(diagonal_params)].copy()
    if plot_metrics.empty:
        return

    agg = (
        plot_metrics
        .groupby(["param_label", "n_train", step_col], dropna=False)
        .agg(
            **{f"{col}_mean": (col, "mean") for _, col in rho_cols},
            n_runs=("manifest_index", "nunique"),
        )
        .reset_index()
    )

    n_trains = sorted(agg["n_train"].dropna().unique())
    params = [param for param in diagonal_params if param in set(agg["param_label"].dropna())]
    if not params:
        return

    ncols = len(rho_cols)
    nrows = len(params)
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_train_colors = {n_train: base_colors[idx % len(base_colors)] for idx, n_train in enumerate(n_trains)}
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(GRID_CELL_WIDTH * ncols + 1.65, GRID_CELL_HEIGHT * nrows + 0.65),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    legend_handles: list[Line2D] = []

    for row_idx, param in enumerate(params):
        row_summary = agg[agg["param_label"] == param]
        for col_idx, (rho_label, col) in enumerate(rho_cols):
            ax = axes[row_idx, col_idx]
            mean_col = f"{col}_mean"
            for n_train in n_trains:
                sub = row_summary[row_summary["n_train"] == n_train].sort_values(step_col)
                sub = sub[(sub[step_col] > 0) & (sub[mean_col] > 0)]
                if sub.empty:
                    continue
                ax.plot(
                    sub[step_col],
                    sub[mean_col],
                    marker="o",
                    markersize=LINE_MARKER_SIZE,
                    linewidth=1.6,
                    color=n_train_colors[n_train],
                    label=format_n_train_value(n_train),
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            if row_idx == nrows - 1:
                ax.set_xlabel(TAU_LABEL)
            if row_idx == 0:
                ax.set_title(rho_label)
            ax.grid(True, alpha=0.3)

    for n_train in n_trains:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=n_train_colors[n_train],
                marker="o",
                linewidth=1.8,
                markersize=LEGEND_MARKER_SIZE,
                label=format_n_train_value(n_train),
            )
        )

    fig.legend(
        handles=legend_handles,
        title=N_TRAIN_LABEL,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )
    fig.suptitle(r"07b $F_{\mathrm{mem}}$ threshold sweep along training by $N_{\mathrm{train}}$", y=0.985)
    fig.subplots_adjust(left=0.17, right=0.78, top=0.86, bottom=0.12, hspace=0.42, wspace=0.18)
    fig.text(0.035, 0.5, r"$F_{\mathrm{mem}}$", rotation=90, va="center", ha="center", fontsize=10)
    for row_idx, param in enumerate(params):
        row_bbox = axes[row_idx, 0].get_position()
        fig.text(
            0.085,
            0.5 * (row_bbox.y0 + row_bbox.y1),
            param,
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
        )
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
    fig, axes = plt.subplots(nrows, ncols, figsize=pub_figsize(ncols, nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    seed_col = choose_seed_col(final_df)

    for ax, n_train in zip(axes, n_trains):
        sub = final_df[final_df["n_train"] == n_train]
        if seed_col is not None:
            ax.scatter(sub[FID_COL], sub[FMEM_COL], c=sub[seed_col], cmap="tab10", s=SCATTER_SIZE_MEDIUM, alpha=0.85)
        else:
            ax.scatter(sub[FID_COL], sub[FMEM_COL], s=SCATTER_SIZE_MEDIUM, alpha=0.85)
        ax.set_title(f"{format_n_train_value(n_train)} | runs={len(sub)}")
        ax.set_xlabel("final FID")
        ax.set_ylabel(r"final $F_{\mathrm{mem}}$")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(n_trains):]:
        ax.axis("off")

    fig.suptitle(r"03 final checkpoint FID vs $F_{\mathrm{mem}}$ by $N_{\mathrm{train}}$", y=1.01)
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
    fig, ax = plt.subplots(figsize=pub_figsize(1.15, 1.05))

    for label, sub in best.groupby("param_label", dropna=False):
        color = canonical_param_color(label)
        # Light points show individual seeds/runs.
        ax.scatter(sub["n_train"], sub[FMEM_COL], color=color, s=SCATTER_SIZE_MEDIUM, alpha=0.45, label=f"{label} runs")

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
            color=color,
            marker="o",
            linewidth=2.0,
            capsize=3,
            markersize=LEGEND_MARKER_SIZE,
            label=f"{label} mean",
        )

    apply_n_train_log_ticks(ax, n_train_ticks)
    ax.set_xlabel(N_TRAIN_LABEL)
    ax.set_ylabel(r"$F_{\mathrm{mem}}$ at best FID")
    ax.set_title(r"04 quality-controlled memorization: $F_{\mathrm{mem}}$ at best FID vs $N_{\mathrm{train}}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="upper left", ncol=2, frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.4)
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

    fig, ax = plt.subplots(figsize=pub_figsize(1.6, 1.3))
    scatter = ax.scatter(
        best[FID_COL],
        best[FMEM_COL],
        c=best["n_train"],
        cmap="viridis",
        s=SCATTER_SIZE_LARGE,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.35,
    )

    ax.set_xlabel("best FID")
    ax.set_ylabel(r"$F_{\mathrm{mem}}$ at best FID")
    ax.set_title(r"05 Pareto diagnostic: best quality vs $F_{\mathrm{mem}}$ at best quality")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(N_TRAIN_LABEL)
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


def pick_seed_for_n_train(metrics: pd.DataFrame, seed_col: str | None, seed: int | None, n_train: int) -> int | None:
    """Choose the requested seed when visible, otherwise best parameterization coverage."""
    if seed_col is None:
        return None

    sub = metrics[metrics["n_train"] == n_train]
    if sub.empty:
        return seed

    if seed is not None and (sub[seed_col] == seed).any():
        return int(seed)

    run_col = first_existing(sub.columns, ["manifest_index", "run_id", "metrics_file"])
    agg_spec = {"n_params": ("param_label", "nunique")}
    if run_col is not None:
        agg_spec["n_runs"] = (run_col, "nunique")
    else:
        agg_spec["n_runs"] = (seed_col, "size")

    coverage = (
        sub
        .groupby(seed_col, dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["n_params", "n_runs", seed_col], ascending=[False, False, True])
    )
    if coverage.empty:
        return seed
    return int(coverage[seed_col].iloc[0])


def fig_06_parameterization_trajectories_fixed_seed_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    seed: int | None,
    n_train: int | None,
    show: bool,
) -> None:
    """Fixed seed policy by n_train: each trajectory is one parameterization.

    This directly targets the missing comparison:
        Does parameterization change the FID(tau), F_mem(tau), and
        FID-vs-F_mem trajectory?
    """
    if "n_train" not in metrics.columns or "param_label" not in metrics.columns:
        return

    x_col = choose_tau_plot_col(metrics, step_col)
    seed_col = choose_seed_col(metrics)
    n_train_values = [int(n_train)] if n_train is not None else n_train_ticks_from(metrics)
    if not n_train_values:
        return

    row_entries: list[tuple[int, int | None, pd.DataFrame]] = []
    for n_train_value in n_train_values:
        selected_seed = pick_seed_for_n_train(metrics, seed_col, seed, n_train_value)
        sub = metrics[metrics["n_train"] == n_train_value].copy()
        if seed_col is not None and selected_seed is not None:
            sub = sub[sub[seed_col] == selected_seed].copy()
        if not sub.empty:
            row_entries.append((n_train_value, selected_seed, sub))

    if not row_entries:
        print("[warn] no data for requested parameterization trajectory grid")
        return

    nrows = len(row_entries)
    ncols = 5
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.3 * ncols, 2.35 * nrows),
        sharex=False,
        sharey=False,
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    def positive_x(group: pd.DataFrame) -> pd.DataFrame:
        return group[group[x_col] > 0].copy()

    def compact_log_ticks(ax: plt.Axes, axis: str) -> None:
        target_axis = ax.xaxis if axis == "x" else ax.yaxis
        target_axis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        target_axis.set_minor_formatter(mticker.NullFormatter())

    for row_idx, (n_train_value, selected_seed, sub) in enumerate(row_entries):
        row_axes = axes[row_idx]
        for label, group in sub.groupby("param_label", dropna=False):
            color = canonical_param_color(label)
            group = group.sort_values(x_col)
            x_group = positive_x(group)
            if not x_group.empty:
                row_axes[0].plot(x_group[x_col], x_group[FID_COL], color=color, marker="o", markersize=LINE_MARKER_SIZE, linewidth=1.6, label=str(label))
                row_axes[1].plot(x_group[x_col], x_group[FMEM_COL], color=color, marker="o", markersize=LINE_MARKER_SIZE, linewidth=1.6, label=str(label))

            row_axes[2].plot(group[FID_COL], group[FMEM_COL], color=color, marker="o", markersize=LINE_MARKER_SIZE, linewidth=1.6, label=str(label))
            log_phase = group[group[FID_COL] > 0]
            row_axes[3].plot(log_phase[FID_COL], log_phase[FMEM_COL], color=color, marker="o", markersize=LINE_MARKER_SIZE, linewidth=1.6, label=str(label))
            log_log_phase = group[(group[FID_COL] > 0) & (group[FMEM_COL] > 0)]
            row_axes[4].plot(log_log_phase[FID_COL], log_log_phase[FMEM_COL], color=color, marker="o", markersize=LINE_MARKER_SIZE, linewidth=1.6, label=str(label))

            best_idx = group[FID_COL].idxmin()
            best_row = group.loc[best_idx]
            row_axes[2].scatter(best_row[FID_COL], best_row[FMEM_COL], s=SCATTER_SIZE_HIGHLIGHT, facecolor=color, edgecolor="black", linewidth=1.0)
            row_axes[3].scatter(best_row[FID_COL], best_row[FMEM_COL], s=SCATTER_SIZE_HIGHLIGHT, facecolor=color, edgecolor="black", linewidth=1.0)
            if best_row[FID_COL] > 0 and best_row[FMEM_COL] > 0:
                row_axes[4].scatter(best_row[FID_COL], best_row[FMEM_COL], s=SCATTER_SIZE_HIGHLIGHT, facecolor=color, edgecolor="black", linewidth=1.0)

        row_axes[0].set_ylabel(display_label(FID_COL))
        row_axes[1].set_ylabel(display_label(FMEM_COL))
        row_axes[2].set_ylabel(display_label(FMEM_COL))
        row_axes[3].set_ylabel(display_label(FMEM_COL))
        row_axes[4].set_ylabel(display_label(FMEM_COL))

        for col_idx, ax in enumerate(row_axes):
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc="best")
            ax.tick_params(axis="both", labelsize=6)
            if col_idx in {0, 1}:
                ax.set_xlabel(TAU_LABEL if x_col == "tau" else "training step")
                ax.set_xscale("log")
                if x_col == "tau":
                    apply_tau_axis_for_triptych(ax, sub[x_col])
            else:
                ax.set_xlabel(display_label(FID_COL))

        row_axes[2].set_xscale("linear")
        row_axes[3].set_xscale("log")
        row_axes[4].set_xscale("log")
        row_axes[4].set_yscale("log")
        compact_log_ticks(row_axes[3], "x")
        compact_log_ticks(row_axes[4], "x")
        compact_log_ticks(row_axes[4], "y")

        if row_idx == 0:
            row_axes[0].set_title(r"FID over $\tau$")
            row_axes[1].set_title(r"$F_{\mathrm{mem}}$ over $\tau$")
            row_axes[2].set_title(r"$F_{\mathrm{mem}}$ vs FID")
            row_axes[3].set_title(r"$F_{\mathrm{mem}}$ vs FID (log FID)")
            row_axes[4].set_title(r"$F_{\mathrm{mem}}$ vs FID (log-log)")

    fig.suptitle(r"06 parameterization trajectories by $N_{\mathrm{train}}$", y=0.997)
    fig.subplots_adjust(left=0.12, top=0.94, hspace=0.62, wspace=0.58)
    for row_idx, (n_train_value, selected_seed, _) in enumerate(row_entries):
        row_bbox = axes[row_idx, 0].get_position()
        row_label = format_n_train_value(n_train_value)
        if seed_col is not None and selected_seed is not None:
            row_label += f"\n{seed_col}={selected_seed}"
        fig.text(
            0.035,
            0.5 * (row_bbox.y0 + row_bbox.y1),
            row_label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=8,
        )

    save_figure(fig, out_dir, "06_parameterization_trajectories_fixed_seed_n_train", show=show)


# ============================================================
# 12. Figure: seed-averaged parameterization trajectories by n_train
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


def pick_n_train_values_for_param_grid(metrics: pd.DataFrame, n_train: int | None) -> list[int]:
    """Choose the n_train values to show in the figure-07 grid."""
    if n_train is not None:
        return [int(n_train)]

    if "n_train" not in metrics.columns:
        return []

    return [int(value) for value in sorted(metrics["n_train"].dropna().unique())]


def fig_07_seed_averaged_parameterization_trajectories_fixed_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    n_train: int | None,
    show: bool,
) -> None:
    """Seed-averaged parameterization trajectories arranged by n_train.

    When no explicit n_train is passed, the figure becomes one row per visible
    n_train value. The test-loss panel is kept in the main row set and labeled
    as a training-step trajectory.
    """
    if "n_train" not in metrics.columns:
        return

    n_train_values = pick_n_train_values_for_param_grid(metrics, n_train)
    if not n_train_values:
        return

    sub = metrics[metrics["n_train"].isin(n_train_values)].copy()
    if sub.empty:
        return

    loss_col = first_existing(
        sub.columns,
        [
            "test_fixed_t_loss_mean_10k",
            "test_random_t_loss_10k",
            "test_loss_0_5_10k",
            "test_loss_0_75_10k",
        ],
    )

    metric_cols = [FID_COL, FMEM_COL]
    if loss_col is not None:
        metric_cols.append(loss_col)

    summary = build_trajectory_summary(
        sub,
        step_col,
        ["n_train", "param_label"],
        metric_cols,
    )
    summary = summary.rename(
        columns={
            f"{FID_COL}_mean": "fid_mean",
            f"{FID_COL}_std": "fid_std",
            f"{FMEM_COL}_mean": "fmem_mean",
            f"{FMEM_COL}_std": "fmem_std",
        }
    )

    save_table(
        summary,
        out_dir,
        "07_seed_averaged_parameterization_trajectories_fixed_n_train",
    )

    ncols = 4 + (1 if loss_col is not None else 0)
    nrows = len(n_train_values)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(1.8 * ncols, GRID_CELL_HEIGHT * nrows),
        sharex=False,
        sharey=False,
    )
    axes = np.atleast_2d(axes)

    def plot_with_band(
        ax: plt.Axes,
        group: pd.DataFrame,
        x_col: str,
        mean_col: str,
        std_col: str,
        label: object,
        color: str,
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

        ax.plot(plot_group[x_col], y, color=color, marker="o", markersize=LINE_MARKER_SIZE, linewidth=2.0, label=str(label))
        ax.fill_between(plot_group[x_col], lower, upper, color=color, alpha=0.15)

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

    def plot_directed_phase(ax: plt.Axes, group: pd.DataFrame, label: object, color: str, log_axes: bool) -> None:
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
            color=color,
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
            s=SCATTER_SIZE_MEDIUM,
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
            s=SCATTER_SIZE_HIGHLIGHT,
            facecolor="none",
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )

        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")

    params = sorted(summary["param_label"].dropna().unique(), key=str)
    if not params:
        return

    param_colors = canonical_param_color_map(params)

    for row_idx, n_train_value in enumerate(n_train_values):
        row_summary = summary[summary["n_train"] == n_train_value]
        if row_summary.empty:
            continue

        if row_idx == 0:
            axes[row_idx, 0].set_title("FID", fontsize=8)
            axes[row_idx, 1].set_title(r"$F_{\mathrm{mem}}$", fontsize=8)
            axes[row_idx, 2].set_title(r"$F_{\mathrm{mem}}$ vs FID", fontsize=8)
            axes[row_idx, 3].set_title(r"$F_{\mathrm{mem}}$ vs FID (log-log)", fontsize=8)
            if loss_col is not None:
                axes[row_idx, 4].set_title("test loss (training steps)", fontsize=8)

        for param in params:
            group = row_summary[row_summary["param_label"] == param].sort_values(step_col)
            if group.empty:
                continue
            color = param_colors[param]

            plot_with_band(
                axes[row_idx, 0],
                group,
                step_col,
                "fid_mean",
                "fid_std",
                param,
                color,
                log_x=True,
                log_y=False,
            )
            plot_with_band(
                axes[row_idx, 1],
                group,
                step_col,
                "fmem_mean",
                "fmem_std",
                param,
                color,
                log_x=True,
                log_y=False,
            )
            plot_directed_phase(axes[row_idx, 2], group, param, color, log_axes=False)
            plot_directed_phase(axes[row_idx, 3], group, param, color, log_axes=True)
            if loss_col is not None:
                plot_with_band(
                    axes[row_idx, 4],
                    group,
                    step_col,
                    f"{loss_col}_mean",
                    f"{loss_col}_std",
                    param,
                    color,
                    log_x=True,
                    log_y=True,
                )

        axes[row_idx, 0].set_ylabel(f"{format_n_train_value(n_train_value)}\n{display_label(FID_COL)}", fontsize=8)
        if row_idx == nrows - 1:
            axes[row_idx, 0].set_xlabel(TAU_LABEL, fontsize=8)
            axes[row_idx, 1].set_xlabel(TAU_LABEL, fontsize=8)
            axes[row_idx, 2].set_xlabel(display_label(FMEM_COL), fontsize=8)
            axes[row_idx, 3].set_xlabel(display_label(FMEM_COL), fontsize=8)
            if loss_col is not None:
                axes[row_idx, 4].set_xlabel("training steps", fontsize=8)

        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            ax.grid(True, alpha=0.3)
            ax.tick_params(
                axis="both",
                which="both",
                left=True,
                bottom=True,
                labelleft=True,
                labelbottom=True,
                labelsize=6,
            )

    param_handles = [
        Line2D([0], [0], color=param_colors[param], marker="o", linewidth=1.8, markersize=LEGEND_MARKER_SIZE, label=str(param))
        for param in params
    ]
    fig.legend(
        handles=param_handles,
        title="param",
        fontsize=7,
        title_fontsize=8,
        loc="lower center",
        ncol=min(4, len(param_handles)),
        frameon=False,
    )

    if len(n_train_values) == 1:
        fig.suptitle(rf"07 parameterization trajectories | {format_n_train_value(n_train_values[0])}", y=0.995, fontsize=13)
    else:
        fig.suptitle(r"07 parameterization trajectories by $N_{\mathrm{train}}$", y=0.995, fontsize=13)
    fig.subplots_adjust(top=0.91, bottom=0.09, hspace=0.35, wspace=0.22)
    save_figure(
        fig,
        out_dir,
        "07_seed_averaged_parameterization_trajectories_fixed_n_train",
        show=show,
    )


def fig_07d_all_run_phase_trajectories_by_n_train(
    metrics: pd.DataFrame,
    out_dir: Path,
    step_col: str,
    n_train: int | None,
    show: bool,
) -> None:
    """All individual run trajectories in the Figure-07 phase columns."""
    required = {"n_train", "param_label", FID_COL, FMEM_COL, step_col}
    if not required.issubset(metrics.columns):
        return

    n_train_values = pick_n_train_values_for_param_grid(metrics, n_train)
    if not n_train_values:
        return

    sub = metrics[metrics["n_train"].isin(n_train_values)].copy()
    if sub.empty:
        return

    run_col = first_existing(sub.columns, ["manifest_index", "run_id", "metrics_file"])
    seed_col = choose_seed_col(sub)
    for column in [FID_COL, FMEM_COL, step_col]:
        sub[column] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub.dropna(subset=["n_train", "param_label", FID_COL, FMEM_COL, step_col])
    if sub.empty:
        return

    source_cols = ["n_train", "param_label", step_col, FID_COL, FMEM_COL]
    for column in [run_col, seed_col, "param_label_raw"]:
        if column is not None and column in sub.columns and column not in source_cols:
            source_cols.insert(2, column)
    save_table(
        sub[source_cols].sort_values([c for c in ["n_train", "param_label", run_col, seed_col, step_col] if c in source_cols]),
        out_dir,
        "07d_all_run_phase_trajectories_by_n_train_source",
    )

    params = sorted(sub["param_label"].dropna().unique(), key=str)
    if not params:
        return
    param_colors = canonical_param_color_map(params)

    positive = sub[(sub[FID_COL] > 0) & (sub[FMEM_COL] > 0)].copy()
    if positive.empty:
        return

    fmem_max = float(sub[FMEM_COL].max())
    fid_min = float(sub[FID_COL].min())
    fid_max = float(sub[FID_COL].max())
    fmem_linear_limits = (0.0, fmem_max * 1.06 if fmem_max > 0 else 1.0)
    fid_linear_limits = (
        max(0.0, fid_min - 0.06 * (fid_max - fid_min)),
        fid_max + 0.06 * (fid_max - fid_min),
    )
    fid_log_limits = (
        max(float(positive[FID_COL].min()) / 1.15, np.finfo(float).tiny),
        float(positive[FID_COL].max()) * 1.15,
    )

    nrows = len(n_train_values)
    ncols = 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2, max(2.5, 1.35 * nrows)),
        sharex=False,
        sharey=False,
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    group_cols = ["param_label"]
    if run_col is not None:
        group_cols.append(run_col)
    elif seed_col is not None:
        group_cols.append(seed_col)

    for row_idx, n_train_value in enumerate(n_train_values):
        row_summary = sub[sub["n_train"] == n_train_value]
        if row_summary.empty:
            continue

        ax_linear = axes[row_idx, 0]
        ax_log = axes[row_idx, 1]

        for keys, group in row_summary.groupby(group_cols, dropna=False):
            param = keys[0] if isinstance(keys, tuple) else keys
            color = param_colors[param]
            phase = group.sort_values(step_col)
            if phase.empty:
                continue

            markevery = max(1, len(phase) // 8)
            ax_linear.plot(
                phase[FMEM_COL],
                phase[FID_COL],
                color=color,
                linewidth=0.9,
                alpha=0.34,
                marker="o",
                markersize=1.4,
                markevery=markevery,
            )

            log_phase = phase[(phase[FID_COL] > 0) & (phase[FMEM_COL] > 0)]
            if not log_phase.empty:
                ax_log.plot(
                    log_phase[FMEM_COL],
                    log_phase[FID_COL],
                    color=color,
                    linewidth=0.9,
                    alpha=0.34,
                    marker="o",
                    markersize=1.4,
                    markevery=markevery,
                )

        ax_linear.set_xlim(*fmem_linear_limits)
        ax_linear.set_ylim(*fid_linear_limits)
        ax_log.set_xlim(*fmem_linear_limits)
        ax_log.set_ylim(*fid_log_limits)
        ax_log.set_yscale("log")

        for ax in [ax_linear, ax_log]:
            ax.grid(True, alpha=0.28)
            ax.tick_params(axis="both", which="both", labelsize=6)
            if row_idx == nrows - 1:
                ax.set_xlabel(display_label(FMEM_COL), fontsize=8)
            compact_locator = mticker.LogLocator(base=10, numticks=4)
            if ax is ax_log:
                ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
                ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        ax_linear.set_ylabel(f"{format_n_train_value(n_train_value)}\nFID", fontsize=8)
        if row_idx == 0:
            ax_linear.set_title(r"FID vs $F_{\mathrm{mem}}$", fontsize=9)
            ax_log.set_title(r"FID vs $F_{\mathrm{mem}}$ (log FID)", fontsize=9)

    param_handles = [
        Line2D(
            [0],
            [0],
            color=param_colors[param],
            linewidth=2.0,
            marker="o",
            markersize=LEGEND_MARKER_SIZE,
            label=str(param),
        )
        for param in params
    ]
    fig.legend(
        handles=param_handles,
        title="param",
        fontsize=7,
        title_fontsize=8,
        loc="lower center",
        ncol=min(3, len(param_handles)),
        frameon=False,
    )
    fig.suptitle(
        r"07d all individual run phase trajectories by $N_{\mathrm{train}}$",
        y=0.995,
        fontsize=12,
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.08, hspace=0.36, wspace=0.28)
    save_figure(
        fig,
        out_dir,
        "07d_all_run_phase_trajectories_by_n_train",
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

    Each row is one N_train value. Each subplot is one rho threshold. Each
    curve is one parameterization averaged over visible seeds/runs.
    """
    if "n_train" not in metrics.columns:
        return

    rho_cols = [(label, col) for label, col in FMEM_RHO_COLS if col in metrics.columns]
    if not rho_cols:
        return

    n_train_values = pick_n_train_values_for_param_grid(metrics, n_train)
    if not n_train_values:
        return

    sub = metrics[metrics["n_train"].isin(n_train_values)].copy()
    if sub.empty:
        return

    summary = (
        sub
        .groupby(["n_train", "param_label", step_col], dropna=False)
        .agg(
            **{f"{col}_mean": (col, "mean") for _, col in rho_cols},
            n_runs=("manifest_index", "nunique"),
        )
        .reset_index()
        .sort_values(["n_train", "param_label", step_col])
    )

    save_table(
        summary,
        out_dir,
        "07c_seed_averaged_parameterization_fmem_thresholds_fixed_n_train",
    )

    params = sorted(summary["param_label"].dropna().unique(), key=str)
    if not params:
        return

    param_colors = canonical_param_color_map(params)

    nrows = len(n_train_values)
    ncols = len(rho_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=grid_figsize(ncols, nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for row_idx, n_train_value in enumerate(n_train_values):
        row_summary = summary[summary["n_train"] == n_train_value]
        if row_summary.empty:
            continue

        for col_idx, (rho_label, col) in enumerate(rho_cols):
            ax = axes[row_idx, col_idx]
            mean_col = f"{col}_mean"
            for label, group in row_summary.groupby("param_label", dropna=False):
                group = group.sort_values(step_col)
                group = group[(group[step_col] > 0) & (group[mean_col] > 0)]
                if group.empty:
                    continue
                ax.plot(
                    group[step_col],
                    group[mean_col],
                    color=param_colors[label],
                    marker="o",
                    markersize=LINE_MARKER_SIZE,
                    linewidth=1.8,
                    label="_nolegend_",
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            if row_idx == 0:
                ax.set_title(rho_label)
            if col_idx == 0:
                ax.set_ylabel(f"{format_n_train_value(n_train_value)}\n{display_label(col)}", fontsize=8)
            else:
                ax.set_ylabel(display_label(col))
            if row_idx == nrows - 1:
                ax.set_xlabel(TAU_LABEL)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="both", labelsize=6)

    param_handles = [
        Line2D([0], [0], color=param_colors[param], marker="o", linewidth=1.8, markersize=LEGEND_MARKER_SIZE, label=str(param))
        for param in params
    ]
    fig.legend(
        handles=param_handles,
        title="param",
        fontsize=7,
        title_fontsize=8,
        loc="lower center",
        ncol=min(4, len(param_handles)),
        frameon=False,
    )

    if len(n_train_values) == 1:
        fig.suptitle(rf"07c parameterization $F_{{\mathrm{{mem}}}}$ threshold sweep | {format_n_train_value(n_train_values[0])}", y=1.03)
    else:
        fig.suptitle(r"07c parameterization $F_{\mathrm{mem}}$ threshold sweep by $N_{\mathrm{train}}$", y=1.03)
    fig.subplots_adjust(top=0.90, bottom=0.11, hspace=0.35, wspace=0.18)
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

    fig = plt.figure(figsize=figsize_from_panel_size(FIG_08_PANEL_SIZE, 2, 2, FIG_08_LAYOUT))
    grid = fig.add_gridspec(2, 2)
    axes = np.array(
        [
            fig.add_subplot(grid[0, :]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]),
        ],
        dtype=object,
    )

    for label, sub in summary.groupby("param_label", dropna=False):
        color = canonical_param_color(label)
        sub = sub.sort_values("n_train")
        axes[0].errorbar(
            sub["n_train"],
            sub["best_fid_mean"],
            yerr=sub["best_fid_std"],
            color=color,
            marker="o",
            linewidth=2.6,
            capsize=3.5,
            capthick=1.6,
            elinewidth=1.6,
            markersize=LEGEND_MARKER_SIZE + 1.0,
            label=str(label),
        )
        axes[1].errorbar(
            sub["n_train"],
            sub["fmem_at_best_fid_mean"],
            yerr=sub["fmem_at_best_fid_std"],
            color=color,
            marker="o",
            linewidth=2.6,
            capsize=3.5,
            capthick=1.6,
            elinewidth=1.6,
            markersize=LEGEND_MARKER_SIZE + 1.0,
            label=str(label),
        )
        log_sub = sub[sub["fmem_at_best_fid_mean"] > 0].copy()
        if not log_sub.empty:
            y = pd.to_numeric(log_sub["fmem_at_best_fid_mean"], errors="coerce").to_numpy(dtype=float)
            yerr = (
                pd.to_numeric(log_sub["fmem_at_best_fid_std"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            lower = np.clip(y - yerr, np.finfo(float).tiny, None)
            upper = y + yerr
            axes[2].errorbar(
                log_sub["n_train"],
                y,
                yerr=np.vstack([y - lower, upper - y]),
                color=color,
                marker="o",
                linewidth=2.6,
                capsize=3.5,
                capthick=1.6,
                elinewidth=1.6,
                markersize=LEGEND_MARKER_SIZE + 1.0,
                label=str(label),
            )

    axes[0].set_ylabel("best FID", fontsize=11, fontweight="bold", color=FIG_08_TEXT_COLOR)
    axes[0].set_title("Best FID", fontsize=11, fontweight="bold", color=FIG_08_TEXT_COLOR)
    axes[1].set_ylabel(r"$\mathbf{F}_{\mathbf{mem}}$ at best FID", fontsize=11, fontweight="bold", color=FIG_08_TEXT_COLOR)
    axes[1].set_title(r"$\mathbf{F}_{\mathbf{mem}}$ at best FID", fontsize=11, fontweight="bold", color=FIG_08_TEXT_COLOR)
    axes[2].set_ylabel(r"$\mathbf{F}_{\mathbf{mem}}$ at best FID", fontsize=11, fontweight="bold", color=FIG_08_TEXT_COLOR)
    axes[2].set_title(r"$\mathbf{F}_{\mathbf{mem}}$ at best FID (log)", fontsize=11, fontweight="bold", color=FIG_08_TEXT_COLOR)
    axes[2].set_yscale("log")

    for ax in axes:
        apply_n_train_log_ticks(ax, n_train_ticks)
        ax.set_xlabel(FIG_08_N_TRAIN_LABEL, fontsize=11, fontweight="bold", color=FIG_08_TEXT_COLOR)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=9, colors=FIG_08_TEXT_COLOR, labelcolor=FIG_08_TEXT_COLOR)
        for tick_label in [*ax.get_xticklabels(which="both"), *ax.get_yticklabels(which="both")]:
            tick_label.set_color(FIG_08_TEXT_COLOR)
            tick_label.set_fontweight("semibold")
        for spine in ax.spines.values():
            spine.set_color(FIG_08_TEXT_COLOR)
            spine.set_linewidth(1.2)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        title="param",
        loc="center left",
        bbox_to_anchor=(0.84, 0.50),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_color(FIG_08_TEXT_COLOR)
    for text in legend.get_texts():
        text.set_fontweight("semibold")
        text.set_color(FIG_08_TEXT_COLOR)
    fig.suptitle(
        r"08 parameterization summary: quality optimum vs $\mathbf{F}_{\mathbf{mem}}$ at quality optimum",
        y=0.965,
        fontsize=13,
        fontweight="bold",
        color=FIG_08_TEXT_COLOR,
    )
    fig.subplots_adjust(**FIG_08_LAYOUT)
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
    GENERATED_FIGURES.clear()

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
    metrics = hide_excluded_parameterizations(metrics)
    if metrics.empty:
        raise RuntimeError("No checkpoint metrics remain after hiding excluded parameterizations.")

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

    fig_01_metric_trajectories_by_n_train(metrics, out_dir, step_col, show=args.show)
    fig_02_metric_trajectories_by_param(metrics, out_dir, step_col, show=args.show)
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
    fig_02e_fid_fmem_tau_grid_by_param(metrics, out_dir, step_col, show=args.show)
    fig_12b_fid_fmem_tau_grid_by_param_mid_n_train(metrics, out_dir, step_col, show=args.show)
    fig_12c_fid_fmem_tau_grid_by_param_merged_n_train(metrics, out_dir, step_col, show=args.show)
    fig_07b_fmem_thresholds_by_n_train(metrics, out_dir, step_col, show=args.show)
    fig_03_final_fid_vs_fmem_faceted_by_n_train(metrics, out_dir, step_col, show=args.show)
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
    fig_07d_all_run_phase_trajectories_by_n_train(
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

    print_figure_captions()
    print("done")


if __name__ == "__main__":
    main()
