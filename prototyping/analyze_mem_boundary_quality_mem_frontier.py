#!/usr/bin/env python
"""Paper-objective summaries for quality/memorization frontier evals.

This script is intentionally separate from the plotting prototype. It focuses
on merged analysis tables that answer:

    Does the target with best FID also have lowest F_mem at best FID?

It accepts one or more eval roots. Each root may either contain combined
``all_checkpoint_metrics.csv`` or per-attempt ``checkpoint_metrics.csv`` files.
When multiple roots are merged, ``run_id`` is the stable run key because
``manifest_index`` can restart in delta eval roots.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_REPO = Path("/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par")
DEFAULT_EVAL_ROOTS = [
    Path("outputs/evals/mem_boundary_20260427_all_done_quality_20260501"),
    Path("outputs/evals/mem_boundary_capacity_20260427_full_trajectory_quality_20260503"),
    Path("outputs/evals/mem_boundary_capacity_20260427_full_trajectory_quality_delta_20260503"),
]
DEFAULT_OUT_DIR = Path("prototyping/figures/paper_objective_quality_mem_frontier_20260503")

FID_COL = "fid_10k"
FMEM_COL = "f_mem"
STEP_COL = "step"

SUMMARY_METRICS = [
    "fid_best",
    "f_mem_at_fid_best",
    "step_best_fid",
    "tau_gen",
    "tau_mem",
    "tau_mem_minus_tau_gen",
    "final_fid",
    "final_f_mem",
    "final_test_random_t_loss_10k",
    "test_random_t_loss_at_fid_best",
]


def resolve_under_repo(repo: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo / path


def label_from_root(root: Path) -> str:
    name = root.name
    if "capacity" in name and "delta" in name:
        return "capacity_delta"
    if "capacity" in name:
        return "capacity_base"
    return "n_sweep"


def load_eval_root(root: Path, label: str) -> pd.DataFrame:
    aggregate = root / "all_checkpoint_metrics.csv"
    frames: list[pd.DataFrame] = []

    if aggregate.exists():
        df = pd.read_csv(aggregate)
        df["metrics_file"] = str(aggregate)
        frames.append(df)
    else:
        for path in sorted((root / "attempts").glob("*/checkpoint_metrics.csv")):
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["metrics_file"] = str(path)
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No checkpoint metrics found under {root}")

    out = pd.concat(frames, ignore_index=True)
    out["eval_root_label"] = label
    out["eval_root"] = str(root)
    return out


def parse_capacity_label(run_id: object) -> str:
    match = re.search(r"_(base\d+_m\d+)_", str(run_id))
    if match:
        return match.group(1)
    return "fixed_capacity"


def add_analysis_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    if "run_id" in out.columns:
        out["run_key"] = out["run_id"].astype(str)
    else:
        out["run_key"] = out["eval_root_label"].astype(str) + ":m" + out["manifest_index"].astype(str)

    out["capacity_label"] = out.get("run_id", pd.Series(index=out.index, dtype=object)).map(parse_capacity_label)
    out["analysis_family"] = np.where(out["capacity_label"].eq("fixed_capacity"), "n_sweep", "capacity")

    if "param" not in out.columns:
        raise RuntimeError("Expected a param column in checkpoint metrics.")
    out["param_label"] = out["param"].astype(str)
    return out


def finite_min_row(rows: pd.DataFrame, metric: str) -> pd.Series:
    valid = rows[np.isfinite(rows[metric])]
    if valid.empty:
        raise RuntimeError(f"No finite {metric} values for run {rows['run_key'].iloc[0]}")
    return valid.sort_values([metric, STEP_COL]).iloc[0]


def first_step_where(rows: pd.DataFrame, mask: pd.Series) -> float:
    hits = rows.loc[mask].sort_values(STEP_COL)
    if hits.empty:
        return math.nan
    return float(hits[STEP_COL].iloc[0])


def value_from(row: pd.Series, column: str) -> float:
    if column not in row.index:
        return math.nan
    return float(row[column])


def derive_run_quantities(
    metrics: pd.DataFrame,
    tau_gen_fid_rel: float,
    tau_mem_fmem_threshold: float,
) -> pd.DataFrame:
    rows_out: list[dict[str, object]] = []

    for run_key, rows in metrics.sort_values([STEP_COL]).groupby("run_key", dropna=False):
        rows = rows.copy()
        best = finite_min_row(rows, FID_COL)
        final = rows.sort_values(STEP_COL).iloc[-1]
        fid_best = float(best[FID_COL])

        tau_gen = first_step_where(rows, rows[FID_COL] <= tau_gen_fid_rel * fid_best)
        tau_mem = first_step_where(rows, rows[FMEM_COL] >= tau_mem_fmem_threshold)
        tau_gap = tau_mem - tau_gen if math.isfinite(tau_gen) and math.isfinite(tau_mem) else math.nan

        meta = rows.iloc[0]
        record = {
            "run_key": run_key,
            "run_id": str(meta.get("run_id", run_key)),
            "eval_root_label": meta.get("eval_root_label"),
            "analysis_family": meta.get("analysis_family"),
            "n_train": int(meta["n_train"]),
            "seed": int(meta["seed"]),
            "param": str(meta["param"]),
            "capacity_label": str(meta["capacity_label"]),
            "checkpoint_rows": int(len(rows)),
            "min_step": int(rows[STEP_COL].min()),
            "max_step": int(rows[STEP_COL].max()),
            "fid_best": fid_best,
            "step_best_fid": int(best[STEP_COL]),
            "f_mem_at_fid_best": value_from(best, FMEM_COL),
            "final_step": int(final[STEP_COL]),
            "final_fid": value_from(final, FID_COL),
            "final_f_mem": value_from(final, FMEM_COL),
            "tau_gen": tau_gen,
            "tau_mem": tau_mem,
            "tau_mem_minus_tau_gen": tau_gap,
            "tau_gen_fid_rel": tau_gen_fid_rel,
            "tau_mem_fmem_threshold": tau_mem_fmem_threshold,
        }

        for column in [
            "f_mem_rho0.3",
            "f_mem_rho0.5",
            "f_mem_rho0.7",
            "test_random_t_loss_10k",
            "test_fixed_t_loss_mean_10k",
            "test_loss_0_5_10k",
            "test_loss_0_75_10k",
        ]:
            record[f"{column}_at_fid_best"] = value_from(best, column)
            record[f"final_{column}"] = value_from(final, column)

        rows_out.append(record)

    return pd.DataFrame(rows_out).sort_values(
        ["analysis_family", "n_train", "capacity_label", "param", "seed", "run_key"]
    )


def aggregate_with_uncertainty(run_summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["analysis_family", "n_train", "capacity_label", "param"]
    agg_spec: dict[str, tuple[str, str]] = {
        "n_runs": ("run_key", "nunique"),
        "n_seeds": ("seed", "nunique"),
    }
    for metric in SUMMARY_METRICS:
        if metric in run_summary.columns:
            agg_spec[f"{metric}_mean"] = (metric, "mean")
            agg_spec[f"{metric}_std"] = (metric, "std")
            agg_spec[f"{metric}_sem"] = (metric, "sem")

    return (
        run_summary.groupby(group_cols, dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(group_cols)
    )


def coverage_table(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["eval_root_label", "analysis_family", "n_train", "capacity_label", "param"], dropna=False)
        .agg(
            runs=("run_key", "nunique"),
            rows=(STEP_COL, "size"),
            seeds=("seed", "nunique"),
            checkpoints_min=(STEP_COL, lambda x: x.groupby(metrics.loc[x.index, "run_key"]).nunique().min()),
            checkpoints_max=(STEP_COL, lambda x: x.groupby(metrics.loc[x.index, "run_key"]).nunique().max()),
            final_step_min=(STEP_COL, lambda x: x.groupby(metrics.loc[x.index, "run_key"]).max().min()),
            final_step_max=(STEP_COL, lambda x: x.groupby(metrics.loc[x.index, "run_key"]).max().max()),
            sample_count=("sample_count", "first"),
            fid_reference_count=("fid_reference_count", "first"),
        )
        .reset_index()
        .sort_values(["analysis_family", "n_train", "capacity_label", "param", "eval_root_label"])
    )


def winner_table(group_summary: pd.DataFrame) -> pd.DataFrame:
    winners: list[dict[str, object]] = []
    group_cols = ["analysis_family", "n_train", "capacity_label"]

    for keys, rows in group_summary.groupby(group_cols, dropna=False):
        fid_star = rows.loc[rows["fid_best_mean"].idxmin()]
        mem_star = rows.loc[rows["f_mem_at_fid_best_mean"].idxmin()]
        record = dict(zip(group_cols, keys))
        record.update(
            k_fid_star=fid_star["param"],
            fid_star_value=float(fid_star["fid_best_mean"]),
            fid_star_f_mem_at_fid_best=float(fid_star["f_mem_at_fid_best_mean"]),
            k_mem_star=mem_star["param"],
            mem_star_f_mem_at_fid_best=float(mem_star["f_mem_at_fid_best_mean"]),
            mem_star_fid_value=float(mem_star["fid_best_mean"]),
            mismatch=bool(fid_star["param"] != mem_star["param"]),
        )
        winners.append(record)

    return pd.DataFrame(winners).sort_values(group_cols)


def pareto_table(group_summary: pd.DataFrame) -> pd.DataFrame:
    rows_out: list[dict[str, object]] = []
    group_cols = ["analysis_family", "n_train", "capacity_label"]

    for keys, rows in group_summary.groupby(group_cols, dropna=False):
        rows = rows.copy()
        for _, row in rows.iterrows():
            dominated = False
            dominated_by: list[str] = []
            for _, other in rows.iterrows():
                if other["param"] == row["param"]:
                    continue
                no_worse = (
                    other["fid_best_mean"] <= row["fid_best_mean"]
                    and other["f_mem_at_fid_best_mean"] <= row["f_mem_at_fid_best_mean"]
                )
                strictly_better = (
                    other["fid_best_mean"] < row["fid_best_mean"]
                    or other["f_mem_at_fid_best_mean"] < row["f_mem_at_fid_best_mean"]
                )
                if no_worse and strictly_better:
                    dominated = True
                    dominated_by.append(str(other["param"]))

            record = dict(zip(group_cols, keys))
            record.update(
                param=row["param"],
                fid_best_mean=float(row["fid_best_mean"]),
                f_mem_at_fid_best_mean=float(row["f_mem_at_fid_best_mean"]),
                pareto=not dominated,
                dominated_by=";".join(dominated_by),
            )
            rows_out.append(record)

    return pd.DataFrame(rows_out).sort_values(group_cols + ["param"])


def save_table(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[table] {path}")
    return path


def plot_metric_by_n(
    summary: pd.DataFrame,
    family: str,
    metric: str,
    ylabel: str,
    out_dir: Path,
    name: str,
) -> None:
    sub = summary[summary["analysis_family"] == family].copy()
    if sub.empty:
        return

    capacities = sorted(sub["capacity_label"].unique())
    ncols = min(3, len(capacities))
    nrows = math.ceil(len(capacities) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), sharey=False)
    axes = np.asarray(axes).reshape(-1)

    for ax, capacity in zip(axes, capacities):
        cap_rows = sub[sub["capacity_label"] == capacity]
        for param, rows in cap_rows.groupby("param"):
            rows = rows.sort_values("n_train")
            ax.errorbar(
                rows["n_train"],
                rows[f"{metric}_mean"],
                yerr=rows.get(f"{metric}_sem"),
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=str(param),
            )
        ax.set_title(capacity)
        ax.set_xlabel("n_train")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(cap_rows["n_train"].unique()))
        ax.set_xticklabels([str(int(v)) for v in sorted(cap_rows["n_train"].unique())])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(capacities):]:
        ax.axis("off")

    fig.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[figure] {path}")


def plot_pareto(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    markers = {"fixed_capacity": "o", "base32_m124": "s", "base64_m124": "^", "base64_m1244": "D"}

    for (capacity, param), rows in summary.groupby(["capacity_label", "param"], dropna=False):
        ax.scatter(
            rows["fid_best_mean"],
            rows["f_mem_at_fid_best_mean"],
            s=70,
            marker=markers.get(capacity, "o"),
            alpha=0.78,
            label=f"{capacity} | {param}",
        )

    ax.set_xlabel("mean best FID")
    ax.set_ylabel(r"mean $F_{mem}$ at best FID")
    ax.set_title("Parameterization means: quality vs memorization")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path = out_dir / "13_pareto_parameterization_means.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[figure] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--eval-root", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tau-gen-fid-rel", type=float, default=1.05)
    parser.add_argument("--tau-mem-fmem-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo
    roots = args.eval_root if args.eval_root is not None else DEFAULT_EVAL_ROOTS
    roots = [resolve_under_repo(repo, root) for root in roots]
    out_dir = resolve_under_repo(repo, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for root in roots:
        label = label_from_root(root)
        frame = load_eval_root(root, label)
        frames.append(frame)
        print(f"[load] {label}: rows={len(frame)} root={root}")

    metrics = add_analysis_columns(pd.concat(frames, ignore_index=True))
    duplicate_step_rows = metrics.duplicated(["run_key", STEP_COL]).sum()
    if duplicate_step_rows:
        raise RuntimeError(f"Found {duplicate_step_rows} duplicate run_key/step rows across merged eval roots.")

    coverage = coverage_table(metrics)
    run_summary = derive_run_quantities(
        metrics,
        tau_gen_fid_rel=float(args.tau_gen_fid_rel),
        tau_mem_fmem_threshold=float(args.tau_mem_fmem_threshold),
    )
    group_summary = aggregate_with_uncertainty(run_summary)
    winners = winner_table(group_summary)
    pareto = pareto_table(group_summary)

    save_table(coverage, out_dir, "01_coverage_by_root_group")
    save_table(run_summary, out_dir, "02_run_derived_quantities")
    save_table(group_summary, out_dir, "03_seed_average_by_group")
    save_table(winners, out_dir, "04_quality_mem_winners")
    save_table(pareto, out_dir, "05_pareto_parameterization_means")

    plot_metric_by_n(
        group_summary,
        family="n_sweep",
        metric="fid_best",
        ylabel="mean best FID",
        out_dir=out_dir,
        name="10_n_sweep_best_fid_by_n",
    )
    plot_metric_by_n(
        group_summary,
        family="n_sweep",
        metric="f_mem_at_fid_best",
        ylabel=r"mean $F_{mem}$ at best FID",
        out_dir=out_dir,
        name="11_n_sweep_fmem_at_best_fid_by_n",
    )
    plot_metric_by_n(
        group_summary,
        family="capacity",
        metric="fid_best",
        ylabel="mean best FID",
        out_dir=out_dir,
        name="12_capacity_best_fid_by_n_capacity",
    )
    plot_metric_by_n(
        group_summary,
        family="capacity",
        metric="f_mem_at_fid_best",
        ylabel=r"mean $F_{mem}$ at best FID",
        out_dir=out_dir,
        name="12b_capacity_fmem_at_best_fid_by_n_capacity",
    )
    plot_pareto(group_summary, out_dir)

    print(f"[done] wrote paper-objective analysis to {out_dir}")


if __name__ == "__main__":
    main()
