"""
Direct prototype runner for synthetic data with a time-conditioned MLP.

This file intentionally keeps the synthetic run visible. It builds projected
GMM data, configures GDFM+MLP, runs the optimizer loop, samples periodically,
computes ambient/latent/oracle metrics, writes local artifacts, and optionally
logs to W&B without delegating the full job to the canonical trainer.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


# -----------------------------------------------------------------------------
# Local Source Path
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve()
REPO_ROOT = None
for CANDIDATE in (HERE, *HERE.parents):
    if (CANDIDATE / "src").is_dir() and (CANDIDATE / "prototyping").is_dir():
        REPO_ROOT = CANDIDATE
        break
if REPO_ROOT is None:
    raise RuntimeError("Could not find repo root from {}".format(HERE))

SRC_ROOT = REPO_ROOT / "src"
for PATH in (REPO_ROOT, SRC_ROOT):
    if str(PATH) not in sys.path:
        sys.path.insert(0, str(PATH))


from prototyping.metrics_synthetic_controlled import (
    memorization_metrics_in_spaces,
    oracle_gmm_metrics,
    project_to_subspace,
)


RUNNER_NAME = "synthetic_mlp"
MODULE_NAME = "prototyping.runners.synthetic_mlp"
DEFAULT_OUT_DIR = "outputs/prototype_runners/synthetic_mlp"
DEFAULT_RUN_NAME = "synthetic_projected_gmm_mlp"
PARAMETERIZATION_CHOICES = ("x", "eps", "v")


def parse_args(argv=None):
    """Parse the direct synthetic runner command line."""
    parser = argparse.ArgumentParser(
        description="Train a projected-GMM GDFM MLP and track synthetic metrics."
    )

    # The synthetic runner is intentionally CLI-first: each group below mirrors
    # a visible block in `main` so notebooks can teach the full experiment path.
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu, ...")
    parser.add_argument("--out-dir", "--study-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")

    parser.add_argument(
        "--data-source",
        choices=("canonical", "manual", "both"),
        default="both",
    )
    parser.add_argument("--data-dim", type=int, default=2, help="Ambient dimension D.")
    parser.add_argument("--intrinsic-dim", type=int, default=2, help="Intrinsic dimension d.")
    parser.add_argument("--n-train", type=int, default=10)
    parser.add_argument("--n-reference", "--n-eval", dest="n_reference", type=int, default=1024)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--sigma-z", type=float, default=0.10)
    parser.add_argument("--sigma-eta", type=float, default=1.0e-3)
    parser.add_argument("--geometry-seed", type=int, default=0)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--time-emb-dim", type=int, default=64)
    parser.add_argument(
        "--prediction-space",
        "--pred-space",
        dest="prediction_space",
        choices=PARAMETERIZATION_CHOICES,
        default="v",
    )
    parser.add_argument("--loss-space", choices=PARAMETERIZATION_CHOICES, default="v")

    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument(
        "--batch-cap",
        type=int,
        default=0,
        help="Maximum training batch size. 0 uses all train points.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Legacy alias for --batch-cap.",
    )
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--metric-every", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=2000)

    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--sampling-method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--sampling-steps", type=int, default=128)
    parser.add_argument("--ratio-threshold", type=float, default=0.3)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--sliced-w2-projections", type=int, default=128)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--wandb-project", default="mem_par")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-notes", default="")
    parser.add_argument("--wandb-strict", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size is not None:
        # Legacy alias: after parsing, everything downstream reads `batch_cap`.
        args.batch_cap = args.batch_size
    return args


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def _count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable)}


def _wandb_status_payload(args, wandb_dir, run=None, *, active=False, warning=None):
    return {
        "enabled": bool(args.wandb),
        "active": bool(active),
        "finished": False,
        "mode": str(args.wandb_mode),
        "project": str(args.wandb_project),
        "entity": args.wandb_entity.strip() or None,
        "run_id": getattr(run, "id", None) if run is not None else None,
        "run_name": getattr(run, "name", None) if run is not None else None,
        "run_path": getattr(run, "path", None) if run is not None else None,
        "run_url": getattr(run, "url", None) if run is not None else None,
        "dir": str(Path(wandb_dir).resolve()),
        "warning": warning,
    }


def _wandb_scalar_payload(payload):
    return {key: value for key, value in payload.items() if value is not None}


def _validate_args(args):
    checks = [
        (args.n_train >= 2, "--n-train must be at least 2 for nearest-neighbor metrics."),
        (args.n_reference >= 1, "--n-reference/--n-eval must be at least 1."),
        (args.eval_samples >= 1, "--eval-samples must be at least 1."),
        (args.data_dim >= 1, "--data-dim must be at least 1."),
        (args.intrinsic_dim >= 1, "--intrinsic-dim must be at least 1."),
        (args.intrinsic_dim <= args.data_dim, "--intrinsic-dim must be <= --data-dim."),
        (args.n_clusters >= 1, "--n-clusters must be at least 1."),
        (args.sigma_z > 0.0, "--sigma-z must be positive."),
        (args.sigma_eta >= 0.0, "--sigma-eta must be non-negative."),
        (args.sliced_w2_projections >= 1, "--sliced-w2-projections must be at least 1."),
        (args.n_steps >= 1, "--n-steps must be at least 1."),
        (args.batch_cap >= 0, "--batch-cap must be non-negative; 0 means all train points."),
        (args.log_every >= 1, "--log-every must be at least 1."),
        (args.metric_every >= 1, "--metric-every must be at least 1."),
        (args.ckpt_every >= 1, "--ckpt-every must be at least 1."),
        (args.sampling_steps >= 1, "--sampling-steps must be at least 1."),
        (args.chunk_size >= 1, "--chunk-size must be at least 1."),
    ]
    for ok, message in checks:
        if not ok:
            raise SystemExit(message)


def _manual_projected_gmm(args, torch):
    """Build the projected GMM inline so the data equations stay auditable."""
    # Geometry is shared by train/reference: one latent GMM, one orthonormal
    # projection basis, then independent sample seeds for the two splits.
    geometry_generator = torch.Generator().manual_seed(int(args.geometry_seed))
    basis, _ = torch.linalg.qr(
        torch.randn(args.data_dim, args.intrinsic_dim, generator=geometry_generator)
    )
    basis = basis[:, : args.intrinsic_dim]
    means = torch.randn(args.n_clusters, args.intrinsic_dim, generator=geometry_generator)

    # Train set: choose components in latent space, add latent noise, project to
    # ambient D, then add small off-subspace observation noise.
    train_generator = torch.Generator().manual_seed(int(args.seed))
    train_labels = torch.randint(args.n_clusters, (args.n_train,), generator=train_generator)
    train_latent = means[train_labels] + args.sigma_z * torch.randn(
        args.n_train,
        args.intrinsic_dim,
        generator=train_generator,
    )
    train = train_latent @ basis.T
    train = train + args.sigma_eta * torch.randn(
        args.n_train,
        args.data_dim,
        generator=train_generator,
    )

    # Reference uses the same planted distribution but a different sample seed.
    # This keeps the evaluation target held out without changing geometry.
    reference_generator = torch.Generator().manual_seed(int(args.seed) + 1)
    reference_labels = torch.randint(
        args.n_clusters,
        (args.n_reference,),
        generator=reference_generator,
    )
    reference_latent = means[reference_labels] + args.sigma_z * torch.randn(
        args.n_reference,
        args.intrinsic_dim,
        generator=reference_generator,
    )
    reference = reference_latent @ basis.T
    reference = reference + args.sigma_eta * torch.randn(
        args.n_reference,
        args.data_dim,
        generator=reference_generator,
    )

    return {
        "train": train.contiguous(),
        "reference": reference.contiguous(),
        "train_labels": train_labels,
        "reference_labels": reference_labels,
        "geometry": {"basis": basis.contiguous(), "means": means.contiguous()},
        "train_source_id": "manual_gmm_subspace:seed:{}".format(int(args.seed)),
        "reference_source_id": "manual_gmm_subspace:seed:{}".format(int(args.seed) + 1),
    }


def _compare_bundles(manual_bundle, canonical_bundle, torch):
    """Compare manual equations against `mp_data` for regression safety."""
    # `--data-source both` is a teaching mode: it proves the readable equations
    # in this file still match the canonical data builder byte-for-byte.
    pairs = {
        "train": (manual_bundle["train"], canonical_bundle["train"]),
        "reference": (manual_bundle["reference"], canonical_bundle["reference"]),
        "train_labels": (manual_bundle["train_labels"], canonical_bundle["train_labels"]),
        "reference_labels": (
            manual_bundle["reference_labels"],
            canonical_bundle["reference_labels"],
        ),
        "basis": (
            manual_bundle["geometry"]["basis"],
            canonical_bundle["geometry"]["basis"],
        ),
        "means": (
            manual_bundle["geometry"]["means"],
            canonical_bundle["geometry"]["means"],
        ),
    }
    report = {"data_source": "both", "all_equal": True, "items": {}}
    for name, (left, right) in pairs.items():
        equal = bool(torch.equal(left, right))
        report["items"][name] = {
            "equal": equal,
            "shape": list(left.shape),
            "max_abs_diff": float((left.float() - right.float()).abs().max().item()),
        }
        report["all_equal"] = bool(report["all_equal"] and equal)
    return report


def _summarize_data_bundle(data_bundle):
    summary = {
        "train_shape": list(data_bundle["train"].shape),
        "reference_shape": list(data_bundle["reference"].shape),
        "train_source_id": data_bundle.get("train_source_id"),
        "reference_source_id": data_bundle.get("reference_source_id"),
    }
    if data_bundle.get("train_labels") is not None:
        summary["train_labels_shape"] = list(data_bundle["train_labels"].shape)
    if data_bundle.get("reference_labels") is not None:
        summary["reference_labels_shape"] = list(data_bundle["reference_labels"].shape)
    if data_bundle.get("geometry") is not None:
        summary["geometry"] = {
            key: list(value.shape)
            for key, value in data_bundle["geometry"].items()
            if hasattr(value, "shape")
        }
    return summary


def _metric_record(
    *,
    step,
    samples,
    train_tensor,
    train_latent,
    basis,
    latent_means,
    baseline_mem,
    baseline_oracle_metrics,
    baseline_latent,
    sliced_directions,
    args,
    config,
):
    # Metrics are split into three views:
    # 1. nearest-neighbor memorization in ambient and latent coordinates,
    # 2. oracle GMM likelihood/coverage diagnostics,
    # 3. reference-baseline versions so generated samples have context.
    generated = memorization_metrics_in_spaces(
        samples,
        train_tensor,
        basis,
        train_latent,
        chunk_size=args.chunk_size,
        config=config,
    )
    samples_oracle_metrics = oracle_gmm_metrics(
        samples,
        basis,
        latent_means,
        args.sigma_z,
        args.sigma_eta,
        baseline_latent,
        sliced_directions,
    )
    return {
        "step": int(step),
        "ambient_f_mem": generated["ambient_f_mem"],
        "ambient_mean_nn_ratio": generated["ambient_mean_nn_ratio"],
        "ambient_std_nn_ratio": generated["ambient_std_nn_ratio"],
        "latent_f_mem": generated["latent_f_mem"],
        "latent_mean_nn_ratio": generated["latent_mean_nn_ratio"],
        "latent_std_nn_ratio": generated["latent_std_nn_ratio"],
        "baseline_ambient_f_mem": baseline_mem["ambient_f_mem"],
        "baseline_ambient_mean_nn_ratio": baseline_mem["ambient_mean_nn_ratio"],
        "baseline_ambient_std_nn_ratio": baseline_mem["ambient_std_nn_ratio"],
        "baseline_latent_f_mem": baseline_mem["latent_f_mem"],
        "baseline_latent_mean_nn_ratio": baseline_mem["latent_mean_nn_ratio"],
        "baseline_latent_std_nn_ratio": baseline_mem["latent_std_nn_ratio"],
        "oracle_latent_nll_mean": samples_oracle_metrics["oracle_latent_nll_mean"],
        "oracle_latent_nll_std": samples_oracle_metrics["oracle_latent_nll_std"],
        "oracle_full_nll_mean": samples_oracle_metrics["oracle_full_nll_mean"],
        "off_subspace_energy_fraction": samples_oracle_metrics[
            "off_subspace_energy_fraction"
        ],
        "latent_frechet_distance_to_oracle": samples_oracle_metrics[
            "latent_frechet_distance_to_oracle"
        ],
        "latent_sliced_w2": samples_oracle_metrics["latent_sliced_w2"],
        "component_weight_l1": samples_oracle_metrics["component_weight_l1"],
        "component_weight_kl": samples_oracle_metrics["component_weight_kl"],
        "component_probs": samples_oracle_metrics["component_probs"],
        "residual_norm_mean": samples_oracle_metrics["residual_norm_mean"],
        "baseline_oracle_latent_nll_mean": baseline_oracle_metrics[
            "oracle_latent_nll_mean"
        ],
        "baseline_oracle_full_nll_mean": baseline_oracle_metrics[
            "oracle_full_nll_mean"
        ],
        "baseline_off_subspace_energy_fraction": baseline_oracle_metrics[
            "off_subspace_energy_fraction"
        ],
        "baseline_latent_frechet_distance_to_oracle": baseline_oracle_metrics[
            "latent_frechet_distance_to_oracle"
        ],
        "baseline_component_weight_l1": baseline_oracle_metrics["component_weight_l1"],
        "baseline_component_weight_kl": baseline_oracle_metrics["component_weight_kl"],
        "baseline_component_probs": baseline_oracle_metrics["component_probs"],
        "f_mem": generated["ambient_f_mem"],
        "mean_nn_ratio": generated["ambient_mean_nn_ratio"],
        "std_nn_ratio": generated["ambient_std_nn_ratio"],
        "baseline_f_mem": baseline_mem["ambient_f_mem"],
        "baseline_mean_nn_ratio": baseline_mem["ambient_mean_nn_ratio"],
        "baseline_std_nn_ratio": baseline_mem["ambient_std_nn_ratio"],
    }


def _render_command(args):
    """Render the minimal command needed to recreate this run."""
    # Keep this command focused on scientific knobs; W&B/path extras are left to
    # the full argparse surface above.
    options = {
        "device": args.device,
        "out-dir": args.out_dir,
        "run-name": args.run_name,
        "seed": args.seed,
        "data-source": args.data_source,
        "data-dim": args.data_dim,
        "intrinsic-dim": args.intrinsic_dim,
        "n-train": args.n_train,
        "n-reference": args.n_reference,
        "n-clusters": args.n_clusters,
        "sigma-z": args.sigma_z,
        "sigma-eta": args.sigma_eta,
        "geometry-seed": args.geometry_seed,
        "hidden-dim": args.hidden_dim,
        "n-layers": args.n_layers,
        "time-emb-dim": args.time_emb_dim,
        "prediction-space": args.prediction_space,
        "loss-space": args.loss_space,
        "n-steps": args.n_steps,
        "batch-cap": args.batch_cap,
        "eval-samples": args.eval_samples,
        "sampling-steps": args.sampling_steps,
        "validate-only": bool(args.validate_only),
    }
    command = ["python", "-m", MODULE_NAME]
    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = "--{}".format(key)
        if value is True:
            command.append(flag)
        else:
            command.extend([flag, str(value)])
    return " ".join(command)


def _batching_metadata(n_train, batch_cap, n_steps):
    """Resolve the direct-runner B-cap policy without invoking canonical train."""
    if int(n_train) < 1:
        raise ValueError("--n-train must be at least 1 for training.")
    batch_cap = int(batch_cap)
    if batch_cap <= 0:
        # Synthetic kept `0` as "full batch" before `batch_cap` existed.
        batch_cap = int(n_train)
    if batch_cap < 1:
        raise ValueError(f"--batch-cap must be >= 1, got {batch_cap}")

    effective_batch_size = min(batch_cap, int(n_train))
    drop_last = int(n_train) > batch_cap
    steps_per_epoch = int(n_train) // effective_batch_size if drop_last else 1
    examples_per_epoch = steps_per_epoch * effective_batch_size
    dropped_per_epoch = int(n_train) - examples_per_epoch
    return {
        "batch_cap": int(batch_cap),
        "effective_batch_size": int(effective_batch_size),
        "drop_last": bool(drop_last),
        "steps_per_epoch": int(steps_per_epoch),
        "examples_per_epoch": int(examples_per_epoch),
        "dropped_per_epoch": int(dropped_per_epoch),
        "n_train": int(n_train),
        "n_steps": int(n_steps),
    }


def _make_figures(
    *,
    figures_dir,
    train_history,
    metric_records,
    train_tensor,
    reference_tensor,
    final_samples,
    train_latent,
    reference_latent,
    final_samples_latent,
    final_samples_residual_norm,
    baseline_residual_norm,
    basis,
    latent_means,
    ambient_means,
    args,
    config,
):
    figure_paths = {}
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        warning_path = Path(figures_dir) / "figure_generation_error.txt"
        warning_path.write_text("{}: {}".format(type(exc).__name__, exc), encoding="utf-8")
        return {"figure_generation_error": str(warning_path)}

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    family_label = str(config.framework.family).upper()
    arch_label = "{} layers={} hidden={} time_emb={}".format(
        str(config.network.arch).upper(),
        args.n_layers,
        args.hidden_dim,
        args.time_emb_dim,
    )
    data_label = (
        "GMM D={} d={} n_train={} n_ref={} clusters={} "
        "sigma_z={:.3g} sigma_eta={:.3g}"
    ).format(
        args.data_dim,
        args.intrinsic_dim,
        args.n_train,
        args.n_reference,
        args.n_clusters,
        args.sigma_z,
        args.sigma_eta,
    )
    param_label = "source={} family={} pred_param={} loss_param={} sampler={}x{}".format(
        args.data_source,
        family_label,
        args.prediction_space,
        args.loss_space,
        args.sampling_method,
        args.sampling_steps,
    )
    train_label = "seed={} train_steps={} batch={} lr={:.3g} weight_decay={:.3g}".format(
        args.seed,
        args.n_steps,
        args.batch_cap if args.batch_cap > 0 else args.n_train,
        args.lr,
        args.weight_decay,
    )
    figure_context = "{}\n{} | {}\n{}".format(data_label, arch_label, param_label, train_label)

    if train_history:
        path = figures_dir / "training_loss_vs_step.png"
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(
            [row["step"] for row in train_history],
            [row["loss"] for row in train_history],
            linewidth=1.8,
        )
        ax.set_title("Training loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths["training_loss_vs_step"] = str(path)

    if metric_records:
        steps = [row["step"] for row in metric_records]
        plot_specs = [
            (
                "f_mem_vs_step",
                "Memorization fraction",
                "f_mem",
                [
                    ("generated ambient", [row["ambient_f_mem"] for row in metric_records], "-"),
                    ("generated latent", [row["latent_f_mem"] for row in metric_records], "-"),
                    (
                        "reference ambient",
                        [row["baseline_ambient_f_mem"] for row in metric_records],
                        "--",
                    ),
                    (
                        "reference latent",
                        [row["baseline_latent_f_mem"] for row in metric_records],
                        "--",
                    ),
                ],
            ),
            (
                "oracle_nll_vs_step",
                "Oracle GMM negative log-likelihood",
                "NLL",
                [
                    (
                        "generated latent NLL",
                        [row["oracle_latent_nll_mean"] for row in metric_records],
                        "-",
                    ),
                    (
                        "reference latent NLL",
                        [row["baseline_oracle_latent_nll_mean"] for row in metric_records],
                        "--",
                    ),
                ],
            ),
            (
                "latent_distance_vs_step",
                "Latent distribution distances",
                "distance",
                [
                    (
                        "latent Frechet",
                        [row["latent_frechet_distance_to_oracle"] for row in metric_records],
                        "-",
                    ),
                    (
                        "latent sliced-W2",
                        [row["latent_sliced_w2"] for row in metric_records],
                        "-",
                    ),
                ],
            ),
            (
                "residual_energy_vs_step",
                "Off-subspace energy",
                "residual / total energy",
                [
                    (
                        "generated",
                        [row["off_subspace_energy_fraction"] for row in metric_records],
                        "-",
                    ),
                    (
                        "reference",
                        [row["baseline_off_subspace_energy_fraction"] for row in metric_records],
                        "--",
                    ),
                ],
            ),
        ]
        for name, title, ylabel, series in plot_specs:
            path = figures_dir / "{}.png".format(name)
            fig, ax = plt.subplots(figsize=(8, 4.6))
            for label, values, linestyle in series:
                ax.plot(steps, values, marker="o", linestyle=linestyle, label=label)
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend()
            fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
            fig.tight_layout(rect=(0, 0.17, 1, 1))
            fig.savefig(path, dpi=160)
            plt.close(fig)
            figure_paths[name] = str(path)

        path = figures_dir / "nn_ratio_vs_step.png"
        fig, ax = plt.subplots(figsize=(8, 4.6))
        means = [row["ambient_mean_nn_ratio"] for row in metric_records]
        stds = [row["ambient_std_nn_ratio"] for row in metric_records]
        ax.fill_between(
            steps,
            [max(0.0, mean - std) for mean, std in zip(means, stds)],
            [mean + std for mean, std in zip(means, stds)],
            alpha=0.18,
            label="generated +/- std",
        )
        ax.plot(steps, means, marker="o", label="generated mean")
        ax.plot(
            steps,
            [row["baseline_ambient_mean_nn_ratio"] for row in metric_records],
            marker="o",
            linestyle="--",
            label="reference baseline mean",
        )
        ax.set_title("Nearest-neighbor ratio")
        ax.set_xlabel("step")
        ax.set_ylabel("NN ratio")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths["nn_ratio_vs_step"] = str(path)

    if args.intrinsic_dim >= 2:
        latent_points = [train_latent, reference_latent, final_samples_latent, latent_means]
        limits = __import__("torch").cat(latent_points, dim=0)
        x_min = float(limits[:, 0].min().item()) - 0.5
        x_max = float(limits[:, 0].max().item()) + 0.5
        y_min = float(limits[:, 1].min().item()) - 0.5
        y_max = float(limits[:, 1].max().item()) + 0.5

        for name, title, generated in (
            ("latent_data_scatter", "Data in true latent coordinates", False),
            ("latent_generated_scatter", "Generated samples in true latent coordinates", True),
        ):
            path = figures_dir / "{}.png".format(name)
            fig, ax = plt.subplots(figsize=(5.8, 6.1))
            ax.scatter(train_latent[:, 0].numpy(), train_latent[:, 1].numpy(), s=24, alpha=0.35, label="train")
            if generated:
                ax.scatter(
                    final_samples_latent[:, 0].numpy(),
                    final_samples_latent[:, 1].numpy(),
                    s=14,
                    alpha=0.75,
                    label="generated",
                )
            else:
                ax.scatter(
                    reference_latent[:, 0].numpy(),
                    reference_latent[:, 1].numpy(),
                    s=8,
                    alpha=0.20,
                    label="reference",
                )
            ax.scatter(
                latent_means[:, 0].numpy(),
                latent_means[:, 1].numpy(),
                s=90,
                marker="x",
                linewidths=2.0,
                label="latent means",
            )
            ax.set_title(title)
            ax.set_xlabel("x @ B [0]")
            ax.set_ylabel("x @ B [1]")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.20)
            ax.legend()
            fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
            fig.tight_layout(rect=(0, 0.17, 1, 1))
            fig.savefig(path, dpi=160)
            plt.close(fig)
            figure_paths[name] = str(path)

    if args.data_dim == 2:
        torch = __import__("torch")
        ambient_points = torch.cat([train_tensor, reference_tensor, final_samples, ambient_means], dim=0)
        x_min = float(ambient_points[:, 0].min().item()) - 0.5
        x_max = float(ambient_points[:, 0].max().item()) + 0.5
        y_min = float(ambient_points[:, 1].min().item()) - 0.5
        y_max = float(ambient_points[:, 1].max().item()) + 0.5

        path = figures_dir / "ambient_generated_scatter.png"
        fig, ax = plt.subplots(figsize=(5.8, 6.1))
        ax.scatter(train_tensor[:, 0].numpy(), train_tensor[:, 1].numpy(), s=24, alpha=0.25, label="train")
        ax.scatter(final_samples[:, 0].numpy(), final_samples[:, 1].numpy(), s=14, alpha=0.75, label="generated")
        ax.scatter(
            ambient_means[:, 0].numpy(),
            ambient_means[:, 1].numpy(),
            s=90,
            marker="x",
            linewidths=2.0,
            label="ambient means",
        )
        ax.set_title("Generated samples in ambient coordinates")
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.20)
        ax.legend()
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths["ambient_generated_scatter"] = str(path)

    path = figures_dir / "residual_norm_histogram.png"
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(baseline_residual_norm.numpy(), bins=32, alpha=0.50, label="reference")
    ax.hist(final_samples_residual_norm.numpy(), bins=32, alpha=0.50, label="generated")
    ax.set_title("Off-subspace residual norms")
    ax.set_xlabel("||x - (x @ B) B.T||")
    ax.set_ylabel("count")
    ax.grid(alpha=0.20)
    ax.legend()
    fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths["residual_norm_histogram"] = str(path)

    if metric_records and metric_records[-1].get("component_probs"):
        path = figures_dir / "component_coverage_final.png"
        xs = list(range(args.n_clusters))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.bar(
            [x - width / 2.0 for x in xs],
            metric_records[-1]["baseline_component_probs"],
            width=width,
            label="reference",
        )
        ax.bar(
            [x + width / 2.0 for x in xs],
            metric_records[-1]["component_probs"],
            width=width,
            label="generated",
        )
        ax.axhline(1.0 / float(args.n_clusters), linestyle="--", linewidth=1.2, label="uniform")
        ax.set_title("Final GMM component coverage")
        ax.set_xlabel("component")
        ax.set_ylabel("mean responsibility")
        ax.grid(alpha=0.20, axis="y")
        ax.legend()
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths["component_coverage_final"] = str(path)

    return figure_paths


def main(argv=None):
    """Run one direct synthetic MLP job, or a validate-only component pass."""
    args = parse_args(argv)

    try:
        import torch
        from torch.utils.data import DataLoader
        from gdfm import GDFM
        from mp_config import config_to_dict, get_base_config
        from mp_data import build_data_bundle
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: {}\nCurrent Python: {}\nActivate the project environment first.".format(
                exc.name or exc,
                sys.executable,
            )
        )

    _validate_args(args)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Optimizer step is the x-axis. Batching metadata explains how the selected
    # N is partitioned, but training stops strictly at `n_steps`.
    batching = _batching_metadata(args.n_train, args.batch_cap, args.n_steps)
    batch_size = int(batching["effective_batch_size"])

    torch.manual_seed(int(args.seed))
    generator = torch.Generator().manual_seed(int(args.seed))

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------

    config = get_base_config()
    # This direct runner writes its own artifact tree, but the config still uses
    # the canonical shape so model/data helpers behave exactly as in `src/`.
    config.study.name = RUNNER_NAME
    config.study.description = "Direct synthetic-data GDFM/MLP prototype runner."
    config.outputs.run_name = args.run_name

    config.framework.family = "fm"
    # These two fields are the parameterization comparison surface.
    config.framework.prediction_space = args.prediction_space
    config.framework.loss_space = args.loss_space

    # Synthetic data is fully materialized before training; there is no raw
    # streaming or augmentation hidden behind the training loop.
    config.data.dataset = "gmm_subspace"
    config.data.root = "synthetic"
    config.data.download = False
    config.data.data_source = args.data_source
    config.data.n_train = args.n_train
    config.data.n_eval = args.n_reference
    config.data.data_dim = args.data_dim
    config.data.intrinsic_dim = args.intrinsic_dim
    config.data.n_clusters = args.n_clusters
    config.data.sigma_z = args.sigma_z
    config.data.sigma_eta = args.sigma_eta
    config.data.geometry_seed = args.geometry_seed
    config.data.flatten = False

    config.network.arch = "mlp"
    config.network.hidden_dim = args.hidden_dim
    config.network.n_layers = args.n_layers
    config.network.time_emb_dim = args.time_emb_dim

    config.training.seed = args.seed
    config.training.n_steps = args.n_steps
    config.training.batch_cap = int(batching["batch_cap"])
    # Compatibility mirror for older code that still reads `batch_size`.
    config.training.batch_size = int(batching["batch_cap"])
    config.training.lr = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.ckpt_every = args.ckpt_every
    config.training.log_every = args.log_every
    config.training.optimizer = "adam"

    config.evaluation.n_samples = args.eval_samples
    config.evaluation.batch_size = int(batching["batch_cap"])
    config.evaluation.ratio_threshold = args.ratio_threshold
    config.evaluation.chunk_size = args.chunk_size
    config.evaluation.sample_panel_count = min(8, args.eval_samples)

    config.sampling.method = args.sampling_method
    config.sampling.n_steps = args.sampling_steps
    config.sliced_w2_projections = args.sliced_w2_projections

    config_payload = config_to_dict(config)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    canonical_bundle = None
    manual_bundle = None
    if args.data_source in ("canonical", "both"):
        canonical_bundle = build_data_bundle(config)
    if args.data_source in ("manual", "both"):
        manual_bundle = _manual_projected_gmm(args, torch)

    # `manual` is for inspecting the equations; `canonical` is for testing the
    # shared data path; `both` asserts they are identical before proceeding.
    data_bundle = manual_bundle if args.data_source == "manual" else canonical_bundle
    data_equivalence_report = None
    if args.data_source == "both":
        data_equivalence_report = _compare_bundles(manual_bundle, canonical_bundle, torch)
        if not data_equivalence_report["all_equal"]:
            raise SystemExit("Manual and canonical projected GMM data did not match.")

    train_tensor = data_bundle["train"].contiguous()
    reference_tensor = data_bundle["reference"].contiguous()
    train_labels = data_bundle["train_labels"]
    reference_labels = data_bundle["reference_labels"]
    basis = data_bundle["geometry"]["basis"].contiguous()
    latent_means = data_bundle["geometry"]["means"].contiguous()
    ambient_means = latent_means @ basis.T

    train_parts = project_to_subspace(train_tensor, basis)
    reference_parts = project_to_subspace(reference_tensor, basis)
    # Latent/residual decompositions let us distinguish true latent memorization
    # from off-subspace artifacts in ambient space.
    train_latent = train_parts["latent"]
    reference_latent = reference_parts["latent"]
    train_residual = train_parts["residual"]
    reference_residual = reference_parts["residual"]

    z_eval = torch.randn(args.eval_samples, args.data_dim, generator=generator)
    sliced_generator = torch.Generator().manual_seed(int(args.seed) + 1009)
    # Fixed sliced-W2 directions make metric curves comparable across steps.
    sliced_directions = torch.randn(
        args.sliced_w2_projections,
        args.intrinsic_dim,
        generator=sliced_generator,
    )
    sliced_directions = sliced_directions / sliced_directions.norm(
        dim=1,
        keepdim=True,
    ).clamp(min=1.0e-12)

    baseline_count = min(args.eval_samples, reference_tensor.shape[0])
    baseline_tensor = reference_tensor[:baseline_count]
    # Baseline metrics answer: what would the diagnostics look like for real
    # held-out data sampled from the target distribution?
    baseline_latent = reference_latent[:baseline_count]
    baseline_residual = project_to_subspace(baseline_tensor, basis)["residual"]
    baseline_oracle_metrics = oracle_gmm_metrics(
        baseline_tensor,
        basis,
        latent_means,
        args.sigma_z,
        args.sigma_eta,
        baseline_latent,
        sliced_directions,
    )
    baseline_mem = memorization_metrics_in_spaces(
        baseline_tensor,
        train_tensor,
        basis,
        train_latent,
        chunk_size=args.chunk_size,
        config=config,
    )

    # -------------------------------------------------------------------------
    # Model And Optimizer
    # -------------------------------------------------------------------------

    model = GDFM(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    parameter_counts = _count_parameters(model)

    if args.validate_only:
        # Validate-only performs one real optimizer update and one sampling pass
        # so data/model/loss/metrics can be checked without writing a run.
        model.train()
        validate_loader = DataLoader(
            train_tensor,
            batch_size=batch_size,
            shuffle=True,
            drop_last=bool(batching["drop_last"]),
            generator=torch.Generator().manual_seed(int(args.seed)),
        )
        batch = next(iter(validate_loader)).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0e9)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            samples = model.sample(
                z_eval.clone().to(device),
                n_steps=args.sampling_steps,
                method=args.sampling_method,
                device=device,
            ).cpu()
        record = _metric_record(
            step=0,
            samples=samples,
            train_tensor=train_tensor,
            train_latent=train_latent,
            basis=basis,
            latent_means=latent_means,
            baseline_mem=baseline_mem,
            baseline_oracle_metrics=baseline_oracle_metrics,
            baseline_latent=baseline_latent,
            sliced_directions=sliced_directions,
            args=args,
            config=config,
        )
        payload = {
            "mode": "validate_only",
            "command": _render_command(args),
            "config": {
                "study": config_payload["study"],
                "framework": config_payload["framework"],
                "data": config_payload["data"],
                "network": config_payload["network"],
                "training": config_payload["training"],
                "evaluation": config_payload["evaluation"],
                "sampling": config_payload["sampling"],
            },
            "data": _summarize_data_bundle(data_bundle),
            "manual_vs_canonical": data_equivalence_report,
            "batching": batching,
            "projection": {
                "basis_shape": list(basis.shape),
                "means_shape": list(latent_means.shape),
                "train_latent_shape": list(train_latent.shape),
                "reference_latent_shape": list(reference_latent.shape),
                "train_residual_norm_mean": float(train_residual.norm(dim=1).mean().item()),
                "reference_residual_norm_mean": float(
                    reference_residual.norm(dim=1).mean().item()
                ),
            },
            "model": {
                "class": "{}.{}".format(model.__class__.__module__, model.__class__.__name__),
                "backbone_class": "{}.{}".format(
                    model.net.__class__.__module__,
                    model.net.__class__.__name__,
                ),
                "parameters": parameter_counts,
            },
            "one_update": {
                "batch_shape": list(batch.shape),
                "loss": float(loss.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()),
            },
            "metrics": record,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    # -------------------------------------------------------------------------
    # Output Folders
    # -------------------------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir).expanduser().resolve()
    run_dir = out_root / "{}_seed{}_{}".format(args.run_name, args.seed, timestamp)
    artifacts_dir = run_dir / "artifacts"
    checkpoints_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    logs_dir = run_dir / "logs"
    wandb_dir = logs_dir / "wandb"
    for path in (artifacts_dir, checkpoints_dir, metrics_dir, figures_dir, logs_dir, wandb_dir):
        path.mkdir(parents=True, exist_ok=True)

    latest_link = out_root / "latest"
    latest_checkpoints_link = out_root / "latest_checkpoints"
    for link_path, target_path in (
        (latest_link, run_dir),
        (latest_checkpoints_link, checkpoints_dir),
    ):
        try:
            if link_path.is_symlink():
                link_path.unlink()
            if not link_path.exists():
                link_path.symlink_to(target_path, target_is_directory=True)
        except OSError:
            pass

    train_history_path = metrics_dir / "train_history.json"
    train_history_jsonl_path = logs_dir / "train_history.jsonl"
    metric_records_path = metrics_dir / "metrics_along_training.json"
    metric_records_jsonl_path = logs_dir / "metrics_along_training.jsonl"
    status_path = run_dir / "status.json"

    print("")
    print("----- environment -----")
    print("python           : {}".format(sys.executable))
    print("repo_root        : {}".format(REPO_ROOT))
    print("torch            : {}".format(torch.__version__))
    print("cuda available   : {}".format(torch.cuda.is_available()))
    print("device           : {}".format(device))
    if str(device).startswith("cuda") and torch.cuda.is_available():
        print("cuda name        : {}".format(torch.cuda.get_device_name(0)))
    print("run_dir          : {}".format(run_dir))
    print("latest           : {}".format(latest_link))
    print("latest_checkpts  : {}".format(latest_checkpoints_link))

    print("")
    print("----- config -----")
    print("data             : projected GMM")
    print("data_source      : {}".format(args.data_source))
    print("ambient/intrinsic: D={} d={}".format(args.data_dim, args.intrinsic_dim))
    print("sigma_z/eta      : {}/{}".format(args.sigma_z, args.sigma_eta))
    print("model            : GDFM + MLP")
    print("prediction/loss  : {}/{}".format(args.prediction_space, args.loss_space))
    print("steps            : {}".format(args.n_steps))
    print("batch_cap        : {}".format(batching["batch_cap"]))
    print("effective_batch  : {}".format(batch_size))
    print("drop_last        : {}".format(batching["drop_last"]))
    print("metric_every     : {}".format(args.metric_every))
    print("ckpt_every       : {}".format(args.ckpt_every))

    print("")
    print("----- data -----")
    print("train shape      : {}".format(tuple(train_tensor.shape)))
    print("reference shape  : {}".format(tuple(reference_tensor.shape)))
    print("basis shape      : {}".format(tuple(basis.shape)))
    print("latent means     : {}".format(tuple(latent_means.shape)))
    print(
        "manual==canonical: {}".format(
            data_equivalence_report["all_equal"] if data_equivalence_report else "not checked"
        )
    )

    print("")
    print("----- model -----")
    print("network          : {}.{}".format(model.net.__class__.__module__, model.net.__class__.__name__))
    print("parameters       : {}".format(parameter_counts["total"]))
    print("trainable        : {}".format(parameter_counts["trainable"]))

    torch.save(train_tensor, artifacts_dir / "train_tensor.pt")
    torch.save(reference_tensor, artifacts_dir / "reference_tensor.pt")
    torch.save(train_labels, artifacts_dir / "train_labels.pt")
    torch.save(reference_labels, artifacts_dir / "reference_labels.pt")
    torch.save(basis, artifacts_dir / "basis.pt")
    torch.save(latent_means, artifacts_dir / "latent_means.pt")
    torch.save(ambient_means, artifacts_dir / "ambient_means.pt")
    torch.save(train_latent, artifacts_dir / "train_latent.pt")
    torch.save(reference_latent, artifacts_dir / "reference_latent.pt")
    torch.save(train_residual, artifacts_dir / "train_residual.pt")
    torch.save(reference_residual, artifacts_dir / "reference_residual.pt")
    torch.save(sliced_directions, artifacts_dir / "sliced_w2_directions.pt")
    torch.save(z_eval, artifacts_dir / "z_eval.pt")
    _write_json(artifacts_dir / "config.json", config_payload)

    gmm_params = {
        "data_source": args.data_source,
        "train_source_id": data_bundle.get("train_source_id"),
        "reference_source_id": data_bundle.get("reference_source_id"),
        "data_dim": int(args.data_dim),
        "intrinsic_dim": int(args.intrinsic_dim),
        "n_clusters": int(args.n_clusters),
        "sigma_z": float(args.sigma_z),
        "sigma_eta": float(args.sigma_eta),
        "geometry_seed": int(args.geometry_seed),
        "sample_seed": int(args.seed),
        "sliced_w2_projections": int(args.sliced_w2_projections),
    }
    _write_json(artifacts_dir / "gmm_params.json", gmm_params)
    if data_equivalence_report is not None:
        _write_json(artifacts_dir / "data_equivalence_report.json", data_equivalence_report)

    # -------------------------------------------------------------------------
    # Tracking
    # -------------------------------------------------------------------------

    wandb_run = None
    wandb_module = None
    wandb_metadata = _wandb_status_payload(args, wandb_dir, active=False)
    if args.wandb:
        print("")
        print("----- W&B -----")
        try:
            import wandb

            wandb_module = wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity.strip() or None,
                mode=args.wandb_mode,
                dir=str(wandb_dir),
                name=args.wandb_name.strip() or run_dir.name,
                group=args.wandb_group.strip() or None,
                tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()],
                notes=args.wandb_notes.strip() or None,
                config=config_payload,
            )
            wandb_metadata = _wandb_status_payload(args, wandb_dir, wandb_run, active=True)
            print("wandb mode       : {}".format(wandb_metadata["mode"]))
            print("wandb project    : {}".format(wandb_metadata["project"]))
            if wandb_metadata["run_url"]:
                print("wandb url        : {}".format(wandb_metadata["run_url"]))
            else:
                print("wandb local dir  : {}".format(wandb_metadata["dir"]))
            if args.wandb_mode == "offline":
                print("wandb sync       : wandb sync {}".format(wandb_metadata["dir"]))
        except Exception as exc:
            message = "W&B init failed: {}: {}".format(type(exc).__name__, exc)
            wandb_metadata = _wandb_status_payload(args, wandb_dir, active=False, warning=message)
            if args.wandb_strict:
                raise
            print(message)

    status = {
        "state": "training",
        "run_dir": str(run_dir),
        "latest": str(latest_link),
        "latest_checkpoints": str(latest_checkpoints_link),
        "started_at": timestamp,
        "device": str(device),
        "gmm": gmm_params,
        "wandb": wandb_metadata,
        "wandb_enabled": bool(args.wandb),
        "wandb_active": wandb_run is not None,
        "batching": batching,
    }
    _write_json(status_path, status)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    print("")
    print("----- training -----")
    train_history = []
    metric_records = []
    train_start = time.perf_counter()
    model.train()
    # Training mirrors the canonical B-cap rule: keep the full selected tensor
    # on CPU, shuffle DataLoader batches, and cycle epochs until `n_steps`.
    train_loader = DataLoader(
        train_tensor,
        batch_size=batch_size,
        shuffle=True,
        drop_last=bool(batching["drop_last"]),
        generator=generator,
    )
    train_iterator = iter(train_loader)

    for step in range(1, args.n_steps + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            # Epoch boundaries are bookkeeping only; optimizer steps remain the
            # training clock.
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.n_steps:
            loss_value = float(loss.item())
            row = {
                "step": int(step),
                "loss": loss_value,
                "elapsed_sec": float(time.perf_counter() - train_start),
            }
            train_history.append(row)
            _append_jsonl(train_history_jsonl_path, row)
            print("step {:5d}/{} loss {:.6f}".format(step, args.n_steps, loss_value))
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss_value}, step=step)

        if step % args.metric_every == 0 or step == args.n_steps:
            model.eval()
            with torch.no_grad():
                samples = model.sample(
                    z_eval.clone().to(device),
                    n_steps=args.sampling_steps,
                    method=args.sampling_method,
                    device=device,
                ).cpu()
            record = _metric_record(
                step=step,
                samples=samples,
                train_tensor=train_tensor,
                train_latent=train_latent,
                basis=basis,
                latent_means=latent_means,
                baseline_mem=baseline_mem,
                baseline_oracle_metrics=baseline_oracle_metrics,
                baseline_latent=baseline_latent,
                sliced_directions=sliced_directions,
                args=args,
                config=config,
            )
            metric_records.append(record)
            _append_jsonl(metric_records_jsonl_path, record)
            print(
                "metric step {:5d}: ambient_f_mem {:.4f}, latent_f_mem {:.4f}, oracle_latent_nll {:.4f}".format(
                    step,
                    record["ambient_f_mem"],
                    record["latent_f_mem"],
                    record["oracle_latent_nll_mean"],
                )
            )
            if wandb_run is not None:
                wandb_run.log(
                    _wandb_scalar_payload(
                        {
                            "eval/f_mem": record["f_mem"],
                            "eval/mean_nn_ratio": record["mean_nn_ratio"],
                            "eval/std_nn_ratio": record["std_nn_ratio"],
                            "eval/baseline_f_mem": record["baseline_f_mem"],
                            "eval/baseline_mean_nn_ratio": record["baseline_mean_nn_ratio"],
                            "eval/baseline_std_nn_ratio": record["baseline_std_nn_ratio"],
                            "eval/latent_f_mem": record["latent_f_mem"],
                            "eval/latent_mean_nn_ratio": record["latent_mean_nn_ratio"],
                            "eval/latent_std_nn_ratio": record["latent_std_nn_ratio"],
                            "eval/oracle_latent_nll_mean": record["oracle_latent_nll_mean"],
                            "eval/oracle_full_nll_mean": record["oracle_full_nll_mean"],
                            "eval/off_subspace_energy_fraction": record[
                                "off_subspace_energy_fraction"
                            ],
                            "eval/latent_frechet_distance_to_oracle": record[
                                "latent_frechet_distance_to_oracle"
                            ],
                            "eval/latent_sliced_w2": record["latent_sliced_w2"],
                            "eval/component_weight_l1": record["component_weight_l1"],
                            "eval/component_weight_kl": record["component_weight_kl"],
                        }
                    ),
                    step=step,
                )
            model.train()

        if step % args.ckpt_every == 0 or step == args.n_steps:
            checkpoint_path = checkpoints_dir / "step_{:06d}.pt".format(step)
            torch.save(
                {
                    "step": int(step),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config_payload,
                },
                checkpoint_path,
            )

    # -------------------------------------------------------------------------
    # Final Evaluation And Outputs
    # -------------------------------------------------------------------------

    model.eval()
    with torch.no_grad():
        final_samples = model.sample(
            z_eval.clone().to(device),
            n_steps=args.sampling_steps,
            method=args.sampling_method,
            device=device,
        ).cpu()

    final_sample_parts = project_to_subspace(final_samples, basis)
    final_samples_latent = final_sample_parts["latent"]
    final_samples_projected = final_sample_parts["projected"]
    final_samples_residual = final_sample_parts["residual"]
    final_samples_residual_norm = final_sample_parts["residual_norm"]

    torch.save(final_samples, metrics_dir / "final_samples.pt")
    torch.save(final_samples, metrics_dir / "final_samples_ambient.pt")
    torch.save(final_samples_latent, metrics_dir / "final_samples_latent.pt")
    torch.save(final_samples_projected, metrics_dir / "final_samples_projected.pt")
    torch.save(final_samples_residual, metrics_dir / "final_samples_residual.pt")
    torch.save(final_samples_residual_norm, metrics_dir / "final_samples_residual_norm.pt")
    _write_json(train_history_path, train_history)
    _write_json(metric_records_path, metric_records)

    figure_paths = _make_figures(
        figures_dir=figures_dir,
        train_history=train_history,
        metric_records=metric_records,
        train_tensor=train_tensor,
        reference_tensor=reference_tensor,
        final_samples=final_samples,
        train_latent=train_latent,
        reference_latent=reference_latent,
        final_samples_latent=final_samples_latent,
        final_samples_residual_norm=final_samples_residual_norm,
        baseline_residual_norm=baseline_residual.norm(dim=1),
        basis=basis,
        latent_means=latent_means,
        ambient_means=ambient_means,
        args=args,
        config=config,
    )

    final_metrics = metric_records[-1] if metric_records else {}
    status.update(
        {
            "state": "completed",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "training_wall_time_sec": float(time.perf_counter() - train_start),
            "final_loss": train_history[-1]["loss"] if train_history else None,
            "final_metrics": final_metrics,
            "figures": figure_paths,
            "wandb": wandb_metadata,
            "wandb_active": wandb_run is not None,
            "batching": batching,
        }
    )
    _write_json(status_path, status)

    summary = {
        "run_dir": str(run_dir),
        "config": config_payload,
        "batching": batching,
        "final_loss": train_history[-1]["loss"] if train_history else None,
        "final_metrics": final_metrics,
        "wandb": wandb_metadata,
        "paths": {
            "status": str(status_path),
            "train_history": str(train_history_path),
            "metrics_along_training": str(metric_records_path),
            "final_samples": str(metrics_dir / "final_samples.pt"),
            "final_samples_latent": str(metrics_dir / "final_samples_latent.pt"),
            "final_samples_residual": str(metrics_dir / "final_samples_residual.pt"),
            "gmm_params": str(artifacts_dir / "gmm_params.json"),
            "data_equivalence_report": (
                str(artifacts_dir / "data_equivalence_report.json")
                if data_equivalence_report is not None
                else None
            ),
            "figures": str(figures_dir),
            "checkpoints": str(checkpoints_dir),
            "latest": str(latest_link),
            "latest_checkpoints": str(latest_checkpoints_link),
        },
        "figures": figure_paths,
    }
    _write_json(artifacts_dir / "run_summary.json", summary)

    if wandb_run is not None:
        media_payload = {}
        if wandb_module is not None:
            for name, path in figure_paths.items():
                suffix = Path(path).suffix.lower()
                if suffix in (".png", ".jpg", ".jpeg"):
                    media_payload["figures/{}".format(name)] = wandb_module.Image(path)
        wandb_run.log(
            _wandb_scalar_payload(
                {
                    "final/loss": summary["final_loss"],
                    "final/f_mem": final_metrics.get("f_mem"),
                    "final/mean_nn_ratio": final_metrics.get("mean_nn_ratio"),
                    "final/std_nn_ratio": final_metrics.get("std_nn_ratio"),
                    "final/latent_f_mem": final_metrics.get("latent_f_mem"),
                    "final/latent_mean_nn_ratio": final_metrics.get("latent_mean_nn_ratio"),
                    "final/oracle_latent_nll_mean": final_metrics.get(
                        "oracle_latent_nll_mean"
                    ),
                    "final/oracle_full_nll_mean": final_metrics.get(
                        "oracle_full_nll_mean"
                    ),
                    "final/off_subspace_energy_fraction": final_metrics.get(
                        "off_subspace_energy_fraction"
                    ),
                    "final/latent_frechet_distance_to_oracle": final_metrics.get(
                        "latent_frechet_distance_to_oracle"
                    ),
                    "final/latent_sliced_w2": final_metrics.get("latent_sliced_w2"),
                    "final/component_weight_l1": final_metrics.get("component_weight_l1"),
                    "final/component_weight_kl": final_metrics.get("component_weight_kl"),
                    **media_payload,
                }
            ),
            step=args.n_steps,
        )
        wandb_run.finish()
        wandb_metadata = dict(wandb_metadata)
        wandb_metadata["active"] = False
        wandb_metadata["finished"] = True
        status["wandb"] = wandb_metadata
        status["wandb_active"] = False
        summary["wandb"] = wandb_metadata
        _write_json(status_path, status)
        _write_json(artifacts_dir / "run_summary.json", summary)

    print("")
    print("----- done -----")
    print("run_dir          : {}".format(run_dir))
    print("latest           : {}".format(latest_link))
    print("latest_checkpts  : {}".format(latest_checkpoints_link))
    print("status           : {}".format(status_path))
    print("train_history    : {}".format(train_history_path))
    print("metrics          : {}".format(metric_records_path))
    print("final_samples    : {}".format(metrics_dir / "final_samples.pt"))
    print("figures          : {}".format(figures_dir))
    print("checkpoints      : {}".format(checkpoints_dir))
    print("final_loss       : {}".format(summary["final_loss"]))
    print("final_metrics    : {}".format(final_metrics))
    print("figure_files     : {}".format(figure_paths))
    if wandb_metadata["enabled"]:
        print("wandb_url        : {}".format(wandb_metadata["run_url"] or "not available"))
        print("wandb_dir        : {}".format(wandb_metadata["dir"]))
        if wandb_metadata["mode"] == "offline":
            print("wandb_sync       : wandb sync {}".format(wandb_metadata["dir"]))

    return summary


if __name__ == "__main__":
    main()
