"""
Direct prototype runner for projected-GMM synthetic data with an MLP.

The synthetic runner stays intentionally visible: it builds the planted data,
measures against the known geometry, trains a small flow model, and saves the
artifacts needed for quick lab inspection.
"""

import argparse
from pathlib import Path
import sys
import time

# Direct script execution from prototyping/runners should still find repo packages.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototyping.metrics_synthetic_controlled import (
    collect_metric_record,
    project_to_subspace,
)
from prototyping.runners.common import (
    PARAMETERIZATION_CHOICES,
    count_parameters,
    resolve_batch_size,
)
from prototyping.runners.figures import make_synthetic_figures
from prototyping.runners.io_utils import (
    append_jsonl,
    make_run_dirs,
    save_tensors,
    update_latest_links,
    write_json,
)
from prototyping.runners.tracking import Tracker


RUNNER_NAME = "synthetic_mlp"
DEFAULT_OUT_DIR = "outputs/prototype_runners/synthetic_mlp"
DEFAULT_RUN_NAME = "synthetic_projected_gmm_mlp"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a projected-GMM GDFM MLP and track synthetic metrics."
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu, ...")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=0)

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
        choices=PARAMETERIZATION_CHOICES,
        default="v",
    )
    parser.add_argument("--loss-space", choices=PARAMETERIZATION_CHOICES, default="v")

    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--batch-cap", type=int, default=10)
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
    return parser.parse_args(argv)


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
        (args.batch_cap >= 1, "--batch-cap must be at least 1."),
        (args.log_every >= 1, "--log-every must be at least 1."),
        (args.metric_every >= 1, "--metric-every must be at least 1."),
        (args.ckpt_every >= 1, "--ckpt-every must be at least 1."),
        (args.sampling_steps >= 1, "--sampling-steps must be at least 1."),
        (args.chunk_size >= 1, "--chunk-size must be at least 1."),
    ]
    for ok, message in checks:
        if not ok:
            raise SystemExit(message)


def build_config(args, batch_size):
    from mp_config import config_to_dict, get_base_config

    config = get_base_config()
    config.study.name = RUNNER_NAME
    config.study.description = "Direct synthetic-data GDFM/MLP prototype runner."
    config.outputs.run_name = args.run_name

    config.framework.family = "fm"
    config.framework.prediction_space = args.prediction_space
    config.framework.loss_space = args.loss_space

    config.data.dataset = "gmm_subspace"
    config.data.root = "synthetic"
    config.data.download = False
    config.data.data_source = "canonical"
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
    config.training.batch_cap = args.batch_cap
    config.training.batch_size = args.batch_cap
    config.training.lr = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.ckpt_every = args.ckpt_every
    config.training.log_every = args.log_every
    config.training.optimizer = "adam"

    config.evaluation.n_samples = args.eval_samples
    config.evaluation.batch_size = batch_size
    config.evaluation.ratio_threshold = args.ratio_threshold
    config.evaluation.chunk_size = args.chunk_size
    config.evaluation.sample_panel_count = min(8, args.eval_samples)

    config.sampling.method = args.sampling_method
    config.sampling.n_steps = args.sampling_steps
    config.sliced_w2_projections = args.sliced_w2_projections
    return config, config_to_dict(config)


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


def _next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _sample(model, z_eval, args, device):
    return model.sample(
        z_eval.clone().to(device),
        n_steps=args.sampling_steps,
        method=args.sampling_method,
        device=device,
    ).cpu()


def _print_field(label, value):
    print("{:<18} {}".format(label + ":", value))


def main(argv=None):
    args = parse_args(argv)

    try:
        import torch
        from torch.utils.data import DataLoader
        from gdfm import GDFM
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

    batch_size = resolve_batch_size(args.n_train, args.batch_cap)
    drop_last = int(args.n_train) > int(args.batch_cap)
    torch.manual_seed(int(args.seed))
    generator = torch.Generator().manual_seed(int(args.seed))

    config, config_payload = build_config(args, batch_size)

    data_bundle = build_data_bundle(config)
    train_tensor = data_bundle["train"].contiguous()
    reference_tensor = data_bundle["reference"].contiguous()
    train_labels = data_bundle["train_labels"]
    reference_labels = data_bundle["reference_labels"]
    basis = data_bundle["geometry"]["basis"].contiguous()
    latent_means = data_bundle["geometry"]["means"].contiguous()
    ambient_means = latent_means @ basis.T

    train_parts = project_to_subspace(train_tensor, basis)
    reference_parts = project_to_subspace(reference_tensor, basis)
    train_latent = train_parts["latent"]
    reference_latent = reference_parts["latent"]

    z_eval = torch.randn(args.eval_samples, args.data_dim, generator=generator)
    sliced_generator = torch.Generator().manual_seed(int(args.seed) + 1009)
    sliced_directions = torch.randn(
        args.sliced_w2_projections,
        args.intrinsic_dim,
        generator=sliced_generator,
    )
    sliced_directions = sliced_directions / sliced_directions.norm(
        dim=1,
        keepdim=True,
    ).clamp(min=1.0e-12)
    baseline_tensor = reference_tensor[: min(args.eval_samples, reference_tensor.shape[0])]

    model = GDFM(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    out_root, paths, timestamp = make_run_dirs(args.out_dir, args.run_name, args.seed)
    link_warnings = update_latest_links(
        out_root,
        {
            "latest": paths["run"],
            "latest_checkpoints": paths["checkpoints"],
        },
    )
    tracker = Tracker(args, paths["wandb"], config_payload)

    _print_field("python", sys.executable)
    _print_field("torch", torch.__version__)
    _print_field("device", device)
    _print_field("run_dir", paths["run"])
    _print_field("train", tuple(train_tensor.shape))
    _print_field("reference", tuple(reference_tensor.shape))
    _print_field("geometry", "D={} d={} clusters={}".format(args.data_dim, args.intrinsic_dim, args.n_clusters))
    _print_field("model", "GDFM + MLP ({:,} params)".format(count_parameters(model)["total"]))
    _print_field("batch", "cap={} effective={} drop_last={}".format(args.batch_cap, batch_size, drop_last))
    if tracker.metadata.get("warning"):
        _print_field("wandb warning", tracker.metadata["warning"])
    for warning in link_warnings:
        _print_field("link warning", warning)

    save_tensors(
        torch,
        {
            paths["artifacts"] / "train_tensor.pt": train_tensor,
            paths["artifacts"] / "reference_tensor.pt": reference_tensor,
            paths["artifacts"] / "train_labels.pt": train_labels,
            paths["artifacts"] / "reference_labels.pt": reference_labels,
            paths["artifacts"] / "basis.pt": basis,
            paths["artifacts"] / "latent_means.pt": latent_means,
            paths["artifacts"] / "ambient_means.pt": ambient_means,
            paths["artifacts"] / "train_latent.pt": train_latent,
            paths["artifacts"] / "reference_latent.pt": reference_latent,
            paths["artifacts"] / "train_residual.pt": train_parts["residual"],
            paths["artifacts"] / "reference_residual.pt": reference_parts["residual"],
            paths["artifacts"] / "sliced_w2_directions.pt": sliced_directions,
            paths["artifacts"] / "z_eval.pt": z_eval,
        },
    )
    write_json(paths["artifacts"] / "config.json", config_payload)
    write_json(
        paths["artifacts"] / "gmm_params.json",
        {
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
        },
    )

    status = {
        "state": "training",
        "started_at": timestamp,
        "run_dir": str(paths["run"]),
        "device": str(device),
        "batch": {
            "batch_cap": int(args.batch_cap),
            "effective_batch_size": int(batch_size),
            "drop_last": bool(drop_last),
        },
        "data": _summarize_data_bundle(data_bundle),
        "wandb": tracker.metadata,
        "link_warnings": link_warnings,
    }
    write_json(paths["run"] / "status.json", status)

    train_history = []
    metric_records = []
    train_start = time.perf_counter()
    train_loader = DataLoader(
        train_tensor,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
    )
    train_iterator = iter(train_loader)

    model.train()
    for step in range(1, args.n_steps + 1):
        batch, train_iterator = _next_batch(train_loader, train_iterator)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.n_steps:
            row = {
                "step": int(step),
                "loss": float(loss.item()),
                "elapsed_sec": float(time.perf_counter() - train_start),
            }
            train_history.append(row)
            append_jsonl(paths["logs"] / "train_history.jsonl", row)
            _print_field("step", "{}/{} loss={:.6f}".format(step, args.n_steps, row["loss"]))
            tracker.log_scalars({"train/loss": row["loss"]}, step=step)

        if step % args.metric_every == 0 or step == args.n_steps:
            model.eval()
            with torch.no_grad():
                samples = _sample(model, z_eval, args, device)
            record = collect_metric_record(
                step=step,
                samples=samples,
                baseline_samples=baseline_tensor,
                train_tensor=train_tensor,
                train_latent=train_latent,
                basis=basis,
                latent_means=latent_means,
                sigma_z=args.sigma_z,
                sigma_eta=args.sigma_eta,
                reference_latent=reference_latent[: baseline_tensor.shape[0]],
                sliced_directions=sliced_directions,
                chunk_size=args.chunk_size,
                config=config,
            )
            metric_records.append(record)
            append_jsonl(paths["logs"] / "metrics_along_training.jsonl", record)
            _print_field(
                "metric",
                "step={} ambient_f_mem={:.4f} latent_f_mem={:.4f} nll={:.4f}".format(
                    step,
                    record["ambient_f_mem"],
                    record["latent_f_mem"],
                    record["oracle_latent_nll_mean"],
                ),
            )
            tracker.log_scalars(
                {
                    "eval/ambient_f_mem": record["ambient_f_mem"],
                    "eval/ambient_mean_nn_ratio": record["ambient_mean_nn_ratio"],
                    "eval/latent_f_mem": record["latent_f_mem"],
                    "eval/oracle_latent_nll_mean": record["oracle_latent_nll_mean"],
                    "eval/oracle_full_nll_mean": record["oracle_full_nll_mean"],
                    "eval/off_subspace_energy_fraction": record["off_subspace_energy_fraction"],
                    "eval/latent_frechet_distance_to_oracle": record["latent_frechet_distance_to_oracle"],
                    "eval/latent_sliced_w2": record["latent_sliced_w2"],
                    "eval/component_weight_l1": record["component_weight_l1"],
                    "eval/component_weight_kl": record["component_weight_kl"],
                },
                step=step,
            )
            model.train()

        if step % args.ckpt_every == 0 or step == args.n_steps:
            torch.save(
                {
                    "step": int(step),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config_payload,
                },
                paths["checkpoints"] / "step_{:06d}.pt".format(step),
            )

    model.eval()
    with torch.no_grad():
        final_samples = _sample(model, z_eval, args, device)
    final_parts = project_to_subspace(final_samples, basis)

    save_tensors(
        torch,
        {
            paths["metrics"] / "final_samples.pt": final_samples,
            paths["metrics"] / "final_samples_ambient.pt": final_samples,
            paths["metrics"] / "final_samples_latent.pt": final_parts["latent"],
            paths["metrics"] / "final_samples_projected.pt": final_parts["projected"],
            paths["metrics"] / "final_samples_residual.pt": final_parts["residual"],
            paths["metrics"] / "final_samples_residual_norm.pt": final_parts["residual_norm"],
        },
    )
    write_json(paths["metrics"] / "train_history.json", train_history)
    write_json(paths["metrics"] / "metrics_along_training.json", metric_records)

    figure_context = (
        "GMM D={} d={} N={} ref={} clusters={} sigma_z={:.3g} sigma_eta={:.3g}\n"
        "MLP layers={} hidden={} time_emb={} pred={} loss={} sampler={}x{}\n"
        "seed={} steps={} batch={} lr={:.3g} wd={:.3g}"
    ).format(
        args.data_dim,
        args.intrinsic_dim,
        args.n_train,
        args.n_reference,
        args.n_clusters,
        args.sigma_z,
        args.sigma_eta,
        args.n_layers,
        args.hidden_dim,
        args.time_emb_dim,
        args.prediction_space,
        args.loss_space,
        args.sampling_method,
        args.sampling_steps,
        args.seed,
        args.n_steps,
        batch_size,
        args.lr,
        args.weight_decay,
    )
    figure_paths = make_synthetic_figures(
        figures_dir=paths["figures"],
        train_history=train_history,
        metric_records=metric_records,
        train_tensor=train_tensor,
        reference_tensor=reference_tensor,
        final_samples=final_samples,
        train_latent=train_latent,
        reference_latent=reference_latent,
        final_samples_latent=final_parts["latent"],
        final_samples_residual_norm=final_parts["residual_norm"],
        baseline_residual_norm=project_to_subspace(baseline_tensor, basis)["residual_norm"],
        latent_means=latent_means,
        ambient_means=ambient_means,
        context=figure_context,
    )

    final_metrics = metric_records[-1] if metric_records else {}
    summary = {
        "run_dir": str(paths["run"]),
        "config": config_payload,
        "batch": {
            "batch_cap": int(args.batch_cap),
            "effective_batch_size": int(batch_size),
            "drop_last": bool(drop_last),
        },
        "final_loss": train_history[-1]["loss"] if train_history else None,
        "final_metrics": final_metrics,
        "wandb": tracker.metadata,
        "paths": {
            "status": str(paths["run"] / "status.json"),
            "train_history": str(paths["metrics"] / "train_history.json"),
            "metrics_along_training": str(paths["metrics"] / "metrics_along_training.json"),
            "final_samples": str(paths["metrics"] / "final_samples.pt"),
            "gmm_params": str(paths["artifacts"] / "gmm_params.json"),
            "figures": str(paths["figures"]),
            "checkpoints": str(paths["checkpoints"]),
            "latest": str(out_root / "latest"),
            "latest_checkpoints": str(out_root / "latest_checkpoints"),
        },
        "figures": figure_paths,
    }
    tracker.log_scalars(
        {
            "final/loss": summary["final_loss"],
            "final/ambient_f_mem": final_metrics.get("ambient_f_mem"),
            "final/latent_f_mem": final_metrics.get("latent_f_mem"),
            "final/oracle_latent_nll_mean": final_metrics.get("oracle_latent_nll_mean"),
            "final/oracle_full_nll_mean": final_metrics.get("oracle_full_nll_mean"),
            "final/off_subspace_energy_fraction": final_metrics.get("off_subspace_energy_fraction"),
            "final/latent_frechet_distance_to_oracle": final_metrics.get("latent_frechet_distance_to_oracle"),
            "final/latent_sliced_w2": final_metrics.get("latent_sliced_w2"),
            "final/component_weight_l1": final_metrics.get("component_weight_l1"),
            "final/component_weight_kl": final_metrics.get("component_weight_kl"),
        },
        step=args.n_steps,
    )
    tracker.log_images(figure_paths, step=args.n_steps)
    summary["wandb"] = tracker.finish()
    status.update(
        {
            "state": "completed",
            "training_wall_time_sec": float(time.perf_counter() - train_start),
            "final_loss": summary["final_loss"],
            "final_metrics": final_metrics,
            "figures": figure_paths,
            "wandb": summary["wandb"],
        }
    )
    write_json(paths["run"] / "status.json", status)
    write_json(paths["artifacts"] / "run_summary.json", summary)

    _print_field("final loss", summary["final_loss"])
    if final_metrics:
        _print_field(
            "final f_mem",
            "ambient={:.4f} latent={:.4f}".format(
                final_metrics["ambient_f_mem"],
                final_metrics["latent_f_mem"],
            ),
        )
    _print_field("metrics", summary["paths"]["metrics_along_training"])
    _print_field("figures", paths["figures"])
    _print_field("checkpoints", paths["checkpoints"])
    return summary


if __name__ == "__main__":
    main()
