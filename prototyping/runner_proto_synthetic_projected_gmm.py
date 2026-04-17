"""
Prototype runner for controlled projected-GMM GDFM/MLP training.

This script intentionally keeps the whole run visible in `main()`: make a
known GMM in intrinsic coordinates, optionally project it into a larger ambient
space, build the model, train it, compute memorization and oracle GMM metrics,
and write local artifacts.
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

# Find the repo root from this file, then make local imports predictable.
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


# -----------------------------------------------------------------------------
# Synthetic Metrics
# -----------------------------------------------------------------------------

from prototyping.metrics_synthetic_controlled import (
    memorization_metrics_in_spaces,
    oracle_gmm_metrics,
    project_to_subspace,
)


def wandb_status_payload(args, wandb_dir, run=None, *, active=False, warning=None):
    """
    Keep enough W&B metadata on disk to find the run from a shell later.
    """
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
        "dir": str(wandb_dir.resolve()),
        "warning": warning,
    }


def wandb_scalar_payload(payload):
    """
    W&B accepts many values, but filtering None keeps singular-case metrics clean.
    """
    return {key: value for key, value in payload.items() if value is not None}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    """
    Run one plain projected-GMM training job.
    """
    parser = argparse.ArgumentParser(description="Train a projected GMM GDFM MLP and track metrics.")

    # Basic run knobs.
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu, auto, ...")
    parser.add_argument("--out-dir", default="tmp/runner_proto_synthetic_projected_gmm")
    parser.add_argument("--run-name", default="synthetic_projected_gmm_mlp")
    parser.add_argument("--seed", type=int, default=0)

    # Projected GMM data knobs.
    parser.add_argument("--data-source", choices=("canonical", "manual", "both"), default="both")
    parser.add_argument("--data-dim", type=int, default=2, help="Ambient dimension D.")
    parser.add_argument("--intrinsic-dim", type=int, default=2, help="Intrinsic dimension d.")
    parser.add_argument("--n-train", type=int, default=10)
    parser.add_argument("--n-reference", type=int, default=1024)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--sigma-z", type=float, default=0.10)
    parser.add_argument("--sigma-eta", type=float, default=1.0e-3)
    parser.add_argument("--geometry-seed", type=int, default=0)

    # Model knobs.
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--time-emb-dim", type=int, default=64)
    parser.add_argument("--prediction-space", choices=("x", "eps", "v"), default="v")
    parser.add_argument("--loss-space", choices=("x", "eps", "v"), default="v")

    # Training knobs. The default is intentionally a real run, not a tiny check.
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses all train points; positive values cap at n_train.")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--metric-every", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=2000)

    # Sampling and metric knobs.
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--sampling-method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--sampling-steps", type=int, default=128)
    parser.add_argument("--ratio-threshold", type=float, default=0.3)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--sliced-w2-projections", type=int, default=128)

    # Optional W&B tracking.
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--wandb-project", default="mem_par")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-name", default="", help="Optional W&B run name. Defaults to the output folder name.")
    parser.add_argument("--wandb-group", default="", help="Optional W&B group for comparing related runs.")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-notes", default="", help="Optional W&B notes.")
    parser.add_argument("--wandb-strict", action="store_true")
    args = parser.parse_args()

    # Import after path setup so this checkout wins over any installed package.
    try:
        import torch
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

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.n_train < 2:
        raise SystemExit("--n-train must be at least 2 for nearest-neighbor metrics.")
    if args.n_reference < 1:
        raise SystemExit("--n-reference must be at least 1.")
    if args.data_dim < 1:
        raise SystemExit("--data-dim must be at least 1.")
    if args.intrinsic_dim < 1:
        raise SystemExit("--intrinsic-dim must be at least 1.")
    if args.intrinsic_dim > args.data_dim:
        raise SystemExit("--intrinsic-dim must be less than or equal to --data-dim.")
    if args.n_clusters < 1:
        raise SystemExit("--n-clusters must be at least 1.")
    if args.sigma_z <= 0.0:
        raise SystemExit("--sigma-z must be positive.")
    if args.sigma_eta < 0.0:
        raise SystemExit("--sigma-eta must be non-negative.")
    if args.sliced_w2_projections < 1:
        raise SystemExit("--sliced-w2-projections must be at least 1.")
    batch_size = args.n_train if args.batch_size <= 0 else min(args.batch_size, args.n_train)

    torch.manual_seed(int(args.seed))
    generator = torch.Generator().manual_seed(int(args.seed))

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
        if link_path.is_symlink():
            link_path.unlink()
        if not link_path.exists():
            link_path.symlink_to(target_path, target_is_directory=True)

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

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------

    # Use the repo's normal config shape, but keep all choices visible here.
    config = get_base_config()
    config.study.name = "runner_proto_synthetic_projected_gmm"
    config.study.description = "Plain projected-GMM GDFM MLP runner."
    config.outputs.run_name = args.run_name

    config.framework.family = "fm"
    config.framework.prediction_space = args.prediction_space
    config.framework.loss_space = args.loss_space

    config.data.dataset = "gmm_subspace"
    config.data.data_source = args.data_source
    config.data.n_train = args.n_train
    config.data.n_eval = args.n_reference
    config.data.data_dim = args.data_dim
    config.data.intrinsic_dim = args.intrinsic_dim
    config.data.n_clusters = args.n_clusters
    config.data.sigma_z = args.sigma_z
    config.data.sigma_eta = args.sigma_eta
    config.data.geometry_seed = args.geometry_seed

    config.network.arch = "mlp"
    config.network.hidden_dim = args.hidden_dim
    config.network.n_layers = args.n_layers
    config.network.time_emb_dim = args.time_emb_dim

    config.training.seed = args.seed
    config.training.n_steps = args.n_steps
    config.training.batch_size = batch_size
    config.training.lr = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.ckpt_every = args.ckpt_every
    config.training.log_every = args.log_every

    config.evaluation.n_samples = args.eval_samples
    config.evaluation.ratio_threshold = args.ratio_threshold
    config.evaluation.chunk_size = args.chunk_size

    config.sampling.method = args.sampling_method
    config.sampling.n_steps = args.sampling_steps

    config_payload = config_to_dict(config)
    (artifacts_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    print("")
    print("----- config -----")
    print("data             : projected GMM")
    print("data_source      : {}".format(args.data_source))
    print("ambient/intrinsic: D={} d={}".format(args.data_dim, args.intrinsic_dim))
    print("sigma_z/eta      : {}/{}".format(args.sigma_z, args.sigma_eta))
    print("model            : GDFM + MLP")
    print("prediction/loss  : {}/{}".format(config.framework.prediction_space, config.framework.loss_space))
    print("steps            : {}".format(args.n_steps))
    print("batch_size       : {}".format(batch_size))
    print("metric_every     : {}".format(args.metric_every))
    print("ckpt_every       : {}".format(args.ckpt_every))

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    # The canonical path uses the repo loader. The manual path mirrors the same
    # equations inline so this script stays educational and directly auditable.
    canonical_bundle = None
    manual_bundle = None
    if args.data_source in ("canonical", "both"):
        canonical_bundle = build_data_bundle(config)

    if args.data_source in ("manual", "both"):
        geometry_generator = torch.Generator().manual_seed(int(args.geometry_seed))
        manual_basis, _ = torch.linalg.qr(
            torch.randn(args.data_dim, args.intrinsic_dim, generator=geometry_generator)
        )
        manual_basis = manual_basis[:, : args.intrinsic_dim]
        manual_means = torch.randn(args.n_clusters, args.intrinsic_dim, generator=geometry_generator)

        train_generator = torch.Generator().manual_seed(int(args.seed))
        manual_train_labels = torch.randint(args.n_clusters, (args.n_train,), generator=train_generator)
        manual_train_latent = manual_means[manual_train_labels] + args.sigma_z * torch.randn(
            args.n_train,
            args.intrinsic_dim,
            generator=train_generator,
        )
        manual_train_tensor = manual_train_latent @ manual_basis.T
        manual_train_tensor = manual_train_tensor + args.sigma_eta * torch.randn(
            args.n_train,
            args.data_dim,
            generator=train_generator,
        )

        reference_generator = torch.Generator().manual_seed(int(args.seed) + 1)
        manual_reference_labels = torch.randint(
            args.n_clusters,
            (args.n_reference,),
            generator=reference_generator,
        )
        manual_reference_latent = manual_means[manual_reference_labels] + args.sigma_z * torch.randn(
            args.n_reference,
            args.intrinsic_dim,
            generator=reference_generator,
        )
        manual_reference_tensor = manual_reference_latent @ manual_basis.T
        manual_reference_tensor = manual_reference_tensor + args.sigma_eta * torch.randn(
            args.n_reference,
            args.data_dim,
            generator=reference_generator,
        )

        manual_bundle = {
            "train": manual_train_tensor.contiguous(),
            "reference": manual_reference_tensor.contiguous(),
            "train_labels": manual_train_labels,
            "reference_labels": manual_reference_labels,
            "geometry": {
                "basis": manual_basis,
                "means": manual_means,
            },
            "train_source_id": "manual_gmm_subspace:seed:{}".format(int(args.seed)),
            "reference_source_id": "manual_gmm_subspace:seed:{}".format(int(args.seed) + 1),
        }

    if args.data_source == "manual":
        data_bundle = manual_bundle
    else:
        data_bundle = canonical_bundle

    data_equivalence_report = None
    if args.data_source == "both":
        comparison_pairs = {
            "train": (manual_bundle["train"], canonical_bundle["train"]),
            "reference": (manual_bundle["reference"], canonical_bundle["reference"]),
            "train_labels": (manual_bundle["train_labels"], canonical_bundle["train_labels"]),
            "reference_labels": (manual_bundle["reference_labels"], canonical_bundle["reference_labels"]),
            "basis": (manual_bundle["geometry"]["basis"], canonical_bundle["geometry"]["basis"]),
            "means": (manual_bundle["geometry"]["means"], canonical_bundle["geometry"]["means"]),
        }
        data_equivalence_report = {
            "data_source": args.data_source,
            "all_equal": True,
            "items": {},
        }
        for name, pair in comparison_pairs.items():
            left, right = pair
            equal = bool(torch.equal(left, right))
            max_abs_diff = float((left.float() - right.float()).abs().max().item())
            data_equivalence_report["items"][name] = {
                "equal": equal,
                "shape": list(left.shape),
                "max_abs_diff": max_abs_diff,
            }
            data_equivalence_report["all_equal"] = bool(data_equivalence_report["all_equal"] and equal)
        (artifacts_dir / "data_equivalence_report.json").write_text(
            json.dumps(data_equivalence_report, indent=2)
        )
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
    train_latent = train_parts["latent"]
    reference_latent = reference_parts["latent"]
    train_projected = train_parts["projected"]
    reference_projected = reference_parts["projected"]
    train_residual = train_parts["residual"]
    reference_residual = reference_parts["residual"]

    z_eval = torch.randn(args.eval_samples, args.data_dim, generator=generator)
    sliced_generator = torch.Generator().manual_seed(int(args.seed) + 1009)
    sliced_directions = torch.randn(
        args.sliced_w2_projections,
        args.intrinsic_dim,
        generator=sliced_generator,
    )
    sliced_directions = sliced_directions / sliced_directions.norm(dim=1, keepdim=True).clamp(min=1.0e-12)

    torch.save(train_tensor, artifacts_dir / "train_tensor.pt")
    torch.save(reference_tensor, artifacts_dir / "reference_tensor.pt")
    torch.save(train_labels, artifacts_dir / "train_labels.pt")
    torch.save(reference_labels, artifacts_dir / "reference_labels.pt")
    torch.save(basis, artifacts_dir / "basis.pt")
    torch.save(latent_means, artifacts_dir / "latent_means.pt")
    torch.save(ambient_means, artifacts_dir / "ambient_means.pt")
    torch.save(train_latent, artifacts_dir / "train_latent.pt")
    torch.save(reference_latent, artifacts_dir / "reference_latent.pt")
    torch.save(train_projected, artifacts_dir / "train_projected.pt")
    torch.save(reference_projected, artifacts_dir / "reference_projected.pt")
    torch.save(train_residual, artifacts_dir / "train_residual.pt")
    torch.save(reference_residual, artifacts_dir / "reference_residual.pt")
    torch.save(sliced_directions, artifacts_dir / "sliced_w2_directions.pt")
    torch.save(z_eval, artifacts_dir / "z_eval.pt")
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
    (artifacts_dir / "gmm_params.json").write_text(json.dumps(gmm_params, indent=2))

    print("")
    print("----- data -----")
    print("train shape      : {}".format(tuple(train_tensor.shape)))
    print("reference shape  : {}".format(tuple(reference_tensor.shape)))
    print("basis shape      : {}".format(tuple(basis.shape)))
    print("latent means     : {}".format(tuple(latent_means.shape)))
    print("manual==canonical: {}".format(data_equivalence_report["all_equal"] if data_equivalence_report else "not checked"))

    baseline_count = min(args.eval_samples, reference_tensor.shape[0])
    baseline_tensor = reference_tensor[:baseline_count]
    baseline_latent = reference_latent[:baseline_count]
    baseline_residual = project_to_subspace(baseline_tensor, basis)["residual"]
    uniform_component_weight = 1.0 / float(args.n_clusters)

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
    n_params = sum(parameter.numel() for parameter in model.parameters())
    n_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    print("")
    print("----- model -----")
    print("network          : {}.{}".format(model.net.__class__.__module__, model.net.__class__.__name__))
    print("parameters       : {}".format(n_params))
    print("trainable        : {}".format(n_trainable))

    # -------------------------------------------------------------------------
    # Tracking
    # -------------------------------------------------------------------------

    wandb_run = None
    wandb_module = None
    wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    wandb_metadata = wandb_status_payload(args, wandb_dir, active=False)
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
                tags=wandb_tags,
                notes=args.wandb_notes.strip() or None,
                config=config_payload,
            )
            wandb_metadata = wandb_status_payload(args, wandb_dir, wandb_run, active=True)
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
            wandb_metadata = wandb_status_payload(args, wandb_dir, active=False, warning=message)
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
    }
    status_path.write_text(json.dumps(status, indent=2))

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    print("")
    print("----- training -----")
    train_history = []
    metric_records = []
    train_start = time.perf_counter()
    model.train()

    for step in range(1, args.n_steps + 1):
        # This is the actual update: choose data, compute loss, backprop, update.
        batch_indices = torch.randint(0, train_tensor.shape[0], (batch_size,), generator=generator)
        batch = train_tensor[batch_indices].to(device)

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
            with train_history_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row))
                handle.write("\n")

            print("step {:5d}/{} loss {:.6f}".format(step, args.n_steps, loss_value))
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss_value}, step=step)

        if step % args.metric_every == 0 or step == args.n_steps:
            # Periodic metrics show whether generated samples are moving toward data.
            model.eval()
            with torch.no_grad():
                samples = model.sample(
                    z_eval.clone().to(device),
                    n_steps=args.sampling_steps,
                    method=args.sampling_method,
                    device=device,
                ).cpu()

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

            record = {
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
                "off_subspace_energy_fraction": samples_oracle_metrics["off_subspace_energy_fraction"],
                "latent_frechet_distance_to_oracle": samples_oracle_metrics["latent_frechet_distance_to_oracle"],
                "latent_sliced_w2": samples_oracle_metrics["latent_sliced_w2"],
                "component_weight_l1": samples_oracle_metrics["component_weight_l1"],
                "component_weight_kl": samples_oracle_metrics["component_weight_kl"],
                "component_probs": samples_oracle_metrics["component_probs"],
                "residual_norm_mean": samples_oracle_metrics["residual_norm_mean"],
                "baseline_oracle_latent_nll_mean": baseline_oracle_metrics["oracle_latent_nll_mean"],
                "baseline_oracle_full_nll_mean": baseline_oracle_metrics["oracle_full_nll_mean"],
                "baseline_off_subspace_energy_fraction": baseline_oracle_metrics["off_subspace_energy_fraction"],
                "baseline_latent_frechet_distance_to_oracle": baseline_oracle_metrics[
                    "latent_frechet_distance_to_oracle"
                ],
                "baseline_component_weight_l1": baseline_oracle_metrics["component_weight_l1"],
                "baseline_component_weight_kl": baseline_oracle_metrics["component_weight_kl"],
                "baseline_component_probs": baseline_oracle_metrics["component_probs"],
                # Backward-compatible aliases for the simple plots and W&B panels.
                "f_mem": generated["ambient_f_mem"],
                "mean_nn_ratio": generated["ambient_mean_nn_ratio"],
                "std_nn_ratio": generated["ambient_std_nn_ratio"],
                "baseline_f_mem": baseline_mem["ambient_f_mem"],
                "baseline_mean_nn_ratio": baseline_mem["ambient_mean_nn_ratio"],
                "baseline_std_nn_ratio": baseline_mem["ambient_std_nn_ratio"],
            }
            metric_records.append(record)
            with metric_records_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record))
                handle.write("\n")

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
                    wandb_scalar_payload(
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
                            "eval/off_subspace_energy_fraction": record["off_subspace_energy_fraction"],
                            "eval/latent_frechet_distance_to_oracle": record["latent_frechet_distance_to_oracle"],
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
    train_history_path.write_text(json.dumps(train_history, indent=2))
    metric_records_path.write_text(json.dumps(metric_records, indent=2))

    # -------------------------------------------------------------------------
    # Final Figures
    # -------------------------------------------------------------------------

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    figure_paths = {}

    family_label = str(config.framework.family).upper()
    arch_label = "{} layers={} hidden={} time_emb={}".format(
        str(config.network.arch).upper(),
        args.n_layers,
        args.hidden_dim,
        args.time_emb_dim,
    )
    data_label = "GMM D={} d={} n_train={} n_ref={} clusters={} sigma_z={:.3g} sigma_eta={:.3g}".format(
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
        batch_size,
        args.lr,
        args.weight_decay,
    )
    figure_context = "{}\n{} | {}\n{}".format(data_label, arch_label, param_label, train_label)

    if train_history:
        steps = [row["step"] for row in train_history]
        losses = [row["loss"] for row in train_history]
        loss_figure_path = figures_dir / "training_loss_vs_step.png"

        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(steps, losses, linewidth=1.8)
        ax.set_title("Training loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(loss_figure_path, dpi=160)
        plt.close(fig)
        figure_paths["training_loss_vs_step"] = str(loss_figure_path)

    if metric_records:
        metric_steps = [row["step"] for row in metric_records]
        f_mem_figure_path = figures_dir / "f_mem_vs_step.png"
        nn_ratio_figure_path = figures_dir / "nn_ratio_vs_step.png"
        oracle_nll_figure_path = figures_dir / "oracle_nll_vs_step.png"
        latent_distance_figure_path = figures_dir / "latent_distance_vs_step.png"
        residual_energy_figure_path = figures_dir / "residual_energy_vs_step.png"

        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(metric_steps, [row["ambient_f_mem"] for row in metric_records], marker="o", label="generated ambient")
        ax.plot(metric_steps, [row["latent_f_mem"] for row in metric_records], marker="o", label="generated latent")
        ax.plot(
            metric_steps,
            [row["baseline_ambient_f_mem"] for row in metric_records],
            marker="o",
            linestyle="--",
            label="reference ambient",
        )
        ax.plot(
            metric_steps,
            [row["baseline_latent_f_mem"] for row in metric_records],
            marker="o",
            linestyle="--",
            label="reference latent",
        )
        ax.set_title("Memorization fraction")
        ax.set_xlabel("step")
        ax.set_ylabel("f_mem")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(f_mem_figure_path, dpi=160)
        plt.close(fig)
        figure_paths["f_mem_vs_step"] = str(f_mem_figure_path)

        generated_nn_mean = [row["ambient_mean_nn_ratio"] for row in metric_records]
        generated_nn_std = [row["ambient_std_nn_ratio"] for row in metric_records]
        generated_nn_lower = [
            max(0.0, mean_value - std_value)
            for mean_value, std_value in zip(generated_nn_mean, generated_nn_std)
        ]
        generated_nn_upper = [
            mean_value + std_value
            for mean_value, std_value in zip(generated_nn_mean, generated_nn_std)
        ]

        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.fill_between(
            metric_steps,
            generated_nn_lower,
            generated_nn_upper,
            alpha=0.18,
            label="generated +/- std",
        )
        ax.plot(
            metric_steps,
            generated_nn_mean,
            marker="o",
            label="generated mean",
        )
        ax.plot(
            metric_steps,
            [row["baseline_ambient_mean_nn_ratio"] for row in metric_records],
            marker="o",
            linestyle="--",
            label="reference baseline mean",
        )
        ax.set_title("Nearest-neighbor ratio")
        ax.set_xlabel("step")
        ax.set_ylabel("r_NN ratio")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(nn_ratio_figure_path, dpi=160)
        plt.close(fig)
        figure_paths["nn_ratio_vs_step"] = str(nn_ratio_figure_path)

        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(
            metric_steps,
            [
                row["oracle_full_nll_mean"]
                if row["oracle_full_nll_mean"] is not None
                else float("nan")
                for row in metric_records
            ],
            marker="o",
            label="generated full NLL",
        )
        ax.plot(
            metric_steps,
            [row["oracle_latent_nll_mean"] for row in metric_records],
            marker="o",
            label="generated latent NLL",
        )
        ax.plot(
            metric_steps,
            [
                row["baseline_oracle_full_nll_mean"]
                if row["baseline_oracle_full_nll_mean"] is not None
                else float("nan")
                for row in metric_records
            ],
            linestyle="--",
            label="reference full NLL",
        )
        ax.plot(
            metric_steps,
            [row["baseline_oracle_latent_nll_mean"] for row in metric_records],
            linestyle="--",
            label="reference latent NLL",
        )
        ax.set_title("Oracle GMM negative log-likelihood")
        ax.set_xlabel("step")
        ax.set_ylabel("NLL")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(oracle_nll_figure_path, dpi=160)
        plt.close(fig)
        figure_paths["oracle_nll_vs_step"] = str(oracle_nll_figure_path)

        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(
            metric_steps,
            [row["latent_frechet_distance_to_oracle"] for row in metric_records],
            marker="o",
            label="latent Frechet",
        )
        ax.plot(
            metric_steps,
            [row["latent_sliced_w2"] for row in metric_records],
            marker="o",
            label="latent sliced-W2",
        )
        ax.plot(
            metric_steps,
            [row["baseline_latent_frechet_distance_to_oracle"] for row in metric_records],
            linestyle="--",
            label="reference Frechet",
        )
        ax.set_title("Latent distribution distances")
        ax.set_xlabel("step")
        ax.set_ylabel("distance")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(latent_distance_figure_path, dpi=160)
        plt.close(fig)
        figure_paths["latent_distance_vs_step"] = str(latent_distance_figure_path)

        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(
            metric_steps,
            [row["off_subspace_energy_fraction"] for row in metric_records],
            marker="o",
            label="generated",
        )
        ax.plot(
            metric_steps,
            [row["baseline_off_subspace_energy_fraction"] for row in metric_records],
            linestyle="--",
            label="reference",
        )
        ax.set_title("Off-subspace energy")
        ax.set_xlabel("step")
        ax.set_ylabel("residual / total energy")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(residual_energy_figure_path, dpi=160)
        plt.close(fig)
        figure_paths["residual_energy_vs_step"] = str(residual_energy_figure_path)

    if args.intrinsic_dim >= 2:
        latent_plot_points = torch.cat([train_latent, reference_latent, final_samples_latent, latent_means], dim=0)
        latent_x_min = float(latent_plot_points[:, 0].min().item()) - 0.5
        latent_x_max = float(latent_plot_points[:, 0].max().item()) + 0.5
        latent_y_min = float(latent_plot_points[:, 1].min().item()) - 0.5
        latent_y_max = float(latent_plot_points[:, 1].max().item()) + 0.5

        latent_data_figure_path = figures_dir / "latent_data_scatter.png"
        fig, ax = plt.subplots(figsize=(5.8, 6.1))
        ax.scatter(train_latent[:, 0].numpy(), train_latent[:, 1].numpy(), s=24, alpha=0.60, label="train")
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
        ax.set_title("Data in true latent coordinates")
        ax.set_xlabel("x @ B [0]")
        ax.set_ylabel("x @ B [1]")
        ax.set_xlim(latent_x_min, latent_x_max)
        ax.set_ylim(latent_y_min, latent_y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.20)
        ax.legend()
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(latent_data_figure_path, dpi=160)
        plt.close(fig)
        figure_paths["latent_data_scatter"] = str(latent_data_figure_path)

        latent_generated_figure_path = figures_dir / "latent_generated_scatter.png"
        fig, ax = plt.subplots(figsize=(5.8, 6.1))
        ax.scatter(train_latent[:, 0].numpy(), train_latent[:, 1].numpy(), s=24, alpha=0.20, label="train")
        ax.scatter(
            final_samples_latent[:, 0].numpy(),
            final_samples_latent[:, 1].numpy(),
            s=14,
            alpha=0.75,
            label="generated",
        )
        ax.scatter(
            latent_means[:, 0].numpy(),
            latent_means[:, 1].numpy(),
            s=90,
            marker="x",
            linewidths=2.0,
            label="latent means",
        )
        ax.set_title("Generated samples in true latent coordinates")
        ax.set_xlabel("x @ B [0]")
        ax.set_ylabel("x @ B [1]")
        ax.set_xlim(latent_x_min, latent_x_max)
        ax.set_ylim(latent_y_min, latent_y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.20)
        ax.legend()
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(latent_generated_figure_path, dpi=160)
        plt.close(fig)
        figure_paths["latent_generated_scatter"] = str(latent_generated_figure_path)

    if args.data_dim == 2:
        ambient_plot_points = torch.cat([train_tensor, reference_tensor, final_samples, ambient_means], dim=0)
        x_min = float(ambient_plot_points[:, 0].min().item()) - 0.5
        x_max = float(ambient_plot_points[:, 0].max().item()) + 0.5
        y_min = float(ambient_plot_points[:, 1].min().item()) - 0.5
        y_max = float(ambient_plot_points[:, 1].max().item()) + 0.5

        data_figure_path = figures_dir / "ambient_data_scatter.png"
        fig, ax = plt.subplots(figsize=(5.8, 6.1))
        ax.scatter(train_tensor[:, 0].numpy(), train_tensor[:, 1].numpy(), s=24, alpha=0.60, label="train")
        ax.scatter(reference_tensor[:, 0].numpy(), reference_tensor[:, 1].numpy(), s=8, alpha=0.20, label="reference")
        ax.scatter(
            ambient_means[:, 0].numpy(),
            ambient_means[:, 1].numpy(),
            s=90,
            marker="x",
            linewidths=2.0,
            label="ambient means",
        )
        ax.set_title("Data in ambient coordinates")
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.20)
        ax.legend()
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(data_figure_path, dpi=160)
        plt.close(fig)
        figure_paths["ambient_data_scatter"] = str(data_figure_path)

        generated_figure_path = figures_dir / "ambient_generated_scatter.png"
        fig, ax = plt.subplots(figsize=(5.8, 6.1))
        ax.scatter(train_tensor[:, 0].numpy(), train_tensor[:, 1].numpy(), s=24, alpha=0.20, label="train")
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
        fig.savefig(generated_figure_path, dpi=160)
        plt.close(fig)
        figure_paths["ambient_generated_scatter"] = str(generated_figure_path)

    residual_hist_path = figures_dir / "residual_norm_histogram.png"
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(baseline_residual.norm(dim=1).numpy(), bins=32, alpha=0.50, label="reference")
    ax.hist(final_samples_residual_norm.numpy(), bins=32, alpha=0.50, label="generated")
    ax.set_title("Off-subspace residual norms")
    ax.set_xlabel("||x - (x @ B) B.T||")
    ax.set_ylabel("count")
    ax.grid(alpha=0.20)
    ax.legend()
    fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(residual_hist_path, dpi=160)
    plt.close(fig)
    figure_paths["residual_norm_histogram"] = str(residual_hist_path)

    if metric_records and metric_records[-1].get("component_probs"):
        component_coverage_path = figures_dir / "component_coverage_final.png"
        final_component_probs = metric_records[-1]["component_probs"]
        baseline_component_probs = metric_records[-1]["baseline_component_probs"]
        xs = list(range(args.n_clusters))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.bar([x - width / 2.0 for x in xs], baseline_component_probs, width=width, label="reference")
        ax.bar([x + width / 2.0 for x in xs], final_component_probs, width=width, label="generated")
        ax.axhline(uniform_component_weight, linestyle="--", linewidth=1.2, label="uniform")
        ax.set_title("Final GMM component coverage")
        ax.set_xlabel("component")
        ax.set_ylabel("mean responsibility")
        ax.grid(alpha=0.20, axis="y")
        ax.legend()
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(component_coverage_path, dpi=160)
        plt.close(fig)
        figure_paths["component_coverage_final"] = str(component_coverage_path)

    # Re-run the sampler while keeping each intermediate state for animations.
    animation_count = min(128, z_eval.shape[0])
    animation_states = []
    z_anim = z_eval[:animation_count].clone().to(device)
    animation_states.append(z_anim.detach().cpu())
    times = torch.linspace(model.t_min, model.t_max, args.sampling_steps + 1, device=device)
    with torch.no_grad():
        for index in range(len(times) - 1):
            dt = times[index + 1] - times[index]
            if args.sampling_method == "euler":
                velocity = model._velocity(z_anim, times[index].item(), z_anim.shape[0])
                z_anim = z_anim + dt * velocity
            else:
                velocity_1 = model._velocity(z_anim, times[index].item(), z_anim.shape[0])
                z_tilde = z_anim + dt * velocity_1
                velocity_2 = model._velocity(z_tilde, times[index + 1].item(), z_anim.shape[0])
                z_anim = z_anim + 0.5 * dt * (velocity_1 + velocity_2)
            animation_states.append(z_anim.detach().cpu())

    # Keep the data and starting noise faint in the background while samples move.
    animation_gif_path = figures_dir / "sampling_animation.gif"
    animation_mp4_path = figures_dir / "sampling_animation.mp4"
    if args.intrinsic_dim >= 2:
        animation_view = "latent"
        animation_train_plot = train_latent
        animation_start_plot = animation_states[0] @ basis
        animation_limits_points = torch.cat(
            [train_latent, reference_latent, final_samples_latent, latent_means, animation_start_plot],
            dim=0,
        )
        animation_xlabel = "x @ B [0]"
        animation_ylabel = "x @ B [1]"
    elif args.data_dim >= 2:
        animation_view = "ambient"
        animation_train_plot = train_tensor
        animation_start_plot = animation_states[0]
        animation_limits_points = torch.cat([train_tensor, reference_tensor, final_samples, z_eval], dim=0)
        animation_xlabel = "x[0]"
        animation_ylabel = "x[1]"
    else:
        animation_view = None

    if animation_view is None:
        warning_path = figures_dir / "sampling_animation_skipped.txt"
        warning_path.write_text("Sampling animation needs at least two plottable dimensions.")
        figure_paths["sampling_animation_skipped"] = str(warning_path)
    else:
        animation_x_min = float(animation_limits_points[:, 0].min().item()) - 0.5
        animation_x_max = float(animation_limits_points[:, 0].max().item()) + 0.5
        animation_y_min = float(animation_limits_points[:, 1].min().item()) - 0.5
        animation_y_max = float(animation_limits_points[:, 1].max().item()) + 0.5

        fig, ax = plt.subplots(figsize=(5.8, 6.1))
        ax.scatter(
            animation_train_plot[:, 0].numpy(),
            animation_train_plot[:, 1].numpy(),
            s=24,
            alpha=0.14,
            label="data",
        )
        ax.scatter(
            animation_start_plot[:, 0].numpy(),
            animation_start_plot[:, 1].numpy(),
            s=12,
            alpha=0.18,
            label="start",
        )
        moving = ax.scatter([], [], s=16, alpha=0.85, label="current")
        ax.set_xlabel(animation_xlabel)
        ax.set_ylabel(animation_ylabel)
        ax.set_xlim(animation_x_min, animation_x_max)
        ax.set_ylim(animation_y_min, animation_y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.20)
        ax.legend(loc="upper right")
        fig.text(0.5, 0.02, figure_context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))

        def update_animation(frame_index):
            state = animation_states[frame_index]
            if animation_view == "latent":
                state = state @ basis
            moving.set_offsets(state[:, :2].numpy())
            ax.set_title("Sampling step {}/{}".format(frame_index, len(animation_states) - 1))
            return (moving,)

        animation = FuncAnimation(
            fig,
            update_animation,
            frames=len(animation_states),
            interval=80,
            blit=False,
        )
        try:
            animation.save(animation_gif_path, writer=PillowWriter(fps=12))
            figure_paths["sampling_animation_gif"] = str(animation_gif_path)
        except Exception as exc:
            warning_path = figures_dir / "sampling_animation_gif_error.txt"
            warning_path.write_text("{}: {}".format(type(exc).__name__, exc))
            figure_paths["sampling_animation_gif_error"] = str(warning_path)
        try:
            animation.save(animation_mp4_path, writer=FFMpegWriter(fps=12, codec="h264"))
            figure_paths["sampling_animation_mp4"] = str(animation_mp4_path)
        except Exception as exc:
            warning_path = figures_dir / "sampling_animation_mp4_error.txt"
            warning_path.write_text("{}: {}".format(type(exc).__name__, exc))
            figure_paths["sampling_animation_mp4_error"] = str(warning_path)
        plt.close(fig)

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
        }
    )
    status_path.write_text(json.dumps(status, indent=2))

    summary = {
        "run_dir": str(run_dir),
        "config": config_payload,
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
            "data_equivalence_report": str(artifacts_dir / "data_equivalence_report.json")
            if data_equivalence_report is not None
            else None,
            "figures": str(figures_dir),
            "checkpoints": str(checkpoints_dir),
            "latest": str(latest_link),
            "latest_checkpoints": str(latest_checkpoints_link),
        },
        "figures": figure_paths,
    }
    (artifacts_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    if wandb_run is not None:
        image_payload = {}
        if wandb_module is not None:
            for name, path in figure_paths.items():
                suffix = Path(path).suffix.lower()
                if suffix in (".gif", ".mp4") and hasattr(wandb_module, "Video"):
                    image_payload["figures/{}".format(name)] = wandb_module.Video(
                        path,
                        fps=12,
                        format=suffix.lstrip("."),
                    )
                elif suffix in (".png", ".jpg", ".jpeg"):
                    image_payload["figures/{}".format(name)] = wandb_module.Image(path)
        wandb_run.log(
            wandb_scalar_payload(
                {
                    "final/loss": summary["final_loss"],
                    "final/f_mem": final_metrics.get("f_mem"),
                    "final/mean_nn_ratio": final_metrics.get("mean_nn_ratio"),
                    "final/std_nn_ratio": final_metrics.get("std_nn_ratio"),
                    "final/latent_f_mem": final_metrics.get("latent_f_mem"),
                    "final/latent_mean_nn_ratio": final_metrics.get("latent_mean_nn_ratio"),
                    "final/oracle_latent_nll_mean": final_metrics.get("oracle_latent_nll_mean"),
                    "final/oracle_full_nll_mean": final_metrics.get("oracle_full_nll_mean"),
                    "final/off_subspace_energy_fraction": final_metrics.get("off_subspace_energy_fraction"),
                    "final/latent_frechet_distance_to_oracle": final_metrics.get(
                        "latent_frechet_distance_to_oracle"
                    ),
                    "final/latent_sliced_w2": final_metrics.get("latent_sliced_w2"),
                    "final/component_weight_l1": final_metrics.get("component_weight_l1"),
                    "final/component_weight_kl": final_metrics.get("component_weight_kl"),
                    **image_payload,
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
        status_path.write_text(json.dumps(status, indent=2))
        (artifacts_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

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


if __name__ == "__main__":
    main()
