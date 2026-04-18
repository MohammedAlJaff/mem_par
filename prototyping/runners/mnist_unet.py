"""
Readable prototype recipe: MNIST images with the canonical UNet adapter.
"""

import argparse
from pathlib import Path

from prototyping.runners.common import (
    PARAMETERIZATION_CHOICES,
    artifact_paths,
    as_int_tuple,
    count_parameters,
    print_json,
    render_command,
    resolve_batch_cap,
)

import torch
from gdfm import GDFM
from mp_config import clone_config, config_to_dict, get_base_config
from mp_data import build_data_bundle
from mp_metrics import compute_memorization_metrics, evaluate_run
from mp_training import train_run


RUNNER_NAME = "mnist_unet"
MODULE_NAME = "prototyping.runners.mnist_unet"
DEFAULT_STUDY_DIR = "outputs/prototype_runners/mnist_unet"
DEFAULT_RUN_NAME = RUNNER_NAME


def default_knobs():
    """Return the editable knobs for this recipe."""
    # Keep this dictionary as the notebook-facing control panel. Every entry
    # should map either to a config field or to a visible CLI flag below.
    return {
        "device": "cpu",
        "study_dir": DEFAULT_STUDY_DIR,
        "run_name": DEFAULT_RUN_NAME,
        "seed": 0,
        "pred_space": "v",
        "loss_space": "v",
        "n_train": 100,
        "n_eval": 100,
        "n_steps": 30000,
        "ckpt_every": 2000,
        "batch_cap": 100,
        "batch_size": None,
        "eval_samples": 128,
        "sampling_steps": 24,
        "root": "tmp/data",
        "download": True,
        "balance_train_classes": True,
        "balance_reference_classes": True,
        "image_size": 32,
        "base_channels": 128,
        "channel_mult": (1, 2, 4),
        "attn_resolutions": (16, 8),
        "num_res_blocks": 2,
        "dropout": 0.1,
        "time_emb_dim": None,
        "wandb": False,
        "wandb_mode": "offline",
        "wandb_project": "mem_par",
        "wandb_entity": "",
        "wandb_notes": "",
    }


def build_config(knobs):
    """Build the canonical config, with every MNIST/UNet choice visible here."""
    # Merge user overrides into defaults first so notebooks can pass a tiny
    # partial dictionary and still get a complete runnable config.
    knobs = {**default_knobs(), **dict(knobs)}
    config = clone_config(get_base_config())

    # Study metadata controls artifact names and status/manifests, not model
    # behavior. Keeping it here makes run folders self-describing.
    config.study.name = RUNNER_NAME
    config.study.description = "MNIST GDFM/UNet prototype recipe."
    config.outputs.run_name = str(knobs["run_name"])

    # These two fields are the parameterization experiment surface: they decide
    # what the network predicts and which target the loss supervises.
    config.framework.family = "fm"
    config.framework.prediction_space = str(knobs["pred_space"])
    config.framework.loss_space = str(knobs["loss_space"])

    # Data config only selects and stores a fixed subset. Augmentations and raw
    # streaming are intentionally absent from this prototype path.
    config.data.dataset = "mnist"
    config.data.root = str(knobs["root"])
    config.data.download = bool(knobs["download"])
    config.data.n_train = int(knobs["n_train"])
    config.data.n_eval = int(knobs["n_eval"])
    config.data.image_size = int(knobs["image_size"])
    config.data.image_channels = 1
    config.data.flatten = False
    config.data.data_dim = int(knobs["image_size"]) * int(knobs["image_size"])
    config.data.balance_train_classes = bool(knobs["balance_train_classes"])
    config.data.balance_reference_classes = bool(knobs["balance_reference_classes"])

    # The adapter behind `GDFM(config)` reads these fields to build the UNet
    # backbone; the GDFM objective stays outside the backbone.
    config.network.arch = "unet"
    config.network.base_channels = int(knobs["base_channels"])
    config.network.channel_mult = tuple(knobs["channel_mult"])
    config.network.attn_resolutions = tuple(knobs["attn_resolutions"])
    config.network.num_res_blocks = int(knobs["num_res_blocks"])
    config.network.dropout = float(knobs["dropout"])
    config.network.time_emb_dim = knobs["time_emb_dim"]

    config.training.seed = int(knobs["seed"])
    config.training.n_steps = int(knobs["n_steps"])
    config.training.ckpt_every = int(knobs["ckpt_every"])
    config.training.log_every = max(1, min(500, int(knobs["ckpt_every"])))
    batch_cap = resolve_batch_cap(knobs)
    # `batch_cap` is the new public B-cap. `batch_size` mirrors it only for
    # older code/tests while the repo finishes migrating.
    config.training.batch_cap = batch_cap
    config.training.batch_size = batch_cap
    config.training.optimizer = "adam"

    # Evaluation uses fixed saved tensors and z_init from the canonical run.
    # `chunk_size` is just a memory guard for pairwise distance metrics.
    config.evaluation.n_samples = int(knobs["eval_samples"])
    config.evaluation.batch_size = batch_cap
    config.evaluation.sample_panel_count = min(8, int(knobs["eval_samples"]))
    config.evaluation.chunk_size = max(
        128,
        int(knobs["eval_samples"]),
        int(knobs["n_eval"]),
        int(knobs["n_train"]),
    )

    config.sampling.method = "heun"
    config.sampling.n_steps = int(knobs["sampling_steps"])

    config.tracking.console.summary_every_steps = int(config.training.log_every)
    config.tracking.console.summary_every_seconds = 60.0
    config.tracking.wandb.enabled = bool(knobs["wandb"])
    config.tracking.wandb.mode = str(knobs["wandb_mode"])
    config.tracking.wandb.project = str(knobs["wandb_project"])
    config.tracking.wandb.entity = str(knobs["wandb_entity"])
    config.tracking.wandb.notes = str(knobs["wandb_notes"])
    return config


def build_data(config):
    """Materialize the canonical MNIST train/reference bundle."""
    return build_data_bundle(config)


def build_model(config, device):
    """Build the canonical GDFM wrapper and its UNet backbone."""
    return GDFM(config).to(device)


def run_one_update(model, train_tensor, config, device):
    """Run one visible optimizer update against the first mini-batch."""
    model.train()
    # One-update checks intentionally use a deterministic prefix batch. The full
    # training path below uses the canonical DataLoader/B-cap loop.
    batch_size = min(int(config.training.batch_cap), int(train_tensor.shape[0]))
    batch = train_tensor[:batch_size].to(device)
    torch.manual_seed(int(config.training.seed))
    # Supplying eps and t makes this smoke update reproducible and easy to
    # compare across prediction/loss parameterizations.
    eps = torch.randn_like(batch)
    t = torch.linspace(
        float(config.schedule.t_min),
        float(config.schedule.t_max),
        batch_size,
        device=batch.device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.training.lr),
        weight_decay=float(config.training.weight_decay),
    )
    optimizer.zero_grad(set_to_none=True)
    loss = model.compute_loss(batch, eps=eps, t=t)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0e9)
    optimizer.step()

    return {
        "batch_shape": list(batch.shape),
        "loss": float(loss.detach().cpu().item()),
        "grad_norm": float(grad_norm.detach().cpu().item()),
        "optimizer": "adam",
        "prediction_space": str(config.framework.prediction_space),
        "loss_space": str(config.framework.loss_space),
    }


def inspect_metrics(config, data_bundle, model=None, device="cpu"):
    """Compute a tiny held-out-reference memory check against training data."""
    del model, device
    # This is a pre-training sanity check: do the chosen reference samples look
    # too close to the selected train subset before we spend time training?
    train = data_bundle["train"].cpu()
    reference = data_bundle["reference"].cpu()
    count = min(int(config.evaluation.n_samples), reference.shape[0])
    metrics = compute_memorization_metrics(
        reference[:count],
        train,
        chunk_size=int(config.evaluation.chunk_size),
        config=config,
    )
    return {
        "data": _summarize_bundle(data_bundle),
        "reference_memory": metrics,
    }


def run_full_job(config, study_dir, device, evaluate=True):
    """Run canonical training, then canonical checkpoint evaluation if requested."""
    # Prototype recipes stop being special here. From this line on, artifacts,
    # checkpoints, status, W&B, and metrics are owned by the canonical `src/`
    # training/evaluation stack.
    run_dir = train_run(config, Path(study_dir), device=device)
    metrics = evaluate_run(run_dir, device=device) if evaluate else []
    return {
        "run_dir": str(run_dir),
        "checkpoint_metrics": metrics,
        "artifacts": artifact_paths(run_dir),
    }


def parse_args(argv=None):
    """Parse the MNIST UNet command line."""
    knobs = default_knobs()
    parser = argparse.ArgumentParser(description="MNIST UNet prototype recipe.")
    parser.add_argument("--device", default=knobs["device"])
    parser.add_argument("--study-dir", default=knobs["study_dir"])
    parser.add_argument("--run-name", default=knobs["run_name"])
    parser.add_argument("--seed", type=int, default=knobs["seed"])
    parser.add_argument("--pred-space", choices=PARAMETERIZATION_CHOICES, default=knobs["pred_space"])
    parser.add_argument("--loss-space", choices=PARAMETERIZATION_CHOICES, default=knobs["loss_space"])
    parser.add_argument("--n-train", type=int, default=knobs["n_train"])
    parser.add_argument("--n-eval", type=int, default=knobs["n_eval"])
    parser.add_argument("--n-steps", type=int, default=knobs["n_steps"])
    parser.add_argument("--ckpt-every", type=int, default=knobs["ckpt_every"])
    parser.add_argument("--batch-cap", type=int, default=knobs["batch_cap"])
    parser.add_argument("--batch-size", type=int, default=None, help="Legacy alias for --batch-cap.")
    parser.add_argument("--eval-samples", type=int, default=knobs["eval_samples"])
    parser.add_argument("--sampling-steps", type=int, default=knobs["sampling_steps"])
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=knobs["wandb"])
    parser.add_argument("--wandb-mode", choices=("offline", "online"), default=knobs["wandb_mode"])
    parser.add_argument("--wandb-project", default=knobs["wandb_project"])
    parser.add_argument("--wandb-entity", default=knobs["wandb_entity"])
    parser.add_argument("--wandb-notes", default=knobs["wandb_notes"])
    parser.add_argument("--root", default=knobs["root"])
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=knobs["download"])
    parser.add_argument("--balance-train-classes", action=argparse.BooleanOptionalAction, default=knobs["balance_train_classes"])
    parser.add_argument("--balance-reference-classes", action=argparse.BooleanOptionalAction, default=knobs["balance_reference_classes"])
    parser.add_argument("--image-size", type=int, default=knobs["image_size"])
    parser.add_argument("--base-channels", type=int, default=knobs["base_channels"])
    parser.add_argument("--channel-mult", default=",".join(str(item) for item in knobs["channel_mult"]))
    parser.add_argument("--attn-resolutions", default=",".join(str(item) for item in knobs["attn_resolutions"]))
    parser.add_argument("--num-res-blocks", type=int, default=knobs["num_res_blocks"])
    parser.add_argument("--dropout", type=float, default=knobs["dropout"])
    parser.add_argument("--time-emb-dim", type=int, default=knobs["time_emb_dim"])
    args = parser.parse_args(argv)
    if args.batch_size is not None:
        # Legacy alias: argparse stores it separately, but the rest of the
        # runner should see one source of truth.
        args.batch_cap = args.batch_size
    args.channel_mult = as_int_tuple(args.channel_mult)
    args.attn_resolutions = as_int_tuple(args.attn_resolutions)
    return args


def main(argv=None):
    """CLI entrypoint for the MNIST UNet recipe."""
    args = parse_args(argv)
    knobs = {**default_knobs(), **vars(args)}
    config = build_config(knobs)

    if args.validate_only:
        # Validate-only is deliberately cheap: load data, build the model, run
        # one update, compute one memory diagnostic, and render the command.
        data_bundle = build_data(config)
        model = build_model(config, args.device)
        update = run_one_update(model, data_bundle["train"], config, args.device)
        metrics = inspect_metrics(config, data_bundle, model, args.device)
        print_json(
            {
                "config": _config_sections(config),
                "model": {
                    "class": "{}.{}".format(
                        model.__class__.__module__,
                        model.__class__.__name__,
                    ),
                    "backbone_class": "{}.{}".format(
                        model.net.__class__.__module__,
                        model.net.__class__.__name__,
                    ),
                    "parameters": count_parameters(model),
                },
                "one_update": update,
                "metrics": metrics,
                "command": command_from_knobs({**knobs, "validate_only": True}),
            }
        )
        return None

    # The normal path performs a full canonical train/evaluate attempt and
    # prints only the artifact map that a user should inspect next.
    result = run_full_job(config, args.study_dir, args.device, evaluate=True)
    print_json(result)
    return result["run_dir"]


def command_from_knobs(knobs):
    """Render the CLI command matching a knob dictionary."""
    knobs = {**default_knobs(), **dict(knobs)}
    # Keep rendered commands aligned with the public CLI. This is what notebooks
    # display before optional full jobs.
    options = {
        "device": knobs["device"],
        "study_dir": knobs["study_dir"],
        "run_name": knobs["run_name"],
        "seed": knobs["seed"],
        "pred_space": knobs["pred_space"],
        "loss_space": knobs["loss_space"],
        "n_train": knobs["n_train"],
        "n_eval": knobs["n_eval"],
        "n_steps": knobs["n_steps"],
        "ckpt_every": knobs["ckpt_every"],
        "batch_cap": resolve_batch_cap(knobs),
        "eval_samples": knobs["eval_samples"],
        "sampling_steps": knobs["sampling_steps"],
        "root": knobs["root"],
        "download": bool(knobs["download"]),
        "balance_train_classes": bool(knobs["balance_train_classes"]),
        "balance_reference_classes": bool(knobs["balance_reference_classes"]),
        "image_size": knobs["image_size"],
        "base_channels": knobs["base_channels"],
        "channel_mult": knobs["channel_mult"],
        "attn_resolutions": knobs["attn_resolutions"],
        "validate_only": bool(knobs.get("validate_only", False)),
    }
    if not bool(knobs["download"]):
        options.pop("download")
        options["no_download"] = True
    return render_command(MODULE_NAME, options)


def _config_sections(config):
    payload = config_to_dict(config)
    return {
        "study": payload["study"],
        "framework": payload["framework"],
        "data": payload["data"],
        "network": payload["network"],
        "training": payload["training"],
        "evaluation": payload["evaluation"],
        "sampling": payload["sampling"],
        "outputs": payload["outputs"],
    }


def _summarize_bundle(bundle):
    summary = {
        "train_shape": list(bundle["train"].shape),
        "reference_shape": list(bundle["reference"].shape),
        "train_source_id": bundle.get("train_source_id"),
        "reference_source_id": bundle.get("reference_source_id"),
        "train_role": bundle.get("train_role"),
        "reference_role": bundle.get("reference_role"),
        "display_shape": bundle.get("display_shape"),
        "data_validation": bundle.get("data_validation"),
        "train_indices_preview": _preview_tensor(bundle.get("train_indices")),
        "reference_indices_preview": _preview_tensor(bundle.get("reference_indices")),
    }
    if bundle.get("train_labels") is not None:
        summary["train_labels_shape"] = list(bundle["train_labels"].shape)
        summary["train_labels_preview"] = _preview_tensor(bundle["train_labels"])
    if bundle.get("reference_labels") is not None:
        summary["reference_labels_shape"] = list(bundle["reference_labels"].shape)
        summary["reference_labels_preview"] = _preview_tensor(bundle["reference_labels"])
    return summary


def _preview_tensor(tensor, limit=10):
    if tensor is None:
        return None
    return [int(item) for item in tensor.flatten()[:limit].tolist()]


if __name__ == "__main__":
    main()
