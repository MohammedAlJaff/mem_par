"""
Readable prototype recipe: CelebA32 grayscale images with the JiT wrapper.
"""

import argparse
from pathlib import Path

from prototyping.runners.common import (
    PARAMETERIZATION_CHOICES,
    artifact_paths,
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


RUNNER_NAME = "celeba32_jit"
MODULE_NAME = "prototyping.runners.celeba32_jit"
DEFAULT_STUDY_DIR = "outputs/prototype_runners/celeba32_jit"
DEFAULT_RUN_NAME = RUNNER_NAME


def default_knobs():
    """Return the editable knobs for this recipe."""
    # Treat this dictionary as the notebook-facing control panel. Every knob
    # should either become config or show up in the rendered CLI.
    return {
        "device": "cpu",
        "study_dir": DEFAULT_STUDY_DIR,
        "run_name": DEFAULT_RUN_NAME,
        "seed": 0,
        "pred_space": "v",
        "loss_space": "v",
        "n_train": 128,
        "n_eval": 128,
        "n_steps": 30000,
        "ckpt_every": 2000,
        "batch_cap": 32,
        "batch_size": None,
        "eval_samples": 32,
        "sampling_steps": 24,
        "root": "tmp/data",
        "download": False,
        "reference_split": "valid",
        "image_size": 32,
        "patch_size": 4,
        "hidden_dim": 128,
        "depth": 4,
        "heads": 4,
        "bottleneck_dim": 64,
        "proj_dropout": 0.0,
        "attn_dropout": 0.0,
        "in_context_len": 8,
        "in_context_start": 1,
        "wandb": False,
        "wandb_mode": "offline",
        "wandb_project": "mem_par",
        "wandb_entity": "",
        "wandb_notes": "",
    }


def build_config(knobs):
    """Build the canonical config, with every CelebA32/JiT choice visible."""
    # Merge notebook/CLI overrides into defaults once, then use `knobs`
    # everywhere so the mapping stays auditable.
    knobs = {**default_knobs(), **dict(knobs)}
    config = clone_config(get_base_config())

    # Study metadata names the artifacts; it does not change model behavior.
    config.study.name = RUNNER_NAME
    config.study.description = "CelebA32 grayscale GDFM/JiT prototype recipe."
    config.outputs.run_name = str(knobs["run_name"])

    # These are the parameterization knobs we sweep for the paper objective.
    config.framework.family = "fm"
    config.framework.prediction_space = str(knobs["pred_space"])
    config.framework.loss_space = str(knobs["loss_space"])

    # The data layer materializes fixed subsets for replay. No augmentation or
    # streaming is hidden inside this runner.
    config.data.dataset = "celeba32_grey"
    config.data.root = str(knobs["root"])
    config.data.download = bool(knobs["download"])
    config.data.reference_split = str(knobs["reference_split"])
    config.data.n_train = int(knobs["n_train"])
    config.data.n_eval = int(knobs["n_eval"])
    config.data.image_size = int(knobs["image_size"])
    config.data.image_channels = 1
    config.data.flatten = False
    config.data.data_dim = int(knobs["image_size"]) * int(knobs["image_size"])

    # JiT-specific architecture choices live under `config.network`; the GDFM
    # wrapper still owns prediction/loss-space semantics.
    config.network.arch = "jit"
    config.network.patch_size = int(knobs["patch_size"])
    config.network.hidden_dim = int(knobs["hidden_dim"])
    config.network.depth = int(knobs["depth"])
    config.network.heads = int(knobs["heads"])
    config.network.bottleneck_dim = int(knobs["bottleneck_dim"])
    config.network.proj_dropout = float(knobs["proj_dropout"])
    config.network.attn_dropout = float(knobs["attn_dropout"])
    config.network.in_context_len = int(knobs["in_context_len"])
    config.network.in_context_start = int(knobs["in_context_start"])

    config.training.seed = int(knobs["seed"])
    config.training.n_steps = int(knobs["n_steps"])
    config.training.ckpt_every = int(knobs["ckpt_every"])
    config.training.log_every = max(1, min(500, int(knobs["ckpt_every"])))
    batch_cap = resolve_batch_cap(knobs)
    # Public B-cap plus legacy mirror. The trainer derives effective batch size
    # from this and the selected N.
    config.training.batch_cap = batch_cap
    config.training.batch_size = batch_cap
    config.training.optimizer = "adam"

    # Pairwise metrics can be memory-heavy, so keep evaluation count and chunk
    # size visible in the recipe.
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
    """Materialize the canonical CelebA32 train/reference bundle."""
    return build_data_bundle(config)


def build_model(config, device):
    """Build the canonical GDFM wrapper and its JiT backbone."""
    return GDFM(config).to(device)


def run_one_update(model, train_tensor, config, device):
    """Run one visible optimizer update against the first mini-batch."""
    model.train()
    # One-update diagnostics use a deterministic prefix batch. Full jobs use
    # canonical DataLoader shuffling and checkpointing.
    batch_size = min(int(config.training.batch_cap), int(train_tensor.shape[0]))
    batch = train_tensor[:batch_size].to(device)
    torch.manual_seed(int(config.training.seed))
    # Explicit eps/t keeps this check reproducible across JiT config changes.
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
    # Pre-training metric check: catches accidental train/reference leakage
    # before running a slower JiT experiment.
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
    # This is the handoff from prototype recipe to canonical infrastructure.
    # Everything saved below follows the same contract as the paper studies.
    run_dir = train_run(config, Path(study_dir), device=device)
    metrics = evaluate_run(run_dir, device=device) if evaluate else []
    return {
        "run_dir": str(run_dir),
        "checkpoint_metrics": metrics,
        "artifacts": artifact_paths(run_dir),
    }


def parse_args(argv=None):
    """Parse the CelebA32 JiT command line."""
    knobs = default_knobs()
    parser = argparse.ArgumentParser(description="CelebA32 JiT prototype recipe.")
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
    parser.add_argument("--reference-split", default=knobs["reference_split"])
    parser.add_argument("--image-size", type=int, default=knobs["image_size"])
    parser.add_argument("--patch-size", type=int, default=knobs["patch_size"])
    parser.add_argument("--hidden-dim", type=int, default=knobs["hidden_dim"])
    parser.add_argument("--depth", type=int, default=knobs["depth"])
    parser.add_argument("--heads", type=int, default=knobs["heads"])
    parser.add_argument("--bottleneck-dim", type=int, default=knobs["bottleneck_dim"])
    parser.add_argument("--proj-dropout", type=float, default=knobs["proj_dropout"])
    parser.add_argument("--attn-dropout", type=float, default=knobs["attn_dropout"])
    parser.add_argument("--in-context-len", type=int, default=knobs["in_context_len"])
    parser.add_argument("--in-context-start", type=int, default=knobs["in_context_start"])
    args = parser.parse_args(argv)
    if args.batch_size is not None:
        # Legacy alias support; after parsing, the runner only uses `batch_cap`.
        args.batch_cap = args.batch_size
    return args


def main(argv=None):
    """CLI entrypoint for the CelebA32 JiT recipe."""
    args = parse_args(argv)
    knobs = {**default_knobs(), **vars(args)}
    config = build_config(knobs)

    if args.validate_only:
        # Validate-only is the quick notebook loop: data, model, one update,
        # memory metric, and rendered command with no full attempt directory.
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

    # Full run: delegate to canonical train/evaluate and report artifact paths.
    result = run_full_job(config, args.study_dir, args.device, evaluate=True)
    print_json(result)
    return result["run_dir"]


def command_from_knobs(knobs):
    """Render the CLI command matching a knob dictionary."""
    knobs = {**default_knobs(), **dict(knobs)}
    # Keep notebook-rendered commands faithful to the parser above.
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
        "reference_split": knobs["reference_split"],
        "download": bool(knobs["download"]),
        "image_size": knobs["image_size"],
        "patch_size": knobs["patch_size"],
        "hidden_dim": knobs["hidden_dim"],
        "depth": knobs["depth"],
        "heads": knobs["heads"],
        "bottleneck_dim": knobs["bottleneck_dim"],
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
