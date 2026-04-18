"""
Readable prototype recipe for 32x32 image datasets with the UNet adapter.
"""

import argparse
from pathlib import Path
import sys

# Direct script execution from prototyping/runners should still find repo packages.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototyping.runners.common import (
    PARAMETERIZATION_CHOICES,
    artifact_paths,
    as_int_tuple,
    resolve_batch_cap,
)
from prototyping.runners.metrics_image32 import (
    DEFAULT_FIXED_TIMES,
    collect_loss_eval_record,
    collect_reference_memory,
    evaluate_image32_run,
)

import torch
from gdfm import GDFM
from mp_config import clone_config, config_to_dict, get_base_config
from mp_data import build_data_bundle
from mp_training import train_run


RUNNER_NAME = "image32_unet"
MODULE_NAME = "prototyping.runners.image32_unet"
DEFAULT_STUDY_DIR = "outputs/prototype_runners/image32_unet"

# Dataset-specific defaults let one Image32 runner cover MNIST, CelebA,
# flowers, animal faces, and similar 32x32 image checks without branching.
DATASET_DEFAULTS = {
    "mnist": {
        "reference_split": "test",
        "allow_same_source_reference": False,
        "dataset_subdir": "",
    },
    "celeba32_grey": {
        "reference_split": "valid",
        "allow_same_source_reference": False,
        "dataset_subdir": "",
    },
    "flowers102": {
        "reference_split": "val",
        "allow_same_source_reference": False,
        "dataset_subdir": "",
    },
    "afhq_v2": {
        "reference_split": "test",
        "allow_same_source_reference": False,
        "dataset_subdir": "afhq_v2",
    },
    "ffhq128": {
        "reference_split": "seeded_partition",
        "allow_same_source_reference": True,
        "dataset_subdir": "ffhq/thumbnails128x128",
    },
}


def default_knobs():
    """Return the editable knobs for this Image32 recipe."""
    # This is the editable recipe surface shared by notebooks and CLI parsing.
    # Keep new knobs here first, then thread them into config/rendering below.
    return {
        "device": "cpu",
        "out_dir": DEFAULT_STUDY_DIR,
        "run_name": RUNNER_NAME,
        "seed": 0,
        "dataset": "celeba32_grey",
        "prediction_space": "v",
        "loss_space": "v",
        "n_train": 128,
        "n_eval": 128,
        "n_steps": 30000,
        "ckpt_every": 2000,
        "batch_cap": 64,
        "eval_samples": 64,
        "sampling_steps": 24,
        "root": "data",
        "dataset_subdir": None,
        "download": False,
        "train_split": "train",
        "reference_split": None,
        "allow_same_source_reference": None,
        "image_size": 32,
        "image_channels": 1,
        "resize_size": 0,
        "balance_train_classes": True,
        "balance_reference_classes": True,
        "loss_eval_times": DEFAULT_FIXED_TIMES,
        "base_channels": 64,
        "channel_mult": (1, 2, 4, 4),
        "attn_resolutions": (8,),
        "num_res_blocks": 2,
        "dropout": 0.0,
        "time_emb_dim": None,
        "wandb": False,
        "wandb_mode": "offline",
        "wandb_project": "mem_par",
        "wandb_entity": "",
        "wandb_notes": "",
    }


def build_config(knobs):
    """Build the canonical config for one Image32 UNet run."""
    # Merge once so every later assignment reads from one complete dictionary.
    knobs = {**default_knobs(), **dict(knobs)}
    dataset = str(knobs["dataset"])
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown Image32 dataset: {dataset}")
    dataset_defaults = DATASET_DEFAULTS[dataset]

    config = clone_config(get_base_config())
    # Study metadata is for artifact labels/manifests; dataset/backbone choices
    # are set explicitly below.
    config.study.name = RUNNER_NAME
    config.study.description = f"{dataset} Image32 GDFM/UNet prototype recipe."
    config.outputs.run_name = str(knobs["run_name"])

    config.framework.family = "fm"
    config.framework.prediction_space = str(knobs["prediction_space"])
    config.framework.loss_space = str(knobs["loss_space"])

    # None-valued split/source knobs mean "use the dataset table default";
    # explicit values let the CLI override without adding per-dataset branches.
    config.data.dataset = dataset
    config.data.root = str(knobs["root"])
    config.data.dataset_subdir = str(
        knobs["dataset_subdir"]
        if knobs["dataset_subdir"] is not None
        else dataset_defaults["dataset_subdir"]
    )
    config.data.download = bool(knobs["download"])
    config.data.train_split = str(knobs["train_split"])
    config.data.reference_split = str(
        knobs["reference_split"]
        if knobs["reference_split"] is not None
        else dataset_defaults["reference_split"]
    )
    config.data.allow_same_source_reference = bool(
        knobs["allow_same_source_reference"]
        if knobs["allow_same_source_reference"] is not None
        else dataset_defaults["allow_same_source_reference"]
    )
    config.data.n_train = int(knobs["n_train"])
    config.data.n_eval = int(knobs["n_eval"])
    config.data.image_size = int(knobs["image_size"])
    config.data.image_channels = int(knobs["image_channels"])
    config.data.resize_size = int(knobs["resize_size"])
    config.data.flatten = False
    config.data.data_dim = (
        int(knobs["image_channels"])
        * int(knobs["image_size"])
        * int(knobs["image_size"])
    )
    config.data.balance_train_classes = bool(knobs["balance_train_classes"])
    config.data.balance_reference_classes = bool(knobs["balance_reference_classes"])

    # Generic image datasets currently share the UNet adapter. Dataset
    # differences stay in data config, not in model construction.
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
    # The canonical trainer still reads the mirror while the runner CLI exposes
    # only the public B-cap.
    config.training.batch_cap = batch_cap
    config.training.batch_size = batch_cap
    config.training.optimizer = "adam"

    # Keep evaluation memory controls visible because image pairwise metrics can
    # become expensive quickly.
    config.evaluation.n_samples = int(knobs["eval_samples"])
    config.evaluation.batch_size = batch_cap
    config.evaluation.sample_panel_count = min(8, int(knobs["eval_samples"]))
    config.evaluation.loss_eval_times = tuple(
        float(item) for item in knobs["loss_eval_times"]
    )
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
    """Materialize the canonical Image32 train/reference bundle."""
    return build_data_bundle(config)


def build_model(config, device):
    """Build the canonical GDFM wrapper and its UNet backbone."""
    return GDFM(config).to(device)


def run_one_update(model, train_tensor, config, device):
    """Run one visible optimizer update against the first mini-batch."""
    model.train()
    # This diagnostic is deterministic; full training uses canonical sampling.
    batch_size = min(int(config.training.batch_cap), int(train_tensor.shape[0]))
    batch = train_tensor[:batch_size].to(device)
    torch.manual_seed(int(config.training.seed))
    eps = torch.randn_like(batch)
    t = torch.linspace(
        float(config.schedule.t_min),
        float(config.schedule.t_max),
        batch_size,
        device=batch.device,
    )

    optimizer = torch.optim.Adam(
        model.parameterfs(),
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
    """Compute held-out memory and optional fixed/random-t loss diagnostics."""
    train = data_bundle["train"].cpu()
    reference = data_bundle["reference"].cpu()
    result = {
        "data": _summarize_bundle(data_bundle),
        "reference_memory": collect_reference_memory(
            reference,
            train,
            chunk_size=int(config.evaluation.chunk_size),
            config=config,
        ),
    }
    if model is not None:
        result["loss_eval"] = collect_loss_eval_record(
            0,
            model,
            train,
            config,
            device,
            fixed_times=tuple(config.evaluation.loss_eval_times),
        )
    return result


def run_full_job(config, out_dir, device, evaluate=True):
    """Run canonical training, then Image32 checkpoint metrics if requested."""
    # Training remains canonical; empirical metrics live beside this runner.
    run_dir = train_run(config, Path(out_dir), device=device)
    evaluation = (
        evaluate_image32_run(
            run_dir,
            device=device,
            fixed_times=tuple(config.evaluation.loss_eval_times),
        )
        if evaluate
        else {"checkpoint_metrics": [], "loss_eval": []}
    )
    return {
        "run_dir": str(run_dir),
        "checkpoint_metrics": evaluation["checkpoint_metrics"],
        "loss_eval": evaluation["loss_eval"],
        "artifacts": artifact_paths(run_dir),
    }


def parse_args(argv=None):
    """Parse the Image32 UNet command line."""
    knobs = default_knobs()
    parser = argparse.ArgumentParser(description="Image32 UNet prototype recipe.")
    parser.add_argument("--device", default=knobs["device"])
    parser.add_argument("--out-dir", default=knobs["out_dir"])
    parser.add_argument("--run-name", default=knobs["run_name"])
    parser.add_argument("--seed", type=int, default=knobs["seed"])
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFAULTS), default=knobs["dataset"])
    parser.add_argument("--prediction-space", choices=PARAMETERIZATION_CHOICES, default=knobs["prediction_space"])
    parser.add_argument("--loss-space", choices=PARAMETERIZATION_CHOICES, default=knobs["loss_space"])
    parser.add_argument("--n-train", type=int, default=knobs["n_train"])
    parser.add_argument("--n-eval", type=int, default=knobs["n_eval"])
    parser.add_argument("--n-steps", type=int, default=knobs["n_steps"])
    parser.add_argument("--ckpt-every", type=int, default=knobs["ckpt_every"])
    parser.add_argument("--batch-cap", type=int, default=knobs["batch_cap"])
    parser.add_argument("--eval-samples", type=int, default=knobs["eval_samples"])
    parser.add_argument("--sampling-steps", type=int, default=knobs["sampling_steps"])
    parser.add_argument("--root", default=knobs["root"])
    parser.add_argument("--dataset-subdir", default=knobs["dataset_subdir"])
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=knobs["download"])
    parser.add_argument("--train-split", default=knobs["train_split"])
    parser.add_argument("--reference-split", default=knobs["reference_split"])
    parser.add_argument("--allow-same-source-reference", action=argparse.BooleanOptionalAction, default=knobs["allow_same_source_reference"])
    parser.add_argument("--image-size", type=int, default=knobs["image_size"])
    parser.add_argument("--image-channels", type=int, choices=(1, 3), default=knobs["image_channels"])
    parser.add_argument("--resize-size", type=int, default=knobs["resize_size"])
    parser.add_argument("--balance-train-classes", action=argparse.BooleanOptionalAction, default=knobs["balance_train_classes"])
    parser.add_argument("--balance-reference-classes", action=argparse.BooleanOptionalAction, default=knobs["balance_reference_classes"])
    parser.add_argument("--loss-eval-times", default=",".join(str(item) for item in knobs["loss_eval_times"]))
    parser.add_argument("--base-channels", type=int, default=knobs["base_channels"])
    parser.add_argument("--channel-mult", default=",".join(str(item) for item in knobs["channel_mult"]))
    parser.add_argument("--attn-resolutions", default=",".join(str(item) for item in knobs["attn_resolutions"]))
    parser.add_argument("--num-res-blocks", type=int, default=knobs["num_res_blocks"])
    parser.add_argument("--dropout", type=float, default=knobs["dropout"])
    parser.add_argument("--time-emb-dim", type=int, default=knobs["time_emb_dim"])
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=knobs["wandb"])
    parser.add_argument("--wandb-mode", choices=("offline", "online"), default=knobs["wandb_mode"])
    parser.add_argument("--wandb-project", default=knobs["wandb_project"])
    parser.add_argument("--wandb-entity", default=knobs["wandb_entity"])
    parser.add_argument("--wandb-notes", default=knobs["wandb_notes"])
    args = parser.parse_args(argv)
    args.channel_mult = as_int_tuple(args.channel_mult)
    args.attn_resolutions = as_int_tuple(args.attn_resolutions)
    args.loss_eval_times = _as_float_tuple(args.loss_eval_times)
    return args


def main(argv=None):
    """CLI entrypoint for the Image32 UNet recipe."""
    args = parse_args(argv)
    knobs = {**default_knobs(), **vars(args)}
    config = build_config(knobs)

    result = run_full_job(config, args.out_dir, args.device, evaluate=True)
    _print_result(result)
    return result["run_dir"]


def _print_result(result):
    print("{:<18} {}".format("run_dir:", result["run_dir"]))
    print("{:<18} {}".format("checkpoints:", result["artifacts"]["checkpoints"]["path"]))
    print("{:<18} {}".format("metrics:", result["artifacts"]["checkpoint_metrics"]["path"]))
    print("{:<18} {}".format("loss_eval:", result["artifacts"]["loss_eval"]["path"]))


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


def _as_float_tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(float(item) for item in value)
    if value is None or str(value).strip() == "":
        return ()
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())


if __name__ == "__main__":
    main()
