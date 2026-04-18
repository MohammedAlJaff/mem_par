"""
Readable prototype recipe for 32x32 image collections with the UNet adapter.
"""

import argparse

from prototyping.runners.common import (
    PARAMETERIZATION_CHOICES,
    as_int_tuple,
    count_parameters,
    print_json,
    render_command,
    resolve_batch_cap,
)
from prototyping.runners.mnist_unet import (
    _config_sections,
    build_data,
    build_model,
    inspect_metrics,
    run_full_job,
    run_one_update,
)

from mp_config import clone_config, get_base_config


RUNNER_NAME = "image_unet"
MODULE_NAME = "prototyping.runners.image_unet"
DEFAULT_STUDY_DIR = "outputs/prototype_runners/image_unet"
# Dataset-specific defaults live here so the generic runner can stay small:
# a user picks `--dataset`, and this table fills the split/source details.
DATASET_DEFAULTS = {
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
    """Return the editable knobs for this image-collection recipe."""
    # This is the editable recipe surface shared by notebooks and CLI parsing.
    # Keep new knobs here first, then thread them into config/rendering below.
    return {
        "device": "cpu",
        "study_dir": DEFAULT_STUDY_DIR,
        "run_name": RUNNER_NAME,
        "seed": 0,
        "dataset": "flowers102",
        "pred_space": "v",
        "loss_space": "v",
        "n_train": 128,
        "n_eval": 128,
        "n_steps": 30000,
        "ckpt_every": 2000,
        "batch_cap": 64,
        "batch_size": None,
        "eval_samples": 64,
        "sampling_steps": 24,
        "root": "tmp/data",
        "dataset_subdir": None,
        "download": False,
        "train_split": "train",
        "reference_split": None,
        "allow_same_source_reference": None,
        "image_size": 32,
        "image_channels": 1,
        "resize_size": 0,
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
    """Build the canonical config for one image-collection UNet run."""
    # Merge once so every later assignment reads from one complete dictionary.
    knobs = {**default_knobs(), **dict(knobs)}
    dataset = str(knobs["dataset"])
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown image dataset: {dataset}")
    dataset_defaults = DATASET_DEFAULTS[dataset]

    config = clone_config(get_base_config())
    # Study metadata is for artifact labels/manifests; dataset/backbone choices
    # are set explicitly below.
    config.study.name = RUNNER_NAME
    config.study.description = f"{dataset} GDFM/UNet prototype recipe."
    config.outputs.run_name = str(knobs["run_name"])

    config.framework.family = "fm"
    config.framework.prediction_space = str(knobs["pred_space"])
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
    config.data.data_dim = int(knobs["image_size"]) * int(knobs["image_size"])

    # Generic image collections currently share the UNet adapter. Dataset
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
    # Public B-cap plus compatibility mirror for old scripts.
    config.training.batch_cap = batch_cap
    config.training.batch_size = batch_cap
    config.training.optimizer = "adam"

    # Keep evaluation memory controls visible because image pairwise metrics can
    # become expensive quickly.
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


def parse_args(argv=None):
    """Parse the image-collection UNet command line."""
    knobs = default_knobs()
    parser = argparse.ArgumentParser(description="Image collection UNet prototype recipe.")
    parser.add_argument("--device", default=knobs["device"])
    parser.add_argument("--study-dir", default=knobs["study_dir"])
    parser.add_argument("--run-name", default=knobs["run_name"])
    parser.add_argument("--seed", type=int, default=knobs["seed"])
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFAULTS), default=knobs["dataset"])
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
    parser.add_argument("--root", default=knobs["root"])
    parser.add_argument("--dataset-subdir", default=knobs["dataset_subdir"])
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=knobs["download"])
    parser.add_argument("--train-split", default=knobs["train_split"])
    parser.add_argument("--reference-split", default=knobs["reference_split"])
    parser.add_argument("--allow-same-source-reference", action=argparse.BooleanOptionalAction, default=knobs["allow_same_source_reference"])
    parser.add_argument("--image-size", type=int, default=knobs["image_size"])
    parser.add_argument("--image-channels", type=int, choices=(1, 3), default=knobs["image_channels"])
    parser.add_argument("--resize-size", type=int, default=knobs["resize_size"])
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
    if args.batch_size is not None:
        # Legacy alias support; downstream code should read `batch_cap`.
        args.batch_cap = args.batch_size
    args.channel_mult = as_int_tuple(args.channel_mult)
    args.attn_resolutions = as_int_tuple(args.attn_resolutions)
    return args


def main(argv=None):
    """CLI entrypoint for the image-collection UNet recipe."""
    args = parse_args(argv)
    knobs = {**default_knobs(), **vars(args)}
    config = build_config(knobs)

    if args.validate_only:
        # Quick path for new datasets: verify loading, model shape, one update,
        # and memory metrics before committing to a full run.
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

    # Full path is a canonical saved attempt; print the resulting artifact map.
    result = run_full_job(config, args.study_dir, args.device, evaluate=True)
    print_json(result)
    return result["run_dir"]


def command_from_knobs(knobs):
    """Render the CLI command matching a knob dictionary."""
    knobs = {**default_knobs(), **dict(knobs)}
    # Notebook helper: show exactly the command that recreates the current
    # knob dictionary.
    options = {
        "device": knobs["device"],
        "study_dir": knobs["study_dir"],
        "run_name": knobs["run_name"],
        "seed": knobs["seed"],
        "dataset": knobs["dataset"],
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
        "dataset_subdir": knobs["dataset_subdir"],
        "image_size": knobs["image_size"],
        "image_channels": knobs["image_channels"],
        "validate_only": bool(knobs.get("validate_only", False)),
    }
    if bool(knobs["download"]):
        options["download"] = True
    else:
        options["no_download"] = True
    return render_command(MODULE_NAME, options)


if __name__ == "__main__":
    main()
