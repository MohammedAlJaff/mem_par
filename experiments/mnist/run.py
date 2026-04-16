"""
Official saved-run entrypoint for the live MNIST study.

This module is intentionally thin. It does not invent a parallel training
stack; it simply chooses a paper-facing MNIST config and then calls the
canonical `src/` pipeline:

1. build a study config
2. run `train_run(...)`
3. replay checkpoints with `evaluate_run(...)`
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mp_artifacts import load_attempt_status, load_config_snapshot, load_json
from mp_config import clone_config, get_base_config
from mp_metrics import evaluate_run
from mp_training import train_run


def build_mnist_config(
    *,
    seed: int = 0,
    root: str = "tmp/data",
    download: bool = True,
    n_train: int = 100,
    n_eval: int | None = None,
    prediction_space: str = "v",
    loss_space: str = "v",
    n_steps: int = 30_000,
    ckpt_every: int = 2_000,
    sampling_steps: int = 24,
    batch_size: int | None = None,
    eval_samples: int | None = None,
    run_name: str = "mnist_checkpoint_replay",
    balance_train_classes: bool = True,
    balance_reference_classes: bool = True,
    summary_every_steps: int | None = None,
    summary_every_seconds: float | None = None,
    wandb_enabled: bool = False,
    wandb_mode: str = "offline",
    wandb_project: str = "mem_par",
    wandb_entity: str = "",
    wandb_notes: str = "",
):
    """
    Build the canonical config for the live MNIST study.

    The defaults now mirror the checkpoint-replay prototype used in the paper
    notebook: a real saved run with balanced MNIST subsets, repeated
    checkpointing, and replayable memorization curves.
    """
    config = clone_config(get_base_config())
    if n_eval is None:
        n_eval = n_train
    if batch_size is None:
        batch_size = min(512, n_train)
    if eval_samples is None:
        eval_samples = min(128, n_eval)

    config.study.name = "mnist"
    config.study.description = (
        "Live MNIST checkpoint-replay prototype for the notebook-first paper workflow."
    )

    config.framework.family = "fm"
    config.framework.prediction_space = prediction_space
    config.framework.loss_space = loss_space

    config.data.dataset = "mnist"
    config.data.root = root
    config.data.download = download
    config.data.n_train = n_train
    config.data.n_eval = n_eval
    config.data.image_size = 32
    config.data.image_channels = 1
    config.data.flatten = False
    config.data.data_dim = 32 * 32
    config.data.balance_train_classes = balance_train_classes
    config.data.balance_reference_classes = balance_reference_classes

    config.network.arch = "unet"
    config.network.base_channels = 128
    config.network.channel_mult = (1, 2, 4)
    config.network.attn_resolutions = (16, 8)
    config.network.num_res_blocks = 2
    config.network.dropout = 0.1
    config.network.time_emb_dim = None

    config.training.seed = seed
    config.training.n_steps = n_steps
    config.training.batch_size = batch_size
    config.training.optimizer = "sgd"
    config.training.lr = 1.0e-2
    config.training.momentum = 0.95
    config.training.ckpt_every = ckpt_every
    config.training.log_every = max(1, min(500, ckpt_every // 2))

    if summary_every_steps is None:
        summary_every_steps = config.training.log_every
    if summary_every_seconds is None:
        summary_every_seconds = 60.0
    config.tracking.console.summary_every_steps = int(summary_every_steps)
    config.tracking.console.summary_every_seconds = float(summary_every_seconds)
    config.tracking.wandb.enabled = bool(wandb_enabled)
    config.tracking.wandb.mode = str(wandb_mode)
    config.tracking.wandb.project = str(wandb_project)
    config.tracking.wandb.entity = str(wandb_entity)
    config.tracking.wandb.notes = str(wandb_notes)

    config.evaluation.n_samples = eval_samples
    config.evaluation.batch_size = batch_size
    config.evaluation.sample_panel_count = min(8, eval_samples)
    config.evaluation.chunk_size = max(128, eval_samples, n_eval)

    config.sampling.method = "heun"
    config.sampling.n_steps = sampling_steps

    config.outputs.run_name = run_name
    return config


def run_mnist_study(study_dir, *, device: str = "cpu", config=None):
    """
    Run the live MNIST study and return `(run_dir, metrics)`.

    This is the official saved-run path. It writes a canonical run directory
    and then replays the saved checkpoints into memorization metrics.
    """
    if config is None:
        config = build_mnist_config()

    run_dir = train_run(config, Path(study_dir), device=device)
    metrics = evaluate_run(run_dir, device=device)
    return run_dir, metrics


def parse_args():
    """
    Parse command-line arguments for the saved-run MNIST entrypoint.
    """
    parser = argparse.ArgumentParser(
        description="Run the live MNIST study and write canonical run artifacts."
    )
    parser.add_argument(
        "--study-dir",
        default="tmp/mnist_study",
        help="Experiment root directory that will receive runs/<run_id>/seeds/<seed>/attempts/...",
    )
    parser.add_argument(
        "--root",
        default="tmp/data",
        help="Where torchvision MNIST data should live.",
    )
    parser.add_argument("--device", default="cpu", help="Training/eval device.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=30_000)
    parser.add_argument("--ckpt-every", type=int, default=2_000)
    parser.add_argument("--sampling-steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--run-name", default="mnist_checkpoint_replay")
    parser.add_argument(
        "--pred-space",
        choices=("x", "eps", "v"),
        default="v",
        help="Prediction parameterization.",
    )
    parser.add_argument(
        "--loss-space",
        choices=("x", "eps", "v"),
        default="v",
        help="Loss parameterization.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether torchvision may download MNIST if it is missing.",
    )
    parser.add_argument(
        "--balance-train-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to enforce exact class balance in the training subset.",
    )
    parser.add_argument(
        "--balance-reference-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to enforce exact class balance in the held-out reference subset.",
    )
    parser.add_argument(
        "--attempt-dir",
        default=None,
        help="Existing attempt directory to replay/evaluate instead of launching a new run.",
    )
    parser.add_argument(
        "--summary-every-steps",
        type=int,
        default=None,
        help="Emit console/attempt-log summaries at least this often in training steps.",
    )
    parser.add_argument(
        "--summary-every-seconds",
        type=float,
        default=None,
        help="Emit console/attempt-log summaries at least this often in wall-clock seconds.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable optional Weights & Biases tracking for this attempt.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online"),
        default="offline",
        help="W&B mode. Offline is recommended for cluster jobs that sync later.",
    )
    parser.add_argument("--wandb-project", default="mem_par", help="W&B project name.")
    parser.add_argument("--wandb-entity", default="", help="Optional W&B entity/team.")
    parser.add_argument("--wandb-notes", default="", help="Optional W&B run notes.")
    return parser.parse_args()


def _print_final_summary(args, config, run_dir, metrics):
    """
    Print the operator-facing summary after one MNIST attempt.
    """
    final_record = metrics[-1]
    status = load_attempt_status(run_dir)
    run_summary_path = run_dir / "artifacts" / "run_summary.json"
    run_summary = load_json(run_summary_path) if run_summary_path.exists() else {}
    figures = status.get("generated_figures") or run_summary.get("generated_figures") or {}
    wandb = status.get("wandb") or {}

    print("experiments.mnist.run")
    print(f"experiment          : {status.get('experiment_name')}")
    print(f"run_id              : {status.get('run_id')}")
    print(f"seed_id             : {status.get('seed_id')}")
    print(f"attempt_id          : {status.get('attempt_id')}")
    print(f"attempt_dir         : {run_dir}")
    print(f"status.json         : {run_dir / 'status.json'}")
    print(f"attempt.log         : {run_dir / 'logs' / 'attempt.log'}")
    print(f"run_summary.json    : {run_summary_path}")
    print(f"checkpoint_metrics  : {run_dir / 'metrics' / 'checkpoint_metrics.json'}")
    print(f"prediction/loss     : {config.framework.prediction_space}/{config.framework.loss_space}")
    print(f"n_train / n_eval    : {config.data.n_train} / {config.data.n_eval}")
    print(f"steps / ckpt_every  : {config.training.n_steps} / {config.training.ckpt_every}")
    print(f"sample steps        : {config.sampling.n_steps}")
    print(f"final checkpoint    : {final_record['step']}")
    print(f"final f_mem         : {final_record['f_mem']:.4f}")
    print(f"baseline f_mem      : {final_record['baseline_f_mem']:.4f}")
    if figures:
        print("figures             :")
        for name, path in sorted(figures.items()):
            print(f"  {name}: {path}")
    if wandb.get("enabled"):
        print(f"wandb mode          : {wandb.get('mode')}")
        print(f"wandb run id        : {wandb.get('run_id')}")
        print(f"wandb dir           : {wandb.get('dir')}")
        if wandb.get("run_url"):
            print(f"wandb url           : {wandb.get('run_url')}")
        if wandb.get("warning"):
            print(f"wandb warning       : {wandb.get('warning')}")
    if args.attempt_dir is not None:
        print("replay source       : existing attempt")


def main():
    """
    Build the config, run the study, and print a compact summary.
    """
    args = parse_args()
    if args.attempt_dir is not None:
        run_dir = Path(args.attempt_dir)
        metrics = evaluate_run(run_dir, device=args.device)
        config = load_config_snapshot(run_dir)
    else:
        config = build_mnist_config(
            seed=args.seed,
            root=args.root,
            download=args.download,
            n_train=args.n_train,
            n_eval=args.n_eval,
            prediction_space=args.pred_space,
            loss_space=args.loss_space,
            n_steps=args.n_steps,
            ckpt_every=args.ckpt_every,
            sampling_steps=args.sampling_steps,
            batch_size=args.batch_size,
            eval_samples=args.eval_samples,
            run_name=args.run_name,
            balance_train_classes=args.balance_train_classes,
            balance_reference_classes=args.balance_reference_classes,
            summary_every_steps=args.summary_every_steps,
            summary_every_seconds=args.summary_every_seconds,
            wandb_enabled=args.wandb,
            wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_notes=args.wandb_notes,
        )

        run_dir, metrics = run_mnist_study(
            args.study_dir,
            device=args.device,
            config=config,
        )
    _print_final_summary(args, config, run_dir, metrics)


if __name__ == "__main__":
    main()
