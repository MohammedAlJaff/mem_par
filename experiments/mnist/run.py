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

from mp_config import clone_config, get_base_config
from mp_metrics import evaluate_run
from mp_training import train_run


def build_mnist_config(
    *,
    seed: int = 0,
    root: str = "tmp/data",
    download: bool = True,
    n_train: int = 64,
    n_eval: int = 128,
    prediction_space: str = "v",
    loss_space: str = "v",
    n_steps: int = 200,
    sampling_steps: int = 24,
    batch_size: int = 16,
    eval_samples: int = 16,
    run_name: str = "mnist_live",
):
    """
    Build the canonical config for the live MNIST study.

    The defaults deliberately stay small enough for a reviewer-friendly first
    run, while still exercising the full artifact-writing path.
    """
    config = clone_config(get_base_config())

    config.study.name = "mnist"
    config.study.description = (
        "Live MNIST paper-study surface for the notebook-first repo workflow."
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

    config.network.arch = "unet"
    config.network.base_channels = 32
    config.network.channel_mult = (1, 2, 2)
    config.network.attn_resolutions = (8,)
    config.network.num_res_blocks = 1
    config.network.dropout = 0.0
    config.network.time_emb_dim = 32

    config.training.seed = seed
    config.training.n_steps = n_steps
    config.training.batch_size = batch_size
    config.training.ckpt_every = max(1, n_steps // 4)
    config.training.log_every = max(1, n_steps // 10)

    config.evaluation.n_samples = eval_samples
    config.evaluation.batch_size = batch_size
    config.evaluation.sample_panel_count = min(8, eval_samples)
    config.evaluation.chunk_size = max(32, eval_samples, n_eval)

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
        help="Study directory that will receive runs/ and figures/.",
    )
    parser.add_argument(
        "--root",
        default="tmp/data",
        help="Where torchvision MNIST data should live.",
    )
    parser.add_argument("--device", default="cpu", help="Training/eval device.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-eval", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--sampling-steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--run-name", default="mnist_live")
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
    return parser.parse_args()


def main():
    """
    Build the config, run the study, and print a compact summary.
    """
    args = parse_args()
    config = build_mnist_config(
        seed=args.seed,
        root=args.root,
        download=args.download,
        n_train=args.n_train,
        n_eval=args.n_eval,
        prediction_space=args.pred_space,
        loss_space=args.loss_space,
        n_steps=args.n_steps,
        sampling_steps=args.sampling_steps,
        batch_size=args.batch_size,
        eval_samples=args.eval_samples,
        run_name=args.run_name,
    )

    run_dir, metrics = run_mnist_study(
        args.study_dir,
        device=args.device,
        config=config,
    )
    final_record = metrics[-1]

    print("experiments.mnist.run")
    print(f"study_dir           : {Path(args.study_dir)}")
    print(f"run_dir             : {run_dir}")
    print(f"prediction/loss     : {args.pred_space}/{args.loss_space}")
    print(f"n_train / n_eval    : {args.n_train} / {args.n_eval}")
    print(f"steps / sample steps: {args.n_steps} / {args.sampling_steps}")
    print(f"final checkpoint    : {final_record['step']}")
    print(f"final f_mem         : {final_record['f_mem']:.4f}")
    print(f"baseline f_mem      : {final_record['baseline_f_mem']:.4f}")


if __name__ == "__main__":
    main()
