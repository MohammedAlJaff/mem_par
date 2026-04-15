import argparse
from pathlib import Path

from mp_artifacts import resolve_run_dir
from mp_config import apply_overrides, clone_config


def _train_parser(study_dir):
    parser = argparse.ArgumentParser(description=f"Train study from {study_dir}")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--prediction-space", default=None)
    parser.add_argument("--loss-space", default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--download", action="store_true")
    return parser


def _evaluate_parser(study_dir):
    parser = argparse.ArgumentParser(description=f"Evaluate study from {study_dir}")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", default=None)
    return parser


def _plot_parser(study_dir):
    parser = argparse.ArgumentParser(description=f"Plot study outputs from {study_dir}")
    parser.add_argument("--run-dir", default=None)
    return parser


def _common_overrides(args):
    return {
        "outputs.run_name": args.run_name,
        "training.seed": args.seed,
        "data.n_train": args.n_train,
        "data.n_eval": args.n_eval,
        "training.n_steps": args.n_steps,
        "training.batch_size": args.batch_size,
        "framework.prediction_space": args.prediction_space,
        "framework.loss_space": args.loss_space,
        "framework.family": args.family,
        "data.root": args.data_root,
        "data.download": True if getattr(args, "download", False) else None,
    }


def train_main(get_config, study_dir):
    from mp_training import train_run

    args = _train_parser(study_dir).parse_args()
    config = clone_config(get_config())
    apply_overrides(config, _common_overrides(args))
    run_dir = train_run(config, Path(study_dir), device=args.device)
    print(run_dir)


def evaluate_main(study_dir):
    from mp_metrics import evaluate_run

    args = _evaluate_parser(study_dir).parse_args()
    run_dir = resolve_run_dir(study_dir, args.run_dir)
    evaluate_run(run_dir, device=args.device)
    print(run_dir)


def plot_main(study_dir):
    from experiments._shared.plot_helpers import plot_run

    args = _plot_parser(study_dir).parse_args()
    run_dir = resolve_run_dir(study_dir, args.run_dir)
    figures_dir = plot_run(run_dir, study_dir=study_dir)
    print(figures_dir)
