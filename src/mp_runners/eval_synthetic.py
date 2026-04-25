"""
CLI for replaying checkpoint metrics on finished synthetic training attempts.

Training writes the exact tensors, geometry, checkpoints, and `z_init.pt`.
This module only validates that saved attempt and calls the synthetic evaluator;
it does not rebuild datasets from current defaults.
"""

import argparse
from pathlib import Path
import sys

import torch

from mp_artifacts import list_checkpoint_paths, load_attempt_status
from mp_runners.common import artifact_paths
from mp_runners.synthetic_eval import evaluate_synthetic_run


DONE_ATTEMPT_PREFIX = "DONE_"
EVAL_READY_STATES = {"trained", "completed"}
EVAL_REQUIRED_FILES = (
    "status.json",
    "artifacts/config.json",
    "artifacts/train_tensor.pt",
    "artifacts/reference_tensor.pt",
    "artifacts/z_init.pt",
    "artifacts/geometry.pt",
    "artifacts/run_summary.json",
)


def _resolve_device(device):
    """Resolve `auto` to an available torch device string."""
    if str(device) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _validate_run_dir(run_dir):
    """Return a replayable DONE attempt directory or raise a clear error."""
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.name.startswith(DONE_ATTEMPT_PREFIX):
        raise ValueError(
            "run_dir must be a finished DONE_ attempt directory; "
            f"got {run_dir}"
        )
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"run_dir is missing status.json: {run_dir}")
    status = load_attempt_status(run_dir)
    state = str(status.get("state", ""))
    if state not in EVAL_READY_STATES:
        raise ValueError(
            "run_dir must have status state 'trained' or 'completed'; "
            f"got {state!r} for {run_dir}"
        )
    missing = [
        str((run_dir / relative_path).resolve())
        for relative_path in EVAL_REQUIRED_FILES
        if not (run_dir / relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "run_dir is missing required saved-run artifacts: " + ", ".join(missing)
        )
    if not list_checkpoint_paths(run_dir):
        raise FileNotFoundError(f"run_dir has no checkpoints: {run_dir / 'checkpoints'}")
    return run_dir


def run_eval_job(run_dir, device):
    """
    Replay synthetic checkpoint metrics for one finished DONE attempt.
    """
    device = _resolve_device(device)
    run_dir = _validate_run_dir(run_dir)
    evaluation = evaluate_synthetic_run(run_dir, device=device)
    return {
        "mode": "run_dir",
        "run_dir": str(run_dir),
        "checkpoint_metrics": evaluation["checkpoint_metrics"],
        "loss_eval": evaluation["loss_eval"],
        "checkpoint_sample_previews": evaluation["checkpoint_sample_previews"],
        "figures": evaluation.get("figures", {}),
        "final_metrics": evaluation.get("final_metrics", {}),
        "final_loss_eval": evaluation.get("final_loss_eval", {}),
        "artifacts": artifact_paths(run_dir),
    }


def parse_args(argv=None):
    """Parse synthetic eval CLI arguments."""
    parser = argparse.ArgumentParser(description="Replay synthetic checkpoint metrics.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(sys.argv[1:] if argv is None else list(argv))


def _print_field(label, value):
    """Print one aligned terminal field."""
    print("{:<18} {}".format(label + ":", value))


def _print_result(result):
    """Print a concise eval result."""
    _print_field("run_dir", result["run_dir"])
    _print_field("checkpoints", len(result["checkpoint_metrics"]))
    _print_field("metrics", result["artifacts"]["checkpoint_metrics"]["path"])
    _print_field("loss_eval", result["artifacts"]["loss_eval"]["path"])


def main(argv=None):
    """CLI entrypoint."""
    args = parse_args(argv)
    result = run_eval_job(args.run_dir, args.device)
    _print_result(result)
    return result


if __name__ == "__main__":
    main()
