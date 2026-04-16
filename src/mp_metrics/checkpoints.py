"""
Checkpoint-replay evaluation for first-party study runs.

Reload a saved run, regenerate samples at each checkpoint, and compute the
memorization curves.
"""

from pathlib import Path

import torch

from gdfm import GDFM
from mp_artifacts import (
    list_checkpoint_paths,
    load_config_snapshot,
    save_json,
    save_tensor,
    update_attempt_status,
)
from mp_metrics.memorization import compute_memorization_metrics, compute_nn_stats


def evaluate_run(run_dir, device="cpu"):
    """
    Replay every checkpoint in a saved run directory and collect metrics.

    For each checkpoint we:
        1. reload model weights
        2. regenerate samples from the saved shared `z_init`
        3. compare those samples against the saved training tensor
        4. compare held-out reference samples against the same training tensor
           as a baseline

    On the final checkpoint we also save a small sample/nearest-neighbor panel
    payload so later plotting code can show concrete examples instead of just
    scalar curves.
    """
    run_dir = Path(run_dir)
    update_attempt_status(run_dir, "evaluating")
    try:
        config = load_config_snapshot(run_dir)

        train_tensor = torch.load(
            run_dir / "artifacts" / "train_tensor.pt",
            map_location="cpu",
        )
        reference_tensor = torch.load(
            run_dir / "artifacts" / "reference_tensor.pt",
            map_location="cpu",
        )
        z_init = torch.load(run_dir / "artifacts" / "z_init.pt", map_location="cpu")

        model = GDFM(config).to(device)
        records = []
        checkpoint_paths = list_checkpoint_paths(run_dir)

        if not checkpoint_paths:
            raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")

        for checkpoint_path in checkpoint_paths:
            # Use the same saved `z_init` at every checkpoint so changes in the
            # curve reflect the model, not evaluation randomness.
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state["model"])
            samples = model.sample(
                z_init.clone().to(device),
                n_steps=int(config.sampling.n_steps),
                method=str(config.sampling.method),
                device=device,
            ).cpu()

            metrics = compute_memorization_metrics(
                samples,
                train_tensor,
                chunk_size=int(config.evaluation.chunk_size),
                config=config,
            )
            baseline = compute_memorization_metrics(
                reference_tensor[: samples.shape[0]],
                train_tensor,
                chunk_size=int(config.evaluation.chunk_size),
                config=config,
            )

            # Keep both generated-sample and held-out-reference metrics at the same
            # checkpoint so later plots can compare them directly.
            record = {
                "step": int(state["step"]),
                "ratio_threshold": metrics["ratio_threshold"],
                "f_mem": metrics["f_mem"],
                "mean_nn_ratio": metrics["mean_nn_ratio"],
                "std_nn_ratio": metrics["std_nn_ratio"],
                "mean_d1": metrics["mean_d1"],
                "mean_d2": metrics["mean_d2"],
                "baseline_f_mem": baseline["f_mem"],
                "baseline_mean_nn_ratio": baseline["mean_nn_ratio"],
                "baseline_std_nn_ratio": baseline["std_nn_ratio"],
            }
            records.append(record)

            if checkpoint_path == checkpoint_paths[-1]:
                # Save the final samples and their nearest neighbors for later
                # panels.
                neighbor_stats = compute_nn_stats(
                    samples,
                    train_tensor,
                    chunk_size=int(config.evaluation.chunk_size),
                )
                panel_count = min(
                    int(config.evaluation.sample_panel_count),
                    samples.shape[0],
                )
                save_tensor(
                    run_dir / "metrics" / "final_samples.pt",
                    samples[:panel_count],
                )
                save_tensor(
                    run_dir / "metrics" / "final_neighbors.pt",
                    train_tensor[neighbor_stats["indices"][:panel_count]],
                )

        save_json(run_dir / "metrics" / "checkpoint_metrics.json", records)
        update_attempt_status(
            run_dir,
            "completed",
            checkpoint_metrics_path=str(
                (run_dir / "metrics" / "checkpoint_metrics.json").resolve()
            ),
            final_samples_path=str((run_dir / "metrics" / "final_samples.pt").resolve()),
            final_neighbors_path=str(
                (run_dir / "metrics" / "final_neighbors.pt").resolve()
            ),
        )
        return records
    except KeyboardInterrupt as exc:
        update_attempt_status(run_dir, "aborted", last_error=f"{type(exc).__name__}: {exc}")
        raise
    except Exception as exc:
        update_attempt_status(run_dir, "failed", last_error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    print("mp_metrics.checkpoints example")

    # `evaluate_run(...)` works on an already-finished run directory rather than
    # on in-memory tensors. The example below shows the expected call pattern.
    print("Usage:")
    print("  from mp_metrics import evaluate_run")
    print("  records = evaluate_run(run_dir, device='cpu')")
    print("")

    # These artifacts are the replay inputs. They fix the data tensors and
    # shared `z_init` so checkpoint comparisons use the same starting point.
    print("Expected artifacts under run_dir/artifacts:")
    print("  - config.json")
    print("  - train_tensor.pt")
    print("  - reference_tensor.pt")
    print("  - z_init.pt")
    print("")

    # These checkpoints are the states that will be replayed and scored.
    print("Expected checkpoints under run_dir/checkpoints:")
    print("  - step_000000.pt")
    print("  - step_000050.pt")
    print("  - ...")
