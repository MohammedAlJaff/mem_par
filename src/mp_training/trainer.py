"""
Canonical training loop for first-party studies.

This file contains the reusable training loop:
- create a run directory
- load the configured data bundle
- save the artifacts needed for deterministic later replay
- build the model and optimizer
- train while logging loss and saving checkpoints

It deliberately avoids study-specific sweep logic or plotting decisions.
"""

import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from gdfm import GDFM
from mp_artifacts import (
    create_run_layout,
    save_checkpoint,
    save_config_snapshot,
    save_json,
    save_tensor,
    update_attempt_status,
)
from mp_data import build_data_bundle


def _running_in_notebook():
    """
    Detect whether the current Python process is running inside IPython/Jupyter.
    """
    try:
        from IPython import get_ipython
    except ModuleNotFoundError:
        return False

    shell = get_ipython()
    if shell is None:
        return False
    return shell.__class__.__name__ in {"ZMQInteractiveShell", "TerminalInteractiveShell"}


def _should_show_progress(config):
    """
    Show tqdm progress in notebooks and interactive terminals, but stay quiet in
    non-interactive test and batch contexts unless explicitly overridden later.
    """
    if not bool(getattr(config.training, "show_progress", True)):
        return False
    return _running_in_notebook() or sys.stderr.isatty()


def _build_optimizer(config, model):
    """
    Instantiate the optimizer requested by the study config.
    """
    name = str(config.training.optimizer).lower()

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
        )

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
            momentum=float(config.training.momentum),
        )

    raise ValueError(f"Unknown optimizer: {config.training.optimizer}")


def _sample_batch(train_tensor, batch_size, generator):
    """
    Draw a random mini-batch from the already-materialized training tensor.
    """
    indices = torch.randint(
        0, train_tensor.shape[0], (batch_size,), generator=generator
    )
    return train_tensor[indices]


def _make_z_init(config, train_tensor):
    """
    Create the shared initial noise tensor used for later evaluation replay.

    Saving this once per run is important for fair comparisons across
    checkpoints and parameterizations: evaluation should start from the same
    initial noise, not a freshly sampled one every time.
    """
    sample_shape = tuple(train_tensor.shape[1:])
    return torch.randn(int(config.evaluation.n_samples), *sample_shape)


def train_run(config, study_dir, device="cpu"):
    """
    Run one complete training job and return the created run directory.

    Phases:
        1. create the run layout on disk
        2. build the canonical data bundle
        3. save replay-critical artifacts such as tensors, labels, indices,
           geometry, config snapshot, and shared `z_init`
        4. build the GDFM model and optimizer
        5. train, log loss, and save checkpoints
        6. write compact run summaries for later inspection
    """
    study_dir = Path(study_dir)
    run_start = time.perf_counter()
    run_dir = create_run_layout(study_dir, config)
    try:
        data_bundle = build_data_bundle(config)

        # -----------------------------------------------------------------
        # Save the exact tensors and metadata needed for later replay.
        # -----------------------------------------------------------------
        train_tensor = data_bundle["train"].cpu()
        reference_tensor = data_bundle["reference"].cpu()
        z_init = _make_z_init(config, train_tensor)

        save_config_snapshot(run_dir, config)
        save_tensor(run_dir / "artifacts" / "train_tensor.pt", train_tensor)
        save_tensor(run_dir / "artifacts" / "reference_tensor.pt", reference_tensor)
        save_tensor(run_dir / "artifacts" / "z_init.pt", z_init)

        if data_bundle.get("train_labels") is not None:
            save_tensor(
                run_dir / "artifacts" / "train_labels.pt",
                data_bundle["train_labels"],
            )
        if data_bundle.get("reference_labels") is not None:
            save_tensor(
                run_dir / "artifacts" / "reference_labels.pt",
                data_bundle["reference_labels"],
            )
        if data_bundle.get("train_indices") is not None:
            save_tensor(
                run_dir / "artifacts" / "train_indices.pt",
                data_bundle["train_indices"],
            )
        if data_bundle.get("reference_indices") is not None:
            save_tensor(
                run_dir / "artifacts" / "reference_indices.pt",
                data_bundle["reference_indices"],
            )
        if data_bundle.get("geometry") is not None:
            save_tensor(run_dir / "artifacts" / "geometry.pt", data_bundle["geometry"])

        data_meta = {
            "dataset": str(config.data.dataset),
            "data_root": str(config.data.root),
            "download": bool(config.data.download),
            "display_shape": data_bundle.get("display_shape"),
            "train_shape": list(train_tensor.shape),
            "reference_shape": list(reference_tensor.shape),
            "train_class_counts": data_bundle.get("train_class_counts"),
            "reference_class_counts": data_bundle.get("reference_class_counts"),
            "balanced_train_classes": data_bundle.get("balanced_train_classes"),
            "balanced_reference_classes": data_bundle.get("balanced_reference_classes"),
            "train_split": data_bundle.get("train_split"),
            "reference_split": data_bundle.get("reference_split")
            or getattr(config.data, "reference_split", None),
        }
        save_json(run_dir / "artifacts" / "data_meta.json", data_meta)
        update_attempt_status(
            run_dir,
            "data_prepared",
            data_meta_path=str((run_dir / "artifacts" / "data_meta.json").resolve()),
        )

        model = GDFM(config).to(device)
        optimizer = _build_optimizer(config, model)
        generator = torch.Generator().manual_seed(int(config.training.seed))

        # -------------------------------------------------------------
        # Training loop
        # -------------------------------------------------------------
        history = []
        save_checkpoint(run_dir, 0, model, optimizer)
        train_start = time.perf_counter()
        update_attempt_status(run_dir, "training")
        progress = tqdm(
            range(1, int(config.training.n_steps) + 1),
            desc=f"train:{config.outputs.run_name}",
            unit="step",
            disable=not _should_show_progress(config),
            dynamic_ncols=True,
        )

        try:
            for step in progress:
                batch = _sample_batch(
                    train_tensor,
                    min(int(config.training.batch_size), train_tensor.shape[0]),
                    generator,
                ).to(device)

                optimizer.zero_grad(set_to_none=True)
                loss = model.compute_loss(batch)
                loss.backward()
                optimizer.step()

                if (
                    step == 1
                    or step % int(config.training.log_every) == 0
                    or step == int(config.training.n_steps)
                ):
                    loss_value = float(loss.item())
                    history.append({"step": int(step), "loss": loss_value})
                    if not progress.disable:
                        progress.set_postfix(loss=f"{loss_value:.4f}", refresh=False)

                if (
                    step % int(config.training.ckpt_every) == 0
                    or step == int(config.training.n_steps)
                ):
                    save_checkpoint(run_dir, step, model, optimizer)
        finally:
            progress.close()

        training_wall_time_sec = time.perf_counter() - train_start
        total_wall_time_sec = time.perf_counter() - run_start
        n_steps = int(config.training.n_steps)

        # -------------------------------------------------------------
        # Final compact summaries
        # -------------------------------------------------------------
        save_json(run_dir / "metrics" / "train_history.json", history)
        run_summary = {
            "study": config.study.name,
            "run_dir": str(run_dir),
            "experiment_dir": str(study_dir.resolve()),
            "run_id": run_dir.parents[3].name,
            "seed_id": run_dir.parents[1].name,
            "attempt_id": run_dir.name,
            "seed": int(config.training.seed),
            "n_steps": n_steps,
            "n_train": int(config.data.n_train),
            "n_eval": int(config.data.n_eval),
            "optimizer": str(config.training.optimizer),
            "training_wall_time_sec": training_wall_time_sec,
            "training_seconds_per_step": (
                training_wall_time_sec / n_steps if n_steps > 0 else 0.0
            ),
            "total_wall_time_sec": total_wall_time_sec,
            "figures_dir": str((run_dir / "figures").resolve()),
            "logs_dir": str((run_dir / "logs").resolve()),
        }
        save_json(run_dir / "artifacts" / "run_summary.json", run_summary)
        update_attempt_status(
            run_dir,
            "trained",
            run_summary_path=str((run_dir / "artifacts" / "run_summary.json").resolve()),
            train_history_path=str((run_dir / "metrics" / "train_history.json").resolve()),
        )
        return run_dir
    except KeyboardInterrupt as exc:
        update_attempt_status(run_dir, "aborted", last_error=f"{type(exc).__name__}: {exc}")
        raise
    except Exception as exc:
        update_attempt_status(run_dir, "failed", last_error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    import tempfile

    from mp_config import get_base_config

    print("mp_training.trainer example")

    # Build the smallest end-to-end training config that still writes a real run.
    config = get_base_config()
    config.study.name = "trainer_demo"
    config.outputs.run_name = "trainer_demo"
    config.data.dataset = "gmm_subspace"
    config.data.n_train = 32
    config.data.n_eval = 16
    config.data.data_dim = 8
    config.data.intrinsic_dim = 2
    config.data.n_clusters = 3
    config.network.arch = "mlp"
    config.network.hidden_dim = 32
    config.network.n_layers = 2
    config.network.time_emb_dim = 16
    config.training.n_steps = 4
    config.training.batch_size = 8
    config.training.ckpt_every = 2
    config.training.log_every = 1
    config.evaluation.n_samples = 8
    config.evaluation.sample_panel_count = 4
    config.evaluation.chunk_size = 32
    config.sampling.n_steps = 4

    # Show the one-line training entrypoint before running the smoke example.
    print("Usage:")
    print("  config = get_base_config()")
    print("  run_dir = train_run(config, study_dir, device='cpu')")

    # Run into a temporary directory so the example is safe to execute anywhere.
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = train_run(config, Path(tmpdir) / "trainer_demo", device="cpu")

        # These paths are the key outputs to inspect after a training run.
        print(f"run_dir              : {run_dir}")
        print(f"checkpoints dir      : {run_dir / 'checkpoints'}")
        print(f"metrics file         : {run_dir / 'metrics' / 'train_history.json'}")
