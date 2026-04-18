"""
Canonical training loop for first-party studies.

Read this file in three layers:

1. Small helpers above `train_run(...)`
   These are boring on purpose: choose an optimizer, resolve DataLoader
   batching, create shared evaluation noise, and save a simple loss figure.

2. `train_run(...)`
   This is the real orchestration point. It does more than gradient descent:
   it creates the run layout, writes the exact artifacts that checkpoint
   evaluation needs, trains the model, saves checkpoints, and records run
   status.

3. The `__main__` block
   This is a tiny runnable example that builds a GMM config and writes one
   throwaway run into a temporary directory.

The most important contract in this module:
    `train_run(...)` returns an attempt directory whose files are later consumed
    by `mp_metrics.evaluate_run(...)`.

Study-specific sweep logic and plotting decisions should live outside this
module. This file should stay focused on the common training lifecycle.
"""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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
from mp_tracking import AttemptTracker


def _build_optimizer(config, model):
    """
    Instantiate the optimizer requested by the study config.

    This is the only place in the trainer where `config.training.optimizer` is
    interpreted. Keeping the branching here makes the main training loop easier
    to read: the loop only needs an object with `zero_grad`, `step`, and a state
    dict for checkpoints.
    """
    name = str(config.training.optimizer).lower()

    if name == "adam":
        # Adam is the default for the current study configs.
        return torch.optim.Adam(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
        )

    if name == "sgd":
        # SGD is kept as a simple alternative for ablations or debugging.
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
            momentum=float(config.training.momentum),
        )

    raise ValueError(f"Unknown optimizer: {config.training.optimizer}")


def _resolve_batch_cap(config):
    """
    Resolve the public batch cap while preserving the old `batch_size` alias.
    """
    training = config.training
    batch_cap = getattr(training, "batch_cap", None)
    if batch_cap is None:
        batch_cap = getattr(training, "batch_size", None)
    if batch_cap is None:
        raise ValueError("Training config must define `batch_cap` or `batch_size`.")

    batch_cap = int(batch_cap)
    if batch_cap < 1:
        raise ValueError(f"training.batch_cap must be >= 1, got {batch_cap}")
    return batch_cap


def _batching_metadata(config, train_tensor):
    """
    Compute the DataLoader batching policy for one fixed saved training tensor.

    Optimizer steps are the official training x-axis. This metadata only
    records how a selected `N` maps to batches under the batch-cap policy.
    """
    n_train = int(train_tensor.shape[0])
    if n_train < 1:
        raise ValueError("Training requires at least one selected training sample.")

    batch_cap = _resolve_batch_cap(config)
    effective_batch_size = min(batch_cap, n_train)
    drop_last = n_train > batch_cap
    if drop_last:
        steps_per_epoch = n_train // effective_batch_size
    else:
        steps_per_epoch = 1
    examples_per_epoch = steps_per_epoch * effective_batch_size
    dropped_per_epoch = n_train - examples_per_epoch

    return {
        "batch_cap": int(batch_cap),
        "effective_batch_size": int(effective_batch_size),
        "drop_last": bool(drop_last),
        "steps_per_epoch": int(steps_per_epoch),
        "examples_per_epoch": int(examples_per_epoch),
        "dropped_per_epoch": int(dropped_per_epoch),
        "n_train": int(n_train),
        "n_steps": int(config.training.n_steps),
    }


def _build_train_loader(train_tensor, batching, generator):
    """
    Build a shuffled DataLoader over the already-materialized training tensor.
    """
    return DataLoader(
        train_tensor,
        batch_size=int(batching["effective_batch_size"]),
        shuffle=True,
        drop_last=bool(batching["drop_last"]),
        generator=generator,
    )


def _make_z_init(config, train_tensor):
    """
    Create the shared initial noise tensor used for later checkpoint evaluation.

    Saving this once per run is important for fair comparisons across
    checkpoints and parameterizations: evaluation should start from the same
    initial noise, not a freshly sampled one every time.

    The shape comes from the training tensor after data loading. That keeps this
    helper agnostic to vectors vs images: GMM gives `[B, D]`, while image data
    gives `[B, C, H, W]`.
    """
    sample_shape = tuple(train_tensor.shape[1:])
    return torch.randn(int(config.evaluation.n_samples), *sample_shape)


def _save_training_loss_figure(history, figures_dir):
    """
    Save the standard loss-vs-step training curve for one attempt.

    The trainer writes both machine-readable loss history and this quick visual
    check. The figure is not required for evaluation, but it is useful when
    opening a run folder by hand.
    """
    if not history:
        return None

    # Import matplotlib lazily so importing the trainer does not require a
    # plotting backend until we actually have a figure to save.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "training_loss_vs_step.png"
    steps = [item["step"] for item in history]
    losses = [item["loss"] for item in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, color="#1f77b4", linewidth=1.8)
    ax.set_title("Training loss vs step")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def train_run(config, study_dir, device="cpu"):
    """
    Run one complete training job and return the created run directory.

    Mental model:
        - `study_dir` is the experiment root chosen by the caller.
        - `create_run_layout(...)` creates one concrete attempt directory under
          that root.
        - Everything saved under that attempt directory is part of the
          evaluation contract for `evaluate_run(...)`.

    Phases:
        1. Create the run layout on disk.
        2. Start attempt tracking.
        3. Build the canonical data bundle.
        4. Save evaluation-critical artifacts: config, tensors, labels,
           indices, geometry, metadata, and shared `z_init`.
        5. Build the GDFM model and optimizer.
        6. Train, log loss, and save checkpoints.
        7. Write compact summaries for later inspection.
        8. Mark failures or interrupts in the attempt status files.
    """
    study_dir = Path(study_dir)
    run_start = time.perf_counter()

    # ---------------------------------------------------------------------
    # Phase 1: allocate the attempt directory.
    #
    # The returned path is not just a loose output folder. It has the canonical
    # layout:
    #
    #   <study_dir>/runs/<run_id>/seeds/<seed_id>/attempts/<attempt_id>/
    #
    # Evaluation and status tooling assume this shape, so all later paths are
    # built relative to `run_dir`.
    # ---------------------------------------------------------------------
    run_dir = create_run_layout(study_dir, config)

    # `tracker` is initialized inside the `try` so the exception handlers can
    # still update status if anything fails after layout creation.
    tracker = None
    try:
        # -----------------------------------------------------------------
        # Phase 2: start operator-facing tracking.
        #
        # AttemptTracker owns status.json, events.jsonl, console summaries, and
        # optional wandb forwarding. The training loop below only calls small
        # tracker methods at lifecycle boundaries.
        # -----------------------------------------------------------------
        tracker = AttemptTracker(config, run_dir)
        tracker.note(f"attempt_dir={run_dir.resolve()}", event_type="attempt_started")

        # -----------------------------------------------------------------
        # Phase 3: materialize the data bundle.
        #
        # `build_data_bundle` dispatches on `config.data.dataset` and returns a
        # standard dictionary with train/reference tensors plus any labels,
        # source indices, class counts, or synthetic geometry.
        # -----------------------------------------------------------------
        data_bundle = build_data_bundle(config)

        # -----------------------------------------------------------------
        # Phase 4: save the exact checkpoint-evaluation inputs.
        #
        # `evaluate_run(...)` does not rebuild the dataset from config. It loads
        # these saved tensors so checkpoint comparisons use the exact same data
        # the model trained against.
        # -----------------------------------------------------------------
        train_tensor = data_bundle["train"].cpu()
        reference_tensor = data_bundle["reference"].cpu()

        # `z_init` is the shared evaluation starting point. Every checkpoint is
        # sampled from this same initial noise tensor during evaluation.
        z_init = _make_z_init(config, train_tensor)

        # Resolve the public batch cap after the exact selected dataset size is
        # known. The legacy `batch_size` field mirrors the cap; derived values
        # live in `batching`.
        batching = _batching_metadata(config, train_tensor)
        config.training.batch_cap = int(batching["batch_cap"])
        config.training.batch_size = int(batching["batch_cap"])

        # Save the resolved config and the two core data tensors first. These
        # three files are the minimum needed to understand what this attempt
        # trained on.
        save_config_snapshot(run_dir, config)
        save_tensor(run_dir / "artifacts" / "train_tensor.pt", train_tensor)
        save_tensor(run_dir / "artifacts" / "reference_tensor.pt", reference_tensor)
        save_tensor(run_dir / "artifacts" / "z_init.pt", z_init)

        # Optional labels are present for MNIST, CelebA, and GMM cluster IDs.
        # They are not used by the unconditional GDFM training loop, but they
        # are useful for later diagnostics and balanced-subset checks.
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

        # Optional source indices let evaluation code and humans verify that train
        # and reference sets did not accidentally overlap when both came from a
        # shared corpus.
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

        # Synthetic GMM runs carry the planted subspace basis and cluster means.
        # Saving geometry here lets metrics explain ambient-vs-latent behavior
        # without reconstructing hidden state from seeds.
        if data_bundle.get("geometry") is not None:
            save_tensor(run_dir / "artifacts" / "geometry.pt", data_bundle["geometry"])

        # `data_meta.json` is the compact human-readable version of the bundle
        # contract. The heavy tensors live in `.pt` files; this JSON file tells
        # you what they are, where they came from, and which leakage checks ran.
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
            "train_role": data_bundle.get("train_role"),
            "reference_role": data_bundle.get("reference_role"),
            "train_source_id": data_bundle.get("train_source_id"),
            "reference_source_id": data_bundle.get("reference_source_id"),
            "data_validation": data_bundle.get("data_validation"),
            "batching": batching,
        }
        save_json(run_dir / "artifacts" / "data_meta.json", data_meta)
        tracker.stage(
            "data_prepared",
            message=(
                "data prepared "
                f"train_shape={list(train_tensor.shape)} "
                f"reference_shape={list(reference_tensor.shape)} "
                f"batch_cap={batching['batch_cap']} "
                f"effective_batch_size={batching['effective_batch_size']} "
                f"drop_last={batching['drop_last']}"
            ),
            data_meta_path=str((run_dir / "artifacts" / "data_meta.json").resolve()),
            batching=batching,
        )

        # -----------------------------------------------------------------
        # Phase 5: build the model and optimizer.
        #
        # GDFM reads the config to choose the prediction/loss spaces and to
        # build the configured backbone: MLP, UNet, or JiT.
        # -----------------------------------------------------------------
        model = GDFM(config).to(device)
        optimizer = _build_optimizer(config, model)

        # This generator controls the DataLoader shuffle order only. The
        # model's internal loss still samples its own times/noise unless
        # callers pass explicit values into `compute_loss`.
        generator = torch.Generator().manual_seed(int(config.training.seed))
        train_loader = _build_train_loader(train_tensor, batching, generator)
        train_iterator = iter(train_loader)

        # -----------------------------------------------------------------
        # Phase 6: train and checkpoint.
        # -----------------------------------------------------------------
        history = []

        # Save step 0 before any optimizer updates so evaluation can measure
        # the untrained model under the same path as later checkpoints.
        checkpoint_path = save_checkpoint(run_dir, 0, model, optimizer)
        tracker.checkpoint_saved(step=0, checkpoint_path=checkpoint_path)
        train_start = time.perf_counter()
        tracker.stage(
            "training",
            message=(
                "training started "
                f"steps={int(config.training.n_steps)} "
                f"batch_cap={batching['batch_cap']} "
                f"effective_batch_size={batching['effective_batch_size']} "
                f"drop_last={batching['drop_last']} "
                f"device={device}"
            ),
            batching=batching,
        )
        progress = tqdm(
            range(1, int(config.training.n_steps) + 1),
            desc=f"train:{config.outputs.run_name}",
            unit="step",
            disable=not tracker.interactive_progress,
            dynamic_ncols=True,
        )

        try:
            for step in progress:
                # Draw from the DataLoader over the fixed training tensor, then
                # move only the batch to the target device. Keeping the stored
                # tensor on CPU avoids pinning the whole dataset in GPU memory.
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)
                batch = batch.to(device)

                # The whole learning rule is these four lines: clear old
                # gradients, compute the GDFM loss, backpropagate, and update
                # parameters.
                optimizer.zero_grad(set_to_none=True)
                loss = model.compute_loss(batch)
                loss.backward()
                optimizer.step()

                # Keep history sparse. For long runs we do not need one JSON
                # record per step; we need enough points for progress, plots,
                # and final summaries.
                if (
                    step == 1
                    or step % int(config.training.log_every) == 0
                    or step == int(config.training.n_steps)
                ):
                    loss_value = float(loss.item())
                    history.append({"step": int(step), "loss": loss_value})
                    if not progress.disable:
                        progress.set_postfix(loss=f"{loss_value:.4f}", refresh=False)
                    tracker.training_progress(
                        step=int(step),
                        total_steps=int(config.training.n_steps),
                        loss=loss_value,
                        elapsed_sec=time.perf_counter() - train_start,
                        force=step == 1 or step == int(config.training.n_steps),
                    )

                # Checkpoints include both model and optimizer state. The final
                # checkpoint is always saved even if `n_steps` is not divisible
                # by `ckpt_every`.
                if (
                    step % int(config.training.ckpt_every) == 0
                    or step == int(config.training.n_steps)
                ):
                    checkpoint_path = save_checkpoint(run_dir, step, model, optimizer)
                    tracker.checkpoint_saved(step=step, checkpoint_path=checkpoint_path)
        finally:
            # Close tqdm cleanly even when training is interrupted or errors.
            progress.close()

        training_wall_time_sec = time.perf_counter() - train_start
        total_wall_time_sec = time.perf_counter() - run_start
        n_steps = int(config.training.n_steps)

        # -----------------------------------------------------------------
        # Phase 7: write final summaries.
        #
        # `train_history.json` is the machine-readable loss curve.
        # `run_summary.json` is the quick operator-facing overview.
        # -----------------------------------------------------------------
        save_json(run_dir / "metrics" / "train_history.json", history)
        training_figure_path = _save_training_loss_figure(history, run_dir / "figures")
        generated_figures = dict(tracker.status.get("generated_figures") or {})
        if training_figure_path is not None:
            tracker.register_figure("training_loss_vs_step", training_figure_path)
            generated_figures = dict(tracker.status.get("generated_figures") or {})

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
            "batching": batching,
            "batch_cap": int(batching["batch_cap"]),
            "effective_batch_size": int(batching["effective_batch_size"]),
            "drop_last": bool(batching["drop_last"]),
            "optimizer": str(config.training.optimizer),
            "training_wall_time_sec": training_wall_time_sec,
            "training_seconds_per_step": (
                training_wall_time_sec / n_steps if n_steps > 0 else 0.0
            ),
            "total_wall_time_sec": total_wall_time_sec,
            "figures_dir": str((run_dir / "figures").resolve()),
            "logs_dir": str((run_dir / "logs").resolve()),
            "status_path": str((run_dir / "status.json").resolve()),
            "attempt_log_path": str((run_dir / "logs" / "attempt.log").resolve()),
            "generated_figures": generated_figures,
            "wandb": tracker.status.get("wandb"),
        }
        save_json(run_dir / "artifacts" / "run_summary.json", run_summary)
        tracker.log_metrics(
            {
                "train/final_loss": history[-1]["loss"] if history else None,
                "train/training_wall_time_sec": training_wall_time_sec,
                "train/training_seconds_per_step": (
                    training_wall_time_sec / n_steps if n_steps > 0 else 0.0
                ),
            },
            step=n_steps,
        )

        # The tracker stage marks training complete, but not full evaluation
        # complete. `evaluate_run(...)` later transitions the same attempt to a
        # completed state after checkpoint metrics are written.
        tracker.stage(
            "trained",
            message=(
                "training complete "
                f"wall_time={training_wall_time_sec:.2f}s "
                f"seconds_per_step={training_wall_time_sec / n_steps if n_steps > 0 else 0.0:.4f}"
            ),
            run_summary_path=str((run_dir / "artifacts" / "run_summary.json").resolve()),
            train_history_path=str((run_dir / "metrics" / "train_history.json").resolve()),
            generated_figures=generated_figures,
        )
        return run_dir
    except KeyboardInterrupt as exc:
        # Preserve an intentional interruption as an explicit attempt state so
        # dashboards and manifests do not confuse it with a successful run.
        if tracker is not None:
            tracker.stage("aborted", last_error=f"{type(exc).__name__}: {exc}")
        else:
            update_attempt_status(run_dir, "aborted", last_error=f"{type(exc).__name__}: {exc}")
        raise
    except Exception as exc:
        # Any unexpected error should still leave the attempt directory
        # inspectable, with status.json pointing at the failure.
        if tracker is not None:
            tracker.stage("failed", last_error=f"{type(exc).__name__}: {exc}")
        else:
            update_attempt_status(run_dir, "failed", last_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # AttemptTracker may own file handles, console tee state, or external
        # logging handles. Close it regardless of success or failure.
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    import tempfile

    from mp_config import get_base_config

    print("mp_training.trainer example")

    # Build the smallest end-to-end training config that still writes a real run.
    config = get_base_config()

    # Study/output names determine the human-facing run labels and become part
    # of the artifact folder names.
    config.study.name = "trainer_demo"
    config.outputs.run_name = "trainer_demo"

    # Use the synthetic GMM path so the example does not need image files on
    # disk. These fields define the size of the ambient space and planted
    # latent mixture.
    config.data.dataset = "gmm_subspace"
    config.data.n_train = 32
    config.data.n_eval = 16
    config.data.data_dim = 8
    config.data.intrinsic_dim = 2
    config.data.n_clusters = 3

    # Pick the tiny MLP backbone. The same `GDFM(config)` call would build a
    # UNet or JiT backbone if `config.network.arch` requested one.
    config.network.arch = "mlp"
    config.network.hidden_dim = 32
    config.network.n_layers = 2
    config.network.time_emb_dim = 16

    # Keep the training schedule short enough for the example while still
    # producing multiple checkpoints and multiple loss-history entries.
    config.training.n_steps = 4
    config.training.batch_cap = 8
    config.training.batch_size = 8
    config.training.ckpt_every = 2
    config.training.log_every = 1

    # These evaluation fields are saved during training because checkpoint
    # evaluation later needs a fixed sample count, panel count, and distance chunk.
    config.evaluation.n_samples = 8
    config.evaluation.sample_panel_count = 4
    config.evaluation.chunk_size = 32
    config.sampling.n_steps = 4

    # Show the one-line training entrypoint before running the tiny example.
    print("Usage:")
    print("  config = get_base_config()")
    print("  run_dir = train_run(config, study_dir, device='cpu')")

    # Run into a temporary directory so the example is safe to execute anywhere.
    with tempfile.TemporaryDirectory() as tmpdir:
        # This is the same call used by experiments and prototype runners:
        # pass a resolved config, choose an experiment root, and pick a device.
        run_dir = train_run(config, Path(tmpdir) / "trainer_demo", device="cpu")

        # These paths are the key outputs to inspect after a training run.
        print(f"run_dir              : {run_dir}")
        print(f"checkpoints dir      : {run_dir / 'checkpoints'}")
        print(f"metrics file         : {run_dir / 'metrics' / 'train_history.json'}")
