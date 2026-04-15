"""
Unified training loop for all experiments.

Handles MLP (synthetic), UNet (images), and JiT (images) with the same
interface. The Trainer is parameterization-agnostic — the FlowMatching
wrapper handles all prediction/loss space logic.

Responsibilities:
    1. Training loop (step-based or epoch-based)
    2. EMA weight maintenance
    3. Learning rate scheduling
    4. Periodic checkpointing (for τ_mem extraction later)
    5. Loss logging
    6. Config saving for reproducibility

What this does NOT do:
    - Evaluation (metrics, sampling, visualization) — that's a separate step
      via evaluate_ckpts.py. This keeps training fast and GPU-efficient.
    - Hyperparameter search — that's handled by sweep configs.
    - Data loading — the Dataset is passed in, already subsampled.

The separation of training and evaluation is intentional:
    - Training runs on GPU, saves checkpoints.
    - Evaluation runs as a separate SLURM job (or locally) on saved checkpoints.
    - This avoids wasting GPU time on metric computation during training.
    - It also means we can add new metrics later without retraining.

Usage:
    from experiments.natural_images.common.training import Trainer
    from experiments.natural_images.common.flow_matching import FlowMatching

    model = FlowMatching(network, pred_space="x", loss_space="v")
    trainer = Trainer(model, dataset, config, run_dir="results/my_run")
    trainer.train()
"""

import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Optional

from .ema import EMA
from .lr_schedule import CosineWarmup
from ..flow_matching import FlowMatching
from ..utils import save_config, save_metrics, log_message


class Trainer:
    """
    Unified training loop.

    Supports two stopping modes:
        max_steps:  stop after N optimizer steps (for synthetic data where
                    n varies and we want fixed compute per run)
        max_epochs: stop after N passes through the dataset (for image data
                    where n is larger and we think in epochs)

    Only one of max_steps / max_epochs should be set.

    Args:
        flow_model: FlowMatching wrapping the network
        train_dataset: torch Dataset (already subsampled to the desired n)
        config: full experiment config dict
        run_dir: where to save checkpoints, logs, config
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        flow_model: FlowMatching,
        train_dataset: Dataset,
        config: dict,
        run_dir: str,
        device: str = "cuda",
    ):
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        # ---- Model ----
        self.flow_model = flow_model.to(device)

        # ---- Training config ----
        tc = config["training"]
        self.lr = tc["lr"]
        self.batch_size = tc["batch_size"]
        self.weight_decay = tc.get("weight_decay", 1e-4)
        self.betas = tuple(tc.get("betas", [0.9, 0.95]))
        self.ema_decay = tc.get("ema_decay", 0.9999)
        self.warmup_fraction = tc.get("warmup_fraction", 0.05)

        # Stopping criterion
        self.max_steps = tc.get("max_steps", None)
        self.max_epochs = tc.get("max_epochs", None)
        assert (self.max_steps is None) != (self.max_epochs is None), \
            "Set exactly one of max_steps or max_epochs in config"

        # Checkpoint intervals
        self.ckpt_interval_steps = tc.get("ckpt_interval_steps", None)
        self.ckpt_interval_epochs = tc.get("ckpt_interval_epochs", None)

        # ---- Data loader ----
        # drop_last=True: consistent batch sizes, avoids edge cases
        # pin_memory=True: faster CPU→GPU transfer
        # num_workers: 0 for synthetic (data is in memory), 4+ for images
        is_image = len(train_dataset[0].shape) > 1  # images have C,H,W
        num_workers = 4 if is_image else 0

        self.loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )

        # ---- Optimizer ----
        self.optimizer = torch.optim.AdamW(
            flow_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )

        # ---- EMA ----
        self.ema = EMA(flow_model, decay=self.ema_decay)

        # ---- LR schedule ----
        total_steps = self._compute_total_steps()
        warmup_steps = int(self.warmup_fraction * total_steps)
        self.scheduler = CosineWarmup(self.optimizer, total_steps, warmup_steps)

        # ---- Logging ----
        self.loss_history = []
        self.global_step = 0
        self.config = config

        # Save config
        save_config(config, str(self.run_dir / "config.json"))

    def _compute_total_steps(self) -> int:
        """Compute total training steps from config."""
        if self.max_steps is not None:
            return self.max_steps
        else:
            steps_per_epoch = max(1, len(self.loader))
            return self.max_epochs * steps_per_epoch

    def train(self) -> dict:
        """
        Main training loop.

        Returns:
            dict with training summary: final_loss, total_steps, wall_time
        """
        log_message(f"Starting training: {self._describe_run()}", self.run_dir)
        log_message(f"  Dataset size: {len(self.loader.dataset)}", self.run_dir)
        log_message(f"  Batch size: {self.loader.batch_size}", self.run_dir)
        log_message(f"  Total steps: {self._compute_total_steps()}", self.run_dir)

        start_time = time.time()
        total_steps = self._compute_total_steps()

        if self.max_steps is not None:
            self._train_by_steps(total_steps)
        else:
            self._train_by_epochs()

        wall_time = time.time() - start_time

        # Save final checkpoint
        self._save_checkpoint("final")

        # Save loss history
        loss_path = self.run_dir / "loss_history.json"
        with open(loss_path, "w") as f:
            json.dump(self.loss_history, f)

        summary = {
            "final_loss": self.loss_history[-1]["loss"] if self.loss_history else float("nan"),
            "total_steps": self.global_step,
            "wall_time_seconds": wall_time,
        }
        save_metrics(summary, str(self.run_dir / "train_summary.json"))
        log_message(f"Training complete. {self.global_step} steps in {wall_time:.0f}s", self.run_dir)

        return summary

    def _train_by_steps(self, total_steps: int) -> None:
        """Training loop counted by optimizer steps."""
        epoch = 0
        loader_iter = iter(self.loader)

        while self.global_step < total_steps:
            # Get next batch, cycling through the dataset
            try:
                batch = next(loader_iter)
            except StopIteration:
                epoch += 1
                loader_iter = iter(self.loader)
                batch = next(loader_iter)

                # Epoch-based checkpointing (if configured alongside step-based training)
                if self.ckpt_interval_epochs and epoch % self.ckpt_interval_epochs == 0:
                    self._save_checkpoint(f"epoch_{epoch:06d}")

            loss = self._train_one_step(batch)

            # Step-based checkpointing
            if self.ckpt_interval_steps and self.global_step % self.ckpt_interval_steps == 0:
                self._save_checkpoint(f"step_{self.global_step:08d}")

            # Logging (every 100 steps or every 10% of training)
            log_interval = max(100, total_steps // 10)
            if self.global_step % log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                log_message(
                    f"  step {self.global_step}/{total_steps} | "
                    f"loss={loss:.6f} | lr={lr:.2e}",
                    self.run_dir,
                )

    def _train_by_epochs(self) -> None:
        """Training loop counted by epochs."""
        for epoch in range(1, self.max_epochs + 1):
            epoch_losses = []

            for batch in self.loader:
                loss = self._train_one_step(batch)
                epoch_losses.append(loss)

            avg_loss = sum(epoch_losses) / len(epoch_losses)

            # Epoch-based checkpointing
            if self.ckpt_interval_epochs and epoch % self.ckpt_interval_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch:06d}")

            # Logging (every 10% of epochs or every 50 epochs)
            log_interval = max(50, self.max_epochs // 10)
            if epoch % log_interval == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                log_message(
                    f"  epoch {epoch}/{self.max_epochs} | "
                    f"loss={avg_loss:.6f} | lr={lr:.2e}",
                    self.run_dir,
                )

    def _train_one_step(self, batch: torch.Tensor) -> float:
        """
        One optimizer step. Returns scalar loss value.

        This is where the actual learning happens. Everything else is
        bookkeeping around this function.
        """
        self.flow_model.train()
        self.optimizer.zero_grad()

        # Move batch to device
        if isinstance(batch, (list, tuple)):
            batch = batch[0]  # handle (img, label) tuples from torchvision
        batch = batch.to(self.device)

        # Compute loss (FlowMatching handles the 3×3 grid logic)
        loss = self.flow_model.compute_loss(batch)

        # Backward + optimize
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update()

        # Record
        loss_val = loss.item()
        self.loss_history.append({
            "step": self.global_step,
            "loss": loss_val,
        })
        self.global_step += 1

        return loss_val

    def _save_checkpoint(self, label: str) -> None:
        """
        Save full training state.

        Saves both the current model AND the EMA model, so we can
        evaluate with EMA parameters without re-running training.
        """
        path = self.run_dir / "checkpoints" / f"{label}.pt"
        torch.save({
            "model_state_dict": self.flow_model.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "loss_history_len": len(self.loss_history),
        }, path)

    def _describe_run(self) -> str:
        """Human-readable run description for logging."""
        fm = self.flow_model
        return (
            f"pred={fm.pred_space}, loss={fm.loss_space}, "
            f"lr={self.lr}, bs={self.batch_size}"
        )
