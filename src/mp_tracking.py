"""
Shared observability helpers for first-party experiment attempts.

This module centralizes the pieces that should feel the same across studies:
- human-readable console and attempt-log output
- structured event records for later debugging
- live status snapshots with progress, latest metrics, and figure paths
- optional Weights & Biases tracking with offline-first defaults

The goal is not to hide training or evaluation logic. The goal is to keep the
operator-facing plumbing consistent so `train_run(...)`, `evaluate_run(...)`,
and study entrypoints can focus on the scientific workflow itself.
"""

from __future__ import annotations

import importlib
import math
import os
import re
import sys
import time
from pathlib import Path

from mp_artifacts import (
    append_attempt_event,
    append_attempt_log,
    load_attempt_status,
    patch_attempt_status,
)
from mp_config import config_to_dict


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


def should_show_progress(config):
    """
    Decide whether interactive progress bars should be shown for this process.
    """
    if not bool(getattr(config.training, "show_progress", True)):
        return False
    return _running_in_notebook() or sys.stderr.isatty()


def format_duration(seconds):
    """
    Format a duration in seconds into a compact `HH:MM:SS` string.
    """
    if seconds is None or not math.isfinite(float(seconds)):
        return "?"
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _wandb_slug(value, fallback):
    """
    Build a compact W&B-safe name fragment while preserving useful underscores.
    """
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-")
    return text or fallback


def _default_wandb_project(config):
    dataset_key = getattr(getattr(config, "study", None), "dataset_key", "")
    dataset = _wandb_slug(dataset_key or getattr(config.data, "dataset", ""), "dataset")
    model = _wandb_slug(getattr(config.network, "arch", ""), "model")
    return os.getenv("WANDB_PROJECT", "").strip() or f"memparam-{dataset}-{model}"


def _default_wandb_entity(fallback):
    return os.getenv("WANDB_ENTITY", "").strip() or fallback


def _default_wandb_run_name(config):
    dataset_key = getattr(getattr(config, "study", None), "dataset_key", "")
    dataset = _wandb_slug(dataset_key or getattr(config.data, "dataset", ""), "dataset")
    model = _wandb_slug(getattr(config.network, "arch", ""), "model")
    n_train = int(getattr(config.data, "n_train", 0))
    prediction = _wandb_slug(getattr(config.framework, "prediction_space", ""), "pred")
    loss = _wandb_slug(getattr(config.framework, "loss_space", ""), "loss")
    seed = int(getattr(config.training, "seed", 0))
    return f"{dataset}-{model}-N{n_train}-{prediction}_{loss}-s{seed}"


def _safe_float(value):
    """
    Convert a numeric-like value to `float` while leaving missing values alone.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_rate(value):
    """
    Format an optional rate for readable logs.
    """
    if value is None or not math.isfinite(float(value)):
        return "?"
    return f"{float(value):.2f}"


def _wandb_figure_caption(metric_name, step):
    """
    Label W&B media with the scientific checkpoint step, not W&B's row index.
    """
    metric_name = str(metric_name)
    if metric_name.startswith("generation_trace/"):
        label = "generation_trace"
    elif metric_name.endswith("/checkpoint_samples_over_train"):
        label = "checkpoint_samples_over_train"
    else:
        label = metric_name
    return f"{label} | grad_step={int(step)}"


class AttemptTracker:
    """
    Coordinate logs, live status snapshots, structured events, and optional W&B.
    """

    def __init__(self, config, attempt_dir, *, enable_console=True):
        """
        Build one tracker bound to a single attempt directory.
        """
        self.config = config
        self.attempt_dir = Path(attempt_dir)
        self.status = load_attempt_status(self.attempt_dir)

        tracking = getattr(config, "tracking", None)
        console = getattr(tracking, "console", None)
        wandb_config = getattr(tracking, "wandb", None)

        default_summary_steps = int(getattr(config.training, "log_every", 1))
        self.console_enabled = bool(getattr(console, "enabled", True)) and enable_console
        self.summary_every_steps = max(
            1,
            int(getattr(console, "summary_every_steps", default_summary_steps)),
        )
        self.summary_every_seconds = max(
            0.0,
            float(getattr(console, "summary_every_seconds", 60.0)),
        )
        self.stage_banners = bool(getattr(console, "stage_banners", True))
        self.tee_to_attempt_log = bool(getattr(console, "tee_to_attempt_log", True))
        self.interactive_progress = should_show_progress(config)

        self._last_training_summary_step = 0
        self._last_training_summary_time = time.perf_counter()
        self._last_eval_summary_index = 0
        self._last_eval_summary_time = time.perf_counter()

        self._wandb_config = wandb_config
        self._wandb_module = None
        self._wandb_run = None
        self._wandb_failed = False

        self._init_wandb()

    def _status_wandb_payload(self, **fields):
        """
        Build the persisted W&B metadata snapshot stored in `status.json`.
        """
        payload = dict(self.status.get("wandb") or {})
        payload.setdefault("enabled", False)
        payload.setdefault("active", False)
        payload.setdefault("mode", None)
        payload.setdefault("project", None)
        payload.setdefault("entity", None)
        payload.setdefault("run_id", None)
        payload.setdefault("run_name", None)
        payload.setdefault("run_path", None)
        payload.setdefault("run_url", None)
        payload.setdefault("dir", None)
        payload.setdefault("warning", None)
        payload.update(fields)
        return payload

    def _wandb_tags(self):
        """
        Build stable W&B tags from the attempt identity and config surface.
        """
        tags = list(getattr(self._wandb_config, "tags", [])) if self._wandb_config else []
        tags.extend(
            [
                f"dataset:{self.config.data.dataset}",
                f"arch:{self.config.network.arch}",
                f"n_train:{self.config.data.n_train}",
                f"prediction:{self.config.framework.prediction_space}",
                f"loss:{self.config.framework.loss_space}",
                f"seed:{self.status['seed_id']}",
                f"lr:{self.config.training.lr}",
                f"batch_cap:{self.config.training.batch_cap}",
                f"steps:{self.config.training.n_steps}",
                f"ckpt_every:{self.config.training.ckpt_every}",
            ]
        )
        deduped = []
        seen = set()
        for tag in tags:
            tag = str(tag)
            if tag not in seen:
                seen.add(tag)
                deduped.append(tag)
        return deduped

    def _normalize_run_path(self, run):
        """
        Convert a W&B run path payload into a stable string when possible.
        """
        path = getattr(run, "path", None)
        if isinstance(path, (list, tuple)):
            return "/".join(str(part) for part in path)
        if path is None:
            return None
        return str(path)

    def _patch_status(self, **fields):
        """
        Persist live status fields and keep the cached copy in sync.
        """
        self.status = patch_attempt_status(self.attempt_dir, **fields)
        return self.status

    def _record_event(self, event_type, **fields):
        """
        Append one structured event to the attempt event stream.
        """
        payload = dict(fields)
        payload.setdefault("attempt_id", self.status["attempt_id"])
        payload.setdefault("run_id", self.status["run_id"])
        payload.setdefault("seed_id", self.status["seed_id"])
        return append_attempt_event(
            self.attempt_dir,
            event_type,
            **payload,
        )

    def _write_log_line(self, line):
        """
        Emit one human-readable line to stdout/stderr and the attempt log.
        """
        if self.console_enabled:
            print(line, flush=True)
        if self.tee_to_attempt_log:
            append_attempt_log(self.attempt_dir, line)

    def _emit(self, line, *, event_type=None, event_fields=None):
        """
        Emit one readable line and optionally mirror it into structured events.
        """
        self._write_log_line(line)
        if event_type is not None:
            fields = {} if event_fields is None else dict(event_fields)
            self._record_event(event_type, message=line, **fields)

    def _handle_wandb_failure(self, exc, *, context):
        """
        Downgrade W&B failures to local-only logging unless strict mode is on.
        """
        self._wandb_failed = True
        warning = f"W&B {context} failed; continuing with local logging: {type(exc).__name__}: {exc}"
        self._wandb_run = None
        self._wandb_module = None
        metadata = self._status_wandb_payload(active=False, warning=warning)
        self._patch_status(wandb=metadata)
        self.warn(warning)
        if bool(getattr(self._wandb_config, "strict", False)):
            raise

    def _define_wandb_metric_axes(self):
        """
        Tell W&B which explicit fields should be used as x-axes.
        """
        if self._wandb_module is None or not hasattr(self._wandb_module, "define_metric"):
            return
        try:
            self._wandb_module.define_metric("train/step")
            self._wandb_module.define_metric("train/*", step_metric="train/step")
            self._wandb_module.define_metric("eval/step")
            self._wandb_module.define_metric("eval/*", step_metric="eval/step")
            self._wandb_module.define_metric("generation_trace/step")
            self._wandb_module.define_metric(
                "generation_trace/*",
                step_metric="generation_trace/step",
            )
        except Exception as exc:
            self.warn(
                "W&B metric-axis definition failed; continuing with default axes: "
                f"{type(exc).__name__}: {exc}"
            )

    def _attach_explicit_step_metric(self, metrics, step):
        """
        Convert an optional caller step into a namespace-specific metric field.
        """
        if step is None:
            return metrics
        if any(key.startswith("generation_trace/") for key in metrics):
            metrics.setdefault("generation_trace/step", int(step))
        elif any(key.startswith("eval/") for key in metrics):
            metrics.setdefault("eval/step", int(step))
        elif any(key.startswith("train/") for key in metrics):
            metrics.setdefault("train/step", int(step))
        return metrics

    def _init_wandb(self):
        """
        Initialize one optional W&B run for this attempt.
        """
        if not bool(getattr(self._wandb_config, "enabled", False)):
            self._patch_status(wandb=self._status_wandb_payload(enabled=False, active=False))
            return

        mode = str(getattr(self._wandb_config, "mode", "offline"))
        configured_project = str(getattr(self._wandb_config, "project", "")).strip()
        project = configured_project or _default_wandb_project(self.config)
        entity = str(
            getattr(self._wandb_config, "entity", _default_wandb_entity("maj90"))
        ).strip() or None
        notes = str(getattr(self._wandb_config, "notes", "")).strip() or None
        configured_name = str(getattr(self._wandb_config, "name", "")).strip()
        configured_group = str(getattr(self._wandb_config, "group", "")).strip()
        run_name = configured_name or _default_wandb_run_name(self.config)
        run_group = configured_group or self.status["run_id"]
        wandb_dir = self.attempt_dir / str(getattr(self._wandb_config, "dir", "logs/wandb"))
        wandb_dir.mkdir(parents=True, exist_ok=True)

        try:
            wandb_module = importlib.import_module("wandb")
            run_id = (self.status.get("wandb") or {}).get("run_id") or self.status["attempt_id"]
            run = wandb_module.init(
                project=project,
                entity=entity,
                dir=str(wandb_dir),
                mode=mode,
                id=run_id,
                resume="allow",
                name=run_name,
                group=run_group,
                job_type=self.config.study.name,
                notes=notes,
                tags=self._wandb_tags(),
                config=config_to_dict(self.config),
                reinit="finish_previous",
            )
            self._wandb_module = wandb_module
            self._wandb_run = run
            self._define_wandb_metric_axes()
            metadata = self._status_wandb_payload(
                enabled=True,
                active=True,
                mode=mode,
                project=project,
                entity=entity,
                run_id=getattr(run, "id", run_id),
                run_name=getattr(run, "name", run_name),
                run_path=self._normalize_run_path(run),
                run_url=getattr(run, "url", None),
                dir=str(wandb_dir.resolve()),
                warning=None,
            )
            self._patch_status(wandb=metadata)
            wandb_target = (
                metadata.get("run_url")
                or metadata.get("run_path")
                or metadata["run_id"]
            )
            self._emit(
                (
                    "wandb active "
                    f"mode={mode} "
                    f"project={project} "
                    f"name={metadata['run_name']} "
                    f"target={wandb_target}"
                ),
                event_type="wandb_initialized",
                event_fields={
                    "mode": mode,
                    "project": project,
                    "entity": entity,
                    "wandb_run_id": metadata["run_id"],
                    "wandb_run_name": metadata["run_name"],
                    "wandb_target": wandb_target,
                },
            )
        except Exception as exc:
            metadata = self._status_wandb_payload(
                enabled=True,
                active=False,
                mode=mode,
                project=project,
                entity=entity,
                run_id=self.status["attempt_id"],
                run_name=self.status["attempt_id"],
                dir=str(wandb_dir.resolve()),
            )
            self._patch_status(wandb=metadata)
            self._handle_wandb_failure(exc, context="initialization")

    def stage(self, state, *, message=None, **status_fields):
        """
        Record a lifecycle state transition and emit an optional stage banner.
        """
        from mp_artifacts import update_attempt_status

        self.status = update_attempt_status(self.attempt_dir, state, **status_fields)
        if self.stage_banners:
            self._emit(
                f"=== {state.upper()} ===",
                event_type="stage_banner",
                event_fields={"state": state},
            )
        if message:
            self._emit(
                message,
                event_type="stage_message",
                event_fields={"state": state},
            )

    def note(self, message, *, event_type="note", **event_fields):
        """
        Emit one general informational line.
        """
        self._emit(message, event_type=event_type, event_fields=event_fields)

    def warn(self, message):
        """
        Persist and emit one warning message for the attempt.
        """
        warnings = list(self.status.get("warnings") or [])
        if message not in warnings:
            warnings.append(message)
        self._patch_status(warnings=warnings)
        self._emit(f"warning: {message}", event_type="warning")

    def _should_emit_periodic_summary(self, step, total, *, last_step, last_time, force):
        """
        Decide whether a periodic summary line should be emitted now.
        """
        if force or step <= 1 or step >= total:
            return True
        now = time.perf_counter()
        if step - last_step >= self.summary_every_steps:
            return True
        if now - last_time >= self.summary_every_seconds:
            return True
        return False

    def training_progress(self, *, step, total_steps, loss, elapsed_sec, force=False):
        """
        Update training progress snapshots and emit periodic summaries.
        """
        loss_value = _safe_float(loss)
        steps_per_sec = (float(step) / elapsed_sec) if elapsed_sec > 0 else None
        eta_sec = (
            (float(total_steps - step) / steps_per_sec)
            if steps_per_sec not in (None, 0.0)
            else None
        )
        progress = {
            "phase": "training",
            "current_step": int(step),
            "total_steps": int(total_steps),
            "percent": (100.0 * float(step) / float(total_steps)) if total_steps else 0.0,
            "elapsed_sec": float(elapsed_sec),
            "eta_sec": eta_sec,
            "steps_per_sec": steps_per_sec,
        }
        latest_metrics = dict(self.status.get("latest_metrics") or {})
        latest_metrics.update({"train_loss": loss_value})
        self._patch_status(progress=progress, latest_metrics=latest_metrics)

        should_emit = self._should_emit_periodic_summary(
            int(step),
            int(total_steps),
            last_step=self._last_training_summary_step,
            last_time=self._last_training_summary_time,
            force=force,
        )
        if should_emit:
            self._last_training_summary_step = int(step)
            self._last_training_summary_time = time.perf_counter()
            line = (
                "train "
                f"step={int(step)}/{int(total_steps)} "
                f"({progress['percent']:.1f}%) "
                f"loss={loss_value:.4f} "
                f"steps/s={_format_rate(steps_per_sec)} "
                f"elapsed={format_duration(elapsed_sec)} "
                f"eta={format_duration(eta_sec)}"
            )
            self._emit(
                line,
                event_type="training_summary",
                event_fields={
                    "step": int(step),
                    "total_steps": int(total_steps),
                    "loss": loss_value,
                    "steps_per_sec": steps_per_sec,
                    "elapsed_sec": float(elapsed_sec),
                    "eta_sec": eta_sec,
                },
            )
            self.log_metrics({"train/loss": loss_value, "train/steps_per_sec": steps_per_sec}, step=int(step))

    def checkpoint_saved(self, *, step, checkpoint_path):
        """
        Record one saved checkpoint in status and logs.
        """
        info = {
            "step": int(step),
            "path": str(Path(checkpoint_path).resolve()),
        }
        self._patch_status(last_checkpoint=info)
        self._emit(
            f"saved checkpoint step={int(step)} -> {info['path']}",
            event_type="checkpoint_saved",
            event_fields=info,
        )

    def evaluation_progress(self, *, checkpoint_index, total_checkpoints, record, elapsed_sec, force=False):
        """
        Update checkpoint-evaluation progress snapshots and emit summaries.
        """
        steps_per_checkpoint = (
            float(checkpoint_index) / elapsed_sec if elapsed_sec > 0 else None
        )
        eta_sec = (
            float(total_checkpoints - checkpoint_index) / steps_per_checkpoint
            if steps_per_checkpoint not in (None, 0.0)
            else None
        )
        progress = {
            "phase": "evaluation",
            "current_checkpoint": int(checkpoint_index),
            "total_checkpoints": int(total_checkpoints),
            "current_step": int(record["step"]),
            "elapsed_sec": float(elapsed_sec),
            "eta_sec": eta_sec,
            "checkpoints_per_sec": steps_per_checkpoint,
        }
        latest_metrics = dict(self.status.get("latest_metrics") or {})
        latest_metrics.update(
            {
                "f_mem": _safe_float(record.get("f_mem")),
                "mean_nn_ratio": _safe_float(record.get("mean_nn_ratio")),
                "std_nn_ratio": _safe_float(record.get("std_nn_ratio")),
                "baseline_f_mem": _safe_float(record.get("baseline_f_mem")),
                "baseline_mean_nn_ratio": _safe_float(record.get("baseline_mean_nn_ratio")),
                "baseline_std_nn_ratio": _safe_float(record.get("baseline_std_nn_ratio")),
            }
        )
        self._patch_status(evaluation_progress=progress, latest_metrics=latest_metrics)

        should_emit = self._should_emit_periodic_summary(
            int(checkpoint_index),
            int(total_checkpoints),
            last_step=self._last_eval_summary_index,
            last_time=self._last_eval_summary_time,
            force=force,
        )
        if should_emit:
            self._last_eval_summary_index = int(checkpoint_index)
            self._last_eval_summary_time = time.perf_counter()
            line = (
                "eval "
                f"checkpoint={int(checkpoint_index)}/{int(total_checkpoints)} "
                f"step={int(record['step'])} "
                f"f_mem={float(record['f_mem']):.4f} "
                f"mean_nn_ratio={float(record['mean_nn_ratio']):.4f} "
                f"elapsed={format_duration(elapsed_sec)} "
                f"eta={format_duration(eta_sec)}"
            )
            self._emit(
                line,
                event_type="evaluation_summary",
                event_fields={
                    "checkpoint_index": int(checkpoint_index),
                    "total_checkpoints": int(total_checkpoints),
                    "step": int(record["step"]),
                    "f_mem": _safe_float(record.get("f_mem")),
                    "mean_nn_ratio": _safe_float(record.get("mean_nn_ratio")),
                    "std_nn_ratio": _safe_float(record.get("std_nn_ratio")),
                    "baseline_f_mem": _safe_float(record.get("baseline_f_mem")),
                    "baseline_mean_nn_ratio": _safe_float(record.get("baseline_mean_nn_ratio")),
                    "baseline_std_nn_ratio": _safe_float(record.get("baseline_std_nn_ratio")),
                    "elapsed_sec": float(elapsed_sec),
                    "eta_sec": eta_sec,
                },
            )
            self.log_metrics(
                {
                    "eval/f_mem": _safe_float(record.get("f_mem")),
                    "eval/mean_nn_ratio": _safe_float(record.get("mean_nn_ratio")),
                    "eval/std_nn_ratio": _safe_float(record.get("std_nn_ratio")),
                    "eval/baseline_f_mem": _safe_float(record.get("baseline_f_mem")),
                    "eval/baseline_mean_nn_ratio": _safe_float(record.get("baseline_mean_nn_ratio")),
                    "eval/baseline_std_nn_ratio": _safe_float(record.get("baseline_std_nn_ratio")),
                },
                step=int(record["step"]),
            )

    def register_figure(self, name, path, *, log_to_wandb=True):
        """
        Register one saved figure in attempt status and optionally log it to W&B.
        """
        figure_path = str(Path(path).resolve())
        figures = dict(self.status.get("generated_figures") or {})
        figures[str(name)] = figure_path
        self._patch_status(generated_figures=figures)
        self._emit(
            f"saved figure {name} -> {figure_path}",
            event_type="figure_saved",
            event_fields={"name": str(name), "path": figure_path},
        )
        if log_to_wandb and self._wandb_run is not None and self._wandb_module is not None:
            try:
                self._wandb_run.log({f"figures/{name}": self._wandb_module.Image(figure_path)})
            except Exception as exc:
                self._handle_wandb_failure(exc, context=f"figure logging ({name})")

    def log_figure_metric(self, metric_name, path, *, step=None):
        """
        Log one saved figure to W&B under a metric namespace with explicit axes.
        """
        if self._wandb_run is None or self._wandb_module is None or self._wandb_failed:
            return
        try:
            image_kwargs = {}
            if step is not None:
                image_kwargs["caption"] = _wandb_figure_caption(metric_name, step)
            payload = {
                str(metric_name): self._wandb_module.Image(
                    str(Path(path).resolve()),
                    **image_kwargs,
                )
            }
            payload = self._attach_explicit_step_metric(payload, step)
            self._wandb_run.log(payload)
        except Exception as exc:
            self._handle_wandb_failure(
                exc,
                context=f"figure metric logging ({metric_name})",
            )

    def log_metrics(self, metrics, *, step=None):
        """
        Send scalar metrics to W&B when it is active.
        """
        if self._wandb_run is None or self._wandb_module is None or self._wandb_failed:
            return
        clean_metrics = {key: value for key, value in metrics.items() if value is not None}
        if not clean_metrics:
            return
        clean_metrics = self._attach_explicit_step_metric(clean_metrics, step)
        try:
            self._wandb_run.log(clean_metrics)
        except Exception as exc:
            self._handle_wandb_failure(exc, context="metric logging")

    def close(self):
        """
        Finish any live W&B run for this attempt.
        """
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.finish()
        except Exception as exc:
            self._handle_wandb_failure(exc, context="finish")
        finally:
            metadata = self._status_wandb_payload(active=False)
            self._patch_status(wandb=metadata)
            self._wandb_run = None
            self._wandb_module = None


if __name__ == "__main__":
    print("mp_tracking example")
    print("Usage:")
    print("  tracker = AttemptTracker(config, attempt_dir)")
    print("  tracker.stage('training')")
    print("  tracker.training_progress(step=10, total_steps=100, loss=0.1234, elapsed_sec=5.0)")
