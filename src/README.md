# src

This folder is the canonical home for reusable first-party code and the active
research runners.

## Packages

- `gdfm/`
  Model-specific code: canonical GDFM, model setup, and backbone modules.

- `mp_config/`
  Shared `ConfigDict` defaults, cloning, and override helpers.

- `mp_data/`
  Canonical dataset loaders and deterministic subset logic.

- `mp_metrics/`
  Metric definitions. Keep this package simple: `loss_eval.py`,
  `metrics_synthetic.py`, and `metrics_images.py`.

- `mp_runners/`
  Runnable workflows and CLI entry points. Use
  `python -m mp_runners.synthetic_runner` and
  `python -m mp_runners.image_runner`.

- `mp_artifacts/`
  Experiment/run/seed/attempt layout, config snapshots, checkpoints, tensors,
  JSON artifacts, status files, and attempt logs.

- `mp_training/`
  Shared training-loop scaffolding and optimizer setup.

- `mp_tracking.py`
  Shared observability support for console summaries, `logs/attempt.log`,
  structured events, figure registration, live `status.json` progress, and
  optional offline-first W&B logging.

## Boundary

`src/` should stay readable and direct. Metrics answer what is measured;
runners answer how an experiment is run.

If functionality from the archived trees is needed again, bring the minimum
needed implementation into `src/` deliberately. Do not recreate imports from
archived paths.
