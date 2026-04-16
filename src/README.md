# src

This folder is the canonical home for reusable first-party code.

Live paper-study workflows now live in `experiments/`. `src/` stays focused on
the reusable kernel those studies build on top of.

## Packages

- `gdfm/`
  Model-specific code: canonical GDFM, model setup, and backbone modules.

- `mp_config/`
  Shared `ConfigDict` defaults, cloning, and override helpers.

- `mp_data/`
  Canonical dataset loaders and deterministic subset logic.

- `mp_metrics/`
  Memorization metrics and checkpoint replay evaluation.
  The canonical `f_mem` threshold lives at `config.evaluation.ratio_threshold`
  and is respected by direct tensor metrics, folder-based metrics, and
  checkpoint replay.

- `mp_artifacts/`
  Experiment/run/seed/attempt layout, config snapshots, checkpoints, tensors,
  JSON artifacts, status files, and attempt logs.

- `mp_training/`
  Shared training-loop scaffolding and optimizer setup.

- `mp_tracking.py`
  Shared observability glue for console summaries, `logs/attempt.log`,
  structured events, figure registration, live `status.json` progress, and
  optional offline-first W&B logging.

## Boundary

`src/` should stay reusable and study-agnostic.

Runnable study workflows belong in `experiments/`, not in `src/`.

If functionality from the archived trees is needed again, bring the minimum
needed implementation into `src/` deliberately. Do not recreate imports from
archived paths.
