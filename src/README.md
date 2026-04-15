# src

This folder is the canonical home for reusable first-party code.

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
  Run layout, config snapshots, checkpoints, tensors, and JSON artifacts.

- `mp_training/`
  Shared training-loop scaffolding and optimizer setup.

## Boundary

`src/` should stay reusable and study-agnostic.

Study-specific scientific choices, sweeps, and figure narratives belong under
`experiments/`, not here.
