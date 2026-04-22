# Proto/Src Functionality Deep Dive

Generated: 2026-04-21T23:26:28Z

This is a functionality-first map of `src/` and `prototyping/runners/`.
It is meant to set up a later test-quality audit by naming the behavior that
tests should protect. It does not score individual tests yet, except to record
current stale-suite signals and obvious places where assertions look structural
rather than behavioral.

Scope notes:

- Included: canonical source under `src/`, prototype runners under
  `prototyping/runners/`, and tests only as attachment/context.
- Included as active work: the current synthetic `torus3d` path and dirty
  synthetic-runner changes.
- Left untouched: synthetic notebook wording, `dunno/`, and
  `prototyping/archive/`.

## Subsystem Map

| Subsystem | Files | Responsibility | Main entry points / contracts |
| --- | --- | --- | --- |
| Model core | `src/gdfm/core.py` | Canonical Gaussian/flow/diffusion model wrapper, target parameterization, losses, and sampling. | `GDFM`, `predict`, `compute_loss`, `sample`, `z_t_interpolate`, `get_true_target`, `convert`, `n_params`. |
| Model setup and backbones | `src/gdfm/model_setup.py`, `src/gdfm/mlp.py`, `src/gdfm/unet.py`, `src/gdfm/jit.py` | Build the trainable network from canonical or legacy config, then expose MLP, UNet, and JiT implementations behind one factory. | `build_network`, `_normalize_network_config`, `TimeCondMLP`, `BonnaireUNet`, `JiT`. |
| Sampler sidecar | `src/gdfm/samplers.py` | Standalone conversion and integration helpers matching `GDFM` semantics. Useful for sampler-focused tests without instantiating full training. | `time_grid`, `predict_in_prediction_space`, `convert_prediction`, `model_to_x`, `model_to_eps`, `model_to_velocity`, `model_to_score`, `sample_euler`, `sample_heun`, `sample_vp_reverse_sde`. |
| Config | `src/mp_config/base.py` | Canonical nested defaults and merge helpers for study, framework, schedule, network, data, training, eval, sampling, outputs, generation trace, runtime, and tracking. | `get_base_config`, `config_to_dict`, `clone_config`, `apply_overrides`. |
| Data bundles | `src/mp_data/*.py` | Build and validate train/reference bundles for synthetic and image datasets, with metadata needed by artifacts and evaluation. | `build_data_bundle`, `validate_data_bundle`, `DataBundle`, `load_gmm_subspace_bundle`, `load_torus3d_bundle`, image dataset loaders. |
| Training | `src/mp_training/trainer.py` | Canonical run writer: creates attempt layout, persists data/config/model artifacts, trains `GDFM`, saves checkpoints/history, optionally traces generation, and marks successful attempts `DONE_...`. | `train_run`, `_batching_metadata`, `_run_generation_trace`, `_prepare_network_config`. |
| Artifacts/status | `src/mp_artifacts/artifacts.py` | Defines the on-disk attempt layout and all JSON/checkpoint helpers. Handles successful attempt rename and status path rewrites. | `create_run_layout`, `AttemptLayout`, `AttemptPaths`, `AttemptTracker`, `save_checkpoint`, `list_checkpoint_paths`, `mark_attempt_done`, `load_config_snapshot`, `update_attempt_status`. |
| Metrics/evaluation | `src/mp_metrics/*.py` | Generic canonical checkpoint replay and memorization/loss/image metrics. | `evaluate_run`, `evaluate_losses`, `compute_memorization_metrics`, `sample_preview_grid`, image similarity metrics. |
| Runtime | `src/mp_runtime.py` | Device and precision behavior, including TF32/BF16 validation and data-transfer metadata. | `RuntimeConfig`, `configure_runtime`, `autocast_context`, `runtime_metadata`. |
| Tracking | `src/mp_tracking.py`, `prototyping/runners/tracking.py` | Status/events/log/progress/W&B integration. Training and eval update status JSON and may attach figures/metrics to W&B. | `AttemptTracker`, tracker helpers, runner tracking adapters. |
| Visualization | `src/mp_visualization.py`, `prototyping/runners/figures.py` | Shared image grids and synthetic-specific diagnostic plots. | `save_image_grid`, preview helpers, `write_synthetic_figures`, `write_torus_figures`. |
| Synthetic runner | `prototyping/runners/synthetic_runner.py` | CLI/API recipe for controlled synthetic MLP runs using canonical training plus synthetic-specific final/replay evaluation. | `default_settings`, `build_config`, `build_data`, `build_model`, `inspect_metrics`, `run_full_job`, `run_eval_job`, `parse_args`, `main`. |
| Synthetic metrics | `prototyping/runners/metrics_synthetic.py`, `metrics_synthetic_controlled.py` | Controlled GMM/torus metrics, oracle geometry diagnostics, intrinsic aliases, holdout/baseline compatibility, figures, and W&B eval logging. | `evaluate_synthetic_run`, `collect_controlled_synthetic_metric_record`, `memorization_metrics_in_spaces`, torus/GMM geometry helpers. |
| Image runner | `prototyping/runners/image_runner.py`, `image_runner_config_builder.py` | CLI/API recipe for image runs using canonical training plus image-specific final/replay evaluation and dataset/preset handling. | `default_settings`, `build_config`, `build_data`, `build_model`, `inspect_metrics`, `run_full_job`, `run_eval_job`, `compare_numerics`, `parse_args`, `main`. |
| Image metrics | `prototyping/runners/metrics_image32.py` | Runner-level image checkpoint replay, generated-vs-train and heldout-vs-train memory records, sample previews, final sample/neighbor artifacts, and summaries. | `evaluate_image32_run`, `collect_image32_metric_record`, `collect_reference_memory`. |

## Canonical Model Core

`GDFM` is the canonical behavioral center. It wraps a neural network and
normalizes the training/sampling semantics around:

- `family`: currently `fm` or `vp`.
- `prediction`: what the model emits, normally one of `x`, `eps`, `v`, `score`.
- `loss_target`: what loss compares against after conversion.
- `time_sampling`: how training timesteps are drawn.
- `sampling.method`: Euler or Heun for deterministic ODE sampling.

The key convention is `t=0` as source/noise and `t=1` as data/target. The
interpolation has the form `z_t = a(t) * x + b(t) * eps`, and the velocity
target is `v_t = a'(t) * x + b'(t) * eps`. `compute_loss` samples `eps` and
`t`, interpolates to `z_t`, runs the network in prediction space, converts to
the configured loss space, and computes weighted MSE. `sample` starts from
source noise, calls the model through `predict`, converts predictions to
velocity, and integrates by Euler or Heun.

The conversion helpers are high-risk behavior because many downstream metrics
depend on generated samples being semantically comparable across
parameterizations. The tests that matter here should assert algebraic
round-trips and sampler equivalence across prediction/loss spaces, not just
that methods exist.

`build_network` is the canonical factory. It accepts nested config and legacy
aliases, then instantiates:

- `mlp`: `TimeCondMLP`, used by synthetic runners and small vector datasets.
- `unet`: `BonnaireUNet`, used by image presets.
- `jit`: `JiT`, the transformer-style image backbone.

`src/gdfm/samplers.py` mirrors the same conversion/integration semantics as a
sidecar. This is useful for focused tests, but it creates a parity risk:
sampler tests should catch drift between this module and `GDFM.sample`.

## Canonical Run Pipeline

The canonical pipeline is:

1. `get_base_config` creates nested defaults.
2. A runner or caller applies study/data/model/runtime/tracking overrides.
3. `train_run(config, study_dir, device)` creates an attempt layout.
4. `configure_runtime` records precision and device behavior.
5. `build_data_bundle` creates train/reference tensors and metadata.
6. `validate_data_bundle` rejects obvious train/reference leakage.
7. Training saves `config.json`, `train.pt`, `reference.pt`, metadata tensors,
   `geometry.pt` when present, and a fixed `z_init.pt`.
8. `build_network` and `GDFM` define the trainable model.
9. The trainer computes batching metadata, trains, saves step 0 plus periodic
   and final checkpoints, and records history.
10. Optional generation trace samples from the fixed source noise during
    training and writes preview artifacts.
11. A successful attempt is renamed to `DONE_<attempt_id>` by
    `mark_attempt_done`; status, summary, manifest paths, and tracker metadata
    are rewritten to the done path.
12. `evaluate_run(run_dir, device)` or a runner-specific evaluator replays the
    saved config, data, fixed `z_init`, and checkpoints to write metrics and
    figures.

Important artifact layout:

```text
<study_dir>/runs/<run_id>/seeds/<seed_id>/attempts/<attempt_id>/
  artifacts/
    config.json
    train.pt
    reference.pt
    z_init.pt
    geometry.pt                 # synthetic only when geometry exists
    data_meta.json
  checkpoints/
    step_000000.pt
    ...
  metrics/
  figures/
  logs/
  status.json
  run_summary.json
```

`make_run_id` intentionally ignores seed/progress/run-name/study/data-root and
tracking details. That means tests should cover stable grouping by behavioral
config and seed-specific attempt separation.

## Synthetic Functionality

The synthetic path currently supports `gmm_subspace` and active dirty
`torus3d` work.

`gmm_subspace`:

- Builds a linear subspace by QR basis from `geometry_seed`.
- Samples mixture component labels and latent coordinates.
- Projects latent data into ambient dimension and optionally adds residual
  noise `sigma_eta`.
- Uses `training.seed` for train and `training.seed + 1` for reference.
- Saves geometry with basis and mixture means.

`torus3d`:

- Samples a standard 3D torus, then embeds it into ambient dimension by QR
  basis.
- Uses `major_radius`, `minor_radius`, `sigma_eta`, and geometry seed.
- Treats local manifold dimension as 2 but metric features as 4 periodic
  coordinates.
- Saves canonical xyz, angle features, intrinsic coordinates, basis, radii,
  and noise metadata.
- Uses the same train/reference seed split convention as GMM.

`metrics_synthetic_controlled.py` is the synthetic oracle layer. It computes:

- Ambient and intrinsic memorization ratios.
- GMM projected-subspace metrics, latent log probability, component coverage,
  intrinsic Frechet, sliced Wasserstein, residual noise/oracle NLL when
  applicable.
- Torus projection metrics, angle coverage, torus distance, periodic-feature
  intrinsic comparisons, and geometry error.
- `holdout_*` metrics plus compatibility `baseline_*` aliases.

`metrics_synthetic.py` replays a saved controlled-synthetic attempt and writes
checkpoint metrics, long-form trajectories, loss evaluation, previews, final
samples, final intrinsic outputs, GMM compatibility tensors, torus-specific
xyz/angle-feature tensors, figures, and summary/tracker updates.

`figures.py` handles synthetic plots. It has explicit GMM and torus branches,
with compatibility handling for latent/intrinsic column names.

## Image Functionality

The image runner uses canonical training, then a runner-specific replay path for
image metrics.

`image_runner_config_builder.py` owns the dataset and preset surface:

- Dataset keys include `mnist`, `celeba32_grey`, `flowers102`,
  `flowers102_128`, `flowers102_256`, `afhq_v2`, and `ffhq128`.
- Profiles cover 32, 128, and 256 resolution UNet setups.
- Defaults route to `outputs/prototype_runners/image_runner`,
  `celeba32_grey`, `v` prediction/loss, `n_train=128`, `n_eval=128`,
  `n_steps=30000`, `batch_cap=64`, `precision=fp32`, `eval_samples=64`, and
  `sampling_steps=24`.
- W&B defaults are enabled/online. Project names are auto-routed to
  `memparam-{dataset_key}-{model}` unless the caller provides an explicit
  project.
- Preset-owned fields are not clobbered by default-valued raw settings.

`metrics_image32.py` replays checkpoints with fixed `z_init`, computes image
memory metrics against train data, computes a held-out reference baseline,
writes loss evaluation, sample previews, final samples/neighbors, checkpoint
metrics, and summary/status updates. It is named `image32`, but the runner
config also carries resolution/preset metadata for larger image profiles.

`generation_trace` is part of canonical training, not only image eval. It saves
periodic sample previews from fixed source noise. Tests should distinguish
training-time trace behavior from final checkpoint replay behavior.

## Prototype Runner Surfaces

Both `synthetic_runner.py` and `image_runner.py` expose a similar recipe API:

- `default_settings`: human-readable runner defaults.
- `build_config`: convert runner settings into canonical config.
- `build_data`: construct the configured data bundle.
- `build_model`: construct a `GDFM` and network.
- `inspect_metrics`: lightweight metrics surface for existing artifacts.
- `run_full_job`: train, optionally final-evaluate.
- `run_eval_job`: replay evaluation from an existing `DONE_...` attempt.
- `parse_args` and `main`: CLI surface.

Shared runner contracts:

- Train-only is default. Final evaluation requires `--final-eval`.
- Replay evaluation uses `--eval-run-dir`.
- Replay requires a `DONE_...` attempt directory, `status.json` in state
  `trained` or `completed`, required saved artifacts, and at least one
  checkpoint.
- Replay uses saved config/artifacts and must not rebuild fresh data from
  current CLI values.
- Existing W&B run identifiers are reused/resumed during replay.

Synthetic-specific runner behavior:

- Dataset choices are `gmm_subspace` and `torus3d`.
- New dimensions are `ambient_dim` and `intrinsic_data_dim`; legacy
  `data_dim` and `intrinsic_dim` remain compatibility aliases.
- For `torus3d`, ambient dimension must be at least 3, CLI intrinsic-dimension
  overrides are rejected, local manifold dimension is 2, metric feature
  dimension is 4, and legacy intrinsic fields are set to ambient dimension for
  compatibility.

Image-specific runner behavior:

- `compare_numerics` is an in-memory benchmark path. It should not be treated
  as the artifact-producing training/eval path.
- CLI controls include dataset/preset, prediction/loss spaces, train/eval
  counts, checkpoint/log intervals, precision, W&B, data root/download,
  generation trace, final eval, replay eval, and sampling steps.

## End-To-End Flows

### Canonical `train_run` + `evaluate_run`

1. Caller provides canonical config and study directory.
2. `train_run` creates a run/seed/attempt layout and tracker.
3. Runtime precision is configured.
4. Data bundle is built, validated, and persisted.
5. Fixed `z_init.pt` is created and saved for checkpoint replay.
6. Network and `GDFM` are built from config.
7. Training saves checkpoints and metrics/history as it progresses.
8. Successful attempt is renamed to `DONE_...`.
9. `evaluate_run` loads saved config, train/reference tensors, `z_init`, and
   checkpoints from the done attempt.
10. Each checkpoint is sampled from the same `z_init`, compared to train and
    reference data, and recorded in checkpoint/loss/preview artifacts.
11. Final samples/neighbors/figures/summary/status are written and the attempt
    stage becomes `completed`.

### Synthetic Train-Only Run

1. CLI/API settings are parsed by `synthetic_runner`.
2. `build_config` produces canonical config for `gmm_subspace` or `torus3d`.
3. `run_full_job` calls canonical `train_run`.
4. `train_run` writes synthetic data, `geometry.pt`, `z_init`, checkpoints,
   training history, and status.
5. Attempt is renamed to `DONE_...`.
6. No synthetic checkpoint metrics are written unless `--final-eval` is set.

### Synthetic Train + Final Eval

1. Steps 1 to 5 from train-only run occur.
2. Because `--final-eval` is set, `run_full_job` calls
   `evaluate_synthetic_run` on the completed attempt.
3. The evaluator loads saved config, train/reference, geometry, `z_init`, and
   checkpoints.
4. Each checkpoint is sampled from fixed `z_init`.
5. Controlled synthetic metrics are computed for generated data and held-out
   reference data.
6. Checkpoint metrics, long trajectories, loss evaluation, final samples,
   intrinsic/geometry artifacts, figures, run summary, W&B/status updates are
   written.

### Synthetic Replay Eval

1. CLI provides `--eval-run-dir <DONE_...>`.
2. `run_eval_job` validates that the path is a done attempt.
3. Status must be `trained` or `completed`.
4. Required artifacts include saved config, train/reference tensors,
   `geometry.pt`, `z_init.pt`, and checkpoints.
5. `evaluate_synthetic_run` replays from saved state.
6. CLI data/model settings are not used to regenerate the run.

### Image Train-Only Run

1. CLI/API settings are parsed by `image_runner`.
2. `image_runner_config_builder.build_config` merges base defaults, dataset
   defaults, and optional presets.
3. `run_full_job` calls canonical `train_run`.
4. `train_run` writes image data, metadata, fixed `z_init`, checkpoints,
   optional generation trace, history, and status.
5. Attempt is renamed to `DONE_...`.
6. No image checkpoint metrics are written unless `--final-eval` is set.

### Image Train + Final Eval

1. Steps 1 to 5 from image train-only run occur.
2. Because `--final-eval` is set, `run_full_job` calls
   `evaluate_image32_run`.
3. The evaluator loads saved config, train/reference, `z_init`, and
   checkpoints.
4. Each checkpoint is sampled from the fixed `z_init`.
5. Generated memorization and held-out reference memory are computed against
   training data.
6. Loss evaluation, checkpoint metrics, sample previews, final samples,
   nearest-neighbor artifacts, summary, W&B/status updates are written.

### Image Replay Eval

1. CLI provides `--eval-run-dir <DONE_...>`.
2. `run_eval_job` validates done-attempt naming and status.
3. Required saved artifacts and checkpoints must already exist.
4. `evaluate_image32_run` replays using saved config/artifacts.
5. Current CLI dataset/preset/training settings do not create new data or
   training state.

## Behavior Contracts And Invariants

- Target-space conversions: `x`, `eps`, `v`, and score conversions must be
  algebraically consistent for both loss and sampling paths.
- Time convention: `t=0` is source/noise and `t=1` is data/target across
  interpolation, true targets, samplers, and plots.
- Network factory: canonical and legacy config aliases should produce the same
  intended network shape and input/output dimensionality.
- Data bundle: train and reference tensors must both exist, be shape-valid, and
  not be the exact same tensor. Same-source references require disjoint indices
  unless explicitly allowed.
- Synthetic geometry: train/reference samples share geometry but not random
  draws. GMM geometry is basis/means. Torus geometry includes basis, radii,
  canonical xyz, angle features, and intrinsic metadata.
- Batch cap semantics: effective batch size is `min(batch_cap, n_train)`;
  `drop_last` is enabled only when `n_train > batch_cap`; steps-per-epoch and
  epoch count are derived from those values.
- Saved source noise: `z_init.pt` is the fixed source for checkpoint replay.
  Metrics across checkpoints assume differences come from model checkpoints,
  not fresh noise.
- Checkpoints: step 0 plus periodic/final checkpoints define the replay
  timeline. Evaluators should handle multiple checkpoint naming/order cases
  through artifact helpers.
- Done attempts: a successful training attempt is renamed to `DONE_...`.
  Replay refuses non-done attempt dirs. Path rewrites in status/summary/manifest
  are part of the success contract.
- Status stages: replay accepts `trained` or `completed`; final evaluation
  writes completed/eval metadata and should remain resumable.
- Replay isolation: `--eval-run-dir` evaluation must load saved config/data,
  not reinterpret current CLI training settings.
- Metric naming: synthetic metrics intentionally expose both `holdout_*` and
  compatibility `baseline_*` names. GMM keeps latent aliases; torus keeps
  periodic/intrinsic aliases.
- Loss eval: loss metrics are computed separately from memorization metrics and
  should not be used as a proxy for sample memory behavior.
- W&B naming/resume: runners derive project/run metadata from config and
  dataset/model defaults, while replay should reuse saved W&B identity when
  present.
- Runtime precision: BF16 requires CUDA, TF32 toggles CUDA backend flags, and
  data-transfer metadata is recorded for reproducibility.
- Generation trace: training-time previews use fixed settings and should not be
  confused with final/replay sample evaluation.

## Test Attachment Context

Observed likely behavioral tests:

- `tests/test_proto_runners.py` includes fake-data runner smoke tests,
  train-only/final-eval/replay surfaces, W&B/project routing, and synthetic
  torus/GMM runner behavior.
- `tests/test_torus_synthetic.py` and `tests/test_projected_gmm_oracle.py`
  exercise active synthetic geometry and oracle metrics.
- Checkpoint/experiment/loss-eval tests touch canonical artifact and evaluation
  paths.

Observed stale or cosmetic signals:

- Tests still import removed `prototyping/gdfm_components.py`.
- `tests/test_live_tree_boundaries.py` expects an `experiments/` tree that is
  not present in this checkout.
- `tests/test_src_docstrings.py` enforces top-level docstrings and main-block
  examples, which are style/policy checks rather than behavior checks.
- Some prototype-runner tests assert source ordering, text absence/presence, or
  notebook cleanliness. These may be useful guardrails, but they are not strong
  evidence that behavior is protected.

## Risk Register For Later Test Audit

Areas likely to need serious tests:

- `GDFM` conversion algebra and sampler parity across prediction/loss spaces.
- Fixed `z_init` replay, checkpoint ordering, and metric reproducibility.
- `train_run` artifact completeness and `DONE_...` path rewrite behavior.
- Data leakage validation for same-source image/reference datasets.
- Batch-cap/drop-last semantics, especially small `n_train` cases.
- Runner replay contracts: done-path validation, status validation, required
  files, and no accidental data regeneration.
- Synthetic torus projection/intrinsic metrics, where dimension aliases and
  manifold semantics can silently drift.
- GMM oracle metrics with and without residual noise.
- Image preset resolution/profile merging and default-valued setting handling.
- W&B resume behavior in final/replay eval, ideally without making tests depend
  on external network calls.
- Runtime precision validation, especially BF16 CUDA-only behavior.

Areas likely to be over-tested by cosmetic assertions:

- Function/class ordering in source files.
- Docstring presence and main-block policy as a proxy for code quality.
- Notebook output/execution-count cleanliness, unless tied to runnable examples.
- Exact W&B project string assertions that do not also prove routing/resume
  behavior.
- Tests that only assert a file exists without validating contents or behavior.

Stale-test signals already observed:

- Full unittest discovery currently fails because old prototype component tests
  reference `prototyping/gdfm_components.py`.
- Full discovery also fails because a live-tree policy test expects a missing
  `experiments/` directory.
- Style/policy failures currently include one missing top-level docstring and a
  missing `if __name__ == '__main__'` example block.

## Checks And Evidence

Static inspection used:

- `rg --files src prototyping/runners tests`
- `rg -n "^(def|class) " src prototyping/runners tests`
- `rg -n` scans for test functions, skips, mocks, and common assertion shapes.
- Focused reads of `src/gdfm`, `src/mp_config`, `src/mp_data`,
  `src/mp_training`, `src/mp_artifacts`, `src/mp_metrics`,
  `src/mp_runtime`, `src/mp_visualization`, and `prototyping/runners`.
- `wc -l` over source, runner, and test files for size context.

Executed checks:

```bash
module load Mambaforge && mamba activate mem_param_env && MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src:. python -m unittest tests.test_torus_synthetic tests.test_proto_runners tests.test_projected_gmm_oracle
```

Result: 40 focused synthetic/torus/prototype tests passed in 18.945 seconds.

```bash
module load Mambaforge && mamba activate mem_param_env && MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src:. python -m unittest discover tests
```

Result: 104 tests ran in 96.580 seconds with 3 failures and 3 errors.

Errors:

- `tests/test_gdfm_parity_with_prototype.py`,
  `tests/test_prototyping_gdfm.py`, and
  `tests/test_prototyping_gdfm_samplers.py` fail at setup because
  `prototyping/gdfm_components.py` is missing.

Failures:

- `tests/test_live_tree_boundaries.py` expects `experiments/` to exist.
- `tests/test_src_docstrings.py` reports
  `src/mp_config/base.py:_default_wandb_entity` missing a top-level docstring.
- `tests/test_src_docstrings.py` reports `src/mp_data/image_collections.py`
  missing a main-block example.

Existing warning noise observed during tests:

- PyTorch `torch.load(weights_only=False)` FutureWarnings.
- Torch Dynamo/Inductor warnings around missing C++ compiler for JiT-related
  paths.
- NVML unavailable warning.

## Handoff For The Later Test Audit

Recommended classification for the later audit:

- `behavioral`: asserts numerical, artifact, replay, leakage, or contract
  behavior that would catch a real regression.
- `smoke`: proves a real path executes but does not deeply validate outputs.
- `contract`: protects public CLI/API shape, filenames, config keys, or status
  schema that downstream users rely on.
- `cosmetic`: checks source text, ordering, docstrings, notebook cleanliness, or
  style without exercising behavior.
- `stale/dead`: targets removed modules, missing repository layout, or legacy
  workflows no longer represented by live code.

The next pass should map each test file to the subsystems above, then label the
tests by that classification and note missing coverage against the behavior
contracts.
