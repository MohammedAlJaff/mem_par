# Coding Journal

This journal is the shared session log for the repo. Use UTC ISO timestamps,
keep entries practical, and record enough context to resume work without
rereading the whole codebase.

## 2026-04-18T05:56:33Z - Starter State And Objectives

### Current State
- Repo path: `/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par`
- Branch state: `main`, ahead of `origin/main` by 1 commit.
- Latest commit: `742d759 Rename synthetic prototype runner and extract metrics`.
- Active IDE file: `ram_test.py`.
- Dirty files to preserve:
  - modified `experiments/synthetic/gmm_data_sanity.ipynb`
  - untracked `.codex`
  - untracked `ram_test.py`
  - untracked `prototyping/projected_gmm_metric_validation.ipynb`

### Research Objective
- Under fixed architecture, data, and training protocol, study how `x`, `eps`,
  and `v` target parameterizations change memorization dynamics.
- Key measurements: `f_mem`, nearest-neighbor ratio and distance statistics,
  and checkpoint-level memorization trajectories.

### Immediate Working Objective
- Track where we are across sessions.
- Record decisions, commands, results, blockers, and next actions.
- Make it easier to resume without losing context around synthetic GMM
  validation, notebook state, RAM/GPU memory probing, and
  canonical-vs-prototype migration work.

### Next Action
- Add this journal file.
- Add a short `AGENT.md` workflow note requiring future meaningful work to
  append or update a timestamped journal entry.
- Leave unrelated dirty notebook/test files untouched.

## Entry Template

```markdown
## YYYY-MM-DDTHH:MM:SSZ - Short Title

### Context
What part of the repo/workstream this entry belongs to.

### Intent
What we are trying to decide, implement, test, or understand.

### Changed / Observed
Files touched, facts learned, or behavior noticed.

### Commands / Checks
Commands run and what they were meant to verify.

### Results
Outcome, metrics, failures, or confidence level.

### Blockers
Anything unresolved or risky.

### Next
The next concrete action.
```

## 2026-04-18T07:00:16Z - Prototype Runner Lab

### Context
Built a structured prototype-runner lab for first-pass paper-relevant
arch/data combinations while keeping canonical training and checkpoint replay
inside `src/`.

### Intent
Make it easy to inspect and iterate on runner components for GMM+MLP,
MNIST+UNet, CelebA32+UNet, and CelebA32+JiT, with one interactive notebook per
combo.

### Changed / Observed
- Added `prototyping/runners/` with shared helpers, a first-pass runtime map,
  README, and four combo runner modules.
- Added four unexecuted component-playground notebooks under
  `prototyping/runners/notebooks/`.
- Replaced `prototyping/runner_proto_synthetic_projected_gmm.py` with a
  compatibility launcher for `prototyping.runners.gmm_subspace_mlp`.
- Added `tests/test_proto_runners.py` for config matrix, parameterization,
  fake-dataset component checks, GMM oracle diagnostics, and notebook structure.

### Commands / Checks
- `python -m py_compile prototyping/runners/*.py prototyping/runner_proto_synthetic_projected_gmm.py tests/test_proto_runners.py`
- `python -m json.tool` on each new runner notebook.
- Checked notebook execution counts and outputs with a small Python inspection.
- Attempted targeted pytest execution for the new and adjacent runner tests.

### Results
- Python syntax checks passed.
- All four notebooks are valid JSON, have no outputs, and have null execution
  counts.
- Runtime imports and pytest could not run in this shell because `/usr/bin/python`
  is missing project dependencies including `torch` and `pytest`.

### Blockers
- Need the project environment active before running runtime component checks
  and the targeted pytest suite.

### Next
Activate the project environment, run the targeted tests, then decide whether
to commit the runner lab separately from existing unrelated dirty files.

## 2026-04-18T07:37:01Z - Flatten Prototype Runner Recipes

### Context
Refactored the prototype-runner lab after the runner files felt too buried
under shared helpers and indirection.

### Intent
Make each arch+data combo readable top to bottom as a recipe: knobs, config,
data, model, one optimizer update, metrics, optional canonical run, and
artifact inspection.

### Changed / Observed
- Shrunk `prototyping/runners/common.py` to path setup, JSON printing, command
  rendering, parameter counts, and artifact path helpers.
- Removed the runtime map file and made the README table the human map.
- Rewrote `gmm_subspace_mlp.py`, `mnist_unet.py`, `celeba32_unet.py`, and
  `celeba32_jit.py` around explicit recipe functions.
- Rewrote the four notebooks so each mirrors the recipe steps with unexecuted
  cells.
- Updated `tests/test_proto_runners.py` to exercise the recipe-style API.

### Commands / Checks
- `python -m py_compile prototyping/runners/*.py prototyping/runner_proto_synthetic_projected_gmm.py tests/test_proto_runners.py`
- Loaded every runner notebook as JSON and checked code cells for null
  execution counts and empty outputs.
- Searched runner/test files for the old hidden-helper/runtime-map entry
  points.
- Attempted the targeted pytest suite for the runner and adjacent tests.

### Results
- Syntax checks passed.
- All four notebooks load as JSON and remain unexecuted with no stored outputs.
- The old hidden-helper names are gone from runner/test files.
- Targeted pytest could not run in this shell because `/usr/bin/python` does
  not have `pytest` installed.

### Blockers
- Need the project environment active before runtime tests can verify data
  loading, model construction, one-update paths, and metrics end to end.

### Next
Activate the project environment and run the targeted pytest suite, then review
the recipe files from the notebook user's point of view for any remaining
buried details.

## 2026-04-18T08:36:39Z - Comment Trainer Flow

### Context
Focused on `src/mp_training/trainer.py`, which remains the canonical training
orchestration function used by the prototype runners.

### Intent
Make `train_run(...)` easier to understand before deciding whether any runner
should replace or fork the training path.

### Changed / Observed
- Expanded the module docstring to explain the helper layer, `train_run(...)`,
  and the tiny runnable example.
- Added phase comments through `train_run(...)` for layout creation, tracking,
  data loading, replay artifacts, model setup, training, checkpointing,
  summaries, and failure handling.
- Added explanatory helper comments for optimizer selection, mini-batch
  sampling, shared `z_init`, and loss-figure saving.
- Replaced an old wording in the example comment with neutral wording.

### Commands / Checks
- `python -m py_compile src/mp_training/trainer.py`
- Searched `src/mp_training/trainer.py` for the banned wording from prior
  preference.

### Results
- Syntax check passed.
- The banned wording no longer appears in `src/mp_training/trainer.py`.
- No training behavior was intentionally changed.

### Blockers
- Runtime training tests still need the project environment with dependencies
  active.

### Next
Read `trainer.py` once from top to bottom and decide whether the comments now
make the canonical path transparent enough, or whether a prototype-local
training recipe is still needed.

## 2026-04-18T08:48:19Z - Rename Synthetic Runner Surface

### Context
The first prototype runner was named after projected GMM, but the synthetic
runner should eventually hold other controlled datasets such as two-moons.

### Intent
Rename the runner script and notebook from the GMM-specific surface to a
broader synthetic MLP surface without changing current behavior.

### Changed / Observed
- Renamed `prototyping/runners/gmm_subspace_mlp.py` to
  `prototyping/runners/synthetic_mlp.py`.
- Renamed `prototyping/runners/notebooks/gmm_subspace_mlp.ipynb` to
  `prototyping/runners/notebooks/synthetic_mlp.ipynb`.
- Updated the runner name, module path, default output directory, README
  matrix/commands, compatibility launcher, and prototype-runner tests.
- Kept the current default dataset as `gmm_subspace`; GMM-specific oracle
  metric helpers remain in the synthetic runner for now.
- Preserved existing notebook outputs while updating source cells to import the
  new module.

### Commands / Checks
- Moved the runner Python file and notebook file.
- Checked notebook source cells for the old module import and the new module
  import.
- Searched runner/test/docs files for stale references to the old module name.

### Results
- The source-facing runner path is now `prototyping.runners.synthetic_mlp`.
- The renamed notebook source imports `synthetic_mlp`.
- Existing historical journal entries still mention the old name as history.

### Blockers
- `synthetic_mlp.ipynb` currently contains stored outputs from an interactive
  run, so notebook-cleanliness checks will fail until outputs are cleared.

### Next
Decide whether to clear the renamed notebook outputs now or keep them as the
local interactive record while the run artifacts remain on disk.

## 2026-04-18T08:52:35Z - Make Wandb Runs Visible

### Context
The prototype runner already exposes W&B knobs, and `AttemptTracker` is the
central place that initializes W&B and logs training/evaluation metrics.

### Intent
Make it easier to launch a short prototype run and immediately see which W&B
run was created.

### Changed / Observed
- Updated `src/mp_tracking.py` so W&B initialization honors optional configured
  run `name` and `group` values.
- Added a console/event line after W&B initialization that prints mode,
  project, run name, and URL/path/id target.
- Current Codex shell is a bare system Python without project dependencies, so
  the actual online W&B run should be launched from the active project/Jupyter
  environment.

### Commands / Checks
- Inspected `src/mp_tracking.py` and `prototyping/runners/synthetic_mlp.py`.
- Checked UTC timestamp for this entry.

### Results
- The next W&B-enabled run should print a visible `wandb active ... target=...`
  line at startup.

### Blockers
- Online W&B logging needs the project environment with `torch`, `wandb`, and
  the other dependencies installed and authenticated.

### Next
Run a small synthetic MLP job with `--wandb --wandb-mode online` and
`--wandb-strict` from the dependency-ready environment.

## 2026-04-18T09:13:55Z - Clarify Checkpoint Evaluation Naming

### Context
The phrase "checkpoint replay" was confusing because it sounded like failed-run
restart behavior.

### Intent
Clarify that the checkpoint pass is for measuring memorization trajectories,
not for resuming or restarting training.

### Changed / Observed
- Confirmed `evaluate_run(...)` loads each saved checkpoint, generates samples
  from the saved shared `z_init`, computes memorization metrics, and writes
  trajectory artifacts.
- Updated source comments, docstrings, `src/README.md`, prototype README text,
  and status messages to say "checkpoint evaluation" instead of the confusing
  phrase.
- Left runtime behavior unchanged.

### Commands / Checks
- Searched source and prototype runner files for the old wording, excluding
  stored notebook outputs.
- Ran `timeout 30s python3.11 -m py_compile ...` on the touched Python files.
- Ran `git diff --check`.

### Results
- Python syntax check passed with Python 3.11.
- Whitespace check passed.
- No restart/resume training feature was removed because the checkpoint pass is
  not a training-restart feature.

### Blockers
- Stored notebook outputs still contain old historical console text until the
  notebooks are cleared or rerun.

### Next
When launching W&B runs, read the checkpoint-evaluation curves as the
memorization-over-training trajectory.

## 2026-04-18T09:35:13Z - Restore Direct Synthetic Runner Style

### Context
The synthetic prototype runner had become too layered for interactive research.
The preferred reference shape is the older projected-GMM prototype: one visible
script that shows config, data, model, training, metrics, artifacts, and W&B.

### Intent
Make `prototyping.runners.synthetic_mlp` direct and readable while leaving the
MNIST/CelebA prototype runners on their recipe-function API.

### Changed / Observed
- Rewrote `prototyping/runners/synthetic_mlp.py` as a direct old-style runner
  with `main(argv=None)`, visible projected-GMM data construction, direct
  optimizer loop, periodic synthetic metrics, checkpoints, figures, local
  status/summary artifacts, and optional direct W&B logging.
- Preserved useful aliases: `--out-dir/--study-dir`,
  `--prediction-space/--pred-space`, and `--n-reference/--n-eval`.
- Kept `--validate-only` as a fast component path that returns and prints a
  JSON payload with config, data equivalence, projection, one update, and
  synthetic metrics.
- Rewrote `prototyping/runners/notebooks/synthetic_mlp.ipynb` as a clean,
  unexecuted direct-flow playground with no stored outputs.
- Updated `prototyping/runners/README.md` and `tests/test_proto_runners.py`
  so synthetic is tested/documented as direct-style while image runners stay
  recipe-style.

### Commands / Checks
- `timeout 30s python3.11 -m py_compile prototyping/runners/synthetic_mlp.py`
- `timeout 30s python3.11 -m py_compile prototyping/runners/synthetic_mlp.py prototyping/runner_proto_synthetic_projected_gmm.py tests/test_proto_runners.py`
- `python3.11 -m unittest tests/test_proto_runners.py`
- Checked the synthetic notebook has no outputs and no execution counts.
- Searched touched files for the banned wording preference.

### Results
- Python syntax checks passed.
- Prototype-runner unit file ran but all 11 tests skipped in this shell because
  the active Python lacks the project ML dependencies.
- The synthetic notebook cleanliness check passed.
- The banned wording was not found in the touched synthetic runner/notebook,
  tests, runner README, or journal.

### Blockers
- Full validate-only and tiny full-run execution still need the dependency-ready
  project environment with `torch`, `wandb`, and friends.

### Next
Run `prototyping.runners.synthetic_mlp --validate-only` and then a tiny full
run from the project/Jupyter environment to confirm live metrics and W&B output.

## 2026-04-18T09:44:46Z - Record Project Python Environment

### Context
The default shell Python was previously the bare system Python, which skipped
project tests and could not import the ML dependencies.

### Intent
Record the project environment activation command so future checks use the
right interpreter.

### Changed / Observed
- Verified `module load Mambaforge && mamba activate mem_param_env` works in
  this shell.
- Confirmed the activated interpreter is Python 3.10.20 at
  `/home/x_moalj/.conda/envs/mem_param_env/bin/python`.
- Added an `AGENT.md` environment note requiring that activation before
  project-dependent Python commands.
- Updated one older artifact-rule phrase to say checkpoint evaluation.

### Commands / Checks
- `module load Mambaforge && mamba activate mem_param_env && python --version && python -c 'import sys; print(sys.executable)'`
- Inspected `AGENT.md`.

### Results
- Future Python checks should use `mem_param_env` instead of `/usr/bin/python`.

### Blockers
- Environment activation is per shell command, so commands still need to include
  the module/load activation prefix unless the interactive shell is already
  activated.

### Next
Run the synthetic validate-only and tiny full-run checks from `mem_param_env`.

## 2026-04-18T09:55:41Z - Orient Repo Direction

### Context
Read the repo README, AGENT guide, source/prototype READMEs, and recent journal
entries to answer whether the repo should expand or refactor next.

### Intent
Ground the next-step decision in the current dirty worktree, stated paper
objective, and actual runtime checks rather than only the written plan.

### Changed / Observed
- Confirmed the repo objective remains fixed-protocol comparison of `x`, `eps`,
  and `v` parameterizations through memorization metrics and checkpoint-level
  trajectories.
- Confirmed the intended boundaries: canonical reusable code in `src/`,
  reviewer-facing workflows in `experiments/`, transitional runners in
  `prototyping/`, and archived material as historical context only.
- Confirmed the current branch is `main` with local uncommitted work across the
  journal, AGENT guide, source docs/training/tracking/metrics/data files,
  prototype runners, notebooks, and tests.
- Found remaining naming drift: the root README still says "checkpoint replay"
  in a few places after newer docs moved to "checkpoint evaluation".
- Found that `prototyping/runners/notebooks/synthetic_mlp.ipynb` currently has
  stored execution counts/outputs, despite the desired clean-notebook invariant.

### Commands / Checks
- `git status --short --branch`
- `git diff --stat`
- `module load Mambaforge && mamba activate mem_param_env && python --version`
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src python -m prototyping.runners.synthetic_mlp --validate-only`
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src python -m unittest tests/test_proto_runners.py`
- `module load Mambaforge && mamba activate mem_param_env && TORCHDYNAMO_DISABLE=1 PYTHONPATH=src python -m unittest tests.test_proto_runners.TestProtoRunners.test_celeba_jit_recipe_steps_use_fake_dataset`

### Results
- Project environment resolves to Python 3.10.20.
- Synthetic MLP `--validate-only` passes in the project environment, including
  manual-vs-canonical GMM equality, projection diagnostics, metrics, and one
  optimizer update.
- `tests/test_proto_runners.py` does not pass yet: JiT fake-dataset one-update
  fails first on Torch Inductor requiring a working C++ compiler, and with
  Dynamo disabled it still fails on the vendored JiT `unpatchify` shape
  assertion.
- The same test file also fails the notebook cleanliness check because
  `synthetic_mlp.ipynb` has execution counts.

### Blockers
- Need to clean the synthetic MLP notebook outputs or decide that executed
  prototype notebooks are allowed and update the test policy accordingly.
- Need to fix the JiT prototype test/config/wrapper path. The tiny test sets
  `depth=1` while the runner default has `in_context_len=8` and
  `in_context_start=1`; with only one block, the vendor model never inserts
  in-context tokens but still slices them away before `unpatchify`.
- Need either a no-compile JiT test path or a documented compiler/module
  requirement for CPU JiT tests.

### Next
Refactor and stabilize before expanding: clean notebook artifacts, fix or guard
the tiny JiT path, align README wording around checkpoint evaluation, then run
the prototype runner tests again from `mem_param_env`.

## 2026-04-18T10:43:31Z - Implement DataLoader Batch-Cap Training

### Context
The N-sweep x-axis decision is now optimizer steps only. Training still needs
to materialize fixed selected datasets on disk, but batching should use a
DataLoader over that saved tensor with an explicit public `training.batch_cap`.

### Intent
Replace with-replacement tensor sampling in the canonical trainer and prototype
runner with the batch-cap rule:
`B_eff = min(batch_cap, N)` and `drop_last = N > batch_cap`, while keeping
`training.batch_size` as a legacy mirror.

### Changed / Observed
- Added `training.batch_cap` to the base config and mirrored legacy
  `training.batch_size`.
- Added canonical trainer helpers for resolving `batch_cap`, computing batching
  metadata, and constructing a shuffled `DataLoader` over the saved CPU
  `train_tensor`.
- Updated canonical training to cycle the DataLoader until exactly
  `training.n_steps` optimizer updates complete, moving only each batch to the
  target device.
- Persisted batching metadata into `data_meta.json`, `run_summary.json`, and
  attempt status fields.
- Updated MNIST config/CLI with preferred `--batch-cap` plus legacy
  `--batch-size`.
- Updated prototype runner config builders, CLIs, rendered commands, and the
  direct synthetic MLP runner to use `batch_cap`; synthetic keeps `0` as its
  old "all train points" CLI sentinel.
- Updated runner notebooks and the MNIST walkthrough to use `batch_cap`, and
  cleared runner notebook execution counts/outputs.
- Updated docs from "checkpoint replay" wording toward checkpoint evaluation
  where touched.
- Added batching tests for `N < batch_cap`, `N == batch_cap`, `N > batch_cap`,
  artifact/status metadata, exact final optimizer step, and alias coverage.

### Commands / Checks
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src python -m py_compile src/mp_config/base.py src/mp_training/trainer.py experiments/mnist/run.py experiments/mnist/__init__.py prototyping/gdfm_config_utils.py prototyping/runners/common.py prototyping/runners/synthetic_mlp.py prototyping/runners/mnist_unet.py prototyping/runners/celeba32_unet.py prototyping/runners/celeba32_jit.py prototyping/runners/image_unet.py tests/test_experiment_pipeline.py tests/test_mnist_experiment.py tests/test_proto_runners.py`
- `module load Mambaforge && mamba activate mem_param_env && MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python -m unittest tests/test_experiment_pipeline.py tests/test_mnist_experiment.py tests/test_proto_runners.py`
- `module load Mambaforge && mamba activate mem_param_env && MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python -m prototyping.runners.synthetic_mlp --device cpu --out-dir tmp/bcap_synth --run-name bcap_smoke --n-train 8 --n-reference 12 --batch-cap 4 --eval-samples 4 --sampling-steps 2 --data-dim 8 --intrinsic-dim 2 --n-clusters 3 --hidden-dim 16 --n-layers 2 --time-emb-dim 16 --sliced-w2-projections 8 --n-steps 2 --metric-every 1 --ckpt-every 1 --log-every 1`
- Checked `tmp/bcap_synth/latest/status.json`, `artifacts/config.json`,
  `metrics/train_history.json`, and `metrics/metrics_along_training.json`.

### Results
- Syntax checks passed.
- Requested unit suite passed: 19 tests in 39.412 seconds.
- The suite covers tiny canonical GMM and MNIST smoke paths, synthetic
  validate-only, notebook cleanliness, and prototype alias behavior.
- Tiny synthetic full run completed with `batch_cap=4`, `effective_batch_size=4`,
  `drop_last=true`, two optimizer steps, two metric records, and completed
  status.
- Torch emitted expected cluster noise for the vendored JiT path: NVML was
  unavailable and Torch Dynamo fell back after no working C++ compiler was
  found. The tests still passed.

### Blockers
- No functional blocker for the batch-cap refactor.
- Some unrelated dirty and untracked files remain in the working tree from
  adjacent work; they were left in place.

### Next
Run a real non-fake MNIST command on a node with the dataset/cache available,
then use `--batch-cap` sweeps for the first optimizer-step-aligned N-sweep
plots.

## 2026-04-18T10:53:28Z - Add Pedagogical Prototype Runner Comments

### Context
The prototype runner directory is now doing two jobs: it is runnable tooling
and it is the readable teaching surface for how configs, data, one-update
checks, full jobs, and rendered commands fit together.

### Intent
Make the runner code easier to read in notebooks and reviews by adding inline
comments at the contracts that matter, without turning simple assignments into
noisy prose.

### Changed / Observed
- Added explanatory inline comments to `prototyping/runners/common.py` around
  path setup, batch-cap alias resolution, command rendering, and manifest
  traversal.
- Added comments to `mnist_unet.py`, `celeba32_unet.py`, `celeba32_jit.py`,
  and `image_unet.py` explaining knob dictionaries, config sections,
  parameterization fields, fixed-subset data policy, B-cap mirrors,
  validate-only mode, and canonical train/evaluate handoff.
- Added comments to `synthetic_mlp.py` explaining the direct synthetic flow:
  manual-vs-canonical GMM data, planted geometry, metric groups, B-cap
  metadata, latent/residual decomposition, baseline metrics, validate-only
  behavior, and DataLoader epoch cycling.
- Found that imports expecting `prototyping.metrics_synthetic_controlled`
  could not resolve because the implementation currently lives under
  `prototyping/runners/metrics_synthetic_controlled.py`.
- Added a narrow compatibility shim at `prototyping/metrics_synthetic_controlled.py`
  so older notebooks/tests and the synthetic runner import path stay valid.

### Commands / Checks
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src:. python -m py_compile prototyping/metrics_synthetic_controlled.py prototyping/runners/common.py prototyping/runners/mnist_unet.py prototyping/runners/celeba32_unet.py prototyping/runners/celeba32_jit.py prototyping/runners/image_unet.py prototyping/runners/synthetic_mlp.py`
- `module load Mambaforge && mamba activate mem_param_env && MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src:. python -m unittest tests/test_proto_runners.py`
- Direct import check for `synthetic_mlp`, `mnist_unet`, `celeba32_unet`, and
  `celeba32_jit`.

### Results
- Runner syntax checks passed.
- Prototype runner tests passed: 12 tests in 16.903 seconds.
- Direct runner imports now resolve.
- The known Torch/JiT warning about missing C++ compiler still appears during
  the JiT fake-dataset test, but it falls back and the test suite passes.

### Blockers
- No blocker for the comment cleanup.

### Next
Keep comments this style: explain invariants and handoffs, not obvious line
mechanics.

## 2026-04-18T11:11:57Z - Prepare Batch-Cap Runner Work For Push

### Context
The batch-cap trainer refactor and prototype runner cleanup are ready to share.
The local checkout is on `main`, but direct GitHub CLI PR tooling is unavailable
because `gh` is not installed in this environment.

### Intent
Commit and push the feature work with plain `git`, keeping scratch files out of
the commit and leaving PR creation as a later step if needed.

### Changed / Observed
- Confirmed `git` can be used independently of GitHub CLI; the remote is the
  SSH origin at `git@github.com:MohammedAlJaff/mem_par.git`.
- Selected the feature scope for commit: batch-cap canonical training,
  prototype runners, runner comments, tests, notebooks/docs touched by the
  refactor, and the compatibility shim.
- Left local scratch artifacts uncommitted: `.codex`, `ram_test.py`, and
  `prototyping/runners/tea_debug.log`.

### Commands / Checks
- `module load Mambaforge && mamba activate mem_param_env && MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src:. python -m unittest tests/test_experiment_pipeline.py tests/test_mnist_experiment.py tests/test_proto_runners.py`

### Results
- Requested unit suite passed: 19 tests in 47.924 seconds.
- Torch emitted the known cluster warnings about unavailable NVML and no
  working C++ compiler for the vendored JiT path; the fallback still passed.

### Blockers
- `gh` is unavailable, so no local `gh pr create` flow was attempted.
- The sandbox blocks `.git` ref writes, so branch/commit/push need elevated
  git execution.

### Next
Create the feature branch, commit the selected files, and push it to origin
with plain `git`.

## 2026-04-18T11:44:30Z - Confirm Batch-Cap Runner Work Landed

### Context
The previous journal entry ended before the branch/commit/push step was
recorded. A follow-up repo sanity check was requested before continuing runner
review work.

### Changed / Observed
- Confirmed local `main`, `origin/main`, and `origin/HEAD` are all at
  `2dfc6b5 Add batch-cap training and prototype runners`.
- Confirmed the batch-cap canonical training and prototype runner work has
  landed on the remote.
- Observed remaining scratch files in the local working tree:
  `.codex`, `ram_test.py`, and `prototyping/runners/tea_debug.log`.

### Commands / Checks
- `git status --short --branch`
- `git log --oneline --decorate -6`
- `git show --stat --oneline --decorate --summary HEAD`
- `git branch -vv`

### Results
- The repo is up to date with `origin/main`.
- The prior "create branch, commit, and push" next step is now stale history.

### Next
Remove local scratch artifacts that are no longer needed and review
`prototyping/runners/synthetic_mlp.py`.

## 2026-04-19T02:50:59Z - Consolidate Image32 Config Presets

### Context
Image32 UNet runner work was split into separate knob/config presets with
lightweight CLI exposure for sanity testing.

### Intent
Move Image32 UNet config defaults out of the runner body, keep the runner
readable, and add quick cleanup for obsolete scratch files.

### Changed / Observed
- Added `prototyping/runners/default_knobs/image32_unet.py` for shared dataset
  knob recipes and defaults.
- Added `prototyping/runners/image32_unet_config.py` as the canonical config
  builder for image32 runs.
- Added `prototyping/runners/image32_unet_configs/` with shared presets:
  `image32_base.py`, `image32_32.py`, `image32_128.py`, `image32_256.py`.
- Updated `prototyping/runners/image32_unet.py` to import shared knobs/config,
  add `--config-preset`, and use explicit sanity update flags.
- Removed untracked scratch artifacts:
  `prototyping/runners/Untitled Folder`, `.ipynb_checkpoints`, runner `__pycache__`
  directories, and `prototyping/runners/notebooks/image_metrics_scaffold.ipynb`.

### Commands / Checks
- `python -m py_compile prototyping/runners/image32_unet.py prototyping/runners/image32_unet_config.py prototyping/runners/image32_unet_configs/__init__.py prototyping/runners/image32_unet_configs/*.py prototyping/runners/default_knobs/image32_unet.py`
- `git status --short`
- `rg`/import sweep across `prototyping` and `src` to confirm moved references.

### Results
- New Image32 config structure is in place and syntactically valid.
- No active references to removed in-runner dataset/config constants remain.

### Blockers
- Full runtime testing requires the project environment (`mem_param_env`), not
  active in this shell.

### Next
- Stage and commit the cleanup/preset migration and push when ready.

## 2026-04-19T03:03:48Z - Finalize Runner Presets and Push

### Context
Image32 UNet was moved to a cleaner “knob-free” setup using a shared settings
builder and dedicated preset modules.

### Intent
Finish the current runner refactor in version control and capture a compact log
of what moved.

### Changed / Observed
- Removed the old knobs/config modules:
  - `prototyping/runners/default_knobs/`
  - `prototyping/runners/image32_unet_configs/`
- Removed legacy `prototyping/runners/image32_unet_config.py`.
- Added `prototyping/runners/image32_unet_config_builder.py` for explicit settings
  merge + config construction.
- Added `prototyping/runners/config_presets/` with:
  - `image32_base.py`
  - `image32_32.py`
  - `image32_128.py`
  - `image32_256.py`
  - `__init__.py` exposing `list_presets(...)` and `get_default_configs(...)`.
- Updated `prototyping/runners/image32_unet.py` imports to use the new presets
  and builder APIs.

### Commands / Checks
- `git status --short --branch`
- `python -m py_compile prototyping/runners/image32_unet.py prototyping/runners/image32_unet_config_builder.py prototyping/runners/config_presets/__init__.py prototyping/runners/config_presets/image32_*.py`
- `rg "default_knobs.image32_unet|image32_unet_config|image32_unet_configs|config_presets|list_presets" -n prototyping/runners src tests`

### Results
- Runner/config refactor is now represented as a coherent preset+builder structure.
- No remaining in-repo imports reference the removed knob/config modules.

### Blockers
- Multiple unrelated modified/untracked files remain in the worktree; this journal
  entry does not include them in the same commit.

### Next
- Commit only the Image32 preset-builder migration files and push.
