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

## 2026-04-19T03:09:20Z - Default W&B Namespace to memparam_mnist_tests

### Context
User requested all default W&B logging for image prototypes and canonical flows to
target `maj90/memparam_mnist_tests` by default, with env overrides.

### Intent
Make `WANDB_PROJECT`/`WANDB_ENTITY` the first override source, then set hard
defaults to `memparam_mnist_tests`/`maj90`, while preserving CLI flags as
highest-priority overrides.

### Changed / Observed
- Image32 preset stack:
  - `prototyping/runners/image32_unet_config_builder.py` now defaults to
    `memparam_mnist_tests` and `maj90` with `WANDB_*` env fallback support.
  - `prototyping/runners/config_presets/image32_base.py` now applies the same
    env-aware defaults in the preset baseline.
- Synthetic prototype runner:
  - `prototyping/runners/synthetic_mlp.py` now defaults to the same project/entity
    via `WANDB_*` env fallback.
- Canonical tracking surface:
  - `src/mp_config/base.py` uses env-aware defaults for base W&B project/entity.
  - `src/mp_tracking.py` uses env-aware fallback defaults when resolving active
    W&B run metadata.
- Docs and notebook defaults were aligned:
  - `README.md` W&B example project flag.
  - synthetic notebook knob default for `wandb_project`.

### Commands / Checks
- `python -m py_compile ...` for modified runner/config/tracking files (in `mem_param_env`).
- Parser smoke checks:
  - `parse_args([])` defaults for image32 and synthetic show `memparam_mnist_tests`/`maj90`.
  - same check with `WANDB_PROJECT=alt_project WANDB_ENTITY=alt_entity` verified override.
- Runtime smoke:
  - tiny image32 full run (`--no-wandb`) completed successfully.
  - tiny synthetic run with `--wandb` attempted and reached run completion; W&B
    init warning observed due missing token in this environment, but status now
    records configured default project/entity.

### Blockers
- W&B cloud/token is not available in this environment, so synthetic status shows
  `warning` with `ServicePollForTokenError` when `--wandb` is enabled.

### Next
- Add explicit user-facing note for required env vars/token setup where online W&B is
  needed.

## 2026-04-19T03:33:21Z - Fix Image32 Preset Resolution for CLI Defaults

### Context
A `--dataset mnist --config_preset None` style path could pass the literal
`"None"` into the preset resolver and crash before training started.

### Intent
Keep the Image32 prototype launcher from raising a preset lookup error when
`config_preset` is unset while preserving explicit CLI overrides.

### Changed / Observed
- Updated `prototyping/runners/image32_unet_config_builder.py`:
  - `_coalesce(...)` now treats blank strings as unset in addition to `None`.
  - Preset selection now uses `_coalesce(settings["config_preset"], dataset default)`
    instead of direct `str(...)` coercion.
- Existing runtime class-balance guard for MNIST remains enabled and prints a
  helpful warning when non-divisible counts are requested.

### Commands / Checks
- `python -m prototyping.runners.image32_unet --device cuda ...` with `--dataset
  mnist` and `--n-train 4096 --n-eval 4096` now reaches config build instead of
  hitting `Unknown Image32 UNet preset: None`.

### Results
- Preset resolution uses the dataset default (`image32_32`) when unset, so the
  UNet CLI no longer fails on defaulted preset config.

### Blockers
- Project environment `python` entrypoint in this shell resolves to 3.6.8 for
  `py_compile` checks, so syntax validation with future annotations could not be
  fully executed in-place.

### Next
- Push the current session changes once the branch is clean.

## 2026-04-19T03:46:13Z - Image32 Visual Sanity Panels

### Context
The Image32 UNet runner needs an opt-in visual sanity path that shows real train
cells before model work and generated samples beside train examples after a few
updates and after training.

### Intent
Make visual checks visible locally and in W&B without changing normal training
behavior when the flag is off.

### Changed / Observed
- Added `config.visual_sanity` defaults in `src/mp_config/base.py`.
- Added Image32 CLI/config settings:
  - `--visual-sanity`
  - `--visual-sanity-count`
  - `--visual-sanity-updates`
  - `--visual-sanity-show`
- Extended `src/mp_training/trainer.py` so visual sanity runs:
  - saves/logs `figures/visual_sanity_train_grid_before_model.png` after data
    load and before model creation.
  - trains a throwaway model for a few updates and saves/logs
    `figures/visual_sanity_samples_over_train.png`.
  - saves/logs `figures/final_samples_over_train.png` from the trained model.
  - registers all visual sanity figures through `AttemptTracker`, so active W&B
    receives them as images.
- Runner final output now prints the `figures/` directory.
- Updated `prototyping/runners/README.md` with CUDA visual sanity examples.

### Commands / Checks
- `module load Mambaforge && mamba activate mem_param_env && python -m py_compile ...`
- Parser/config smoke for `--visual-sanity`.
- Runtime smoke on CPU because CUDA is unavailable in this tool shell:
  - `python -m prototyping.runners.image32_unet --device cpu --dataset mnist ... --visual-sanity`

### Results
- Visual sanity smoke produced the train grid, throwaway-update sample panel,
  final sample panel, checkpoints, metrics, and final printed `figures:` path.

### Blockers
- CUDA is not visible inside the tool environment (`torch.cuda.is_available()`
  is false), so CUDA runtime verification still needs the user node/session.

### Next
- Run the same smoke command with `--device cuda` on the active GPU node and
  confirm W&B receives the three visual sanity figures.

## 2026-04-19T04:07:22Z - Checkpoint Visual Panels for Image32 Evaluation

### Context
The Image32 visual sanity path should show sample evolution at each checkpoint,
not only before and after training.

### Intent
Generate one samples-over-train panel for every saved checkpoint during
Image32 evaluation, and stop W&B scalar logs from conflicting on the global
step counter.

### Changed / Observed
- Added `src/mp_visualization.py` as the shared image-grid renderer.
- Updated training visual sanity code to use the shared renderer.
- Extended `prototyping/runners/metrics_image32.py`:
  - resumes the attempt tracker during evaluation.
  - logs eval metrics with explicit eval steps.
  - saves/registers `figures/checkpoint_samples_over_train_step_*.png` for
    every checkpoint when `--visual-sanity` is enabled.
  - records `checkpoint_visual_panels` in `run_summary.json`.
- Updated `src/mp_tracking.py`:
  - defines W&B metric axes for train, eval, and visual sanity namespaces.
  - converts caller-provided steps into explicit fields instead of W&B global
    `step=...` logging.
  - uses `reinit=\"finish_previous\"` for W&B initialization.

### Commands / Checks
- `module load Mambaforge && mamba activate mem_param_env && python -m py_compile ...`
- `PYTHONPATH=src:. python -m unittest tests.test_tracking`
- CPU runtime smoke:
  - `python -m prototyping.runners.image32_unet --device cpu --dataset mnist --n-steps 2 --ckpt-every 2 --visual-sanity ...`

### Results
- Smoke run saved checkpoint panels for steps `0` and `2`, plus the existing
  train/preflight/final panels.
- `run_summary.json` records two `checkpoint_visual_panels`.
- W&B online step-axis behavior still needs confirmation on the user CUDA
  node/session because this tool shell cannot initialize W&B service reliably.

### Next
- Run the CUDA online smoke with `--n-steps 200 --ckpt-every 200 --visual-sanity`
  and confirm no W&B out-of-order step warnings.

## 2026-04-19T04:13:26Z - Informative Image Panel Titles

### Context
Visual sanity and checkpoint panels should be readable after the fact in local
figures and W&B media, without needing to cross-reference the run config.

### Changed / Observed
- Added a shared `model_context_title(...)` helper for image panels.
- Panel titles now include:
  - checkpoint or sanity-update context when relevant.
  - training steps and sampling steps.
  - dataset and image size.
  - framework family.
  - prediction space and loss space parameterization.
  - model architecture.
- Training preflight, visual sanity, final samples, and checkpoint panels all
  use the same title format.
- Image-panel rendering now chooses a local `logs/matplotlib` cache next to
  direct figure outputs when the path is not under a run `figures/` directory.

### Commands / Checks
- `module load Mambaforge && mamba activate mem_param_env && python -m py_compile src/mp_visualization.py src/mp_training/trainer.py prototyping/runners/metrics_image32.py`
- Tiny render smoke to `/tmp/image32_title_smoke.png` with a checkpoint-style
  contextual title.

### Results
- Syntax check and render smoke passed in `mem_param_env`.

## 2026-04-19T23:27:07Z - Implement W&B Combo Naming and DONE Attempts

### Context
Followed up on the Image32 visual sanity work with operational naming and run
folder cleanup requirements.

### Intent
Make W&B organization match the comparison workflow and make successfully
trained attempts visible from the filesystem.

### Changed / Observed
- Set the Image32 base preset learning rate to `0.005`.
- Changed default canonical W&B project resolution to
  `memparam-{dataset}-{model}` when neither `--wandb-project` nor
  `WANDB_PROJECT` is provided.
- Changed default W&B run names to
  `{dataset}-{model}-N{n_train}-{prediction_param}_{loss_param}-s{seed}`.
- Added `mark_attempt_done(...)`, which renames successful training attempt
  folders to `DONE_<attempt_id>` and rewrites status, run-summary, and manifest
  paths.
- Updated `train_run(...)` to mark attempts DONE after successful training, so
  Image32 evaluation continues from the renamed path.
- Restored the synthetic metrics compatibility shim so prototype-runner tests
  execute instead of skipping.

### Commands / Checks
- `git diff --check`
- `python -m py_compile ...` for changed source, runner, and test files.
- `python -m json.tool prototyping/runners/notebooks/synthetic_mlp.ipynb`
- `python -m unittest tests.test_tracking tests.test_artifact_layout tests.test_experiment_pipeline tests.test_proto_runners`
- Checked `torch.cuda.is_available()` in this tool shell.

### Results
- Syntax, whitespace, and notebook JSON checks passed.
- Focused test suite passed: 24 tests in 10.055 seconds.
- CUDA is not visible in this tool shell (`torch.cuda.is_available() == False`).

### Blockers
- CUDA/W&B online smoke still needs to run from the active GPU session.

### Next
- Run the Image32 CUDA online smoke with `--visual-sanity`, `--n-steps 200`,
  and `--ckpt-every 200`; confirm W&B project/run names, media panels, no
  out-of-order step warnings, and final printed path using `DONE_`.

## 2026-04-19T23:50:07Z - Shared Checkpoint Loss Evaluation

### Context
Loss values should be comparable across synthetic, image, canonical, and
prototype runs, not only visible as noisy train-step losses.

### Intent
Evaluate held-out/reference-set loss at checkpoint cadence with random-`t` and
fixed-`t` probes, then save and log the same metric shape across run types.

### Changed / Observed
- Added shared `src/mp_metrics/loss_eval.py` with deterministic checkpoint
  probes for random `t`, `t=0.5`, and `t=0.75`.
- Added config defaults:
  - `evaluation.loss_eval_times = (0.5, 0.75)`
  - `evaluation.loss_eval_batch_size = 0`
- Canonical `evaluate_run(...)` now writes `metrics/loss_eval.json`, updates
  run summary fields, and logs loss probes under `eval/*`.
- Image32 evaluation now uses the shared helper and evaluates held-out
  `reference_tensor`, not the training tensor.
- Direct synthetic runner now records `loss_eval.json` at checkpoint steps.
- Artifact status metadata now includes `loss_eval_path`.

### Commands / Checks
- `git diff --check`
- `python -m py_compile ...` for changed source, runner, and test files.
- `python -m unittest tests.test_loss_eval tests.test_experiment_pipeline tests.test_proto_runners tests.test_artifact_layout tests.test_tracking`

### Results
- Whitespace and syntax checks passed.
- Focused suite passed: 25 tests in 10.485 seconds.

### Blockers
- No implementation blocker. CUDA/W&B smoke from the previous entry is still
  pending in a GPU-visible session.

### Next
- In the next CUDA smoke, inspect both W&B `eval/random_t_loss` and
  `eval/loss_t_0_5` / `eval/loss_t_0_75`, plus local `metrics/loss_eval.json`.

## 2026-04-20T00:27:59Z - Checkpoint Sample Previews And Live Fixed-Z Grids

### Context
Image runs need quick visual feedback during training, while checkpoint
evaluation should keep comparable samples for later inspection.

### Intent
Save small checkpoint sample previews consistently, and make live W&B sample
grids easy to watch without changing the noise source across checkpoints.

### Changed / Observed
- Added `src/mp_metrics/checkpoint_previews.py` to save compact preview tensors
  under `metrics/checkpoint_sample_previews/` and optional image panels under
  `figures/`.
- Canonical and Image32 checkpoint evaluation now write
  `metrics/checkpoint_sample_previews.json` and update run summaries/status with
  preview paths.
- Direct synthetic runner now writes equivalent checkpoint preview tensor
  records with `--checkpoint-preview-count`.
- Added live visual sanity checkpoint grids in `train_run(...)`: after step 0
  and each checkpoint, the trainer samples from the saved fixed `z_init` bank,
  saves `figures/live_checkpoint_samples_over_train_step_*.png`, and logs the
  panel to W&B under `eval/live_checkpoint_samples_over_train`.
- Added `--log-every` to the Image32 CLI so train-loss logging cadence can be
  controlled directly, e.g. `--log-every 1` for smoke runs.

### Commands / Checks
- `git diff --check`
- `python -m py_compile ...` for changed source, runner, and test files in
  `mem_param_env`.
- `python -m unittest tests.test_checkpoint_previews tests.test_experiment_pipeline tests.test_proto_runners tests.test_artifact_layout tests.test_tracking tests.test_loss_eval`

### Results
- Whitespace and syntax checks passed.
- Focused suite passed: 30 tests in 13.526 seconds.

### Blockers
- `gh` is not installed in this shell, so a draft GitHub PR cannot be opened
  from here.
- CUDA/W&B online smoke still needs to run from a GPU-visible session.

### Next
- Run the Image32 CUDA smoke with `--visual-sanity --visual-sanity-count 20
  --ckpt-every 200 --log-every 1`; confirm W&B `train/loss` points and
  `eval/live_checkpoint_samples_over_train` media at checkpoint steps.

## 2026-04-20T02:10:00Z - Bonnaire-Close Image32 Runner Cleanup

### Context
The prototype Image32 runner had grown too much CLI surface and still carried
sanity/update terminology that no longer matched the training workflow.

### Intent
Keep the runner simple for paper experiments: Bonnaire-close 32px defaults,
few CLI knobs, fixed-noise generation visuals across training, and clear
definitions for the run-size knobs.

### Changed / Observed
- Removed Image32 sanity-only/update paths and old visual-sanity naming.
- Renamed the live visual path to `generation_trace`, with fixed `z_init.pt`
  panels at step 0, each checkpoint, and final model.
- Updated W&B logging for trace figures to
  `generation_trace/samples_over_train` with explicit `generation_trace/step`.
- Set the default 32px/Image32 UNet profile close to Bonnaire:
  `base_channels=32`, `channel_mult=(1, 2, 4)`,
  `attn_resolutions=(16, 8)`, `num_res_blocks=2`, `dropout=0.1`,
  `time_emb_dim=None`, Adam, and `lr=1e-4`.
- Slimmed the Image32 CLI to real sweep/prototype knobs; rare architecture and
  data-shape knobs now live in dataset profiles or config presets.
- Added plain comments/help text explaining:
  - `n_train`: real images used for optimization.
  - `n_eval`: real held-out/reference images loaded for evaluation.
  - `eval_samples`: generated samples per checkpoint and `z_init.pt` size.
  - `sampling_steps`: solver steps used to generate images from `z_init`.
- Confirmed Bonnaire uses sinusoidal time embeddings expanded to
  `base_channels * 4` and injected into ResNet blocks.
- Kept checkpoint preview figures alongside generation trace figures for now;
  they serve the post-training metric/evaluation bundle.

### Commands / Checks
- `git diff --check`
- `python -m py_compile ...` for changed Image32 runner/config/trainer files.
- `python -m unittest tests.test_proto_runners`
- `python -m unittest tests.test_mp_data_mnist`
- `python -m unittest tests.test_checkpoint_previews tests.test_experiment_pipeline tests.test_artifact_layout tests.test_tracking`

### Results
- Whitespace and syntax checks passed.
- Focused test suites passed:
  - `tests.test_proto_runners`: 16 tests.
  - `tests.test_mp_data_mnist`: 2 tests.
  - Shared artifact/tracking suites: 15 tests.
- Only existing `torch.load(weights_only=False)` FutureWarnings appeared.

### Blockers
- `gh` is not installed in this shell, so PR automation is unavailable.
- CUDA/W&B online smoke still needs to run from a GPU-visible session.

### Next
- Run the proper Image32 CUDA run with `--generation-trace`; inspect both
  `generation_trace_*` training panels and checkpoint metric previews.

## 2026-04-20T03:40:26Z - Generalize Image Runner And Add Runtime Speed/Trace Polish

### Context
The prototype image runner is no longer only a 32px runner: today it was used
for Flowers128 and AFHQ/face-style 256px experiments, and the W&B projects were
getting mixed across dataset/model combinations.

### Intent
Ship the single-process image workflow with clearer naming, visible runtime
numerics, fixed-noise generation traces, and W&B metadata that makes the
scientific training step unambiguous. Keep DDP/multi-GPU out of this runner.

### Changed / Observed
- Hard-renamed the public entrypoint from
  `prototyping.runners.image32_unet` to `prototyping.runners.image_runner`.
- Kept image-family internals where they still fit: `metrics_image32.py` and
  the `image32_32`, `image32_128`, and `image32_256` presets.
- Updated the default output root to
  `outputs/prototype_runners/image_runner`.
- Added `--precision {fp32,tf32,bf16}` and `--compare-numerics`, with runtime
  metadata for TF32, autocast dtype, pinned memory, and nonblocking H2D copies.
- Enabled CUDA `pin_memory` and nonblocking transfer for the already
  materialized train-tensor batches.
- Kept the fixed-`z_init` `generation_trace` figures across training and kept
  the checkpoint preview figure family alongside it.
- Added W&B image captions that show the real gradient step, e.g.
  `generation_trace | grad_step=...` and
  `checkpoint_samples_over_train | grad_step=...`.
- Routed default W&B projects by dataset/model key, such as
  `memparam-flowers102_128-unet`, while treating the old broad
  `memparam-image32-unet` value as legacy auto-routing.
- Flipped Flowers102 profiles so training uses official `test` and reference
  evaluation uses official `train`, which better matches the available split
  sizes for generative experiments.
- Verified the hard rename behavior: the old module fails by design and
  `python -m prototyping.runners.image_runner --help` works in `mem_param_env`.
- Left multi-GPU/DDP intentionally deferred to a separate future runner, likely
  `prototyping.runners.image_runner_ddp`, launched with `torchrun`.
- GPU note from AFHQ 256 BF16 on A100 40GB: batch 16 ran at about
  `3.28 step/s`; batch 24 ran at about `2.24 step/s`, so batch 16 remains the
  better fixed-step default for that setup. An 80GB GPU may help with headroom
  and larger evals, but DDP is the real route to shorter wall-clock for huge
  step counts.

### Commands / Checks
- `python -m unittest tests.test_proto_runners`
- `python -m unittest tests.test_tracking tests.test_mp_runtime`
- `python -m unittest tests.test_checkpoint_previews tests.test_experiment_pipeline tests.test_loss_eval`
- `git diff --check`
- `python -m py_compile ...` for changed runner, runtime, metric, trainer,
  tracking, and test files.
- `python -m prototyping.runners.image_runner --help`
- `python -m prototyping.runners.image32_unet --help`

### Results
- `tests.test_proto_runners`: 19 tests passed.
- `tests.test_tracking tests.test_mp_runtime`: 6 tests passed.
- Checkpoint/experiment/loss-eval suite: 9 tests passed.
- Whitespace and byte-compile checks passed.
- New runner help works inside `mem_param_env`.
- Old `image32_unet` module fails with `No module named`, as intended.
- Only existing `torch.load(weights_only=False)` FutureWarnings appeared.

### Blockers
- None for this ship batch.

### Next
- Push this runtime batch to `origin/main`.
- Run longer Flowers128/AFHQ256 BF16 jobs with dataset-specific W&B projects.
- Start any multi-GPU work in a separate DDP runner rather than expanding
  `image_runner.py`.
