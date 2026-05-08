# Coding Journal

## 2026-05-09T01:36:46+0200 - N-Train tau-scale figure polish

Updated the standalone n-train tau-scale plotting script:

- Kept the threshold-based tau definition fixed at `FID <= 1.10 x run-best FID` for `tau_gen`.
- Kept the main three-column tau layout as `tau_gen`, `tau_mem`, and `Delta tau`, removing the extra log-y `tau_mem` column.
- Added figure-level definition titles while keeping subfigures title-free and driven by axis labels.
- Added one external parameterization legend on the right side of each tau figure.
- Centralized parameter colors so `eps/eps`, `v/v`, `x/v`, and `x/x` use the same colors across all tau figures, including the original N01 grid.
- Added explicit per-subfigure panel sizing so the side legend no longer changes panel flatness.
- Increased and bolded titles, axis labels, tick labels, legend text, line widths, markers, and censored-point markers for readability.
- Increased panel heights after the typography pass so the figures are less flat and have more breathing room.

Verification:

- `python -m py_compile prototyping/plot_n_train_tau_scales.py`
- `MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp conda run -n memcompgen python prototyping/plot_n_train_tau_scales.py`

Generated outputs were refreshed under `prototyping/figures/n_train_tau_scales/`.

## 2026-05-07T16:07:19+02:00 - Capacity Figures and N-Train Tau-Scale Analysis

### Context
Continued the paper-figure cleanup for mem-boundary analyses, with two main
threads: capacity-control figure polish and a new Bonnaire-style tau-scale
analysis focused only on the original `N_train` sweep.

### Capacity Figure Changes
- Updated `prototyping/analyze_mem_boundary_quality_mem_frontier.py` to use
  repo-relative defaults instead of the cluster `/proj/...` path.
- Renamed the capacity outputs into a capacity-specific scheme:
  `C01_capacity_best_fid_and_fmem_by_capacity` and
  `C03_capacity_quality_mem_pareto`.
- Merged the previous separate capacity best-FID and `F_mem@FID_best` plots
  into a single two-row C01 grid with capacity columns, shared scales,
  shared legend, bold axis labels, and parsed capacity titles:
  base channel size plus block channel multipliers.
- Standardized capacity parameterization legends to math labels:
  `\epsilon/\epsilon`, `v/v`, `x/v`, and `x/x`.

### N-Train Tau-Scale Figures
- Added `prototyping/plot_n_train_tau_scales.py`, a standalone script that
  reads only
  `outputs/evals/mem_boundary_20260427_all_done_quality_20260501/all_checkpoint_metrics.csv`.
- The script writes run-level and seed-averaged tau-scale CSV tables plus
  PNG/PDF figures under `prototyping/figures/n_train_tau_scales`.
- Added threshold-based tau figures:
  - `N01_n_train_tau_gen_mem_delta_by_param`;
  - `N02_n_train_tau_gen_mem_delta_overlay_params`;
  - `N03_n_train_tau_gen_threshold_sensitivity_overlay_params`;
  - `N04_n_train_tau_gen_threshold_sensitivity_fmem_0p01_overlay_params`.
- Added Bonnaire-style tau figures:
  - `N05_bonnaire_tau_argmin_fmem_onset_overlay_params`, where
    `tau_gen` is the first checkpoint attaining run-minimum FID and
    `tau_mem` is the first checkpoint with `F_mem > 0`;
  - `N06_bonnaire_tau_scaling_guides_overlay_params`, with reference guides
    for constant `tau_gen` and `tau_mem` proportional to `N_train`.

### Verification
Ran the tau-scale script in the project conda environment:

```bash
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp conda run -n memcompgen python prototyping/plot_n_train_tau_scales.py
```

The run completed successfully. Threshold-based `N01` processed `84` runs with
`24` censored `tau_mem` rows. Bonnaire-style `N05` processed `84` runs with
`0` missing `tau_gen` and only `1` censored `tau_mem` row. The generated `N02`,
`N03`, `N04`, `N05`, and `N06` PNGs were visually inspected for axis scales,
legends, threshold notes, and readability.

## 2026-05-07T07:07:59+02:00 - Figure Plotting Cleanup Checkpoint

### Context
Prepared the current figure-generation work for handoff so it can be pulled
and continued from another machine. The main active script is:
`prototyping/plot_mem_boundary_20260427_quality_memorization.py`.

### Plotting Changes
- Standardized figure output to write both PDF and PNG through the shared
  `save_figure` helper.
- Added compact publication-oriented figure sizing and smaller markers across
  the mem-boundary plotting scripts.
- Normalized visible plot text to use `FID` and math-form
  `N_{\mathrm{train}}` labels where appropriate while leaving internal column
  names such as `fid_10k` unchanged.
- Stopped future generation of figure 04 by removing its call from `main`.
  Existing stale `04_*` output files were intentionally left untouched.
- Reworked figure 06 into a row-per-visible-`N_train` grid with five columns:
  FID over tau, `F_mem` over tau, phase plot, log-FID phase plot, and log-log
  phase plot. Each row selects the requested seed when available, otherwise
  the best-coverage seed for that `N_train`.
- Added margin row labels and compact log tick formatting to keep figure 06
  readable after expanding it to all visible `N_train` values.
- Kept figure 11 and figure 12 cleanup changes in the plotting script:
  smaller subfigures, dashed paired metrics, no markers for figure 11, angled
  x tick labels, in-panel legends, figure 12 row-wise shared `F_mem` scale,
  and one column per parameterization for the FID/`F_mem` tau grid.

### Verification
Ran the legacy plotting script in the project environment:

```bash
module load Mambaforge && mamba activate mem_param_env && MPLCONFIGDIR=/tmp/mpl-mempar python prototyping/plot_mem_boundary_20260427_quality_memorization.py
```

The run completed successfully and regenerated
`06_parameterization_trajectories_fixed_seed_n_train.{pdf,png}` without
printing any new `04_fmem_at_best_fid_vs_n_train_by_parameterization.{pdf,png}`
outputs. The refreshed figure 06 PNG was inspected for row labels, spacing, and
tick readability. A targeted grep over visible label/title callsites found no
remaining exposed `fid_10k`, `FID_best`, or lowercase `fid` labels.

## 2026-05-01T02:50:24+02:00 - Capacity Resume Wave Submitted

### Context
Before touching the unfinished capacity-control outputs, we made a full
directory-level safety snapshot of the capacity output tree:
`outputs/job_sets/mem_gen_boundary_capacity_20260427_BACKUP_20260501_023904`.
The source tree is about `1.3T` while the filesystem had about `887G` free, so
a physical byte-for-byte copy would not fit. The backup is therefore a
hardlink snapshot. Small mutable metadata/log files in the backup were
de-linked after creation so status/manifests/logs preserve the pre-resume
state; large checkpoint/tensor files remain hardlinked.

### Non-DONE Capacity Diagnosis
- Reconciled `jobs/job_sets/mem_gen_boundary_capacity_20260427/manifest.tsv`
  against local attempt `status.json` files and Slurm logs.
- Found `52` non-`DONE_` capacity-control runs.
- All `52` classify as Slurm walltime terminations, not Python/app failures:
  stderr evidence consistently ended with `DUE TO TIME LIMIT`.
- No CUDA OOM, traceback, data/config error, or missing W&B run id was found
  among the non-`DONE_` attempts.
- The stale attempts were generally near completion, around
  `895k-990.5k / 1,000,000` optimizer steps.

### Resume Submission
- Generated resume manifest:
  `jobs/job_sets/mem_gen_boundary_capacity_20260427/resume_undone_20260501.tsv`.
- Generated capped resume wrapper:
  `jobs/job_sets/mem_gen_boundary_capacity_20260427/resume_undone_20260501.sbatch`.
- Submitted Slurm array job `16466870` (`mp_mgb_cap_resume`).
- Resume array details:
  - `52` tasks;
  - task ids:
    `15,16,17,18,19,20,21,24,25,26,27,28,29,30,31,32,33,34,35,50,52,53,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107`;
  - concurrency cap `%8`;
  - walltime `08:00:00`;
  - initial state `PENDING (Priority)`;
  - initial priority about `355382`.
- Task `24` was included intentionally: the scratch no-W&B full resume
  completed under `tmp/resume_full_20260430`, but the canonical capacity tree
  and canonical W&B run still needed a proper online resume/finalization.

### W&B Expectation
- All `52` source attempts have saved W&B run IDs.
- Resume code should pass each source run id into `wandb.init(...,
  resume="allow")`.
- Expected W&B behavior: existing runs in project
  `memparam-mem_gen_boundary_capacity_20260427`, group
  `mem_gen_boundary_capacity_20260427`, should receive appended metrics from
  their source checkpoint through step `1,000,000` and finish.
- Unexpected behavior to watch for: creation of unrelated duplicate W&B runs,
  W&B init warnings, or completed local attempts whose W&B run remains marked
  crashed.

### Monitor
- Queue:
  `squeue -j 16466870`.
- Logs:
  `jobs/job_sets/mem_gen_boundary_capacity_20260427/slurm_resume_20260501/`.

## 2026-04-30T02:17:06+02:00 - Full No-W&B Resume Test Submitted

### Context
We verified that the CelebA capacity-control runs that looked crashed in W&B
were SLURM walltime terminations, not Python/app failures: accounting showed
`TIMEOUT` at the `30:00:00` limit and stderr ended with `DUE TO TIME LIMIT`.
We then tested the resume mechanism on a duplicated copy of one timed-out
attempt.

### Resume Smoke Result
- Source task: capacity array task `24`, CelebA32, `N=256`, `base64_m1244`,
  `x/x`, seed `0`, config hash `1237b733a1`.
- Original attempt used for duplication:
  `outputs/job_sets/mem_gen_boundary_capacity_20260427/runs/mem_gen_boundary_capacity_20260427_celeba32_N256_B256_base64_m1244_x_x_steps1000000_s0-1237b733a1/seeds/seed_0000/attempts/20260428_012613`.
- Fresh duplicated smoke source:
  `tmp/resume_smoke_20260430/source_attempt_task24`.
- Smoke job `16463599` resumed from `step_945000` to `step_950000` with
  W&B disabled and completed successfully:
  - SLURM state `COMPLETED`, exit `0:0`, elapsed `00:11:32`.
  - Attempt state `trained`.
  - W&B status in the new attempt: `enabled=false`, `active=false`.
  - New checkpoint saved: `checkpoints/step_950000.pt`.
- Completed smoke output:
  `tmp/resume_smoke_20260430/outputs/runs/mem_gen_boundary_capacity_20260427_celeba32_N256_B256_base64_m1244_x_x_steps1000000_s0-1237b733a1/seeds/seed_0000/attempts/DONE_20260430_013543`.

### Full Resume Submitted
- Fresh duplicate for the full continuation:
  `tmp/resume_full_20260430/source_attempt_task24`.
- Runner:
  `tmp/resume_full_20260430/run_resume_task24_full.py`.
- Submit script:
  `tmp/resume_full_20260430/resume_task24_full_no_wandb.sbatch`.
- Submitted SLURM job: `16463631` (`mp_resume_full`).
- At journal time, `squeue`/`sacct` showed:
  - state `PENDING` / `PD (Priority)`;
  - timelimit `04:00:00`;
  - no node assigned yet;
  - no stdout/stderr logs yet.
- Expected logs once it starts:
  - `tmp/resume_full_20260430/slurm/mp_resume_full_16463631.out`;
  - `tmp/resume_full_20260430/slurm/mp_resume_full_16463631.err`.
- Expected first useful stdout lines:
  - `wandb_enabled=False`;
  - `resuming source_step=945000 target_step=1000000`;
  - `loaded resume checkpoint step=945000`;
  - `training started steps=945001..1000000`.

### Runtime Estimate
- Exact original task `24` reported at `step=945000`:
  `steps/s=9.18`, ETA `01:39:54`.
- It reached `step=990500` before the 30h walltime termination, with ETA
  `00:17:15`.
- The no-W&B smoke continuation took `660.48s` for `5,000` resumed steps
  (`0.1321s/step`), implying about `2h01m` for the remaining `55,000` steps.
- Practical expectation for the full resume: around `2h`, with a rough range
  of `1h40m` to `2h10m`; the requested `04:00:00` walltime should be safe.

### Notes
- The full resume should read from the duplicated source and write only under
  `tmp/resume_full_20260430/outputs`.
- The copied `source_attempt_task24/status.json` still contains stale absolute
  metadata paths pointing at the original attempt. The trainer's actual resume
  paths are derived from `--resume_from_attempt` and the duplicated
  `source_checkpoint`, so this should not make it write into the original
  folder. If desired later, scrub the duplicate status metadata for cosmetic
  cleanliness.
- Monitor with:
  `squeue -u x_moalj`
  and
  `tail -f tmp/resume_full_20260430/slurm/mp_resume_full_16463631.out`.

## 2026-04-29T04:20:21Z - Reminder: Parameterization as a Memorization Axis

### Context
As we turn the `tmp/eval_prototype_runs` notebook work into real measurements,
treat diffusion parameterization as a first-class experimental dimension for
memorization analysis. In this run family that means comparing `x/x`,
`eps/eps`, `v/v`, and `x/v` while holding the dataset, seed family, checkpoint
step, sampler, and memorization metric fixed where possible.

### Working Hypothesis
- Parameterization is not just a quality/convergence detail. It changes the
  prediction target and loss geometry, and can change target scale, timestep
  weighting, gradient flow, and which image details are easiest for the model to
  fit or reproduce.
- For our CelebA probes, report memorization by parameterization alongside
  train-vs-test loss. Do not only aggregate by `N`, architecture, or checkpoint.
- Keep the caveat explicit: the papers below strongly support other
  memorization axes, while the parameterization-to-memorization link is our
  local hypothesis to test rather than a settled claim.

### Axes To Track
- Parameterization / prediction space: `x`, `eps`, `v`, and mixed settings such
  as `x/v`.
- Model capacity: parameter count, U-Net width/depth, and effective
  over-parameterization.
- Conditionality: unconditional vs labels/prompts/other conditioning signals.
- Training time: checkpoint step, number of epochs/passes, and whether
  memorization appears before obvious overfitting.
- Dataset size and uniqueness: `N`, long-tail examples, outliers, and rare
  subpopulations.
- Duplicates and near-duplicates: exact repeats, near-copy clusters, and
  train-test overlap.
- Metric/eval space: raw pixels, random projection `f_mem`, feature space, and
  threshold choice.

### References
- Ho, Jain, and Abbeel, "Denoising Diffusion Probabilistic Models" (2020):
  https://arxiv.org/abs/2006.11239. Anchor for the baseline diffusion
  training setup and noise-prediction objective.
- Salimans and Ho, "Progressive Distillation for Fast Sampling of Diffusion
  Models" (ICLR 2022): https://openreview.net/forum?id=TIdIXIpzhoI. Anchor for
  diffusion parameterization choices such as velocity-style prediction affecting
  stability/training behavior.
- Zhang et al., "Understanding Deep Learning Requires Rethinking
  Generalization" (ICLR 2017): https://openreview.net/forum?id=Sy8gdB9xx.
  Anchor for capacity/over-parameterization and fitting/memorization behavior.
- Feldman, "Does Learning Require Memorization? A Short Tale about a Long Tail"
  (2020): https://arxiv.org/abs/1906.05271. Anchor for long-tail data,
  outliers, dataset support, and why memorization can be tied to generalization.
- Tirumala et al., "Memorization Without Overfitting: Analyzing the Training
  Dynamics of Large Language Models" (NeurIPS 2022):
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/fa0509f4dab6807e2cb465715bf2d249-Abstract-Conference.html.
  Anchor for training time, model size, dataset size, and memorization dynamics.
- Lee et al., "Deduplicating Training Data Makes Language Models Better" (ACL
  2022): https://aclanthology.org/2022.acl-long.577/. Anchor for duplicates,
  train-test overlap, and emitted memorized training content.
- Carlini et al., "Extracting Training Data from Diffusion Models" (USENIX
  Security 2023):
  https://www.usenix.org/conference/usenixsecurity23/presentation/carlini.
  Anchor for diffusion memorization/extraction and modeling/data decisions.
- Somepalli et al., "Diffusion Art or Digital Forgery? Investigating Data
  Replication in Diffusion Models" (CVPR 2023):
  https://openaccess.thecvf.com/content/CVPR2023/html/Somepalli_Diffusion_Art_or_Digital_Forgery_Investigating_Data_Replication_in_Diffusion_CVPR_2023_paper.html.
  Anchor for image retrieval style replication checks, CelebA/ImageNet/LAION
  settings, and training-set-size effects.

## 2026-04-29T04:13:50Z - Eval Notebook End-of-Day State

### Context
The immediate eval-prototyping goal narrowed to the local notebook under
`tmp/eval_prototype_runs`: compare train-vs-test loss using whole datasets, then
probe model memorization by comparing generated samples against the full saved
train set.

### Notebook State
- Notebook:
  `tmp/eval_prototype_runs/celeba_adam_N16384_s0_allparams/train_loss_curves.ipynb`.
- `tmp/` is ignored, so the notebook edits are local-only for now. This journal
  entry is the pushed record of what exists there.
- Sections currently present:
  - `Train Loss Curves`.
  - `Prototype Metric Loader`.
  - `Sample Grid`.
  - `Memorization Probe`.

### Full Train-vs-Test Loss
- The metric-loader cell now treats CelebA `test` as the real test set for this
  CelebA-only run family.
- It loads the full exact saved run train tensor: `16,384` images.
- It materializes/caches the CelebA test split under the ignored sandbox.
- It keeps the saved validation reference tensor only as provenance/context.
- `LOSS_EVAL_BATCH_SIZE=256` is a streaming chunk size, not a subsample size.
  Every train/test example is included; batching is only for memory.
- The output payload name is
  `prototype_single_checkpoint_full_train_vs_test_loss`.
- The summary includes:
  - `test_minus_train_random_t_loss`.
  - `test_over_train_random_t_loss`.
  - full train/test counts and the streaming batch size.

### Memorization Probe
- Added a new `Memorization Probe` cell.
- It generates or reuses `MEM_SAMPLE_COUNT=64` model samples from the saved
  `z_init` prefix.
- It compares each generated sample against every saved train image.
- The primary memorization statistic is nearest-neighbor ratio
  `r = d1 / d2`, thresholded as `r < rho`.
- It currently computes two spaces:
  - `raw_pixels`.
  - `random_projected_pixels`, using the run's cached projection matrix and
    cached projected train/reference vectors.
- It also computes a same-count CelebA test baseline against the train set, so
  generated `f_mem` can be interpreted relative to real held-out images.
- Intended outputs:
  - `metrics/prototype_memorization_probe_step_<step>_n<N>.json`.
  - `metrics/prototype_memorization_samples_step_<step>_n<N>.pt`.
  - `figures/prototype_memorization_nn_pairs_step_<step>_n<K>.png`.

### Notes For Next Session
- The notebook cells were syntax-checked but not executed in this shell because
  the shell Python did not have `torch` loaded. Run them in the project Jupyter
  kernel / `mem_param_env`.
- Next productive move: execute the metric-loader cell for `eps/eps` final
  checkpoint, then execute the memorization probe. If it behaves, loop over all
  four parameterizations and selected checkpoints.
- Decide later whether to promote this local notebook/prototype into tracked
  code under `scripts/eval/`, a tracked notebook under `notebooks/`, or both.

## 2026-04-29T03:35:22Z - Focused Eval Prototype Inventory

### Context
The thing to preserve from this morning is the local evaluation-prototyping
state under `tmp/eval_prototype_runs`, not only the Slurm/W&B job state. This
entry is the tighter reminder for that workstream.

### Local-Only Sandbox
- `tmp/` is ignored by `.gitignore`; do not `git add tmp/` unless we first
  extract small, intentional source files.
- `tmp/eval_prototype_runs` is about `14G`, mostly copied checkpoints and
  tensors. Treat it as a local scratch/eval artifact cache, not as a pushable
  repo payload.
- Active sandbox:
  `tmp/eval_prototype_runs/celeba_adam_N16384_s0_allparams`.
- Purpose from `SANDBOX_MANIFEST.json`: worst-case CelebA Adam `N=16384`,
  seed `0`, all parameterizations, for local evaluation prototyping.
- Copy style: minimal normal file copies, not hardlinks: artifacts,
  checkpoints, status, and train history copied; fresh metrics/figures/log
  space inside the sandbox.
- W&B is disabled in copied configs/status summaries. Any metric written inside
  this sandbox is a prototype metric, not an original run artifact.

### Copied Attempts
- Four completed `mem_gen_boundary_20260427` Adam CelebA32 attempts are copied:
  `x/x`, `eps/eps`, `v/v`, and `x/v`.
- All four local copied statuses are `trained`.
- Each attempt has `72` checkpoints, including `step_000000.pt` through
  `step_1000000.pt`, plus copied train/reference tensors and projection
  artifacts.
- Each `metrics/train_history.json` has `2001` rows from step `1` to
  `1000000`.
- Final copied train-history losses:
  - `x/x`: `0.06057939678430557`.
  - `eps/eps`: `0.07373814284801483`.
  - `v/v`: `0.18088839948177338`.
  - `x/v`: `0.18286867439746857`.

### Prep Script
- `tmp/eval_prototype_runs/prepare_celeba_adam_N16384_s0_allparams.py`
  creates/syncs the sandbox.
- It copies each source attempt with `rsync -a`, rewrites absolute paths in
  `status.json` and `artifacts/run_summary.json`, disables W&B in
  `artifacts/config.json`, syncs `metrics/train_history.json`, validates that
  paths point inside the sandbox, and writes `SANDBOX_MANIFEST.json` plus
  `PROTOTYPE_COPY.md`.
- It also supports `--sync-train-history-only` for refreshing histories without
  rebuilding the whole 14G sandbox.

### Notebook Prototype
- `tmp/eval_prototype_runs/celeba_adam_N16384_s0_allparams/train_loss_curves.ipynb`
  currently has three sections:
  - `Train Loss Curves`.
  - `Prototype Metric Loader`.
  - `Sample Grid`.
- The train-loss section discovers the four copied attempts and writes:
  - `figures/train_loss_curves.png` (`1280 x 800`).
  - `figures/train_loss_curves.svg`.
- The metric-loader section currently selects `eps/eps` at `step=1000000`,
  runs on CPU, and loads:
  - `train_tensor`: `(16384, 1, 32, 32)`.
  - `reference_tensor`: `(1024, 1, 32, 32)`.
  - `z_init`: `(16384, 1, 32, 32)`.
  - Model class `GDFM` with `3,917,409` parameters.
- The single-checkpoint metric smoke writes
  `metrics/prototype_single_checkpoint_metric.json` inside the copied
  `eps/eps` attempt. Preview values:
  - reference random-t loss: `0.08594270050525665`.
  - train random-t loss: `0.05378539487719536`.
  - reference-minus-train random-t loss: `0.032157305628061295`.
  - fixed `t=0.5`: reference `0.038509249687194824`, train
    `0.028474632650613785`.
  - fixed `t=0.75`: reference `0.11610203981399536`, train
    `0.06398393213748932`.
- The sample-grid smoke uses `eps/eps`, `step=1000000`, Heun sampling with
  `50` steps, `SAMPLE_COUNT=40`, `SAMPLE_BATCH_SIZE=10`, and writes:
  `figures/prototype_samples_eps_eps_step_1000000_n40.png`
  (`2250 x 1216`). An earlier `n2` grid also exists.

### Next Eval Prototype Steps
- Keep the 14G sandbox local and ignored; promote only durable code/notes out
  of it when the workflow stabilizes.
- Turn the notebook metric smoke into a loop over all four parameterizations
  and selected checkpoints, then write a compact comparison table/figure.
- Decide whether the tracked home should be `scripts/eval/`, a notebook under
  `notebooks/`, or both. The prep script and notebook are the likely source
  material to extract.
- Keep W&B disabled for sandbox evaluation until the metric definition is
  settled; then decide whether to log final eval artifacts back to the real
  runs or to a separate eval-only group.

### Commands / Checks
- `find tmp/eval_prototype_runs -maxdepth 5 ...`
- `du -sh tmp/eval_prototype_runs`
- `jq` over `SANDBOX_MANIFEST.json`
- `jq` over `train_loss_curves.ipynb` sections, code cells, and outputs
- `find tmp/eval_prototype_runs/... -name train_history.json ...`
- `file` on generated PNG/SVG artifacts

## 2026-04-29T03:18:27Z - Morning Checkup: Crashed Capacity Runs and Eval Prototypes

### Context
The W&B runs view showed several `mem_gen_boundary_capacity_20260427` rows as
`Crashed`, so I re-checked the repo, journal, Slurm queue/accounting, local
status files, and the ignored evaluation-prototyping sandboxes.

### Current Repo State
- Branch is `main` at `bf8425d` / `origin/main`.
- The worktree is intentionally broad and dirty. Notable groups:
  - Modified tracking/config/metric/trainer files and tests.
  - Deleted legacy top-level job scripts under `jobs/`.
  - Untracked job-set folders, logs, notebooks, scripts, `.codex`,
    `scratch.py`, `tea_debug.log`, and `tests/test_visualization.py`.
- This entry is journal-only; avoid accidentally staging unrelated dirty-tree
  work when pushing reminders.

### Current Job State
- Active queue: capacity array `16443588` plus interactive job `16458204`.
- Capacity array `16443588` is now `40 completed / 61 running / 7 timed out`.
  Task `16443588_22` completed during this check.
- The seven W&B `Crashed` rows are Slurm `TIMEOUT`, not fresh application
  exceptions. They are tasks `15-21`:
  - `base64_m124`, `N=256`, `x/v`, seed `0`.
  - `base64_m124`, `N=256`, `x/x`, seed `1`.
  - `base64_m124`, `N=256`, `eps/eps`, seed `1`.
  - `base64_m124`, `N=256`, `v/v`, seed `1`.
  - `base64_m124`, `N=256`, `x/v`, seed `1`.
  - `base64_m124`, `N=256`, `x/x`, seed `2`.
  - `base64_m124`, `N=256`, `eps/eps`, seed `2`.
- Those timed-out attempts ended around `967.5k-975k / 1e6` training steps,
  roughly `46-60` minutes short. Slurm killed them at the 30h walltime, so
  local `status.json` files still say `training`; trust Slurm/W&B over those
  stale status files until they are resumed or reconciled.
- The remaining running capacity tasks are in ranges `23-35`, `48-71`, and
  `84-107`. Several are still likely to time out: for example task `107` was
  near `898k / 1e6` with ETA about `3h06m` while the Slurm elapsed clock was
  already past `27h`.
- Other known states remain:
  - Adam CelebA boundary sweep `16443304` / `mem_gen_boundary_20260427`:
    `84/84` complete.
  - SGD+momentum sweep `16443527`: diagonal tasks complete; started `x/v`
    tasks cancelled after NaNs and local statuses are stale `training`.
  - Lower-LR SGD+momentum smoke `16443574_3`: complete.
  - Historical FFHQ/Flowers/sanity failures are still visible in accounting
    and local statuses; no new active jobs from those sets were found.

### Eval Prototype Sandboxes
- Repo-local ignored sandbox:
  `tmp/eval_prototype_runs/celeba_adam_N16384_s0_allparams`.
  It is a minimal copy of four completed Adam CelebA32 `N=16384`, seed `0`
  attempts: `x/x`, `eps/eps`, `v/v`, and `x/v`.
- The sandbox copies are normal files, not hardlinks; W&B is disabled in copied
  configs. Do not treat metrics written there as original run artifacts.
- Each copied attempt has `72` checkpoints, plus copied status/artifact files
  and fresh local metrics/figures/log space.
- Prototype files currently present:
  - `SANDBOX_MANIFEST.json` and `PROTOTYPE_COPY.md`.
  - `prepare_celeba_adam_N16384_s0_allparams.py`.
  - `train_loss_curves.ipynb`.
  - `figures/train_loss_curves.svg` and `figures/train_loss_curves.png`.
- Additional `/tmp` evaluation/prototype leftovers checked:
  - `/tmp/train_loss_curves.validated.json`: validated notebook JSON for the
    train-loss-curves workflow.
  - `/tmp/synthetic_train_runner_ipynb.json` and
    `/tmp/image_train_runner_ipynb.json`: runner notebook prototypes.
  - `/tmp/synthetic_runner_help.txt` and `/tmp/image_runner_help.txt`: captured
    runner CLI help.
  - `/tmp/eval_e1_per_timestep.py`: older ResNKF E1 per-timestep test-MSE
    plotting script, outputting `scripts/eval/e1_test_mse_per_timestep`.
  - `/tmp/attack_generator_smoke`: contains generator/phase checkpoints and
    `offline_eval/summary.json` with smoke metrics
    `attack_success_rate=1.0`, `phase_specificity=1.0`, and zero perturbation
    norms/deviations.
  - `/tmp/ag_eval_help.txt` and `/tmp/atk_eval_help.txt`: captured
    `offline_eval.py --help`; the newer `ag` version includes
    `--attack-surface {image,state}` and
    `--diffusion-grad-last-n-steps`.

### Reminders
- After `mem_gen_boundary_capacity_20260427` stops moving, decide whether to
  resume timed-out tasks `15-21` and any new timeouts using:
  `RESUME_FROM_ATTEMPT=/path/to/attempt sbatch --array=<task_id> jobs/job_sets/mem_gen_boundary_capacity_20260427/submit.sh`.
- Do not launch the lower-LR full SGD+momentum array or capacity resumes
  without an explicit compute decision.
- Keep surfacing the larger-model fill-in decision until resolved: after the
  capacity sweep finishes, decide whether to add larger-model `N=512,1024,4096,8192`
  points or accept the current `256/2048/16384` subset.
- Later cleanup should reconcile stale local `training` statuses caused by
  Slurm cancellation/timeouts so the local status summary stops under-reporting
  crashes.

### Commands / Checks
- `git status --short --branch`
- `tail -n 80 CODING_JOURNAL.md`
- `squeue -u x_moalj`
- `sacct -S 2026-04-25 -u x_moalj --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS --parsable2`
- `sacct -j 16443588 --format=JobID,State,Elapsed --parsable2`
- `find tmp/eval_prototype_runs ...`
- `find /tmp ...`
- `jq` status summaries over `outputs/job_sets/*/status.json`

## 2026-04-28T13:46:18Z - Daily Option: Larger-Model CelebA N-Sweep Fill-In

### Context
During the crash/rerun review, we checked whether the desired CelebA32
`n_train` sweep with larger UNets already exists locally.

### Observed
- The original CelebA32 Adam boundary sweep, `mem_gen_boundary_20260427`,
  already completed the full `N={256,512,1024,2048,4096,8192,16384}` grid for
  `x/x`, `eps/eps`, `v/v`, and `x/v` with 3 seeds using the baseline
  `base32_m124` UNet (`3.9M` params).
- The active capacity-control sweep,
  `mem_gen_boundary_capacity_20260427`, already covers a sensible subset for
  larger models: `N={256,2048,16384}`, all four parameterizations, 3 seeds,
  and model labels `base32_m124`, `base64_m124`, and `base64_m1244`.
- The two larger models in that sweep are about `15.6M` params
  (`base64_m124`) and `25.8M` params (`base64_m1244`).

### Daily Option Until Decided
- Keep bringing up this option in morning checkups until explicitly decided:
  after `mem_gen_boundary_capacity_20260427` finishes, decide whether the
  three-point larger-model subset is enough or whether to launch a fill-in
  sweep for the missing larger-model N points: `512`, `1024`, `4096`, and
  `8192`.
- Recommended default: wait for the active capacity sweep to finish first.
  Only fill in the missing N points if the `256/2048/16384` results show a
  moving boundary or otherwise leave the larger-model story ambiguous.

## 2026-04-27T02:47:12Z - Lower-LR SGD+Momentum x/v Smoke Queued

### Context
The Bonnaire SGD settings (`lr=0.01`, momentum `0.95`) were stable for the
diagonal parameterizations but caused systematic NaNs for the `x/v` slice. We
created a lower-learning-rate sibling set to test whether this is an LR-driven
stability issue.

### Changed / Observed
- Added `jobs/job_sets/mem_gen_boundary_sgdm_lr0p001_20260427`.
- The new job set mirrors the 84-row SGD grid but uses `lr=0.001`.
- It writes to a distinct output/group identity:
  `mem_gen_boundary_sgdm_lr0p001_20260427`.
- It intentionally uses the same W&B project as the `lr=0.01` SGD set:
  `memparam-mem_gen_boundary_sgdm_20260427`.
- Run names include `sgd_m0p95_lr0p001` to avoid collision with the
  `sgd_m0p95_lr0p01` runs.
- Submitted an offline smoke for task `3`, the smallest `x/v` run, as Slurm job
  `16443574`.
- The smoke is currently pending on Slurm priority.

### Commands / Checks
- `bash -n jobs/job_sets/mem_gen_boundary_sgdm_lr0p001_20260427/submit.sh`
- Manifest validation: 84 rows, fixed `optimizer=sgd`, `lr=0.001`,
  `weight_decay=0.0`, `momentum=0.95`.
- Smoke submit:
  `WANDB_MODE=offline N_TRAINING_STEPS=2000 CKPT_SCHEDULE=interval CKPT_EVERY=500 sbatch --array=3 jobs/job_sets/mem_gen_boundary_sgdm_lr0p001_20260427/submit.sh`.

### Next
- Wait for smoke job `16443574_3` to run.
- Accept the smoke only if it reaches `2000/2000`, confirms `lr=0.001`,
  writes checkpoints/monitor figures, and has no NaN/error signatures.
- Submit the full lower-LR array only after this `x/v` smoke passes.

## 2026-04-27T02:36:43Z - SGD+Momentum Array Launched; x/v Cancelled

### Context
After the required offline smoke passed, launched the full CelebA32
SGD+momentum sister array to anchor the Bonnaire optimizer comparison.

### Changed / Observed
- Submitted `jobs/job_sets/mem_gen_boundary_sgdm_20260427/submit.sh` as Slurm
  array job `16443527`.
- Early logs confirmed online W&B logging, run names containing
  `sgd_m0p95_lr0p01`, and the intended optimizer settings:
  `optimizer=sgd`, `lr=0.01`, `momentum=0.95`, `weight_decay=0.0`.
- The `x/v` tasks produced systematic `loss=nan` early under SGD+momentum,
  while diagonal tasks (`x/x`, `eps/eps`, `v/v`) remained finite.
- Cancelled only the `x/v` slice to avoid wasting GPU time:
  tasks `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79,83`.
- Left the diagonal SGD comparisons running; the current Slurm array still has
  active/pending non-`x/v` tasks.

### Commands / Checks
- `sbatch jobs/job_sets/mem_gen_boundary_sgdm_20260427/submit.sh`
- `squeue -u x_moalj -j 16443527 -o "%.200i %.2t %.10M %.40R"`
- Grep/rg scan of SGD logs for `loss=nan`, `loss=inf`, traceback, OOM,
  runtime error, and failed markers.
- Filtered the log scan to non-`x/v` tasks after cancellation.

### Results
- Full array submission succeeded as job `16443527`.
- The instability is specific to the launched `x/v` slice so far.
- No NaN/inf/runtime markers were found in the remaining diagonal task logs.
- Latest diagonal task losses were finite, with early tasks around
  `8500-9500/1000000` steps and newer tasks around `1500-2000/1000000`.

### Next
- Continue monitoring the diagonal SGD runs.
- Treat `x/v` with Bonnaire SGD settings as unstable until separately tuned
  or explicitly documented as failing at `lr=0.01`, momentum `0.95`.

## 2026-04-27T02:05:40Z - SGD+Momentum Smoke Passed

### Context
Ran the required offline smoke task before considering the full SGD+momentum
CelebA32 sister array.

### Changed / Observed
- Submitted `jobs/job_sets/mem_gen_boundary_sgdm_20260427/submit.sh` as Slurm
  job `16443452`, array task `0`, with:
  `WANDB_MODE=offline N_TRAINING_STEPS=2000 CKPT_SCHEDULE=interval CKPT_EVERY=500`.
- Slurm accounting reports `COMPLETED` with exit code `0:0`.
- Stdout confirms `optimizer=sgd`, `lr=0.01`, `momentum=0.95`,
  `weight_decay=0.0`, and run name containing `sgd_m0p95_lr0p01`.
- Training reached `2000/2000`; the attempt was renamed to `DONE_20260427_035957`.
- Checkpoints were written at steps `0`, `500`, `1000`, `1500`, and `2000`.
- Generation monitor figures and `training_loss_vs_step.png` were written.
- Status and run summary record `state=trained`, `optimizer=sgd`, `lr=0.01`,
  `momentum=0.95`, and `completed_step=2000`.

### Commands / Checks
- `sacct -j 16443452 --format=JobID,JobName%16,State,Elapsed,Timelimit,ExitCode -P`
- Grep of smoke stdout/stderr for traceback, OOM, failed, cancellation,
  runtime error, missing-file, `nan`, and `inf` signatures.
- Artifact checks for expected checkpoint and figure files.

### Results
- Smoke passed. No error signatures were found in stdout/stderr.

### Next
- The full SGD+momentum 84-task array is ready to submit when compute budget is
  approved:
  `sbatch jobs/job_sets/mem_gen_boundary_sgdm_20260427/submit.sh`.

## 2026-04-27T01:50:32Z - SGD+Momentum CelebA32 Sister Sweep

### Context
Reviewer-facing concern: Bonnaire's main CelebA results use SGD with momentum,
while the active `mem_gen_boundary_20260427` sweep uses Adam. We need a direct
optimizer-control sister set that anchors the central comparison to Bonnaire's
optimizer.

### Changed / Observed
- Added public image-runner optimizer controls: `--optimizer`, `--lr`,
  `--weight_decay`, and `--momentum`.
- Preserved existing image defaults: Adam, `lr=1e-4`, `weight_decay=0.0`, and
  default momentum `0.9`.
- Added optimizer, learning-rate, weight-decay, and momentum fields to status
  files, W&B tags, and local run summaries.
- Added `jobs/job_sets/mem_gen_boundary_sgdm_20260427` as an 84-task mirror of
  the active Adam CelebA32 boundary grid, with SGD, `lr=0.01`, momentum `0.95`,
  `weight_decay=0.0`, and `batch_cap=256`.
- Documented the Bonnaire capacity sanity check: our active CelebA32 UNet has
  `3,917,409` trainable parameters; vendored Bonnaire `run_Unet.py -W 32` has
  `4,055,233` total and `4,023,233` trainable parameters.

### Commands / Checks
- `bash -n jobs/job_sets/mem_gen_boundary_sgdm_20260427/submit.sh`
- Manifest validation: 84 data rows, 16 TSV fields, fixed SGD optimizer fields.
- Manifest comparison against `mem_gen_boundary_20260427`: 84 matched Adam-grid
  rows after accounting for the added optimizer columns.
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src:. python -m py_compile src/mp_runners/image_train_runner.py src/mp_runners/image_config.py src/mp_training/trainer.py src/mp_tracking.py src/mp_artifacts/artifacts.py`
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src:. python -m unittest tests.test_proto_runners tests.test_tracking`
- `git diff --check`

### Results
- Shell syntax, manifest validation/comparison, and byte-compile passed.
- Focused proto-runner and tracking tests passed: 51 tests.
- `git diff --check` passed.

### Next
- Run the required offline smoke task before submitting the full SGD array:
  `WANDB_MODE=offline N_TRAINING_STEPS=2000 CKPT_SCHEDULE=interval CKPT_EVERY=500 sbatch --array=0 jobs/job_sets/mem_gen_boundary_sgdm_20260427/submit.sh`.
- Submit the full array only after smoke stdout confirms `optimizer=sgd`,
  `lr=0.01`, `momentum=0.95`, and reaches `2000/2000` without error markers.

## 2026-04-27T00:13:44Z - Mem-Gen-Boundary Submitted And Sanity Checked

### Context
After drafting the CelebA32 mem-gen-boundary sweep, we removed the Slurm array
concurrency cap and launched the full 84-run job array.

### Changed / Observed
- Submitted `jobs/job_sets/mem_gen_boundary_20260427/submit.sh` as Slurm job
  `16443304` with array range `0-83`.
- Kept the job unconstrained by node class so Slurm can allocate any suitable
  Berzelius GPU node.
- Confirmed all 84 tasks reached running state and created status files.
- Confirmed all 84 status files reported `training`, active W&B, and 72
  resolved Bonnaire checkpoints.
- Checked stdout/stderr for `Traceback`, runtime errors, CUDA OOM, failed
  markers, and exception markers; none were present.

### Next
- Monitor the long-running array in W&B and Slurm.
- If a task times out or is killed, resume it with `RESUME_FROM_ATTEMPT` and
  the same array task id.

## 2026-04-26T23:56:12Z - Mem-Gen-Boundary CelebA32 Sweep Draft

### Context
We defined the first core CelebA32 grayscale boundary sweep for long
memorization/generalization runs. The goal is to compare training-set size,
parameterization, and seed while keeping the data recipe and optimizer-step
horizon fixed.

### Changed / Observed
- Added `jobs/job_sets/mem_gen_boundary_20260427` with a Slurm array script,
  README, and 84-row manifest.
- Fixed the run family to `dataset=celeba`, `image_size=32`, the internal
  `bonnaire_celeba32` recipe, `n_test=1024`, `batch_cap=256`, and
  `n_training_steps=1000000`.
- Swept `N_train` over `256, 512, 1024, 2048, 4096, 8192, 16384`, seeds
  `0, 1, 2`, and parameterizations `x/x`, `eps/eps`, `v/v`, and `x/v`.
- Kept `ckpt_schedule=bonnaire-1m` as the active checkpointing policy, with
  `ckpt_every=5000` available only when switching to interval checkpointing.
- Left node class unconstrained so Slurm can place jobs on any suitable GPU.
- Exposed `RESUME_FROM_ATTEMPT`, `RESUME_FROM_CHECKPOINT`, and
  `STOP_AFTER_STEP` environment overrides for manual continuation after
  walltime or interruption.

### Next
- Syntax-check the new submit script and confirm the manifest has 84 tasks.
- Optionally submit a single offline smoke task before launching the full
  online array.

## 2026-04-25T05:32:01Z - Parallel Slurm Job Set Fixes

### Context
We started moving from one-off GPU smoke commands to Slurm job sets that keep
related runs in one output root and one W&B project. The first fully parallel
Flowers64 sweep exposed two parallel-launch issues before any result sweep
should be trusted.

### Issue
- Slurm copies submitted scripts into its internal job directory. Computing the
  repository root from `${BASH_SOURCE[0]}` made jobs try to write `outputs/`
  under `/var/lib/slurm/...`, which failed with permission errors.
- Running all array tasks in parallel made many attempts start in the same
  second. The timestamp-only attempt id was reused as the W&B backend run id,
  so different parameterizations could upload monitor figures into the same W&B
  run.
- The same parallel launch also let multiple tasks read/write shared manifest
  JSON files concurrently, producing corrupted `experiment_manifest.json` reads.

### Changed / Observed
- Added dedicated job sets under `jobs/job_sets/`:
  - `image_sanity_20260425`
  - `synthetic_sanity_20260425`
  - `flowers64_n_sweep_20260425`
- The Flowers64 set fixes `dataset=flowers,image_size=64` and sweeps
  `N_train` across `128, 256, 512, 1024, 2048, 4096` with diagonal
  `x/x`, `eps/eps`, and `v/v` parameterizations.
- Submit/debug scripts now use the fixed repository path
  `/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par` and
  absolute Slurm log paths, with `MP_REPO_ROOT` as an override.
- JSON artifact writes are now protected by per-file advisory locks and
  atomic temp-file replacement.
- W&B backend run ids now include `config_hash`, `seed_id`, and `attempt_id`
  instead of only the timestamp attempt id.
- Existing human-readable W&B run names still include the dataset, image size,
  model, `N_train`, batch cap, parameterization, steps, and seed.
- Added ignores for Slurm `.out/.err` files plus JSON lock/temp files.

### Commands / Checks
- `python -m unittest tests.test_artifact_layout tests.test_tracking`
- `python -m py_compile src/mp_artifacts/artifacts.py src/mp_tracking.py`
- `bash -n jobs/job_sets/image_sanity_20260425/submit.sh jobs/job_sets/synthetic_sanity_20260425/submit.sh jobs/job_sets/flowers64_n_sweep_20260425/submit.sh jobs/job_sets/image_sanity_20260425/debug_one.sh jobs/job_sets/synthetic_sanity_20260425/debug_one.sh jobs/job_sets/flowers64_n_sweep_20260425/debug_one.sh`
- `git diff --check`

### Results
- Artifact-layout and tracking tests passed: 10 tests.
- Byte-compile passed for the artifact and tracking modules.
- Job scripts passed shell syntax checks.
- `git diff --check` passed.

### Important Note
The W&B runs and local outputs from the failed Flowers64 array job
`16434340` are not clean research artifacts. They may contain mixed monitor
media because several parallel tasks shared timestamp-only W&B ids. The clean
path is to cancel/ignore that array, move the partial output directory aside,
and resubmit after this fix.

### Next
- Re-run the Flowers64 job set and verify that each W&B run id/name pair maps
  to exactly one parameterization.
- Submit the image and synthetic sanity job sets after the Flowers64 set is
  stable for a few minutes.

## 2026-04-25T02:34:33Z - FM-Only Experiment Scope

### Context
Before launching larger NeurIPS-facing sweeps, clarified whether current
runner outputs should be treated as FM, VP, or DDPM experiments.

### Intent
Keep the active paper story clean: current image and synthetic train/eval
runs are FM runs with deterministic ODE sampling, while VP/DDPM-like code stays
as dormant lower-level infrastructure unless explicitly promoted later.

### Changed / Observed
- Documented that image and synthetic train runners hard-code
  `config.framework.family = "fm"`.
- Documented that image generation uses deterministic Heun sampling from the
  saved fixed `z_init` bank, and synthetic generation exposes only Euler/Heun.
- Marked `vp_sde` / VP reverse-SDE helpers in `gdfm.samplers` as tests/examples
  infrastructure, not part of normal train/eval runs.
- Clarified that Bonnaire-close CelebA32/Image32 means model/data recipe
  compatibility, not Bonnaire's integer-timestep DDPM objective or sampler.

### Commands / Checks
- `git diff --check`

### Results
- Docs-only whitespace check passed.

### Next
- Proceed with FM experiment sweeps.

## 2026-04-25T01:45:22Z - Pre-Submission Train/Eval Runner Split

### Context
Cleaned up runner boundaries before deleting local outputs and starting larger
NeurIPS-facing training sweeps.

### Intent
Make train runners train-only, make checkpoint evaluation replay from saved
artifacts, and fix reproducibility issues that could silently contaminate
comparisons across parameterizations and runs.

### Changed / Observed
- Hard-renamed `mp_runners.image_runner` to `mp_runners.image_train_runner`
  and `mp_runners.synthetic_runner` to `mp_runners.synthetic_train_runner`
  with no compatibility wrappers.
- Added `mp_runners.eval_synthetic --run_dir ...` for synthetic checkpoint
  metric replay and removed synthetic inline eval flags/paths.
- Seeded training through shared trainer setup so `training.seed` controls
  torch CPU/CUDA RNGs, model init, DataLoader order, training noise/t sampling,
  and a fixed saved `z_init` bank.
- Added `evaluation.sample_bank_size` / `--eval_sample_bank_size` with default
  `1024`; `z_init` now covers eval plus monitor requirements without changing
  step-0 model initialization.
- Removed public image `--config_preset`; image architecture/data profiles are
  derived internally from `--dataset` and `--image_size`.
- Marked `dataset=celeba,image_size=32` runs with `data.recipe =
  bonnaire_celeba32`.
- Moved saved train/reference leakage checks into a shared eval helper used by
  image eval, synthetic eval, and generic checkpoint eval.
- Updated docs, notebooks, and tests for the new train/eval module names and
  underscore-only public CLI.

### Commands / Checks
- `PYTHONPATH=src:. python -m py_compile src/mp_runners/image_train_runner.py src/mp_runners/synthetic_train_runner.py src/mp_runners/eval_synthetic.py src/mp_runners/image_config.py src/mp_runners/evaluate_checkpoints.py src/mp_training/trainer.py src/mp_visualization.py src/mp_training/checkpoint_schedule.py tests/test_proto_runners.py tests/test_torus_synthetic.py tests/test_live_tree_boundaries.py`
- `PYTHONPATH=src:. python -m unittest tests.test_proto_runners`
- `PYTHONPATH=src:. python -m unittest tests.test_torus_synthetic`
- `PYTHONPATH=src:. python -m unittest tests.test_live_tree_boundaries`
- `PYTHONPATH=src:. python -m unittest tests.test_checkpoint_schedule`
- `PYTHONPATH=src:. python -m unittest tests.test_tracking`
- `PYTHONPATH=src:. python -m unittest tests.test_mp_metrics`
- `PYTHONPATH=src:. python -m unittest tests.test_src_examples`
- `PYTHONPATH=src:. python -m unittest tests.test_checkpoint_previews`
- `PYTHONPATH=src:. python -m unittest tests.test_data_leakage_guards`
- `PYTHONPATH=src:. python -m unittest tests.test_experiment_pipeline`
- `python -m json.tool prototyping/runners/notebooks/image_train_runner.ipynb`
- `python -m json.tool prototyping/runners/notebooks/synthetic_train_runner.ipynb`

### Results
- Focused checks passed.
- Same-seed tiny CPU train runs now reproduce `z_init`, step-0 weights, and
  final checkpoint weights.
- Changing eval bank size no longer perturbs step-0 model initialization.
- Tests still emit PyTorch `torch.load(weights_only=False)` FutureWarnings.

### Next
Push this state to `origin/main`, then remove local generated outputs and tmp
artifacts separately.

## 2026-04-24T22:05:38Z - Image Runner API Cleanup And Projected f_mem Monitor

### Context
Cleaned up the canonical image training runner before longer image sweeps.
The user wanted dataset names to be public and simple, evals separated from
training, and projected f_mem treated as a train-time sanity monitor.

### Intent
Make the image runner a clean training API: users type datasets like `flowers`
or `ffhq`, choose resolution with `--image_size`, and get checkpoint-time
monitoring without mixing full checkpoint evaluation into the train runner.

### Changed / Observed
- Public image datasets are now `mnist`, `celeba`, `flowers`, `afhq`, and
  `ffhq`; legacy loader/profile names remain internal and are rejected at the
  CLI.
- Replaced `data.n_eval` with `data.n_test` and training horizon
  `training.n_steps` with `training.n_training_steps`; kept
  `sampling.n_steps` unchanged.
- Converted the image runner to underscore flags such as `--run_name`,
  `--image_size`, `--n_test`, `--n_training_steps`, and `--ckpt_every`.
- Removed image-runner final-eval and compare-numerics paths.
- Renamed `generation_trace` to `generation_monitor` across config, trainer,
  artifacts, tracking, docs, and tests.
- Added always-on image train-time `random_projected_f_mem` monitor with saved
  projection matrix/metadata and `monitor/random_projected_f_mem_*` logging.
- Hard-renamed metrics modules to `mp_metrics.metrics_images` and
  `mp_metrics.metrics_synthetic`.

### Commands / Checks
- `PYTHONPATH=src:. python -m unittest tests.test_proto_runners tests.test_experiment_pipeline tests.test_checkpoint_schedule tests.test_tracking tests.test_checkpoint_previews tests.test_artifact_layout tests.test_src_examples tests.test_mp_metrics tests.test_torus_synthetic`
- `PYTHONPATH=src:. python -m py_compile src/mp_runners/image_runner.py src/mp_runners/image_config.py src/mp_training/trainer.py src/mp_tracking.py src/mp_runners/image_eval.py tests/test_proto_runners.py tests/test_experiment_pipeline.py tests/test_checkpoint_previews.py tests/test_artifact_layout.py tests/test_tracking.py`
- `PYTHONPATH=src:. python -m mp_runners.image_runner --help`
- `git diff --check`

### Results
- Focused test suite passed: 83 tests.
- Help output now shows only clean image datasets and underscore image flags.
- `git diff --check` passed.

### Next
Use the cleaned image runner for short smoke jobs, then submit larger training
sweeps and run full checkpoint evaluation separately on completed attempts.

## 2026-04-22T01:36:40Z - Image Data Bundle Gaps

### Context
The user asked to address missing image data-bundle support around Flowers102
64px, count validation, and a user-facing data summary.

### Changed / Observed
- Added `flowers102_64` as a public image runner dataset key.
- Added `image32_64` as the matching UNet preset.
- Added config-time count validation for known dataset splits.
- Added `--data-summary` to materialize and print the selected data bundle
  without training.

### Commands / Checks
- `python -m py_compile src/mp_runners/image_config.py src/mp_runners/image_presets.py src/mp_runners/image_runner.py tests/test_proto_runners.py`
- `python -m unittest tests.test_proto_runners`
- `python -m mp_runners.image_runner --help`
- `git diff --check`

### Results
- Runner tests passed: 29 tests.
- Help now lists `flowers102_64`, `image32_64`, and `--data-summary`.

### Next
- Use `--data-summary` before long image runs when checking subset sizes and
  train/reference split choices.

## 2026-04-22T01:28:56Z - Short Response Preference

### Context
The user clarified that responses should not be long-winded.

### Changed / Observed
- Updated `AGENT.md` and `CLAUDE.md` communication rules.
- New rule: keep answers short by default, answer first, and add detail only
  when requested or needed to avoid ambiguity.

### Commands / Checks
- `rg -n "short|long|slang|research|wording|terminology|communication|preference" AGENT.md CLAUDE.md CODING_JOURNAL.md .codex 2>/dev/null`

### Results
- The short-answer preference is recorded with the existing research wording
  preference.

### Next
- Use short answers by default.

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

## 2026-04-20T04:29:38Z - Make Image Runner Final Evaluation Replayable

### Context
For long 128px/256px runs, doing checkpoint metrics immediately after training
is not always worth the wall-clock cost because the run already saves all
artifacts needed to evaluate later.

### Intent
Keep training checkpoint-first and make evaluation an explicit choice: skip it
by default for long jobs, opt into it when desired, and allow metrics to be
replayed later from finished `DONE_...` attempts.

### Changed / Observed
- Added `--final-eval/--no-final-eval` to `image_runner`; the default is now
  `--no-final-eval`.
- Added `--eval-run-dir` to replay image checkpoint metrics from an existing
  finished attempt directory.
- Enforced the replay contract: the eval path accepts only `DONE_...` attempt
  folders with `status.json` state `trained` or `completed`, required saved-run
  artifacts, and at least one checkpoint.
- Kept training artifacts unchanged: config snapshot, train/reference tensors,
  `z_init.pt`, checkpoints, train history, and generation trace remain written
  during training.
- Kept immediate final evaluation available with `--final-eval`; replay eval
  uses the saved config and existing W&B run id.

### Commands / Checks
- `python -m unittest tests.test_proto_runners`
- `python -m unittest tests.test_checkpoint_previews tests.test_experiment_pipeline tests.test_loss_eval`
- `git diff --check`
- `python -m py_compile prototyping/runners/image_runner.py prototyping/runners/image_runner_config_builder.py prototyping/runners/metrics_image32.py tests/test_proto_runners.py`
- `python -m prototyping.runners.image_runner --help`

### Results
- `tests.test_proto_runners`: 22 tests passed.
- Checkpoint/experiment/loss-eval suite: 9 tests passed.
- Whitespace, byte-compile, and CLI help checks passed.
- Only existing `torch.load(weights_only=False)` FutureWarnings appeared.

### Next
- Run a tiny CUDA train-only job, then replay it with `--eval-run-dir` to verify
  W&B resume behavior in the UI.

## 2026-04-21T23:16:56Z - Status Check Synthetic Torus Worktree

### Context
Performed a read-only status pass across the repo, current journal, and active
synthetic-runner work before deciding what to touch next.

### Intent
Separate landed image-runner history from the current dirty synthetic/torus
work, verify whether the active tree is runnable, and record the chosen cleanup
policy.

### Changed / Observed
- Current branch is `main` at `fafa28b Make image runner final eval replayable`,
  tracking `origin/main`.
- Active dirty work is focused on controlled synthetic runs: `torus3d` data
  loading, synthetic final-eval replay from `DONE_...` attempts, canonical
  `train_run(...)` usage in `synthetic_mlp`, and dataset-neutral
  `intrinsic_*` metrics with compatibility aliases.
- The untracked `dunno/` notes outline a broader synthetic benchmark agenda
  around GMM versus torus, support distance, calibrated copy rate, PRDC/MMD/LID,
  and checkpoint-wise evaluation.
- Decision: skip synthetic notebook wording cleanup for now, leave `dunno/` and
  `prototyping/archive/` untouched as TBD, and only record this journal entry.

### Commands / Checks
- `git status --short --branch`
- `git log -5 --oneline --decorate`
- `git diff --stat`
- Read `README.md`, `AGENT.md`, `src/README.md`, `prototyping/runners/README.md`,
  `CODING_JOURNAL.md`, and the `dunno/` notes.
- `module load Mambaforge && mamba activate mem_param_env && MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src:. python -m unittest tests.test_torus_synthetic tests.test_proto_runners tests.test_projected_gmm_oracle`
- `git diff --check`
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src:. python -m py_compile src/mp_data/torus.py prototyping/runners/metrics_synthetic.py prototyping/runners/metrics_synthetic_controlled.py prototyping/runners/figures.py prototyping/runners/synthetic_mlp.py tests/test_torus_synthetic.py tests/test_proto_runners.py tests/test_projected_gmm_oracle.py`
- `module load Mambaforge && mamba activate mem_param_env && python -m json.tool prototyping/runners/notebooks/synthetic_mlp.ipynb`

### Results
- Focused synthetic/torus/prototype suite passed: 40 tests in 18.945 seconds.
- Whitespace check, byte-compile check, and notebook JSON validation passed.
- Only known `torch.load(weights_only=False)` FutureWarnings appeared.

### Blockers
- No blocker for the active synthetic/torus code path found in the focused
  checks.
- Root README and `src/README.md` still mention an `experiments/` surface that
  is not present in this checkout; this is separate documentation drift.

### Next
- Continue from the current dirty synthetic/torus worktree without polishing
  the synthetic notebook wording yet.
- Decide later whether `dunno/` and `prototyping/archive/` should become tracked
  source material, local references, or cleanup candidates.

## 2026-04-21T23:26:28Z - Proto Src Functionality Deep Dive

### Context
Mapped live functionality in `src/` and `prototyping/runners/` to prepare for a
later audit of whether tests protect real behavior or only cosmetic structure.

### Intent
Create a subsystem and flow handoff that names the canonical model core,
training/evaluation pipeline, synthetic and image runner surfaces, behavior
contracts, and current stale-test signals.

### Changed / Observed
- Added `PROTO_SRC_FUNCTIONALITY_DEEP_DIVE.md`.
- Documented `GDFM`, model setup/backbones, sampler conversions, config/data,
  training artifacts, status handling, checkpoint evaluation, runtime/tracking,
  image runner replay, and synthetic GMM/torus metrics.
- Included the active dirty `torus3d` path and synthetic replay work as live
  functionality.
- Left synthetic notebook wording, `dunno/`, and `prototyping/archive/`
  untouched.

### Commands / Checks
- `rg --files src prototyping/runners tests`
- `rg -n "^(def|class) " src prototyping/runners tests`
- Focused file reads across `src/gdfm`, `src/mp_*`, and
  `prototyping/runners`.
- `module load Mambaforge && mamba activate mem_param_env && MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src:. python -m unittest discover tests`

### Results
- Deep-dive handoff captures subsystem responsibilities, end-to-end flows,
  behavior contracts, risk areas for the later test audit, and evidence.
- Full unittest discovery currently ran 104 tests and failed with 3 failures
  and 3 errors.
- Failures/errors are from stale `prototyping/gdfm_components.py` tests,
  missing `experiments/`, and docstring/main-block policy checks.

### Blockers
- No blocker for creating the functionality map.
- The later test-quality audit should treat the current full-suite failures as
  stale/policy context before judging the rest of the suite.

### Next
- Classify tests by subsystem and by value: behavioral, smoke, contract,
  cosmetic, or stale/dead.

## 2026-04-22T00:02:40Z - Communication Preference

### Context
The user clarified the preferred communication style for all future work.

### Intent
Record the preference in repo-visible guidance so future assistants and
contributors use the correct wording.

### Changed / Observed
- Added a communication rule to `AGENT.md`.
- Added the same rule to `CLAUDE.md`.
- Preference: the user is a researcher; use simple wording and precise
  research terminology. Do not use slang, hype wording, playful labels, or
  informal phrases.
- Added explicit examples to the repo guidance:
  - Avoid: "Got it - let's clean this up."
    Use: "Understood. I will remove the unused metric fields and update the call sites."
  - Avoid: "This is a nice little smoke test."
    Use: "This is a short runtime test that checks whether the path executes without errors."
  - Avoid: "The code has some weird leftovers."
    Use: "The code still contains stale compatibility aliases from the previous metric interface."
  - Avoid: "This metric is doing oracle shit."
    Use: "This metric uses the known synthetic data-generating distribution."
  - Avoid: "The runner wires everything together."
    Use: "The runner constructs the config, launches training, and optionally runs checkpoint evaluation."
  - Avoid: "This is probably cosmetic test coverage."
    Use: "This test checks structure or naming, but does not verify behavioral correctness."
  - Avoid: "We can hard prune this."
    Use: "We can remove this interface without preserving backward compatibility."
  - Avoid: "The old names are noisy."
    Use: "The old names are ambiguous because they do not state the measured quantity or comparison set."
  - Avoid: "The figure plumbing is now aligned."
    Use: "The figure filenames, figure map keys, run summary entries, and W&B keys now use the same names."

### Commands / Checks
- Read `AGENT.md`, `CLAUDE.md`, and `CODING_JOURNAL.md`.
- Ran `date -u +%Y-%m-%dT%H:%M:%SZ` to timestamp this entry.

### Results
- The communication preference is now recorded in the repo guide and journal.

### Blockers
- None.

## 2026-04-22T03:56:49Z - Prepare Push Branch

### Context
The user asked to push the current work and ensure the journal is updated.

### Changed / Observed
- Prepared branch `codex/runner-metric-cleanup-and-image-resolver` from `main`.
- Staged the repo changes from the runner move, metric cleanup, checkpoint
  schedule work, W&B identity changes, and image dataset/size/model resolver.
- Left local-only `.codex` and `tea_debug.log` unstaged.
- `gh` is not installed in this environment, so a PR cannot be opened here.

### Commands / Checks
- `git status -sb`
- `git diff --stat`
- `git switch -c codex/runner-metric-cleanup-and-image-resolver`
- `git add -A`
- `git restore --staged .codex tea_debug.log`

### Results
- Branch is ready for commit and push.

### Blockers
- PR creation is blocked by missing `gh`; pushing the branch with `git` can
  still proceed.

## 2026-04-22T03:57:44Z - Pushed Runner Refactor Branch

### Context
Follow-up to the requested push.

### Changed / Observed
- Committed the staged runner/metric/config refactor as
  `e1319da refactor runners metrics and image config`.
- Pushed branch `codex/runner-metric-cleanup-and-image-resolver` to `origin`.
- Left local-only `.codex` and `tea_debug.log` untracked.

### Commands / Checks
- `git commit -m "refactor runners metrics and image config"`
- `git push -u origin codex/runner-metric-cleanup-and-image-resolver`

### Results
- Remote branch is available at:
  `origin/codex/runner-metric-cleanup-and-image-resolver`.
- GitHub suggested PR URL:
  `https://github.com/MohammedAlJaff/mem_par/pull/new/codex/runner-metric-cleanup-and-image-resolver`.

### Blockers
- PR creation remains blocked in this environment because `gh` is not installed.

## 2026-04-22T03:52:34Z - Image Runner Dataset Size Model Resolver

### Context
The user wanted image runs to be configured from only dataset, image size, and
model, with the remaining profile choices resolved automatically.

### Changed / Observed
- Added `--image-size` and `--model` to `mp_runners.image_runner`.
- Added image config resolution so `--dataset flowers102 --image-size 64
  --model unet` resolves to the `flowers102_64` profile.
- Kept older dataset profile keys such as `flowers102_128`.
- Added dynamic simple profiles for scalable datasets such as MNIST, so
  `--dataset mnist --image-size 64 --model unet` resolves to `mnist_64`.
- Kept programmatic custom image sizes valid for small tests and debugging.
- Updated runner docs and tests.

### Commands / Checks
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src:. python -m unittest tests.test_proto_runners`
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src:. python -m unittest tests.test_proto_runners tests.test_tracking`
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src:. python -m py_compile src/mp_runners/image_config.py src/mp_runners/image_runner.py tests/test_proto_runners.py`
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src:. python -m mp_runners.image_runner --help`
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src:. python -m unittest tests.test_src_docstrings tests.test_src_examples`
- Manual resolver check for Flowers102-64 and MNIST-64.
- `git diff --check`

### Results
- Focused runner/tracking tests passed: 41 tests.
- Docstring/example policy tests passed: 5 tests.
- Byte-compile and image runner help passed.
- Manual resolver check printed:
  - `flowers102_64 flowers102 64 unet 64`
  - `mnist_64 mnist 64 unet 64`
- `git diff --check` passed.

### Blockers
- None.

### Next
- Apply this communication style in all future responses and repo notes.

## 2026-04-22T00:30:44Z - Runner Move And Metric Simplification

### Context
The active prototype runners and runner metric code were promoted into `src`.
The user asked for a clean, simple layout before reducing duplicated code.

### Intent
Keep runner workflow code separate from metric definitions:
- metrics define what is measured;
- runners define how an experiment is run;
- small duplicated helper code is acceptable when it keeps files readable.

### Changed / Observed
- Added canonical runner modules under `src/mp_runners`.
- Simplified metric layout to `mp_metrics/loss_eval.py`,
  `mp_metrics/synthetic.py`, and `mp_metrics/images.py`.
- Moved generic checkpoint replay workflow to `mp_runners.evaluate_checkpoints`.
- Removed active prototype runner and metric source files from
  `prototyping/runners`.
- Updated tests, runner notebooks, and runner README to use `mp_runners` and
  canonical metric modules.

### Commands / Checks
- `rg -n "prototyping\\.runners|mp_metrics\\.(memorization|checkpoints|checkpoint_previews)" src tests`
- `python -m py_compile src/mp_metrics/*.py src/mp_runners/*.py`
- `python -m mp_runners.synthetic_mlp --help`
- `python -m mp_runners.image_runner --help`
- `python -m unittest discover tests`

### Results
- Static import grep passed for `src` and `tests`; only a negative assertion in
  `tests/test_proto_runners.py` still mentions the old image32 path.
- `python3.11 -m py_compile src/mp_metrics/*.py src/mp_runners/*.py` passed.
- `git diff --check` passed.
- Focused runner/metric tests ran 52 tests, all skipped because this shell does
  not have `torch`.
- Full unittest discovery ran 115 tests with 107 skips, 12 failures, and 1
  error. Failures are from missing local dependencies (`torch`, `numpy`,
  `ml_collections`), missing `experiments/`, and pre-existing docstring/main
  policy checks outside the moved runner files.

### Blockers
- None expected.

## 2026-04-27 - End-Of-Day Sweep Status And Same-UNet Pilot Plan

### Context
- Launched the CelebA32 Adam mem-gen-boundary baseline and prepared the
  capacity-control follow-up for checking whether the observed boundary is
  stable under larger UNet capacity.
- Discussed a Li-and-He-inspired MNIST image-size pilot where image size changes
  ambient dimension while the digit semantics remain roughly fixed.

### Job Status Snapshot
- Main Adam CelebA32 boundary job `16443304`: `84/84` running, healthy.
- Progress: `210000` to `228500` steps out of `1M`.
- Speed min/avg/max: `14.58 / 15.43 / 15.90` steps/s.
- Slurm log scan: `0` error hits.
- Capacity-control Adam sweep `16443588_[0-107]`: pending, reason `Priority`,
  with no local status files yet.
- SGDM sweep: `52` training, `1` trained, with no Slurm error hits.
- SGDM lr `1e-3`: pending, with no local status files yet.

### Same-Big-UNet MNIST Pilot Plan
- Dataset/settings: MNIST, image sizes `32`, `64`, and `256`, fixed
  `n_train=4096`, `n_test=1024`, `batch_cap=16`, Adam, TF32, and `1000`
  pilot training steps.
- Parameterizations for the pilot: `x/v` and `v/v`, seed `0`.
- Arm 1, exact fixed-capacity UNet: use the same roughly `108M` parameter UNet
  at all image sizes, with `base_channels=64` and
  `channel_mult=1,2,4,4,8,8`.
- Arm 2, bottleneck-matched UNets: use approximately `8x8` bottlenecks with
  `15.6M`, `25.8M`, and `108.4M` parameter UNets for `32`, `64`, and `256`
  respectively.
- Pilot size: `12` jobs.
- Decision rule: use the exact `108M` UNet for the full fixed-capacity MNIST
  image-size sweep if fit and speed are tolerable; otherwise use the
  bottleneck-matched arm and report parameter count as an explicit covariate.

### Checks
- Planned before commit: `bash -n` on the capacity submit script.
- Planned before commit: confirm the capacity manifest has `108` data rows.
- Planned before commit: run the targeted image-runner UNet override unit test.

### Next
- Re-check queue health tomorrow.
- If the capacity-control sweep starts cleanly, let it run behind the Adam
  baseline.
- Implement the 12-job MNIST same-big-UNet pilot as a separate job set.

## 2026-04-24T22:53:11Z - Image Runner W&B Monitor Cleanup

### Context
The image train runner was being cleaned up for larger experiment sweeps. The
training runner should stay focused on training and lightweight sanity monitors,
while full checkpoint evaluation remains separate.

### Intent
- Make public W&B identity include image size so mixed-resolution image sweeps
  do not land in the same project/run group.
- Remove the unused figure-logging switch and treat W&B media logging as always
  on when W&B itself is enabled.
- Keep image runs online by default.
- Make the train-time random projected f_mem monitor log only the dashboard
  signals that are useful during training.
- Avoid the final generation monitor panel colliding with checkpoint sample
  panels in W&B.

### Changed / Observed
- Default W&B project names now use `memparam-{dataset}-{image_size}-{model}`
  for image configs, for example `memparam-flowers-128-unet`.
- Default W&B run names and groups now include the image size for image configs.
- Removed `--wandb_log_figures`, `--no_wandb_log_figures`, and the stale
  `tracking.wandb.log_figures` image-runner config field.
- Added `std_r` to image memorization summaries, including projected image
  f_mem records.
- Random projected f_mem now logs only `f_mem`, `mean_r`, and `std_r` to W&B
  under the `monitor/` namespace; richer distance summaries remain in saved
  metric JSON for debugging.
- Final generation monitor panels now log to
  `generation_monitor/final_samples`; checkpoint panels stay under
  `generation_monitor/samples`.
- Updated runner docs and tests for the new W&B naming and monitor payloads.

### Commands / Checks
- `PYTHONPATH=src:. python -m unittest tests.test_proto_runners tests.test_tracking`
- `PYTHONPATH=src:. python -m unittest tests.test_mp_metrics tests.test_proto_runners tests.test_tracking`
- `git diff --check`

### Results
- Focused proto-runner and tracking tests passed: 43 tests.
- Focused metric, proto-runner, and tracking tests passed: 58 tests.
- `git diff --check` passed.

### Blockers
- None.

### Next
- Consider attaching the random projected f_mem projection matrix and metadata
  as W&B artifacts, not only local run artifacts.

## 2026-04-22T05:28:26Z - End-Of-Day Sweep And Tracking Polish

### Context
Before launching larger memorization sweeps, the runner output and job scripts
needed to be predictable enough to trust on Berzelius and in W&B.

### Intent
- Make automatic W&B project/name/group values reflect the final resolved
  config unless the user passes explicit W&B identity flags.
- Keep torus visual outputs focused on the figures needed for review.
- Add reproducible SLURM scripts for the synthetic GMM sweep and first image
  sweeps.
- Record data-count choices directly in the scripts so future runs do not
  rediscover dataset split limits the hard way.

### Changed / Observed
- Fixed automatic W&B identity resolution so changed objective flags such as
  `--prediction-space eps --loss-space eps` no longer leave stale `v_v` names.
- Logged final synthetic figures against the final checkpoint step to avoid W&B
  chart indexing issues.
- Simplified torus outputs to a train-set 3D scatter and per-checkpoint faded
  train plus generated 3D overlays; removed extra generic 2D intrinsic figures
  for torus runs.
- Added synthetic GMM SLURM scripts for the 72-job diagonal sweep over
  `N = 128..16384`, three seeds, and `x/x`, `eps/eps`, `v/v`.
- Added image SLURM scripts for MNIST32, CelebA32, and Flowers64 sweeps with
  one thin-node GPU per task.
- Checked actual image loaders and encoded the resulting count rules:
  - MNIST32 balanced reference cap is `8920`.
  - CelebA32 valid split supports `16384` reference examples for this sweep.
  - Flowers64 uses the large official test split for training and the small
    official train split, `1020` images, for held-out reference.

### Commands / Checks
- `python -m unittest tests.test_tracking`
- `python -m unittest tests.test_proto_runners`
- `python -m unittest tests.test_torus_synthetic`
- `python -m py_compile src/mp_tracking.py src/mp_runners/image_config.py src/mp_runners/synthetic_eval.py src/mp_visualization.py`
- `bash -n jobs/debug_synthetic_gmm_one_run.sh jobs/submit_synthetic_gmm_sweep.sh`
- `bash -n jobs/debug_image_mnist32_one_run.sh jobs/debug_image_celeba32_one_run.sh jobs/debug_image_flowers64_one_run.sh jobs/submit_image_mnist32_sweep.sh jobs/submit_image_celeba32_sweep.sh jobs/submit_image_flowers64_sweep.sh`
- `python -m mp_runners.image_runner --data-summary ...` for the largest
  MNIST32, CelebA32, and Flowers64 planned cases.
- Programmatic image config validation for every generated MNIST32, CelebA32,
  and Flowers64 array job.
- `git diff --check`

### Results
- Tracking, proto-runner, and torus synthetic tests passed.
- Byte-compile passed for touched Python files.
- All shell scripts passed Bash syntax checks.
- Image data summaries materialized the selected maximum cases:
  - MNIST32: train `[8192, 1, 32, 32]`, reference `[8920, 1, 32, 32]`.
  - CelebA32: train/reference `[16384, 1, 32, 32]`.
  - Flowers64: train `[4096, 3, 64, 64]`, reference `[1020, 3, 64, 64]`.
- Config validation passed for 63 MNIST32 jobs, 72 CelebA32 jobs, and 54
  Flowers64 jobs.

### Blockers
- SLURM scripts still contain `YOUR_PROJECT_ACCOUNT`; replace it before
  submitting.
- On Berzelius Hopper, remove `#SBATCH -C thin` if that constraint is not
  available.

## 2026-04-22T03:41:26Z - Automatic W&B Identity From Config

### Context
W&B project/name/group should be automatic unless the CLI explicitly provides
`--wandb-project`, `--wandb-name`, or `--wandb-group`. Runs are compared within
datasets; normal sweep axes are prediction/loss spaces, batch cap, seed, and
`N`.

### Changed / Observed
- Added config-derived W&B identity resolution in `src/mp_tracking.py`.
- Default project is `memparam-{dataset_key}-{model_arch}`.
- Default run name now includes dataset, model, `N`, batch cap,
  prediction/loss spaces, training steps, and seed.
- Default group is dataset, model, and training steps only, so `x`, `eps`,
  `v`, batch-size, seed, and `N` sweeps remain comparable inside one group.
- Preserved explicit W&B project/name/group values exactly.
- Removed silent `WANDB_PROJECT` override for automatic project selection.
- Recorded the rule in `README.md`, `PYTHON_STYLE.md`, `AGENT.md`,
  `CLAUDE.md`, and `prototyping/runners/README.md`.

### Commands / Checks
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src:. python -m unittest tests.test_tracking tests.test_proto_runners`
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src:. python -m py_compile src/mp_tracking.py src/mp_training/trainer.py src/mp_runners/image_config.py src/mp_runners/image_presets.py src/mp_runners/synthetic_runner.py tests/test_tracking.py tests/test_proto_runners.py`
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src:. python -m mp_runners.synthetic_runner --help`
- `module load Mambaforge && mamba activate mem_param_env && PYTHONPATH=src:. python -m mp_runners.image_runner --help`
- Manual config identity check for Flowers102-64 and torus3d settings.
- `git diff --check`

### Results
- Focused tracking and runner tests passed: 39 tests.
- Byte-compile passed for changed Python files.
- Synthetic and image runner CLI help passed.
- Manual config identity check produced dataset/model projects, detailed run
  names, and dataset/model/steps groups.
- `git diff --check` passed.

### Blockers
- None.

## 2026-04-22T02:49:18Z - Rename Synthetic Runner Module

### Context
The canonical image runner uses the general module name `image_runner`. The
synthetic runner still used `synthetic_mlp`, which mixed the workflow name with
one model choice.

### Changed / Observed
- Renamed `src/mp_runners/synthetic_mlp.py` to
  `src/mp_runners/synthetic_runner.py`.
- Updated the runner name and default output directory to `synthetic_runner`.
- Updated live tests, runner docs, and the synthetic runner notebook import to
  use `mp_runners.synthetic_runner`.

### Commands / Checks
- `python -m py_compile src/mp_runners/synthetic_runner.py tests/test_proto_runners.py tests/test_torus_synthetic.py tests/test_live_tree_boundaries.py`
- `python -m unittest tests.test_proto_runners tests.test_torus_synthetic tests.test_live_tree_boundaries tests.test_src_docstrings tests.test_src_examples`
- `PYTHONPATH=src python -m mp_runners.synthetic_runner --help`
- `python -m json.tool prototyping/runners/notebooks/synthetic_runner.ipynb`
- `rg -n "synthetic_mlp|mp_runners\\.synthetic_mlp" src tests prototyping/runners README.md PROTO_SRC_FUNCTIONALITY_DEEP_DIVE.md --glob '!**/__pycache__/**'`
- `git diff --check`

### Results
- Byte-compile passed.
- Focused runner, torus, live-boundary, docstring, and example tests passed:
  42 tests.
- New CLI help passed for `mp_runners.synthetic_runner`.
- Synthetic runner notebook JSON is valid.
- Active-code/docs grep found no `synthetic_mlp` references.
- `git diff --check` passed.

### Next
- Audit tests for behavioral value after the runner move is stable.

## 2026-04-22T02:09:04Z - Image Metric Notation Cleanup

### Context
The image metric code needed to match the paper notation before larger
experiment sweeps.

### Intent
- Use paper symbols in metric code and records: `rho`, `r`, `d1`, `d2`, `N`,
  `tau`, and `t`.
- Remove duplicate memorization aliases from image metric return dictionaries
  and W&B payloads.
- Keep the compatibility config field `evaluation.ratio_threshold`, but resolve
  it to `rho` inside metric code.

### Changed / Observed
- Updated `src/mp_metrics/images.py` to return canonical keys only for image
  memorization metrics: `rho`, `N`, `mean_r`, `median_r`, `median_d1`,
  `median_d2`, and `nn_idx`.
- Added `delta_f_mem` and rho-sweep fields for `rho` values 0.3, 0.5, and 0.7
  in image checkpoint records and W&B logs.
- Removed the unused `DEFAULT_FEATURE_ENCODER` export rather than changing
  default image metric behavior.
- Added an `image_range='auto'` warning and debug logging for dict tensor
  payload key selection.
- Updated shared evaluation/tracking code to use `mean_r`, `median_r`, and
  `delta_f_mem`.
- Added the math-first naming rule to `PYTHON_STYLE.md`, `AGENT.md`, and
  `CLAUDE.md`.

### Commands / Checks
- `python -m unittest tests.test_mp_metrics`
- `python -m py_compile src/mp_metrics/images.py src/mp_metrics/__init__.py src/mp_tracking.py src/mp_runners/evaluate_checkpoints.py tests/test_mp_metrics.py`
- `git diff --check`
- `python -m unittest tests.test_src_docstrings`
- `python -m unittest tests.test_experiment_pipeline`
- `python -m unittest tests.test_proto_runners`
- `rg -n "DEFAULT_FEATURE_ENCODER|compute_f_mem_by_threshold|mean_nn_ratio|std_nn_ratio|nn_ratio|\"ratios\"|\"indices\"|ratio_threshold" src/mp_metrics/images.py src/mp_metrics/__init__.py src/mp_tracking.py src/mp_runners/evaluate_checkpoints.py tests/test_mp_metrics.py AGENT.md CLAUDE.md PYTHON_STYLE.md`

### Results
- `tests.test_mp_metrics` passed: 12 tests.
- Byte-compile passed for the touched Python files.
- `git diff --check` passed.
- `tests.test_src_docstrings` passed: 4 tests.
- `tests.test_experiment_pipeline` passed: 5 tests.
- `tests.test_proto_runners` passed: 29 tests.
- The grep only reports the compatibility config field
  `evaluation.ratio_threshold` and explicit style/test text that mentions
  deprecated alias names.

### Blockers
- None.

### Next
- Apply the same notation cleanup to synthetic metrics separately if desired.

## 2026-04-22T02:40:05Z - Metric Module Boundary Refactor

### Context
The metric modules had grown to include checkpoint replay, file loading,
preview artifacts, figure generation, run-summary updates, and tracking payload
formatting.

### Intent
- Keep `mp_metrics` limited to metric computation from in-memory tensors or
  feature matrices.
- Move saved-run evaluation to `mp_runners`.
- Move figure generation to `mp_visualization`.
- Move folder/file metric convenience wrappers outside `mp_metrics`.
- Move tracker payload naming outside metric computation.

### Changed / Observed
- Added image and synthetic evaluation modules under `src/mp_runners`.
- Moved image checkpoint previews and image reference-memory collection out of
  `mp_metrics.images`.
- Moved synthetic checkpoint replay out of `mp_metrics.synthetic`.
- Moved synthetic figure generation into `src/mp_visualization.py`.
- Moved folder/file metric wrappers into `src/mp_runners/metric_io.py`.
- Moved `loss_eval_metric_payload` from `mp_metrics.loss_eval` to
  `mp_tracking`.
- Updated the style guides with the metric-boundary rule.

### Commands / Checks
- `python -m py_compile src/mp_metrics/*.py src/mp_runners/*.py src/mp_visualization.py src/mp_tracking.py`
- `python -m unittest tests.test_mp_metrics tests.test_loss_eval tests.test_checkpoint_previews`
- `python -m unittest tests.test_experiment_pipeline tests.test_proto_runners tests.test_torus_synthetic tests.test_src_docstrings`
- `python -m unittest tests.test_projected_gmm_oracle tests.test_src_examples`
- `PYTHONPATH=src python -m mp_runners.synthetic_mlp --help`
- `PYTHONPATH=src python -m mp_runners.image_runner --help`
- `rg -n "evaluate_.*run|make_.*figures|checkpoint_preview|run_summary|compute_.*folder|_load_collection|load_metric_collection|loss_eval_metric_payload" src/mp_metrics`
- `git diff --check`

### Results
- Byte-compile passed.
- Focused metric, runner, torus, docstring, and example tests passed.
- Runner CLI help passed when `PYTHONPATH=src` was set for this checkout.
- The `src/mp_metrics` boundary grep returned no matches.
- `git diff --check` passed.

### Blockers
- None expected.

## 2026-05-01 - Mem-boundary checkpoint eval runner

Implemented a standalone read-only checkpoint eval path for the main Adam mem-boundary N=16384 seed=0 runs.

Artifacts added:
- `src/mp_runners/mem_boundary_eval.py`
- `jobs/evals/mem_boundary_n16384_s0_eval_smoke.sbatch`
- `jobs/evals/mem_boundary_n16384_s0_eval.sbatch`
- `jobs/evals/mem_boundary_n16384_s0_eval_combine.sbatch`

Runner behavior:
- Reads completed source attempts under `outputs/job_sets/mem_gen_boundary_20260427` without writing into them.
- Defaults to output dir `outputs/evals/mem_boundary_n16384_s0_quality_20260501`.
- Supports params `x_x,eps_eps,v_v,x_v`, Heun-50 sampling, cached FID reference stats, append-safe per-checkpoint CSV/JSONL, final CSVs, combined CSVs, and requested plots.
- Skips completed checkpoint rows by `(param, step, sample_count, fid_reference_count, sampler, sampling_steps)` unless `--overwrite` is passed.
- Full Slurm array tasks write per-param outputs with `--skip_combined`; the separate combine job rebuilds global CSVs/plots after all param tasks finish.

Validation done:
- Python syntax passed in `mem_param_env`.
- Shell syntax passed for smoke, full-array, and combine sbatch scripts.
- CLI imports and parses successfully.
- Source resolution found exactly one DONE attempt for each of `x_x`, `eps_eps`, `v_v`, `x_v`; each has 72 checkpoints and final step `1000000`.
- `torch_fidelity` is available in `mem_param_env` as version `0.4.0-beta`.

No smoke or full eval job has been submitted yet.

## 2026-05-01 - Expanded mem-boundary eval to all DONE attempts

Implemented the manifest-driven all-DONE eval plan for `outputs/job_sets/mem_gen_boundary_20260427`.

Key updates:
- `src/mp_runners/mem_boundary_eval.py` now supports `--all_done` manifest discovery over every completed attempt in the job set.
- The all-DONE manifest resolves 84 attempts: 7 train sizes x 3 seeds x 4 parameterizations.
- Each manifest row records N, seed, param, run id, attempt id/path, checkpoint count, final step, sampler, and sampling steps.
- Eval source artifacts are read from `attempt/artifacts/` and source attempts remain read-only.
- Eval samples now use exactly 10,000 deterministic z initializations generated from `config.training.seed + 2003`, rather than being capped by saved `z_init.pt`.
- Held-out eval tensors are rebuilt as 10,000 CelebA valid images using the same 32x32 grayscale preprocessing and `seeded_indices(..., seed + 1)` selection policy as training.
- Held-out tensors are cached globally by seed/split/preprocessing under the eval output cache.
- FID reference stats are cached globally by seed/split/reference count/feature extractor under the eval output cache.
- Cache writes use lock files with stale-lock cleanup to avoid array-task races.
- Skip/resume keys now include N, seed, param, step, sample count, FID reference count, heldout split, sampler, and sampling steps.
- Per-attempt and combined outputs include N/seed/run/attempt/heldout/z-source metadata.
- Combined plots are grouped by N/seed/param and final-by-N aggregate comparison plots are produced.

Slurm updates:
- Existing eval scripts now target all-DONE mode and the new output root `outputs/evals/mem_boundary_20260427_all_done_quality_20260501`.
- Added clearer aliases: `jobs/evals/mem_boundary_all_done_eval_smoke.sbatch`, `jobs/evals/mem_boundary_all_done_eval.sbatch`, and `jobs/evals/mem_boundary_all_done_eval_combine.sbatch`.
- Smoke runs manifest indices `0,72` over checkpoints `0,945000,1000000`.
- Full eval runs `--array=0-83%4`, one single-node 1-GPU task per attempt, with `--skip_combined`.
- Combine is CPU-only and rebuilds manifest, combined CSVs, final CSVs, and plots.

Validation done:
- Python syntax passed in `mem_param_env`.
- Shell syntax passed for legacy and all-DONE smoke/full/combine sbatch scripts.
- `--all_done --manifest_only` successfully wrote an 84-row manifest to `/tmp/mem_boundary_manifest_check` and validated checkpoint/final-step/sampler invariants.

No smoke or full eval job has been submitted yet.

## 2026-05-01 - Smoke eval job status check

Checked all-DONE mem-boundary smoke eval job `16466887`.

Current scheduler state:
- Job `16466887` / `mp_mgb_all_smoke` is `PENDING` with reason `Priority`.
- `sacct` reports `PENDING`, `ExitCode=0:0`, no start/end time yet, requested `cpu=8`, `gres/gpu=1`, `mem=63600M`, `node=1`.
- Existing capacity resume array `16466870` is also pending on priority.

Operational note:
- Created `jobs/evals/logs/` because the Slurm stdout/stderr paths point there and the directory was not present yet.
- No smoke log files exist yet because the job has not started.

## 2026-05-01 - Replaced heavy smoke with light smoke

Implemented the fast-smoke recovery plan.

Actions taken:
- Cancelled pending heavy smoke job `16466887` only.
- Left capacity resume array `16466870` untouched.
- Updated `jobs/evals/mem_boundary_all_done_eval_smoke.sbatch` and the legacy alias `jobs/evals/mem_boundary_n16384_s0_eval_smoke.sbatch` to use lighter resources:
  - `--cpus-per-task=4`
  - `--mem=32G`
  - `--gpus=1`
  - `--time=02:00:00`
  - manifest rows `0,72`
  - checkpoint steps `0,1000000`
  - `--sample_count 10000`
  - `--fid_reference_count 10000`
  - `--heldout_split valid`
- Shell syntax checks passed for both smoke scripts.
- Submitted replacement light smoke job `16466891`.

Scheduler state after resubmission:
- `16466887` / `mp_mgb_all_smoke`: `CANCELLED by 13657`, requested `cpu=8`, `gres/gpu=1`, `mem=63600M`, `node=1`, never started.
- `16466891` / `mp_mgb_all_smoke`: `PENDING`, reason `Priority`, requested `cpu=4`, `gres/gpu=1`, `mem=32G`, `node=1`.
- Capacity resume array `16466870` remains `PENDING` on `Priority`.
- No smoke log files exist yet because `16466891` has not started.

## 2026-05-01 - Smoke FID range failure fixed

Checked smoke job `16466891` after it started and failed.

Observed state:
- `16466891` started at `2026-05-01T04:34:58`, failed at `2026-05-01T04:38:51`, elapsed `00:03:53`, exit `1:0`.
- It successfully evaluated manifest row `0` checkpoint `step_000000.pt` and wrote one row with finite metrics: `f_mem=0`, `fid_10k=423.481`, `test_loss=0.272654`.
- It failed on `step_1000000.pt` during generated-sample FID encoding because `image_range='auto'` could not infer range for samples with min/max around `-1.132/1.138`.

Fix applied:
- Set runner FID encoding to explicit `FID_IMAGE_RANGE = "-1_1"` in `src/mp_runners/mem_boundary_eval.py` for both reference FID stats and generated sample FID features.
- This maps normalized CelebA/model tensors from `[-1, 1]` to `[0, 1]` and clamps slight out-of-range model samples instead of failing auto inference.

FID implementation notes:
- The runner uses `torch_fidelity:inception-v3-compat` through `_TorchFidelityFeatureEncoder` in `src/mp_metrics/metrics_images.py`.
- The wrapper converts grayscale tensors to RGB uint8; installed torch-fidelity internally resizes inputs to Inception's `299x299` using TensorFlow-compatible bilinear interpolation.
- No manual upscaling is needed in the runner.

Validation:
- `python -m py_compile src/mp_runners/mem_boundary_eval.py` passed in `mem_param_env` after the fix.

Smoke has not been resubmitted yet after this fix.

## 2026-05-01 - Resubmitted fixed light smoke eval

Checked scheduler state and submitted the fixed all-DONE eval smoke after the FID image range patch.

Pre-submit observations:
- No eval smoke job was queued or running.
- Capacity resume array `16466870` remained pending on `Priority`.
- Earlier capacity resume tasks `16466870_15` through `16466870_21` had failed quickly with exit `127:0`; this is separate from the eval smoke path and still needs follow-up before relying on capacity resumes.
- Previous eval smoke states: `16466887` cancelled, `16466891` failed due to FID `image_range='auto'` inference.

Submitted:
- Fixed light smoke job `16467047` using `jobs/evals/mem_boundary_all_done_eval_smoke.sbatch`.

Post-submit scheduler state:
- `16467047` / `mp_mgb_all_smoke`: `PENDING`, reason `Priority`, requested `billing=4,cpu=4,gres/gpu=1,mem=32G,node=1`.
- Capacity resume array `16466870` still pending on `Priority`.

## 2026-05-01 - Submitted canary capacity resume and full eval array

Performed the post-smoke recovery actions.

Capacity resume wrapper:
- Patched `jobs/job_sets/mem_gen_boundary_capacity_20260427/resume_undone_20260501.sbatch` line 22 from broken `{AWK:-awk}` syntax to plain `awk` command substitution.
- `bash -n` passed for the patched resume wrapper.
- Submitted one canary resume task: job `16473993`, array index `107%1`.

Eval jobs:
- `bash -n` passed for `jobs/evals/mem_boundary_all_done_eval.sbatch`.
- Submitted full all-DONE eval array: job `16473994`, array `0-83%4`.
- Initial combine submission failed because Berzelius rejected a no-GPU job with invalid TRES / no GPU allocation.
- Patched `jobs/evals/mem_boundary_all_done_eval_combine.sbatch` and legacy alias to request `--gpus=1`.
- `bash -n` passed for both combine scripts after the patch.
- Submitted dependent combine job: `16473995`, dependency `afterok:16473994`.

Current scheduler state immediately after submissions:
- `16473993_[107%1]` / `mp_mgb_cap_resume`: `PENDING`, reason `Priority`, requested `billing=1,cpu=1,gres/gpu=1,mem=7950M,node=1`.
- `16473994_[0-83%4]` / `mp_mgb_all_eval`: `PENDING`, reason `Priority`, requested `billing=8,cpu=8,gres/gpu=1,mem=63600M,node=1`.
- `16473995` / `mp_mgb_all_combine`: `PENDING`, reason `Dependency`, requested `billing=2,cpu=2,gres/gpu=1,mem=15900M,node=1`.
- There is also a pending interactive job `16473992` on priority.

Important guardrail:
- Did not submit the remaining full capacity resume array yet. Wait for canary `16473993_107` to start and pass the wrapper/training setup before resubmitting all 52 capacity resume rows.

## 2026-05-01 - End-of-day operational summary: eval smoke, full eval, and capacity recovery

Summary of the last operational cycle:
- Built and validated a manifest-driven mem-boundary checkpoint eval path for all 84 DONE attempts in `outputs/job_sets/mem_gen_boundary_20260427`.
- Smoke eval initially exposed an FID image-range bug: generated samples at the final checkpoint could exceed `[-1, 1]` slightly, causing `image_range='auto'` to fail.
- Fixed FID encoding to use explicit `FID_IMAGE_RANGE = "-1_1"`, which maps normalized CelebA/model tensors to `[0, 1]` and clips slight overshoot.
- Confirmed no manual upscaling is required for FID: the repo wrapper repeats grayscale to RGB uint8 and torch-fidelity internally resizes to Inception `299x299`.
- Re-ran light smoke job `16467047`; it completed successfully in `00:04:59` with finite `f_mem`, `fid_10k`, and 10K held-out test-loss metrics.
- Smoke output root: `outputs/evals/mem_boundary_20260427_all_done_quality_20260501_smoke`.
- Smoke combined metrics: `all_checkpoint_metrics.csv` has 4 rows; `all_final_metrics.csv` has 2 rows; manifest has 84 rows.
- Submitted full all-DONE eval array `16473994_[0-83%4]`; it is pending on scheduler priority.
- Submitted dependent combine job `16473995` with `afterok:16473994`; it is pending on dependency.
- Combine job had to request `--gpus=1` because Berzelius rejected a no-GPU Slurm submission for this account/partition.

Capacity resume recovery:
- Previous capacity resume array `16466870` failed before training with exit `127:0`.
- Inspected 52/52 resume stderr logs; every failure was the same shell-wrapper typo: `{AWK:-awk}: command not found`.
- Patched `jobs/job_sets/mem_gen_boundary_capacity_20260427/resume_undone_20260501.sbatch` to use plain `awk` command substitution.
- Submitted one capacity resume canary: `16473993_[107%1]`; it is pending on scheduler priority.
- Did not submit the remaining 52 capacity resume tasks yet; wait for canary `16473993_107` to start and prove the wrapper/training setup works.

Current known scheduler state when this entry was written:
- `16473993_[107%1]` / `mp_mgb_cap_resume`: pending on `Priority`.
- `16473994_[0-83%4]` / `mp_mgb_all_eval`: pending on `Priority`.
- `16473995` / `mp_mgb_all_combine`: pending on `Dependency`.

Push scope for this recovery branch:
- Eval runner and Slurm scripts.
- Capacity resume wrapper fix and its manifest.
- Coding journal/log updates.
- Exclude generated Slurm logs, eval outputs, checkpoint outputs, and unrelated pre-existing worktree changes.

## 2026-05-01 - Paper objective locked for internal experiment planning

Created `PAPER_OBJECTIVE.md` as the internal north-star document for the paper.

Core thesis to preserve across evals, plots, and writing:
- Do not frame the paper as `epsilon/epsilon memorizes less`.
- Frame it as: target parameterization changes the learning dynamics by which diffusion/flow models move from generation to memorization.
- Strong claim: at matched architecture, dataset size, optimizer, and sample quality, different target parameterizations enter different memorization regimes.
- Contribution is the intersection of Bonnaire-style generation-to-memorization dynamics, Li-He target-geometry/finite-capacity parameterization effects, and George-style train/test learning curves.

Operational implications:
- Prioritize quality-controlled memorization: `F_mem(tau)` plus `FID(tau)` for every checkpoint.
- Main derived quantities: `F_mem@FID_best`, matched-FID memorization, `tau_gen`, `tau_mem`, and `tau_mem - tau_gen`.
- Avoid centering raw native training loss comparisons, final `F_mem` alone, or generic Pareto fronts without showing parameterization shifts.
- Every metrics table should group by parameterization, dataset size, capacity, optimizer, seed, and checkpoint.

The detailed paper skeleton, figure plan, reviewer-objection framing, and implementation target table now live in `PAPER_OBJECTIVE.md`.

## 2026-05-02 - Pivoted paper objective after 2601.21419

Updated the internal paper objective after noting that Jin and Wang, "Revisiting Diffusion Model Predictions Through Dimensionality" (arXiv:2601.21419), substantially covers the intrinsic/ambient-dimension target-selection story.

New objective:
- Do not frame novelty as target parameterization depending on intrinsic versus ambient dimension.
- Frame novelty as target parameterization changing memorization dynamics in flow matching.
- Main question: whether the quality-optimal target `k_FID_star` equals the memorization-optimal target `k_mem_star` under `F_mem@FID_best` or matched-quality criteria.
- Central line: the target that is optimal for sample quality may not be optimal for memorization.

Implementation implications recorded in `PAPER_OBJECTIVE.md`:
- Use the continuous target family `y_k = k*x - (1-k)*eps` for new experiments.
- Treat `k=0` as `-eps`, not legacy `+eps`; do not silently mix signs when mapping old `eps/eps` runs.
- Add fixed-`k` sweeps over at least `k in {0, 0.5, 1}` and preferably `{0, 0.25, 0.5, 0.75, 1}`.
- Track checkpoint-level FID, `F_mem`, `F_mem@FID_best`, `tau_gen`, `tau_mem`, and `tau_mem - tau_gen`.
- Keep dataset-size and capacity sweeps, but use them to test whether target choice shifts the generation-memorization frontier, not as standalone novelty.

README framing was adjusted to describe the repository as a flow-matching memorization project and to mark dimensionality sweeps as stress tests rather than the core novelty claim.

## 2026-05-02 - Interactive one-GPU eval queue helper

Added a CSV-backed interactive eval queue helper for rare one-GPU interactive allocations.

Files and output root:
- Queue helper: `outputs/evals/mem_boundary_20260427_interactive_queue_20260502/mem_eval_queue.py`
- Queue spreadsheet/ledger: `outputs/evals/mem_boundary_20260427_interactive_queue_20260502/eval_queue.csv`
- Per-item logs: `outputs/evals/mem_boundary_20260427_interactive_queue_20260502/queue_logs/`
- Per-attempt metrics: `outputs/evals/mem_boundary_20260427_interactive_queue_20260502/attempts/*/checkpoint_metrics.csv` and `.jsonl`

Design:
- One queue row is exactly one `(manifest_index, checkpoint_step, sample_count, fid_reference_count, heldout_split, sampler, sampling_steps)` eval unit.
- The helper delegates metric computation to `python -m mp_runners.mem_boundary_eval` rather than creating a second eval implementation.
- Re-running is safe: the helper checks existing `checkpoint_metrics.csv` rows and marks matching queue rows `done`.
- Interrupted rows become `interrupted`; stale `running` rows are reset to `pending` on status/run.
- `--n_items` controls how many checkpoint eval units to consume in the interactive session.

Interactive setup:

```bash
cd /proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par
module load Mambaforge
mamba activate mem_param_env
export PYTHONPATH=src:.
export OUT_DIR="outputs/evals/mem_boundary_20260427_interactive_queue_20260502"
```

Single final-checkpoint test for `N=16384, seed=0, x_x`:

```bash
python "$OUT_DIR/mem_eval_queue.py" init \
  --out_dir "$OUT_DIR" \
  --manifest_indices 72 \
  --checkpoint_steps 1000000

python "$OUT_DIR/mem_eval_queue.py" status --out_dir "$OUT_DIR"

python "$OUT_DIR/mem_eval_queue.py" run \
  --out_dir "$OUT_DIR" \
  --n_items 1 \
  --device cuda
```

Seed-0 N-sweep queue for `N=256`, `N=2048`, and `N=16384`, all 4 params, 5 checkpoints:

```bash
python "$OUT_DIR/mem_eval_queue.py" init \
  --out_dir "$OUT_DIR" \
  --manifest_indices "0-3,36-39,72-75" \
  --checkpoint_steps "0,100000,300000,595000,1000000"

python "$OUT_DIR/mem_eval_queue.py" run \
  --out_dir "$OUT_DIR" \
  --n_items 4 \
  --device cuda
```

Combined output refresh:

```bash
python "$OUT_DIR/mem_eval_queue.py" combine --out_dir "$OUT_DIR"
```

Scope note:
- This helper does not add common-space `x/eps/v` losses or `k`-target support. It only makes the existing all-DONE eval runner easy to consume one checkpoint or N checkpoints at a time on a single interactive GPU.

## 2026-05-02T06:22:35+02:00 - End-of-night job handoff for tomorrow

Final scheduler/accounting snapshot before stopping for the night.

Eval array status:
- Full all-DONE mem-boundary eval array `16473994_[0-83%4]` is actively progressing.
- Completed cleanly so far: array tasks `0` through `11`, all exit `0:0`.
- Currently running: `16473994_12`, `16473994_13`, `16473994_14`, and `16473994_15`.
- Remaining tasks `16-83` are pending with reason `JobArrayTaskLimit`, which is expected because the array was submitted with concurrency cap `%4`.
- Dependent combine job `16473995` is pending on `Dependency`; it should start only after the full eval array completes successfully.

Capacity resume status:
- Capacity resume canary `16473993_107` completed cleanly with exit `0:0` after `02:01:12`.
- This confirms the patched resume wrapper got past the previous `{AWK:-awk}` shell error and the canary path is viable.
- Tomorrow action: inspect canary output/status, then submit the remaining capacity resume array if the attempt state is sane.

Interactive status:
- Interactive job `16474429` timed out normally after `03:00:25`.
- Interactive queue helper remains available at `outputs/evals/mem_boundary_20260427_interactive_queue_20260502/mem_eval_queue.py` if another interactive allocation is obtained.

Tomorrow checklist:
1. Check `sacct -X -j 16473994,16473995` and confirm whether the full eval array and combine job finished.
2. If combine completed, inspect `outputs/evals/mem_boundary_20260427_all_done_quality_20260501/all_checkpoint_metrics.csv` and `all_final_metrics.csv` row counts.
3. Confirm full eval acceptance target: `6048` checkpoint rows and `84` final rows.
4. Inspect capacity canary `16473993_107` output and status files under `outputs/job_sets/mem_gen_boundary_capacity_20260427`.
5. If canary state is good, submit the remaining capacity resume tasks from `jobs/job_sets/mem_gen_boundary_capacity_20260427/resume_undone_20260501.tsv` using the patched sbatch wrapper.
6. Keep source attempts read-only; all eval artifacts should stay under `outputs/evals/`.

## 2026-05-03 - Plotting prototype notes for F_mem eval figures

Added and iterated on the prototype plotting script for the all-DONE mem-boundary eval:

- `prototyping/plot_mem_boundary_20260427_quality_memorization.py`
- Output root: `prototyping/figures/mem_boundary_20260427_all_done_quality_20260501_figures/`

Important interpretation notes:

- The CSV metric column is still named `f_mem`, but plot labels should use paper notation `F_mem`.
- The plotted `F_mem` is nearest-neighbor ratio memorization against the full saved training tensor for that run, not against held-out data.
- The default `F_mem` column corresponds to `rho=0.5`; the eval CSVs also include `f_mem_rho0.3`, `f_mem_rho0.5`, and `f_mem_rho0.7`.
- The `F_mem` curves can appear only after they become non-zero on log-scaled plots. Early zero-valued points are intentionally absent from log-y panels because log scale cannot display zero.
- Current eval outputs use non-projected image/tensor nearest-neighbor metrics for the post-hoc `F_mem`; the older train-time random-projected monitor is separate.
- Parameterization comparisons are paired at fixed `(n_train, seed)`: current visible outputs reuse the same held-out/FID cache and the same saved train tensor across `eps_eps`, `v_v`, `x_v`, and `x_x` for a fixed seed and dataset size.

## 2026-05-03 - Plotting prototype and capacity-resume sanity checkpoint

Prepared the plotting prototype for commit/push:

- Script: `prototyping/plot_mem_boundary_20260427_quality_memorization.py`.
- Scope: all-DONE mem-boundary eval plotting for `mem_boundary_20260427_all_done_quality_20260501`.
- Main figures now include quality/memorization trajectories, `F_mem@FID_best`, parameterization comparisons, threshold-specific `F_mem` panels for `rho in {0.3, 0.5, 0.7}`, exact `n_train` log ticks, and `tau` axis labels.
- Plot labels use paper notation `F_mem`; raw CSV fields remain lowercase `f_mem`.

Capacity-resume sanity jobs submitted for W&B/local resume verification:

- Job `16478119`, array tasks `24` and `26`.
- Task `24`: `base64_m1244`, `N=256`, seed `0`, `x/x`.
- Task `26`: `base64_m1244`, `N=256`, seed `0`, `v/v`.
- Both were healthy when last checked: running, writing to new local attempt directories, and W&B reported `Resuming run ...` against the original run IDs rather than creating unrelated runs.

Commit scope for this checkpoint should remain limited to the plotting prototype and journal updates; generated figures, Slurm logs, notebooks, and unrelated dirty worktree files should stay out of the commit.

## 2026-05-03 - FFHQ output cleanup and repo-main consolidation plan

Deleted only the FFHQ output result directories after confirming they were disposable preflight/speed-grid artifacts to redo later:

- `outputs/job_sets/ffhq64_param_sweep_20260426`
- `outputs/job_sets/ffhq_speed_grid_20260426_fat`
- `outputs/job_sets/ffhq_speed_grid_20260426_thin`

Storage after deletion:

- Free filesystem space improved from about `865G` to about `1.8T`.
- Remaining `outputs/job_sets` size is about `1.7T`.

Cleanup policy for bringing the repo back to `main`:

- Keep source, docs, tests, job specifications/manifests/scripts, real notebooks, and useful helper scripts.
- Exclude generated logs, scratch files, notebook checkpoints, local Codex state, pyc caches, and output/data/tmp result artifacts.
- Restore deleted legacy job launcher scripts rather than committing their deletion.
- Restore `tea_debug.log` rather than committing transient debug output.

## 2026-05-03 - Capacity final-checkpoint eval launch

Implemented and smoke-tested the capacity final-checkpoint eval path.

Safety and consistency checks:

- Kept capacity eval outputs isolated from the older all-DONE `n_train` sweep outputs.
- Confirmed the existing `n_train` sweep used `train_tensor.pt` from `torchvision_celeba:train` for `F_mem`, and `torchvision_celeba:valid` as the held-out/FID reference.
- Kept the capacity full eval consistent with the existing sweep by using `--heldout_split valid`.
- Confirmed `F_mem` is computed against each run's full saved training tensor, not the held-out/FID reference set.
- Confirmed the eval records threshold-specific memorization columns for `rho in {0.3, 0.5, 0.7}` in addition to the default `f_mem`.
- Added capacity-safe output naming because the capacity jobset has multiple model labels for the same `(n_train, seed, param)`.

Smoke jobs:

- Tiny final-checkpoint smoke, `sample_count=64`, `fid_reference_count=64`.
- Short final-checkpoint smoke, `sample_count=1024`, `fid_reference_count=1024`.
- Both a full 72-checkpoint source attempt and a resumed 2-checkpoint tail attempt completed cleanly.

Full eval launch:

- Job: `16479222`.
- Script: `jobs/evals/mem_boundary_capacity_final_eval.sbatch`.
- Array: `0-96` with no throttle, per deadline pressure.
- Frozen manifest: `outputs/evals/mem_boundary_capacity_20260427_final_quality_20260503/manifest.csv`.
- Output root: `outputs/evals/mem_boundary_capacity_20260427_final_quality_20260503`.
- Settings: final checkpoint only, `step_1000000`, `sample_count=10000`, `fid_reference_count=10000`, `heldout_split=valid`, `device=cuda`.
- The frozen manifest had `97` completed capacity attempts at launch time.
- The large resume array still had `11` tasks running at launch time; those should be picked up by a later small eval launch after their DONE attempts appear.

## 2026-05-03 - Capacity full-trajectory merged eval launch

Corrected course from final-checkpoint-only eval to the required full-trajectory capacity eval:

- Goal: all currently completed capacity runs, all seeds, all 72 checkpoints per run.
- Key issue: resumed capacity attempts often contain only tail checkpoints, so full trajectories must merge the original source attempt checkpoints with the resumed DONE attempt checkpoints.
- Implemented frozen-manifest support for merged checkpoint source directories.
- Added atomic manifest writes and an empty-manifest guard so Slurm array tasks cannot race on manifest creation.
- Kept scientific settings aligned with the existing `n_train` sweep:
  - `sample_count=10000`
  - `fid_reference_count=10000`
  - `heldout_split=valid`
  - `F_mem` against each run's saved `train_tensor.pt`
  - threshold columns for `rho in {0.3, 0.5, 0.7}`

Merged trajectory smoke:

- Job: `16479322_1`.
- Run: `N=256`, seed `0`, `x/x`, `base64_m1244`.
- Output: `outputs/evals/tmp_capacity_full_trajectory_merged_smoke_20260503`.
- Settings: `sample_count=64`, `fid_reference_count=64`.
- Result: completed cleanly in `00:09:44`.
- Verified `72` checkpoint metric rows, first step `0`, final step `1000000`.
- Verified early/mid checkpoints came from the original source attempt and the final checkpoint came from the resumed DONE attempt.

Real full-trajectory launch:

- Job: `16479328`.
- Script: `jobs/evals/mem_boundary_capacity_full_trajectory_eval.sbatch`.
- Frozen manifest: `outputs/evals/mem_boundary_capacity_20260427_full_trajectory_quality_20260503/manifest.csv`.
- Output root: `outputs/evals/mem_boundary_capacity_20260427_full_trajectory_quality_20260503`.
- Manifest had `97` rows, all with `checkpoint_count=72` and `final_step=1000000`.
- Manifest composition: `41` merged resumed trajectories and `56` single-source full trajectories.
- Slurm array: `0-96`, no throttle, per deadline pressure.
- Time limit: `05:00:00` per task based on old full-trajectory eval evidence.
- Health after launch: all `97` tasks running, no failures seen, no metric files yet at the first early check.

Related running jobs at launch checkpoint:

- Capacity resume array `16479052`: `39` completed, `10` still running, no failures seen.
- Old all-DONE `n_train` sweep eval `16473994`: `76` completed, `4` running, `2` pending, no failures seen; combine job `16473995` still dependency-pending.
