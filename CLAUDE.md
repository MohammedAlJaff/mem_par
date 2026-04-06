# mem_par — Parameterization & Memorization Phase Boundary

Paper: "Equivalent in Principle, Unequal in Practice"
Question: do prediction parameterizations (`x`, `eps`, `v`) shift the memorization-generalization phase boundary in flow matching?
Key quantities: `f_mem` (memorization fraction), `n*` (critical training-set size), `tau_mem` (first epoch/step where `f_mem` crosses onset threshold)

---

## Commands

### Self-test individual modules (run from project root)
```bash
python -m experiments.synthetic.high_d.gmm_subspace.run --help
python -m experiments.synthetic.high_d.torus.run --help
python -m experiments.synthetic.low_d.spiral_mem_demo.run --help
python -m experiments.natural_images.celeba32_grey_unet.run --help
python -m baselines.flow_matching.run --help
```

### Run a single experiment (debug / CPU-friendly)
```bash
python -m experiments.synthetic.high_d.gmm_subspace.run \
    --d 4 --n 500 --pred-space x --loss-space v --seed 0 \
    --device cpu
```

### Run a lab script
```bash
python lab/01_primitives.py
```

---

## Source Layout

```text
experiments/
  synthetic/
    low_d/
      common/           low-dimensional synthetic shared code
      spiral_mem_demo/  standalone visual demo
    high_d/
      common/           high-dimensional synthetic shared code
      gmm_subspace/     canonical linear-subspace experiment
      torus/            canonical curved-manifold experiment
    mnist/
      common/           synthetic image-generation scaffolding
  natural_images/
    common/             real-image shared code
    celeba32_grey_unet/
    afhq_cat_64_jit/
    afhq_cat_64_unet/
    afhq_cat_128_jit/
    afhq_cat_256_jit/

baselines/
  flow_matching/        vendored flow-matching sandbox + wrapper
  jit/                  vendored JiT baseline + wrapper
  bonnaire/             vendored Bonnaire baseline + wrapper

lab/                    walkthroughs, sanity checks, reference extras
lab/reference/          non-canonical archived material
third_party/            preserved third-party vendor code
```

---

## Placement Rules

- If the latent generator is known and controlled, the experiment belongs in `experiments/synthetic/`.
- If the dataset is observational and collected from the world, the experiment belongs in `experiments/natural_images/`.
- If the synthetic geometry is directly visualizable in 2D or 3D, use `experiments/synthetic/low_d/`.
- If the synthetic process has known latent structure in high ambient dimension, use `experiments/synthetic/high_d/`.
- If the observations are image-like but still generated from a known latent process, use `experiments/synthetic/mnist/`.
- Do not put synthetic image experiments in `experiments/natural_images/`.
- Do not recreate `experiments/synthetic/common/`.

---

## The 3x3 Prediction/Loss Grid

```text
Interpolant:  z_t = t·x + (1−t)·eps,   v = x − eps
Network outputs one of: x (clean data), eps (noise), v (velocity)
Loss measured in one of:  x,            eps,         v
```

- `DENOM_CLIP = 0.05` clips denominators `(1−t)` and `t` near endpoints
- `T_CLIP = [0.001, 0.999]`
- Time sampling: logit-normal, `mu=-0.8`, `sigma=0.8` (Li & He default)
- Primary axis of the paper: `pred_space`
- `loss_space` is held at `"v"` in Tier 0 unless an experiment explicitly studies the loss axis

---

## Canonical Experiment Contract

Every canonical experiment folder owns:

- `README.md`
- `config.yaml`
- `paths.example.yaml`
- `run.py`
- `sweep.py`
- `aggregate.py`
- `runs/`
- `figures/`

Additional expectations by subfamily:

- `synthetic/high_d/`: make the latent structure explicit; prefer `data_model.py`, `dataset.py`, and `ground_truth.py` when the experiment grows beyond a minimal one-file setup.
- `synthetic/low_d/`: keep geometry, visualization, and evaluation tightly coupled to the experiment so the scientific picture is easy to read.
- `synthetic/mnist/`: treat the generator as first-class; prefer `data_model.py`, `dataset.py`, `ground_truth.py`, and optionally `renderer.py`.
- `natural_images/`: use family-local shared loaders and models where appropriate, but keep experiment-specific assumptions inside the owning experiment folder.

---

## Configs

- Each experiment folder owns an explicit `config.yaml`.
- Machine paths and cluster settings live in `paths.example.yaml` plus ignored local variants.
- Sweep axes may live as YAML lists such as `pred_space: [x, eps, v]`.
- Local `sweep.py` expands those lists into a run manifest.
- Canonical configs must not use `_base_` inheritance.
- Canonical experiment entrypoints must not route through a repo-wide registry or generic dispatch layer.

Run directory naming should encode the active sweep axes for that experiment.

Examples:

- `runs/predx_lossv_d4_n500_seed0/`
- `runs/predv_lossv_n200_seed1/`
- `runs/<experiment-specific-axis-name>/...` when `d` is not part of the design

Each run directory should contain the reproducibility-critical artifacts for that experiment, typically:

```text
config.json         full resolved config
run.log             timestamped log
train_data.pt       exact training set tensor
train_indices.pt    indices into parent dataset when relevant
dataset_info.json   data-generation parameters
loss_history.json   per-step or per-epoch loss
metrics.json        f_mem, d1, d2, r, train summary
checkpoints/        model, EMA, optimizer state
samples/            generated samples
figures/            run-local plots
```

---

## Constraints — Do Not Violate

### 1. Training set persistence
Every run must call `save_dataset_artifact(dataset, run_dir, params)` once, immediately after `build_dataset()`, before training begins.
All evaluation must use `load_train_data(run_dir)` and must never rebuild the dataset ad hoc.

Why: `f_mem`, `d1`, `d2`, and `r` are measured against the exact training set.
If the evaluation set differs by even one sample, the metric is wrong.
For `natural_images`, filesystem order can differ across machines, so `train_data.pt` is the only safe record of what was trained on.

### 2. Always use Dataset + DataLoader
- Every data source must be a `torch.utils.data.Dataset`.
- Training must go through `torch.utils.data.DataLoader`.
- Evaluation-side artifact collection should also use DataLoader batching.
- Do not use `torch.stack([dataset[i] for i in range(n)])` outside the artifact-saving path.

### 3. No training-time augmentation
Random crops, flips, color jitter, or any stochastic transform applied after dataset construction confound `n`.
A model trained on `n=1000` images with random flipping effectively sees more than 1000 observations.

Important distinction:

- Declared stochasticity inside the canonical synthetic generator is allowed.
- That randomness must be controlled by the experiment seed and fixed before `save_dataset_artifact(...)`.
- After the dataset artifact is written, training must not apply additional stochastic transforms.

### 4. Reproducibility
Call `seed_everything(seed)` once at the top of every run, before any data generation or model initialization.
`cuDNN.deterministic = True` stays on for canonical experiments.

### 5. EMA pattern
Call `ema.apply()` before sampling or evaluation.
Call `ema.restore()` immediately after.

### 6. Parameterization separation
The network maps `(z_t, t) -> R^D` and does not know what `pred_space` means.
`FlowMatching` assigns that meaning.
Do not bake parameterization semantics into network code.

### 7. Preserve the experiment-centric structure
- Do not recreate `src/`, root `configs/`, root `scripts/`, or a repo-wide pipeline layer.
- Do not create `experiments/synthetic/common/`.
- If code is shared only within one synthetic subfamily, keep it in that subfamily's `common/`.
- If code might be shared across synthetic subfamilies, duplicate first unless the abstraction is clearly stable and scientifically justified.

### 8. Non-canonical areas stay non-canonical
- `lab/` is for exploration, walkthroughs, and sanity checks.
- `lab/reference/` is archived context, not canonical code.
- Canonical experiments must not import from `lab/reference/`.
- Vendored code under `baselines/*/vendor` and `third_party/` is read-only unless explicitly instructed otherwise.

### 9. Never push to remote or amend past commits without explicit instruction.

---

## Workflow

1. Explore new ideas in `lab/`.
2. Graduate stable work into:
   - `experiments/synthetic/<subfamily>/<experiment>/`, or
   - `experiments/natural_images/<experiment>/`
3. Production runs go through the experiment-local `run.py`, `sweep.py`, and `aggregate.py`.
4. New shared code may exist only in:
   - `experiments/synthetic/low_d/common/`
   - `experiments/synthetic/high_d/common/`
   - `experiments/synthetic/mnist/common/`
   - `experiments/natural_images/common/`
5. `lab/reference/`, `baselines/`, and `third_party/` are context or baselines, not the source of truth for paper experiments.

---

## Cluster (Berzelius, NSC)

- NVIDIA A100 80GB, project storage `/proj/berzelius-2025-NNN/`
- Use each experiment-local `sweep.py` to generate manifests and SLURM scripts.
- `synthetic/low_d` and `synthetic/high_d` usually generate data on the fly.
- `synthetic/mnist` should stage or generate data according to its declared latent generator, not by borrowing the `natural_images` workflow.
- `natural_images` may need dataset staging to `/scratch/local/${SLURM_JOB_ID}/`.
- Pre-download all model weights before submitting; compute nodes have no internet.
