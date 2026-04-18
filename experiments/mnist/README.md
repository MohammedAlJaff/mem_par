# MNIST Study

`experiments/mnist/` is the first polished live study surface in the paper
repo.

## Entry Points

- `mnist_walkthrough.ipynb`
  The primary reviewer path. It is written as an interactive, line-by-line
  walkthrough, but it now launches the real saved-run train/eval path and then
  reads the written artifacts back from disk.

- `python -m experiments.mnist.run`
  The official saved-run path. This is the artifact-writing route that uses the
  stable `src/` kernel to create an experiment/run/seed/attempt directory,
  checkpoints, manifests, status metadata, and checkpoint evaluation metrics.

## Notebook-First Usage

Open the notebook from the repo root:

```bash
jupyter lab experiments/mnist/mnist_walkthrough.ipynb
```

The notebook now acts as a checkpoint-evaluation prototype:

- it builds balanced train and reference subsets explicitly
- it writes a real attempt directory with saved labels, indices, tensors,
  checkpoints, manifests, and run summaries
- it evaluates every checkpoint into memorization metrics and plots the curves
- it surfaces the attempt-local `status.json`, `logs/attempt.log`, replay
  metrics, and core-generated scalar figures

For a quick smoke pass, launch Jupyter with `MEMPAR_FAST_NOTEBOOK=1`.

## Official Saved-Run Usage

From the repo root:

```bash
python -m experiments.mnist.run --download --device cpu
```

If you are running directly from a checkout that has not been installed in
editable mode yet, use:

```bash
PYTHONPATH=src python -m experiments.mnist.run --download --device cpu
```

The default study output goes under `tmp/mnist_study/`, which is already
ignored by git. Under that root, each hyperparameter run gets its own
seed-specific attempt directories rather than one flat run folder.

Useful operator flags:

```bash
PYTHONPATH=src python -m experiments.mnist.run \
  --download \
  --device cuda \
  --batch-cap 512 \
  --summary-every-steps 500 \
  --summary-every-seconds 60
```

Optional offline W&B tracking:

```bash
PYTHONPATH=src python -m experiments.mnist.run \
  --download \
  --device cuda \
  --wandb \
  --wandb-mode offline \
  --wandb-project mem_par
```

W&B is disabled by default. When enabled, the local W&B directory is inside the
attempt under `logs/wandb/`, so cluster runs remain self-contained and can be
synced later.

Each completed attempt should contain:

- `status.json`
- `logs/events.jsonl`
- `logs/attempt.log`
- `artifacts/config.json`
- `artifacts/run_summary.json`
- `artifacts/data_meta.json`
- `metrics/train_history.json`
- `metrics/checkpoint_metrics.json`
- `figures/training_loss_vs_step.png`
- `figures/f_mem_vs_tau.png`
- `figures/nn_ratio_vs_tau.png`
