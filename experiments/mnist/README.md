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
  checkpoints, manifests, status metadata, and replay metrics.

## Notebook-First Usage

Open the notebook from the repo root:

```bash
jupyter lab experiments/mnist/mnist_walkthrough.ipynb
```

The notebook now acts as a checkpoint-replay prototype:

- it builds balanced train and reference subsets explicitly
- it writes a real attempt directory with saved labels, indices, tensors,
  checkpoints, manifests, and run summaries
- it replays every checkpoint into memorization metrics and plots the curves
- it stores the notebook-generated figures inside that attempt's `figures/`
  directory

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
