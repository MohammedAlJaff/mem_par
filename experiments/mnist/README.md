# MNIST Study

`experiments/mnist/` is the first polished live study surface in the paper
repo.

## Entry Points

- `mnist_walkthrough.ipynb`
  The primary reviewer path. It is written as an interactive, line-by-line
  walkthrough rather than a thin wrapper around hidden helper functions.

- `python -m experiments.mnist.run`
  The official saved-run path. This is the artifact-writing route that uses the
  stable `src/` kernel to create a run directory, checkpoints, and replay
  metrics.

## Notebook-First Usage

Open the notebook from the repo root:

```bash
jupyter lab experiments/mnist/mnist_walkthrough.ipynb
```

The notebook is intentionally in-memory by default. It demonstrates the study
step by step without requiring a canonical run directory.

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
ignored by git.
