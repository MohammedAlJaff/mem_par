# MNIST n-Sweep

Canonical MNIST sweep over training-set size.

Default configuration:

- family: `fm`
- prediction/loss: `x/x`
- backbone: `mlp`
- default train set size in config: `64`

Entry points:

```bash
python -m experiments.mnist_nsweep.train --help
python -m experiments.mnist_nsweep.evaluate --help
python -m experiments.mnist_nsweep.plot --help
```

This study is the canonical place to scale the MNIST memorization pipeline
beyond the smallest proof-of-concept run.
