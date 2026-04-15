# MNIST N=4 Diagonal

Canonical proof-of-concept MNIST study for the diagonal parameterization
comparison at extremely small `N`.

Default configuration:

- family: `fm`
- prediction/loss: `x/x`
- backbone: `mlp`
- train set size: `4`

Entry points:

```bash
python -m experiments.mnist_n4_diagonal.train --help
python -m experiments.mnist_n4_diagonal.evaluate --help
python -m experiments.mnist_n4_diagonal.plot --help
```

Use this study when you want the smallest canonical memorization pipeline in
the new stack.
