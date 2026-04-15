# GMM Subspace

Canonical high-dimensional synthetic study built around a Gaussian mixture on a
linear subspace.

Default configuration:

- family: `fm`
- prediction/loss: `x/x`
- backbone: `mlp`
- ambient dimension: `256`
- intrinsic dimension: `4`

Entry points:

```bash
python -m experiments.gmm_subspace.train --help
python -m experiments.gmm_subspace.evaluate --help
python -m experiments.gmm_subspace.plot --help
```

This study is the current canonical synthetic high-dimensional path in the new
experiment stack.
