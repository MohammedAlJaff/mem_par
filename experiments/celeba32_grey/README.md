# CelebA32 Grey

Canonical real-image study for 32x32 greyscale CelebA.

Default configuration:

- family: `vp`
- prediction/loss: `eps/eps`
- backbone: `unet`
- optimizer: `sgd`

Entry points:

```bash
python -m experiments.celeba32_grey.train --help
python -m experiments.celeba32_grey.evaluate --help
python -m experiments.celeba32_grey.plot --help
```

This is the current canonical image-study surface in the new stack.
