# CelebA32 Grey UNet

Archived CelebA32 validation experiment from the older natural-images stack.

New canonical image work now lives at `experiments/celeba32_grey/`.

## Commands

```bash
python -m experiments.natural_images.celeba32_grey_unet.run --device cpu --n 200
python -m experiments.natural_images.celeba32_grey_unet.sweep --dry-run
python -m experiments.natural_images.celeba32_grey_unet.aggregate
```
