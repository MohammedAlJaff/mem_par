# Parameterization Boundary

**Does prediction parameterization shift the memorization–generalization phase boundary in flow matching models?**

This repo is organized around experiments as the primary scientific unit.

## Quick Start

```bash
pip install -r requirements.txt

# Canonical synthetic smoke test
python -m experiments.synthetic.high_d.gmm_subspace.run --device cpu --d 4 --n 500

# Sweep generation
python -m experiments.synthetic.high_d.gmm_subspace.sweep --dry-run

# Aggregate local results
python -m experiments.synthetic.high_d.gmm_subspace.aggregate

# Baseline sandbox
python -m baselines.flow_matching.run train-2d --dataset checkerboard
```

## Layout

```text
parameterization_boundary/
├── experiments/
│   ├── synthetic/
│   │   ├── low_d/
│   │   │   ├── common/
│   │   │   └── spiral_mem_demo/
│   │   ├── high_d/
│   │   │   ├── common/
│   │   │   ├── gmm_subspace/
│   │   │   └── torus/
│   │   └── mnist/
│   │       └── common/
│   └── natural_images/
│       ├── common/
│       ├── celeba32_grey_unet/
│       ├── afhq_cat_64_jit/
│       ├── afhq_cat_64_unet/
│       ├── afhq_cat_128_jit/
│       └── afhq_cat_256_jit/
├── baselines/
│   ├── flow_matching/
│   ├── jit/
│   └── bonnaire/
├── lab/
├── notebooks/
├── output/                # legacy artifacts only
├── requirements.txt
├── pyproject.toml
└── setup.py
```

## Canonical Experiment Modules

- `python -m experiments.synthetic.high_d.gmm_subspace.run`
- `python -m experiments.synthetic.high_d.torus.run`
- `python -m experiments.synthetic.low_d.spiral_mem_demo.run`
- `python -m experiments.synthetic.mnist.<experiment>.run`
- `python -m experiments.natural_images.celeba32_grey_unet.run`
- `python -m experiments.natural_images.afhq_cat_64_jit.run`
- `python -m experiments.natural_images.afhq_cat_64_unet.run`
- `python -m experiments.natural_images.afhq_cat_128_jit.run`
- `python -m experiments.natural_images.afhq_cat_256_jit.run`

Each experiment folder contains:

- `config.yaml` for scientific settings and sweep axes
- `paths.example.yaml` for machine-specific roots and SLURM settings
- `run.py`, `sweep.py`, and `aggregate.py`
- local `runs/` and `figures/`

## Notes

- New experiment outputs stay inside the owning experiment folder.
- Root `output/` is treated as legacy history and is not part of the canonical workflow.
- Labs and baselines remain first-class, but they are not the source of truth for paper experiments.

## Citation

```bibtex
@article{aljaff2026equivalent,
  title={Equivalent in Principle, Unequal in Practice: How Parametrization 
         Shapes the Memorization--Generalization Phase Boundary in 
         Score-Based Generative Models},
  author={Al-Jaff, Mohammad and Henter, Gustav Eje and Azizpour, Hossein 
          and Kragic, Danica},
  journal={arXiv preprint},
  year={2026}
}
```
