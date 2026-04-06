# Equivalent in Principle, Unequal in Practice

**How target parameterization shapes memorization and generalization in diffusion and flow-based models**

This repo is organized around experiments as the primary scientific unit for studying how target parameterization affects memorization and generalization.

## Layout

```text
mem_par/
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

## Setup

The repo-level environment target is Python 3.10. A good remote setup flow is:

```bash
mamba env create -f environment.yml
mamba activate mem-par
```

Then install the PyTorch build that matches the machine:

```bash
# Apple Silicon or CPU-only Linux
pip install "torch>=2.1" "torchvision>=0.16"

# NVIDIA Linux with CUDA 12.4
pip install "torch>=2.1" "torchvision>=0.16" --index-url https://download.pytorch.org/whl/cu124

# NVIDIA Linux with CUDA 12.1
pip install "torch>=2.1" "torchvision>=0.16" --index-url https://download.pytorch.org/whl/cu121
```

Finally install the repo requirements:

```bash
pip install -r requirements.txt
```

That installs:

- the main package in editable mode via `-e .`
- the vendored `third_party/torch-fidelity` package for the JiT baseline
- evaluation packages such as `clean-fid` and `lpips`
- baseline extras such as `natsort` and `pytorch-fid`

If the environment already exists, update it with:

```bash
mamba env update -f environment.yml --prune
mamba activate mem-par
pip install -r requirements.txt
```

Quick sanity checks:

```bash
python -c "import torch, torchvision, yaml, sklearn, cleanfid, lpips, natsort, pytorch_fid; print(torch.__version__)"
python -m experiments.synthetic.low_d.spiral_mem_demo.run --help
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
  title={Equivalent in Principle, Unequal in Practice: How Target Parameterization
         Shapes Memorization and Generalization in Diffusion and Flow-Based Models},
  author={Al-Jaff, Mohammad and Henter, Gustav Eje and Azizpour, Hossein 
          and Kragic, Danica},
  journal={arXiv preprint},
  year={2026}
}
```
