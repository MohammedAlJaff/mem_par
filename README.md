# Equivalent in Principle, Unequal in Practice

**How target parameterization shapes memorization and generalization in diffusion and flow-based models**

This repo now has a clearer ownership split:

- `src/` is the canonical umbrella for reusable first-party code
- `src/gdfm/` owns the model core and backbone setup
- `src/mp_*/` owns thin reusable study support code such as config, data, metrics, artifacts, and training
- `experiments/` is the canonical surface for experiment-specific code and scripts
- `baselines/` and `third_party/` remain preserved vendor/reference code
- `prototyping/`, `lab/`, and `notebooks/` remain reference material

## Layout

```text
mem_par/
├── src/
│   ├── gdfm/
│   ├── mp_config/
│   ├── mp_data/
│   ├── mp_metrics/
│   ├── mp_artifacts/
│   └── mp_training/
├── experiments/
│   ├── _shared/
│   ├── mnist_n4_diagonal/
│   ├── mnist_nsweep/
│   ├── gmm_subspace/
│   ├── celeba32_grey/
│   └── legacy/
├── baselines/
│   ├── flow_matching/
│   ├── jit/
│   └── bonnaire/
├── lab/                   # pedagogy and sanity material
├── notebooks/             # exploratory notebooks
├── prototyping/           # prototype fallback/reference material
├── output/                # legacy artifacts only
├── outputs/               # scratch outputs only
├── requirements.txt
├── pyproject.toml
└── setup.py
```

## Setup

The repo-level environment target is Python 3.10. A simple setup flow is:

```bash
mamba env create -n mem_param_env -f environment.yml
mamba activate mem_param_env
python -m pip install -r requirements.txt
```

`requirements.txt` installs:

- the main package in editable mode via `-e .`
- the vendored `third_party/torch-fidelity` package for the JiT baseline
- evaluation packages such as `clean-fid` and `lpips`
- notebook tooling such as `jupyterlab`
- baseline extras such as `natsort` and `pytorch-fid`

### GPU PyTorch on clusters

`environment.yml` intentionally keeps PyTorch out of the base environment so each machine can
install the right build afterward.

For NVIDIA Linux clusters, install a CUDA-enabled PyTorch build after activating the env:

```bash
CONDA_OVERRIDE_CUDA=11.8 mamba install -y -c conda-forge "pytorch==2.5.1=cuda*" "torchvision=*=cuda*"
python -m pip install -r requirements.txt
```

For CPU-only Linux or Apple Silicon, just activate the env and install the requirements:

```bash
python -m pip install -r requirements.txt
```

If the environment already exists, update it with:

```bash
mamba env update -n mem_param_env -f environment.yml --prune
mamba activate mem_param_env
python -m pip install -r requirements.txt
```

Quick sanity checks:

```bash
python -c "import torch, torchvision, yaml, sklearn, cleanfid, lpips, natsort, pytorch_fid, ml_collections; print(torch.__version__)"
python -m experiments.gmm_subspace.train --help
```

## Canonical Experiment Modules

- `python -m experiments.mnist_n4_diagonal.train`
- `python -m experiments.mnist_n4_diagonal.evaluate`
- `python -m experiments.mnist_n4_diagonal.plot`
- `python -m experiments.mnist_nsweep.train`
- `python -m experiments.gmm_subspace.train`
- `python -m experiments.celeba32_grey.train`

Each experiment folder contains:

- `config.py` returning a canonical `ConfigDict`
- `train.py`, `evaluate.py`, and `plot.py`
- local `runs/` and `figures/`

## Notes

- New canonical outputs stay inside the owning experiment folder.
- Root `output/` and `outputs/` are legacy/scratch only and are not part of the canonical workflow.
- Older experiment code is preserved under `experiments/legacy/`.
- Labs, notebooks, and prototypes remain useful references, but they are not canonical runtime surfaces.

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
