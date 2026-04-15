# Equivalent in Principle, Unequal in Practice

**How target parameterization shapes memorization and generalization in diffusion and flow-based models**

This repository studies whether equivalent-in-principle parameterizations
(`x`, `eps`, `v`) remain unequal in practice when we hold architecture, data,
and training protocol fixed.

The repo is now organized around three clear surfaces:

- `src/`
  reusable first-party code
- `experiments/`
  canonical study-specific entrypoints and outputs
- `baselines/` and `third_party/`
  preserved external code

Non-canonical but useful material lives under `reference/` and `prototyping/`.

## Current Ownership Split

- `src/gdfm/`
  canonical GDFM model core, model setup, and backbones
- `src/mp_config/`
  base `ConfigDict` defaults and override helpers
- `src/mp_data/`
  canonical data loading and subset logic
- `src/mp_metrics/`
  memorization metrics and checkpoint replay
- `src/mp_artifacts/`
  run layout, config snapshots, checkpoints, and saved tensors
- `src/mp_training/`
  shared training loop scaffolding
- `experiments/<study>/`
  thin study-specific `config.py`, `train.py`, `evaluate.py`, `plot.py`
- `experiments/legacy/`
  preserved pre-reset experiment stack
- `reference/`
  walkthroughs, sanity checks, exploratory notebooks, and archive material
- `prototyping/`
  fallback and transitional prototype code

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
│   ├── mnist_n4_diagonal/
│   ├── mnist_nsweep/
│   ├── gmm_subspace/
│   ├── celeba32_grey/
│   └── legacy/
├── baselines/
├── third_party/
├── reference/
│   ├── walkthroughs/
│   ├── sanity/
│   ├── notebooks/
│   └── archive/
├── prototyping/
├── output/                # legacy artifacts only
├── outputs/               # scratch outputs only
├── requirements.txt
├── pyproject.toml
└── setup.py
```

## Setup

The repo-level environment target is Python 3.10.

```bash
mamba env create -n mem_param_env -f environment.yml
mamba activate mem_param_env
python -m pip install -r requirements.txt
```

`requirements.txt` installs the project in editable mode and pulls in the
supporting evaluation, notebook, and baseline packages used by the current
stack.

### GPU PyTorch on clusters

`environment.yml` intentionally leaves the PyTorch build choice to the machine.

For NVIDIA Linux clusters:

```bash
CONDA_OVERRIDE_CUDA=11.8 mamba install -y -c conda-forge "pytorch==2.5.1=cuda*" "torchvision=*=cuda*"
python -m pip install -r requirements.txt
```

For CPU-only Linux or Apple Silicon:

```bash
python -m pip install -r requirements.txt
```

If you are running directly from a checkout that has not been installed in
editable mode yet, prefix local commands with `PYTHONPATH=src`.

## Canonical Study Entry Points

```bash
python -m experiments.mnist_n4_diagonal.train --help
python -m experiments.mnist_n4_diagonal.evaluate --help
python -m experiments.mnist_n4_diagonal.plot --help
python -m experiments.mnist_nsweep.train --help
python -m experiments.gmm_subspace.train --help
python -m experiments.celeba32_grey.train --help
```

Each active study folder owns:

- `config.py`
- `train.py`
- `evaluate.py`
- `plot.py`
- local `runs/`
- local `figures/`

## Safety And Reference Material

- `prototyping/gdfm_components.py` remains the fallback reference implementation
  while canonical GDFM hardens in `src/gdfm/core.py`.
- `reference/` contains pedagogy, sanity checks, notebooks, and archived
  context. Useful, but non-canonical.
- `experiments/legacy/` keeps the old experiment stack readable and importable,
  but it is not the source of truth for new first-party work.

## Notes

- Canonical outputs stay inside the owning experiment folder.
- Root `output/` and `outputs/` are legacy or scratch only.
- `baselines/` and `third_party/` stay separate from the canonical first-party
  stack.

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
