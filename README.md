# Equivalent in Principle, Unequal in Practice

**How target parameterization shapes memorization and generalization in diffusion and flow-based models**

This repository studies whether equivalent-in-principle parameterizations
(`x`, `eps`, `v`) remain unequal in practice when we hold architecture, data,
and training protocol fixed.

The repo is now organized around four clear surfaces:

- `src/`
  reusable first-party code
- `baselines/` and `third_party/`
  preserved external code
- `prototyping/`
  fallback and transitional prototype code
- `archives/`
  zipped snapshots of removed experiment and reference trees

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
- `prototyping/`
  fallback and transitional prototype code
- `archives/`
  zip files preserving the removed `experiments/` and `reference/` trees

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
├── baselines/
├── third_party/
├── prototyping/
├── archives/
│   ├── experiments_YYYYMMDD.zip
│   └── reference_YYYYMMDD.zip
├── tests/
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

## Live Code Surface

There is intentionally no live first-party `experiments/` tree right now.

The repo is in a tighter phase:
- build and harden reusable code in `src/`
- keep `prototyping/` as the scientific fallback
- recover older context only from the zip snapshots in `archives/`

## Safety And Reference Material

- `prototyping/gdfm_components.py` remains the fallback reference implementation
  while canonical GDFM hardens in `src/gdfm/core.py`.
- `archives/experiments_20260416.zip` preserves the removed experiment tree.
- `archives/reference_20260416.zip` preserves the removed reference tree.

## Notes

- `src/` is now the only live first-party code surface.
- If functionality from the archived trees is needed again, bring it back into
  `src/` deliberately instead of importing from old paths.
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
