# Experiments

This tree is the canonical home for experiment-specific code and scripts.

New first-party experiments should live at the top level of this folder and
stay thin:
- `config.py`
- `train.py`
- `evaluate.py`
- `plot.py`
- local `runs/`
- local `figures/`

Shared reusable experiment machinery lives under `src/`.

Older experiment code is preserved under `experiments/legacy/`.
