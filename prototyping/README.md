# Prototyping

This folder contains prototype and transition material.

It is still valuable because it holds:

- the cleaned `gdfm_components.py` style and notation exemplar
- the fallback GDFM reference implementation
- sidecar prototype samplers and small transitional helpers

It is not the canonical home for new first-party study infrastructure.

Canonical new work belongs in:

- `src/gdfm/` for model-core code
- `src/mp_*/` for shared study infrastructure
- `experiments/` for runnable first-party study modules

When canonical GDFM changes, keep the prototype-vs-canonical parity tests green
so this folder remains a trustworthy fallback.
