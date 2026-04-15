# Legacy Experiments

This subtree preserves the older experiment stack as reference material.

Use it for:

- reading old implementation ideas
- understanding historical runs and layouts
- recovering archived utilities that may still inform the new system

Do not use it as the source of truth for new first-party experiment code.

New work should instead go to:

- `src/gdfm/` for model-core changes
- `src/mp_*` for reusable study infrastructure
- top-level `experiments/<study>/` for active canonical studies

Compatibility shims at `experiments.synthetic` and `experiments.natural_images`
continue to point here so archived imports remain readable.
