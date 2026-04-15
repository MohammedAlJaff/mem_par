# Notebooks

These notebooks are reference-facing interactive assets.

They are useful for local investigation, sanity checks, and pedagogical
inspection, but they are not the canonical runtime surface for the repo.

New paper-facing study logic should live in:

- `src/` for reusable code
- `experiments/` for active study entrypoints

Current notebooks of interest:

- `skimage_memorization_sanity.ipynb`
  Interactive sanity check for the canonical `mp_metrics` memorization stack
  using real grayscale images from `skimage.data`.
- `mnist_minimal_run.ipynb`
  Older exploratory notebook kept for historical context.
- `proto_play.ipynb`
  Older exploratory notebook kept for historical context.

If a notebook turns into a stable teaching asset, prefer moving the idea into
`reference/walkthroughs/` or `reference/sanity/`.
