# Synthetic Data Sanity

This folder contains notebook-first checks for the first-party synthetic data
generators.

The current notebook is:

- `gmm_data_sanity.ipynb`
  A line-by-line sanity notebook for the canonical `gmm_subspace` loader. It
  checks a true 2D GMM, a projected high-dimensional GMM, and a fixed `d/D`
  sweep where intrinsic dimension increases with ambient dimension.

This is intentionally data-focused. It does not train models; it verifies that
the generated tensors, labels, subspace geometry, projections, embeddings, and
clustering behavior match what we expect before these datasets are used in
training sweeps.
