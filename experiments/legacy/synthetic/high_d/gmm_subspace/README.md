# GMM Subspace

Archived GMM subspace experiment from the older synthetic stack.

These commands still resolve through the legacy compatibility shim, but new
paper-facing work should use the top-level canonical study in
`experiments/gmm_subspace/`.

## Commands

```bash
python -m experiments.synthetic.high_d.gmm_subspace.run --device cpu --d 4 --n 500
python -m experiments.synthetic.high_d.gmm_subspace.sweep --dry-run
python -m experiments.synthetic.high_d.gmm_subspace.aggregate
```
