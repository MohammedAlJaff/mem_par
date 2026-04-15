"""
Lab 02 — Data blocks: GMMSubspace and TorusEmbedding.

What this covers:
    - Constructing both synthetic datasets with various settings
    - Inspecting data shapes, statistics, subspace geometry
    - DataLoader integration
    - The geometry_seed / data_seed separation (critical for fair comparison)
    - How ψ = D/d (manifold gap) varies across the d sweep

Run from project root:
    python reference/walkthroughs/02_data_blocks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from experiments.synthetic.high_d.common.data import GMMSubspace, TorusEmbedding
from experiments.synthetic.high_d.common.utils import seed_everything


def demo_gmm_shapes():
    print("\n── 1. GMMSubspace: shapes and construction ───────────────────────")
    D, d, K, n = 256, 16, 10, 500
    ds = GMMSubspace(D=D, d=d, K=K, sigma_z=0.5, sigma_eta=1e-3, n=n, geometry_seed=0, data_seed=0)

    print(f"  Params: D={D}, d={d}, K={K}, n={n}")
    print(f"  len(ds)          = {len(ds)}")
    print(f"  ds[0].shape      = {tuple(ds[0].shape)}")
    print(f"  data.shape       = {tuple(ds.data.shape)}")
    print(f"  intrinsic_dim    = {ds.intrinsic_dim}")
    print(f"  subspace_basis U : shape {tuple(ds.subspace_basis.shape)}  (D×d orthonormal)")
    print(f"  data mean        : {ds.data.mean():.4f}  (expect ~0)")
    print(f"  data std         : {ds.data.std():.4f}")

    # Variance explained by subspace
    U = ds.subspace_basis
    proj = ds.data @ U  # (n, d) coordinates in subspace
    x_rec = proj @ U.T  # (n, D) reconstructed
    var_in_subspace = x_rec.var().item()
    var_total = ds.data.var().item()
    print(f"  Variance in subspace : {var_in_subspace:.4f} / {var_total:.4f} = {var_in_subspace/var_total:.2%}")


def demo_gmm_d_sweep():
    print("\n── 2. GMMSubspace: the D/d manifold gap sweep ────────────────────")
    D = 256
    d_values = [4, 16, 32, 64, 128, 256]
    print(f"  Fixed D={D}, varying d → ψ = D/d is the manifold gap")
    print(f"  {'d':>6}  {'ψ=D/d':>8}  {'data.std':>9}  {'subspace var %':>15}")
    for d in d_values:
        ds = GMMSubspace(D=D, d=d, K=10, n=200, geometry_seed=0, data_seed=0)
        U = ds.subspace_basis
        proj = ds.data @ U
        x_rec = proj @ U.T
        subspace_var_frac = x_rec.var().item() / ds.data.var().item()
        psi = D / d
        print(f"  {d:>6}  {psi:>8.1f}  {ds.data.std().item():>9.4f}  {subspace_var_frac:>14.2%}")
    print("  ψ=1 → no gap (null control). ψ≫1 → strong gap (test case).")


def demo_gmm_seed_separation():
    print("\n── 3. GMMSubspace: geometry_seed ↔ data_seed separation ─────────")
    # Same geometry, different data → subspaces match, samples differ
    ds_a = GMMSubspace(D=64, d=4, K=5, n=100, geometry_seed=0, data_seed=0)
    ds_b = GMMSubspace(D=64, d=4, K=5, n=100, geometry_seed=0, data_seed=1)
    ds_c = GMMSubspace(D=64, d=4, K=5, n=100, geometry_seed=1, data_seed=0)

    same_geom = torch.allclose(ds_a.subspace_basis, ds_b.subspace_basis)
    diff_geom = not torch.allclose(ds_a.subspace_basis, ds_c.subspace_basis)
    diff_data = not torch.allclose(ds_a.data, ds_b.data)
    print(f"  Same geometry_seed, diff data_seed → same U?  {same_geom}  ✓")
    print(f"  Diff geometry_seed, same data_seed → diff U?  {diff_geom}  ✓")
    print(f"  Same geometry_seed, diff data_seed → diff data? {diff_data}  ✓")
    print("  KEY: in the paper all 9 pred×loss combos use geometry_seed=0, data_seed=run_seed.")
    print("       This guarantees IDENTICAL training sets — only the target differs.")


def demo_dataloader():
    print("\n── 4. DataLoader integration ─────────────────────────────────────")
    ds = GMMSubspace(D=32, d=4, K=5, n=100, geometry_seed=0, data_seed=0)
    loader = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
    batch = next(iter(loader))
    print(f"  Dataset size: {len(ds)}")
    print(f"  Batch size:   {batch.shape[0]}")
    print(f"  Batch shape:  {tuple(batch.shape)}")
    print(f"  Batch dtype:  {batch.dtype}")
    print(f"  Batches/epoch: {len(loader)}")


def demo_torus():
    print("\n── 5. TorusEmbedding ─────────────────────────────────────────────")
    D, d, n = 64, 2, 300
    tds = TorusEmbedding(D=D, d=d, hidden_mult=4, n=n, geometry_seed=0, data_seed=0)
    print(f"  Params: D={D}, d={d}, n={n}")
    print(f"  len(tds)       = {len(tds)}")
    print(f"  tds[0].shape   = {tuple(tds[0].shape)}")
    print(f"  intrinsic_dim  = {tds.intrinsic_dim}")
    print(f"  data mean      : {tds.data.mean():.4f}")
    print(f"  data std       : {tds.data.std():.4f}")
    print(f"  MLP W1.shape   : {tuple(tds.W1.shape)}  (frozen, geometry_seed=0)")
    print(f"  Topology       : d-torus (angles → cos/sin → frozen MLP → R^D)")
    print("  Why: tests that the parameterisation effect holds for NONLINEAR manifolds.")


def demo_torus_d_sweep():
    print("\n── 6. TorusEmbedding: d sweep ────────────────────────────────────")
    D = 64
    print(f"  Fixed D={D}, varying d (torus dimension)")
    print(f"  {'d':>4}  {'ψ=D/d':>8}  {'data.std':>9}")
    for d in [2, 4, 8]:
        tds = TorusEmbedding(D=D, d=d, hidden_mult=4, n=200, geometry_seed=0, data_seed=0)
        print(f"  {d:>4}  {D/d:>8.1f}  {tds.data.std().item():>9.4f}")


if __name__ == "__main__":
    seed_everything(0)
    print("=" * 65)
    print("  Lab 02: Data blocks (GMMSubspace + TorusEmbedding)")
    print("=" * 65)

    demo_gmm_shapes()
    demo_gmm_d_sweep()
    demo_gmm_seed_separation()
    demo_dataloader()
    demo_torus()
    demo_torus_d_sweep()

    print("\n" + "=" * 65)
    print("  Done. Data blocks understood.")
    print("=" * 65)
