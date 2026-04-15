"""
Sanity check: train GDFM on 2D toy data and visualize samples.

Usage:
    python train_2d.py                          # default: ring, fm, x-pred, v-loss
    python train_2d.py --dataset moons          # two moons
    python train_2d.py --pred eps --loss eps     # eps-pred, eps-loss
    python train_2d.py --family vp              # VP diffusion
    python train_2d.py --all_cells              # train all 9 cells, one plot

This is a Tier-0 sanity check. If any cell fails to learn a 2D distribution
where d = D = 2, something is broken. The manifold-gap hypothesis predicts
NO difference between parameterizations here.

All cells sample from the SAME initial noise for fair visual comparison.
"""

import argparse
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from gdfm_components import GDFM, GDFMConfig


# ============================================================
# Datasets
# ============================================================


def sample_ring(n, noise=0.05):
    """Unit circle with Gaussian noise."""
    theta = torch.rand(n) * 2 * math.pi
    x = torch.stack([theta.cos(), theta.sin()], dim=1)
    return x + noise * torch.randn_like(x)


def sample_moons(n, noise=0.05):
    """Two moons (sklearn)."""
    from sklearn.datasets import make_moons

    data, _ = make_moons(n_samples=n, noise=noise)
    return torch.tensor(data, dtype=torch.float32)


def sample_gaussians(n, n_clusters=8, std=0.05):
    """Ring of Gaussians."""
    angles = torch.linspace(0, 2 * math.pi, n_clusters + 1)[:-1]
    centers = torch.stack([angles.cos(), angles.sin()], dim=1) * 2.0
    idx = torch.randint(0, n_clusters, (n,))
    return centers[idx] + std * torch.randn(n, 2)


DATASETS = {
    "ring": sample_ring,
    "moons": sample_moons,
    "gaussians": sample_gaussians,
}


# ============================================================
# Training
# ============================================================


def train(model, data, n_steps=5000, lr=3e-4, batch_size=256):
    """Simple training loop. Returns loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = data.shape[0]
    losses = []

    for step in range(n_steps):
        idx = torch.randint(0, n, (batch_size,))
        loss = model.compute_loss(data[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 1000 == 0:
            avg = np.mean(losses[-100:])
            print(f"  step {step + 1:5d}  loss={avg:.4f}")

    return losses


# ============================================================
# Visualization
# ============================================================


def plot_samples(data_np, samples_np, title, ax):
    """Plot training data + generated samples on a single axis."""
    ax.scatter(data_np[:, 0], data_np[:, 1], s=1, alpha=0.3, c="gray", label="data")
    ax.scatter(
        samples_np[:, 0],
        samples_np[:, 1],
        s=1,
        alpha=0.3,
        c="tab:blue",
        label="generated",
    )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.legend(fontsize=6, loc="upper right")


def run_single(args):
    """Train one configuration and plot."""
    print(f"\n{'=' * 50}")
    print(
        f"family={args.family}  pred={args.pred}  loss={args.loss}  data={args.dataset}"
    )
    print(f"{'=' * 50}")

    data = DATASETS[args.dataset](n=args.n_data, noise=args.noise)

    cfg = GDFMConfig(
        family=args.family,
        prediction_space=args.pred,
        loss_space=args.loss,
        data_dim=2,
        hidden_dim=128,
        n_layers=3,
        time_emb_dim=32,
    )
    model = GDFM(cfg)
    print(model)

    losses = train(
        model, data, n_steps=args.n_steps, lr=args.lr, batch_size=args.batch_size
    )

    # shared initial noise
    z_init = torch.randn(2000, 2)
    samples = model.sample(z_init.clone(), n_steps=200, method="euler").numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_samples(data.numpy(), samples, f"pred={args.pred}, loss={args.loss}", axes[0])

    # loss curve
    axes[1].plot(losses, alpha=0.3, c="gray")
    window = min(100, len(losses))
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
    axes[1].plot(smoothed, c="tab:blue")
    axes[1].set_title("Training loss")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("loss")

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    fname = out / f"gdfm_{args.family}_{args.pred}_{args.loss}_{args.dataset}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close()


def run_all_cells(args):
    """Train all 9 cells of the 3x3 grid, plot in one figure."""
    print(f"\n{'=' * 50}")
    print(f"All 9 cells | family={args.family}  data={args.dataset}")
    print(f"{'=' * 50}")

    data = DATASETS[args.dataset](n=args.n_data, noise=args.noise)
    data_np = data.numpy()
    spaces = ["x", "eps", "v"]

    # same initial noise for all 9 cells — fair comparison
    z_init = torch.randn(2000, 2)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(
        f"3x3 grid | family={args.family} | data={args.dataset} | "
        f"n={args.n_data} | {args.n_steps} steps",
        fontsize=12,
    )

    for i, pred in enumerate(spaces):
        for j, loss_space in enumerate(spaces):
            ax = axes[i, j]
            print(f"\npred={pred}, loss={loss_space}")

            cfg = GDFMConfig(
                family=args.family,
                prediction_space=pred,
                loss_space=loss_space,
                data_dim=2,
                hidden_dim=128,
                n_layers=3,
                time_emb_dim=32,
            )
            model = GDFM(cfg)
            train(
                model,
                data,
                n_steps=args.n_steps,
                lr=args.lr,
                batch_size=args.batch_size,
            )

            # same z_init for every cell
            samples = model.sample(z_init.clone(), n_steps=200, method="euler").numpy()
            plot_samples(data_np, samples, f"pred={pred}, loss={loss_space}", ax)

            # row/column labels
            if i == 0:
                ax.set_title(f"loss={loss_space}", fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"pred={pred}", fontsize=10, fontweight="bold")

    fig.tight_layout()
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    fname = out / f"gdfm_3x3_{args.family}_{args.dataset}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fname}")
    plt.close()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDFM 2D sanity check")
    parser.add_argument("--dataset", type=str, default="ring", choices=DATASETS.keys())
    parser.add_argument("--family", type=str, default="fm", choices=["fm", "vp"])
    parser.add_argument("--pred", type=str, default="x", choices=["x", "eps", "v"])
    parser.add_argument("--loss", type=str, default="x", choices=["x", "eps", "v"])
    parser.add_argument("--n_data", type=int, default=2000)
    parser.add_argument("--n_steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--noise", type=float, default=0.001)
    parser.add_argument("--all_cells", action="store_true", help="Train all 9 cells")
    args = parser.parse_args()

    if args.all_cells:
        run_all_cells(args)
    else:
        run_single(args)
