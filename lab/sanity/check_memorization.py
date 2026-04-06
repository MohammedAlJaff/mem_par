"""
Structured 3x3 grid sanity check with light memorization-oriented diagnostics.

Despite the filename, this is best read as a parameterization sanity script.
Historically it grew out of memorization experiments, so it still reports a
sample-to-data nearest-neighbor statistic, but it is not a formal detector of
memorization.

The main teaching goal is to separate three questions:
    1. Can every cell in the 3x3 grid train and sample at all?
    2. Do runs that share a `pred_space` behave similarly across rows?
    3. Do different `pred_space` choices lead to measurably different outputs?

This script trains all 9 (pred_space, loss_space) combinations on the same
toy 2D dataset for a short run and checks a few coarse end-to-end behaviors:

    1. All 9 configurations train stably (loss decreases)
    2. Configurations sharing a pred_space have correlated loss curves
    3. Different pred_spaces produce measurably different sample clouds
    4. Diagonal entries (pred=loss) are often easier to optimize, so we
       report whether they tend to converge faster, but this is only a
       heuristic sanity signal, not a hard requirement

What this script is good for:
    - Catching obvious wiring bugs in the 3×3 grid
    - Catching catastrophic conversion mistakes that break training/sampling

What this script is NOT good for:
    - Proving the conversion functions are algebraically correct
    - Proving any optimization theorem about diagonal entries

For exact correctness of conversions, add a separate round-trip algebraic test.

Usage:
    python lab/sanity/check_memorization.py --device cpu
    python lab/sanity/check_memorization.py --device cuda --save-dir outputs/sanity_3x3
"""

import sys
from pathlib import Path
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.synthetic.low_d.common.data.toy2d import sample_8gaussians
from experiments.synthetic.low_d.common.flow_matching import FlowMatching
from experiments.synthetic.low_d.common.networks import TimeConditionedMLP
from experiments.synthetic.low_d.common.utils import seed_everything


PRED_SPACES = ["x", "eps", "v"]
LOSS_SPACES = ["x", "eps", "v"]


def moving_average(x, window=25):
    """Smooth a noisy loss curve so the plot shows trend rather than jitter."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def curve_corr(a, b, burn_in=25):
    """
    Compare the shapes of two loss curves after removing the startup transient.

    We ignore the first few iterations because random initialization can create
    noisy, uninformative movement before the optimization dynamics settle.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a = a[burn_in:]
    b = b[burn_in:]

    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]

    if n < 5:
        return np.nan

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def pairwise_mean_distance(sample_sets):
    """
    sample_sets: dict[pred_space] -> ndarray of shape (N, D)

    Returns pairwise distances between the mean of each sample cloud.

    This is intentionally a very cheap summary statistic. It will not tell us
    whether the full distributions differ, but it is good enough to catch the
    degenerate case where every pred_space produces almost the same cloud.
    """
    means = {k: v.mean(axis=0) for k, v in sample_sets.items()}
    keys = sorted(means.keys())
    dists = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            d = float(np.linalg.norm(means[k1] - means[k2]))
            dists[(k1, k2)] = d
    return dists


def sample_quality_stats(samples, data):
    """
    Very crude checks:
      - finiteness
      - boundedness
      - average nearest distance to data

    The nearest-neighbor distance is only a rough alarm bell:
        - very large values suggest the sampler missed the data manifold
        - very small values can mean either good alignment or overfitting
    so it must be interpreted with care.
    """
    finite = bool(torch.isfinite(samples).all().item())
    in_range = bool((samples.abs() < 20).all().item())

    # For tiny toy datasets, this is a fast way to quantify whether generated
    # points land anywhere near the support of the data.
    dmat = torch.cdist(samples, data)
    nn_dist = dmat.min(dim=1).values.mean().item()

    return {
        "finite": finite,
        "in_range": in_range,
        "mean_nn_dist_to_data": float(nn_dist),
    }


def train_one_config(
    pred_space,
    loss_space,
    data,
    device="cpu",
    n_steps=400,
    lr=5e-4,
    hidden_dim=256,
    depth=3,
    time_embed_dim=32,
    sample_n=500,
    sample_steps=100,
    sample_method="heun",
):
    """
    Train one cell of the 3x3 grid and return both optimization and sample stats.

    Keeping this logic in one function makes it easier to compare cells fairly:
    every configuration uses the same network size, optimizer, seed, and data.
    """
    # Re-seed every run so differences across cells are attributable to the
    # grid choice rather than random initialization.
    seed_everything(0)

    net = TimeConditionedMLP(
        input_dim=2,
        hidden_dim=hidden_dim,
        depth=depth,
        time_embed_dim=time_embed_dim,
    )
    model = FlowMatching(
        net,
        pred_space=pred_space,
        loss_space=loss_space,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    grad_norms = []

    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = model.compute_loss(data)
        loss.backward()

        # Early gradient norms are a lightweight proxy for how "hard" the
        # optimization path is under each parameterization choice.
        total_sq_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total_sq_norm += g.pow(2).sum().item()
        grad_norms.append(total_sq_norm**0.5)

        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        samples = model.sample(
            shape=(sample_n, 2),
            n_steps=sample_steps,
            method=sample_method,
            device=device,
        )

    initial = float(np.mean(losses[:10]))
    final = float(np.mean(losses[-10:]))
    rel_drop = (initial - final) / max(abs(initial), 1e-12)
    decreased = final < initial

    stats = sample_quality_stats(samples, data)

    return {
        "model": model,
        "losses": losses,
        "grad_norms": grad_norms,
        "initial_loss": initial,
        "final_loss": final,
        "relative_loss_drop": float(rel_drop),
        "decreased": bool(decreased),
        "samples": samples.detach().cpu(),
        **stats,
    }


def plot_dataset(data, save_path=None):
    """Plot the toy dataset once so all sample plots can be interpreted against it."""
    x = data.detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(x[:, 0], x[:, 1], alpha=0.6, s=12)
    plt.title("8 Gaussians dataset")
    plt.axis("equal")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_loss_grid(results, save_path=None):
    """Show one loss curve per grid cell so row/column patterns are easy to inspect."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)

    for i, pred in enumerate(PRED_SPACES):
        for j, loss in enumerate(LOSS_SPACES):
            ax = axes[i, j]
            curve = results[(pred, loss)]["losses"]
            smooth = moving_average(curve, window=25)
            ax.plot(smooth)
            ax.set_title(f"pred={pred}, loss={loss}")
            ax.set_xlabel("step")
            ax.set_ylabel("loss")

    fig.suptitle("Loss curves across 3×3 grid", y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_samples_grid(results, data, save_path=None):
    """Overlay generated samples on the dataset for each prediction/loss combination."""
    data_np = data.detach().cpu().numpy()
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)

    for i, pred in enumerate(PRED_SPACES):
        for j, loss in enumerate(LOSS_SPACES):
            ax = axes[i, j]
            samples = results[(pred, loss)]["samples"].numpy()
            ax.scatter(data_np[:, 0], data_np[:, 1], alpha=0.15, s=8, label="data")
            ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=8, label="samples")
            ax.set_title(f"pred={pred}, loss={loss}")
            ax.axis("equal")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Generated samples across 3×3 grid", y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main(device="cpu", save_dir=None):
    """
    Run the structured grid diagnostic and optionally save the resulting figures.

    The output is intentionally "teaching friendly": each block prints what is
    being checked and how strongly we should trust that signal.
    """
    print("=" * 72)
    print("SANITY CHECK: 3×3 prediction/loss grid")
    print("=" * 72)

    seed_everything(0)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    print("Generating 8 Gaussians dataset...")
    data = sample_8gaussians(500, seed=0).to(device)

    plot_dataset(
        data,
        save_path=(save_dir / "dataset.png") if save_dir is not None else None,
    )

    # `results[(pred, loss)]` stores everything we need for the summary checks
    # and the final plots.
    results = {}
    all_passed = True

    for pred, loss in product(PRED_SPACES, LOSS_SPACES):
        label = f"pred={pred}, loss={loss}"
        print(f"\nTraining: {label}")

        out = train_one_config(
            pred_space=pred,
            loss_space=loss,
            data=data,
            device=device,
            n_steps=400,
            lr=5e-4,
            hidden_dim=256,
            depth=3,
            time_embed_dim=32,
            sample_n=500,
            sample_steps=100,
            sample_method="heun",
        )

        passed = out["decreased"] and out["finite"] and out["in_range"]
        all_passed = all_passed and passed

        results[(pred, loss)] = out

        status = "✓" if passed else "✗"
        print(
            f"  {status} {label:20s} | "
            f"loss: {out['initial_loss']:.4f} -> {out['final_loss']:.4f} | "
            f"rel_drop={out['relative_loss_drop']:.3f} | "
            f"finite={out['finite']} | in_range={out['in_range']} | "
            f"mean_nn_to_data={out['mean_nn_dist_to_data']:.4f}"
        )

    print("\n" + "-" * 72)
    print("CHECK 1: Stable training")
    print("-" * 72)
    print(
        f"{'PASS' if all_passed else 'FAIL'}: all 9 configs decreased loss and sampled stably"
    )

    print("\n" + "-" * 72)
    print("CHECK 2: Loss-curve correlation within shared pred_space")
    print("-" * 72)
    corr_summary = {}
    for pred in PRED_SPACES:
        # Within a fixed column, we vary only the loss space. If Li & He's
        # observation is roughly right, these curves should still resemble each
        # other more than curves from different columns.
        pairs = [("x", "eps"), ("x", "v"), ("eps", "v")]
        vals = []
        for a, b in pairs:
            c = curve_corr(
                results[(pred, a)]["losses"],
                results[(pred, b)]["losses"],
                burn_in=25,
            )
            vals.append(c)
            print(f"pred={pred:3s} | corr(loss={a}, loss={b}) = {c:.4f}")
        corr_summary[pred] = float(np.nanmean(vals))

    mean_within_pred_corr = float(np.nanmean(list(corr_summary.values())))
    corr_pass = mean_within_pred_corr > 0.5
    print(
        f"Mean within-pred correlation = {mean_within_pred_corr:.4f} "
        f"| {'PASS' if corr_pass else 'WARN'}"
    )

    print("\n" + "-" * 72)
    print("CHECK 3: Different pred_spaces give different sample clouds")
    print("-" * 72)

    # Concatenate rows within each column to get one coarse "pred_space cloud".
    pred_space_samples = {}
    for pred in PRED_SPACES:
        pred_space_samples[pred] = np.concatenate(
            [results[(pred, loss)]["samples"].numpy() for loss in LOSS_SPACES],
            axis=0,
        )

    mean_dists = pairwise_mean_distance(pred_space_samples)
    for (a, b), d in mean_dists.items():
        print(f"mean-cloud distance between pred={a} and pred={b}: {d:.4f}")

    sample_diff_pass = any(d > 0.05 for d in mean_dists.values())
    print(
        f"{'PASS' if sample_diff_pass else 'WARN'}: pred spaces are not identical in sample mean"
    )

    print("\n" + "-" * 72)
    print("CHECK 4: Diagonal entries as easier optimization paths (heuristic only)")
    print("-" * 72)

    # Diagonal cells avoid a conversion in the final loss computation, so they
    # are sometimes easier to optimize. This is a suggestive pattern, not a law.
    diagonal = [(s, s) for s in PRED_SPACES]
    off_diagonal = [(p, l) for p, l in product(PRED_SPACES, LOSS_SPACES) if p != l]

    diag_final = np.mean([results[k]["final_loss"] for k in diagonal])
    off_final = np.mean([results[k]["final_loss"] for k in off_diagonal])

    diag_drop = np.mean([results[k]["relative_loss_drop"] for k in diagonal])
    off_drop = np.mean([results[k]["relative_loss_drop"] for k in off_diagonal])

    diag_grad = np.mean([np.mean(results[k]["grad_norms"][:50]) for k in diagonal])
    off_grad = np.mean([np.mean(results[k]["grad_norms"][:50]) for k in off_diagonal])

    print(f"avg diagonal final loss      = {diag_final:.4f}")
    print(f"avg off-diagonal final loss  = {off_final:.4f}")
    print(f"avg diagonal relative drop   = {diag_drop:.4f}")
    print(f"avg off-diagonal rel. drop   = {off_drop:.4f}")
    print(f"avg diagonal grad norm (early)= {diag_grad:.4f}")
    print(f"avg off-diagonal grad norm    = {off_grad:.4f}")

    diag_heuristic = (diag_final <= off_final) and (diag_drop >= off_drop)
    print(
        f"{'PASS' if diag_heuristic else 'WARN'}: diagonal configs look easier to optimize"
    )

    if save_dir is not None:
        plot_loss_grid(results, save_path=save_dir / "loss_grid.png")
        plot_samples_grid(results, data, save_path=save_dir / "samples_grid.png")
    else:
        plot_loss_grid(results)
        plot_samples_grid(results, data)

    print("\n" + "=" * 72)
    if all_passed:
        print("OVERALL RESULT: PASS")
    else:
        print("OVERALL RESULT: FAIL")
        print("At least one configuration did not train/sample stably.")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run a structured 3x3 prediction/loss-grid sanity check and "
            "optionally save summary figures."
        )
    )
    parser.add_argument("--device", default="cpu", help="Torch device to run on.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional directory where dataset/loss/sample figures are saved.",
    )
    args = parser.parse_args()

    main(device=args.device, save_dir=args.save_dir)
