"""
Manual, visual-first sanity check for the 3x3 prediction/loss grid.

Pedagogical goal:
    Hold the architecture fixed and vary only two choices:
        1. `pred_space`: what the network is asked to predict
        2. `loss_space`: which target space the MSE is computed in

    Running all 9 combinations side by side helps answer:
        - Do all cells train at all?
        - Do some columns (`pred_space`) behave noticeably differently?
        - Do samples stay finite and roughly on the same scale as the data?

This script is intentionally interactive and visual. With the default
settings it uses a tiny dataset and a long optimization run, so it behaves
more like an overfit stress test than a quick benchmark. That makes gross
conversion bugs or target-construction mistakes easy to spot by eye.

For a shorter and more structured diagnostic with summary plots and
heuristic metrics, see `reference/sanity/check_memorization.py`.

Usage:
    python reference/sanity/check_3x3_grid.py
    python reference/sanity/check_3x3_grid.py --device cuda --train-steps 2000
"""

import sys
from pathlib import Path
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

sns.set_style("dark")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.synthetic.low_d.common.data.toy2d import sample_8gaussians
from experiments.synthetic.low_d.common.flow_matching import FlowMatching
from experiments.synthetic.low_d.common.networks import TimeConditionedMLP
from experiments.synthetic.low_d.common.utils import seed_everything


PRED_SPACES = ["x", "eps", "v"]
LOSS_SPACES = ["x", "eps", "v"]


def main(
    device: str = "cpu",
    dataset_size: int = 5,
    train_steps: int = 50_000,
    sample_count: int = 500,
):
    """Run the manual 3x3 smoke test and show a plot for each configuration."""
    print("=" * 60)
    print("SANITY CHECK: 3×3 prediction/loss grid")
    print("=" * 60)

    # Fix the global seed once so the dataset and plotting order are reproducible.
    seed_everything(0)

    print("Generating 8 Gaussians dataset...")
    # The default dataset is intentionally tiny: if a grid cell is wired
    # incorrectly, the failure tends to show up quickly and dramatically.
    data = sample_8gaussians(dataset_size, seed=0)

    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title(f"8 Gaussians Dataset ({dataset_size} training points)")
    plt.show()

    # Store a compact summary for the final pass/fail report.
    results = {}
    all_passed = True

    for pred, loss in product(PRED_SPACES, LOSS_SPACES):
        label = f"pred={pred}, loss={loss}"

        print(f"\nTraining: {label}")

        # Re-seeding here ensures that differences across cells come from the
        # parameterization choice rather than a lucky initialization.
        seed_everything(0)

        print(f"Initializing model with pred_space={pred}, loss_space={loss}")
        net = TimeConditionedMLP(
            input_dim=2, hidden_dim=256, depth=3, time_embed_dim=32
        )
        model = FlowMatching(net, pred_space=pred, loss_space=loss).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

        losses = []
        data_device = data.to(device)
        for _ in tqdm(range(train_steps), desc=f"Training {label}"):
            optimizer.zero_grad()
            loss = model.compute_loss(data_device)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Compare the first and last few steps rather than single points.
        # This keeps the smoke test from overreacting to noisy iterations.
        initial = sum(losses[:10]) / 10
        final = sum(losses[-10:]) / 10
        decreased = final < initial * 0.5

        print(f"Generating samples with pred_space={pred}, loss_space={loss}")
        model.eval()
        with torch.no_grad():
            samples = model.sample(
                shape=(sample_count, 2), n_steps=100, method="heun", device=device
            )
            print(f"Samples shape: {samples.shape}")

            plt.figure(figsize=(10, 8))
            plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
            plt.title(f"Generated Samples - {label}")
            plt.show()

        print("Checking samples are finite and in reasonable range")
        finite = torch.isfinite(samples).all().item()
        # This bound is intentionally crude: it is only meant to catch
        # exploding conversions, not to score sample quality precisely.
        in_range = (samples.abs() < 20).all().item()

        passed = decreased and finite and in_range
        if not passed:
            all_passed = False

        status = "✓" if passed else "✗"
        results[(pred, loss)] = {
            "initial_loss": initial,
            "final_loss": final,
            "decreased": decreased,
            "finite": finite,
            "in_range": in_range,
        }

        print(
            f"  {status}  {label:20s} | loss: {initial:.4f} → {final:.4f} | "
            f"finite={finite} range={in_range}"
        )

    print(f"\n  {'✓ ALL PASSED' if all_passed else '✗ FAILURES DETECTED'}")

    if not all_passed:
        print("\n  Check conversion functions in flow_matching.py")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Visual smoke test for the full 3x3 prediction/loss grid. "
            "Useful for building intuition and spotting glaring failures."
        )
    )
    parser.add_argument("--device", default="cpu", help="Torch device to run on.")
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=5,
        help="Number of 2D training samples. The default is intentionally tiny.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=50_000,
        help="Number of optimizer steps per grid cell.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=500,
        help="Number of generated samples to plot per grid cell.",
    )
    args = parser.parse_args()
    main(
        device=args.device,
        dataset_size=args.dataset_size,
        train_steps=args.train_steps,
        sample_count=args.sample_count,
    )
