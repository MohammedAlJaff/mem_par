"""
Sanity Check 3: Does the UNet train on image data?

Creates a tiny batch of random images (or real CelebA if available),
runs a few hundred training steps, and checks that:
    1. Forward pass produces correct output shape
    2. Loss decreases
    3. Generated samples are not all zeros/NaN

This catches shape mismatches, broken skip connections, time conditioning
bugs, and normalization issues.

Pedagogical note:
    This is a plumbing smoke test, not an image-quality benchmark. Random data
    is enough to verify that the architecture, time conditioning, loss path,
    and sampler all produce sane tensors. Real data is optional when you want a
    slightly more realistic end-to-end pass.

Usage:
    python lab/sanity/check_unet.py
    python lab/sanity/check_unet.py --use-real-data --data-root /path/to/celeba
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.natural_images.common.flow_matching import FlowMatching
from experiments.natural_images.common.networks import make_unet_for_resolution
from experiments.natural_images.common.utils import seed_everything


def main(device: str = "cpu", use_real_data: bool = False, data_root: str = None):
    """Run a small end-to-end training/sampling smoke test for the image UNet."""
    print("=" * 60)
    print("SANITY CHECK: UNet on image data")
    print("=" * 60)

    seed_everything(0)

    # Testing more than one resolution catches a lot of shape bugs that only
    # appear when the encoder/decoder changes depth.
    for resolution, in_ch in [(32, 1), (64, 3)]:
        print(f"\n  --- Resolution {resolution}×{resolution}, channels={in_ch} ---")

        unet = make_unet_for_resolution(resolution, in_channels=in_ch)
        n_params = sum(p.numel() for p in unet.parameters())
        print(f"  Params: {n_params:,}")

        # Random data is enough to validate tensor plumbing. Real data is only
        # used when requested so this script stays lightweight by default.
        if use_real_data and data_root and resolution == 32 and in_ch == 1:
            from experiments.natural_images.common.data import CelebA32Grey

            dataset = CelebA32Grey(data_root, split="train", n=100, seed=0)
            data = torch.stack([dataset[i] for i in range(min(32, len(dataset)))])
        else:
            data = torch.randn(32, in_ch, resolution, resolution) * 0.5

        # Using pred=x and loss=v exercises a non-trivial conversion path inside
        # FlowMatching, which makes this a slightly stronger integration test.
        model = FlowMatching(unet, pred_space="x", loss_space="v").to(device)
        data = data.to(device)

        # A clean forward pass isolates architecture issues from optimization.
        t_test = torch.rand(data.shape[0], device=device)
        with torch.no_grad():
            out = unet(data, t_test)
        shape_ok = out.shape == data.shape
        print(f"  Forward shape: {out.shape} (expected {data.shape}) {'✓' if shape_ok else '✗'}")

        # We only need a short run here: the question is "can it optimize at
        # all?" rather than "how good is the final model?"
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        losses = []
        for step in range(200):
            optimizer.zero_grad()
            loss = model.compute_loss(data)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        initial = sum(losses[:10]) / 10
        final = sum(losses[-10:]) / 10
        decreased = final < initial * 0.8
        print(f"  Loss: {initial:.4f} → {final:.4f} {'✓' if decreased else '✗'}")

        # Sampling checks that the ODE path also works end to end, not just the
        # training loss.
        model.eval()
        with torch.no_grad():
            samples = model.sample(
                shape=(4, in_ch, resolution, resolution),
                n_steps=20, method="euler", device=device
            )
        finite = torch.isfinite(samples).all().item()
        nonzero = (samples.abs() > 1e-6).any().item()
        print(f"  Samples: finite={finite}, nonzero={nonzero} {'✓' if finite and nonzero else '✗'}")

        passed = shape_ok and decreased and finite and nonzero
        print(f"  {'✓ PASS' if passed else '✗ FAIL'}")

        if not passed:
            sys.exit(1)

    print(f"\n  ✓ ALL RESOLUTIONS PASSED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the image UNet with a short training and sampling run."
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run on.",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Load CelebA32Grey when available instead of synthetic random tensors.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root used only when --use-real-data is set.",
    )
    args = parser.parse_args()
    main(args.device, args.use_real_data, args.data_root)
