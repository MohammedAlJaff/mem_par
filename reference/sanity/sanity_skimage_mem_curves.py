"""
Sanity-check memorization curves on real images from `skimage.data`.

For each selected source image:
    1. Use that image as the generated prototype.
    2. Use the full selected image set as the reference set (including itself).
    3. Add increasing Gaussian noise to the prototype.
    4. Compute memorization metrics at each noise level.
    5. Plot one f_mem curve per source image.

Run:
    python reference/sanity/sanity_skimage_mem_curves.py
    python reference/sanity/sanity_skimage_mem_curves.py --ratio-threshold 0.33
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mp_metrics import compute_memorization_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot f_mem-vs-noise curves using canonical mp_metrics on skimage images."
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=["camera", "coins", "moon", "page"],
        help="skimage.data image names to use.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Target square size for resized images.",
    )
    parser.add_argument(
        "--n-levels",
        type=int,
        default=12,
        help="Number of noise levels.",
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=0.0,
        help="Minimum Gaussian noise sigma.",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=0.50,
        help="Maximum Gaussian noise sigma.",
    )
    parser.add_argument(
        "--samples-per-level",
        type=int,
        default=64,
        help="Number of noisy draws per noise level.",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.5,
        help="f_mem threshold in r = d1 / d2.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for nearest-neighbor distance computation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for noise draws.",
    )
    parser.add_argument(
        "--output",
        default="reference/sanity/figures/skimage_fmem_vs_noise.png",
        help="Output path for the summary plot.",
    )
    return parser.parse_args()


def _load_skimage_image(name, size):
    try:
        from skimage import color, data, transform, util
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-image is required for this sanity script. "
            "Install it in your active environment, then rerun."
        ) from exc

    if not hasattr(data, name):
        raise ValueError(
            f"Unknown skimage.data image '{name}'. "
            "Choose from valid skimage.data function names."
        )

    image_fn = getattr(data, name)
    image = image_fn()
    image = util.img_as_float32(image)

    if image.ndim == 3:
        image = color.rgb2gray(image)

    image = transform.resize(
        image,
        (size, size),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    return torch.from_numpy(image).unsqueeze(0)  # [1, H, W]


def build_reference_set(image_names, size):
    images = []
    for name in image_names:
        tensor = _load_skimage_image(name, size)
        images.append(tensor)
    return torch.stack(images, dim=0), list(image_names)  # [N, 1, H, W]


def noisy_batch(base_image, sigma, n_samples, generator):
    noise = torch.randn(
        n_samples,
        *base_image.shape,
        generator=generator,
        dtype=base_image.dtype,
    )
    noisy = base_image.unsqueeze(0) + sigma * noise
    return noisy.clamp(0.0, 1.0)


def main():
    args = parse_args()
    generator = torch.Generator().manual_seed(args.seed)

    reference, labels = build_reference_set(args.images, args.size)
    sigma_values = np.linspace(args.sigma_min, args.sigma_max, args.n_levels)

    curve_data = {}
    print(
        f"Reference set: {len(labels)} images, resized to {args.size}x{args.size}, "
        f"ratio_threshold={args.ratio_threshold:.3f}"
    )

    for idx, label in enumerate(labels):
        base = reference[idx]
        rows = []
        for sigma in sigma_values:
            samples = noisy_batch(
                base,
                sigma=float(sigma),
                n_samples=args.samples_per_level,
                generator=generator,
            )
            metrics = compute_memorization_metrics(
                samples,
                reference,
                ratio_threshold=args.ratio_threshold,
                chunk_size=args.chunk_size,
            )
            rows.append(
                {
                    "sigma": float(sigma),
                    "f_mem": metrics["f_mem"],
                    "mean_nn_ratio": metrics["mean_nn_ratio"],
                    "mean_d1": metrics["mean_d1"],
                }
            )
        curve_data[label] = rows

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for label, rows in curve_data.items():
        sigmas = [row["sigma"] for row in rows]
        f_mem = [row["f_mem"] for row in rows]
        mean_r = [row["mean_nn_ratio"] for row in rows]
        axes[0].plot(sigmas, f_mem, marker="o", linewidth=1.5, label=label)
        axes[1].plot(sigmas, mean_r, marker="o", linewidth=1.5, label=label)

    axes[0].set_title("f_mem vs Noise Sigma")
    axes[0].set_xlabel("sigma")
    axes[0].set_ylabel("f_mem")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Mean NN Ratio vs Noise Sigma")
    axes[1].set_xlabel("sigma")
    axes[1].set_ylabel("mean(d1/d2)")
    axes[1].grid(alpha=0.3)

    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"skimage Memorization Sanity | threshold={args.ratio_threshold:.3f} | "
        f"samples/level={args.samples_per_level}"
    )
    fig.savefig(output_path, dpi=170)
    plt.close(fig)

    print(f"Saved plot: {output_path}")
    print("\nPer-image f_mem at min/max sigma:")
    for label, rows in curve_data.items():
        first = rows[0]
        last = rows[-1]
        print(
            f"  {label:12s}  sigma={first['sigma']:.3f} -> f_mem={first['f_mem']:.3f} ; "
            f"sigma={last['sigma']:.3f} -> f_mem={last['f_mem']:.3f}"
        )


if __name__ == "__main__":
    main()
