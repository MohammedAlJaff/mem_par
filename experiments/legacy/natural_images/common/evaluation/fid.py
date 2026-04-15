"""
FID (Fréchet Inception Distance) computation.

FID is a SECONDARY metric in this paper. The PRIMARY metric is f_mem
(memorization fraction from NN ratio). FID measures overall sample
quality, not memorization specifically. A model can have low FID while
memorizing (if it memorizes well) or high FID while generalizing (if
it generalizes poorly).

We report FID for completeness and comparison with prior work (Li & He
report FID as their main metric). But our conclusions about the
memorization boundary are based on f_mem, not FID.

Uses clean-fid (pip install clean-fid), which handles:
    - Correct Inception v3 preprocessing (bilinear resize to 299×299)
    - Consistent statistics across different image sizes
    - Feature caching for repeated evaluation

Alternative: pytorch-fid (pip install pytorch-fid)
    Pro: simpler API
    Con: known issues with resize interpolation mode
    Con: less maintained
Alternative: torch-fidelity
    Pro: also computes IS and KID
    Con: heavier dependency
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional


def save_samples_as_images(
    samples: torch.Tensor,
    output_dir: str,
    prefix: str = "sample",
) -> Path:
    """
    Save generated samples as PNG files for FID computation.

    clean-fid requires a directory of images, not a tensor.
    This function handles the conversion.

    Args:
        samples: (M, C, H, W) in [-1, 1] (our standard format)
        output_dir: where to save the PNGs
        prefix: filename prefix

    Returns:
        Path to the output directory
    """
    from torchvision.utils import save_image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert from [-1, 1] to [0, 1] for save_image
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0, 1)

    for i in range(samples.shape[0]):
        save_image(samples[i], output_dir / f"{prefix}_{i:06d}.png")

    return output_dir


def compute_fid(
    generated_dir: str,
    reference_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
) -> float:
    """
    Compute FID between two directories of images.

    Args:
        generated_dir: path to directory of generated images (PNGs)
        reference_dir: path to directory of reference images (PNGs)
        device: 'cuda' or 'cpu' (GPU is ~10× faster)
        batch_size: batch size for Inception feature extraction

    Returns:
        FID score (lower is better, 0 = identical distributions)

    Typical values:
        FID ≈ 0–5:   excellent (close to training distribution)
        FID ≈ 5–20:  good
        FID ≈ 20–50: mediocre
        FID > 100:   poor (model hasn't learned the distribution)
        FID > 300:   catastrophic failure (Li & He's ε-pred at large patches)
    """
    try:
        from cleanfid import fid as cfid
    except ImportError:
        raise ImportError(
            "clean-fid is required for FID computation.\n"
            "Install with: pip install clean-fid"
        )

    score = cfid.compute_fid(
        generated_dir,
        reference_dir,
        device=torch.device(device),
        batch_size=batch_size,
    )
    return float(score)


def compute_fid_from_tensors(
    generated: torch.Tensor,
    reference_dir: str,
    tmp_dir: str = "/tmp/fid_generated",
    device: str = "cuda",
) -> float:
    """
    Convenience wrapper: compute FID from a tensor of generated images.

    Saves to a temp directory, computes FID, optionally cleans up.

    Args:
        generated: (M, C, H, W) in [-1, 1]
        reference_dir: path to reference images
        tmp_dir: where to save generated images temporarily
        device: compute device
    """
    gen_dir = save_samples_as_images(generated, tmp_dir)
    score = compute_fid(str(gen_dir), reference_dir, device=device)
    return score
