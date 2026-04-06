"""
Image dataset loaders with deterministic subsampling.

CRITICAL DESIGN RULE: NO DATA AUGMENTATION. ANYWHERE. EVER.

Why? We are measuring memorization as a function of n (number of unique
training examples). Data augmentation (random crops, flips, color jitter)
effectively increases n by creating near-duplicate variants. This would
confound our measurement: a model trained on n=1000 images with flipping
sees 2000 effective images. We need a clean n → f_mem relationship.

This rule applies to ALL experiments: synthetic, CelebA, AFHQ, Flowers.
If a reviewer asks about augmentation, we note it as a deliberate choice
and point to the confound.

Classes:
    CelebA32Grey      — Bonnaire et al. preprocessing (32×32 greyscale)
    ImageFolderSubset — Generic loader for AFHQ / Flowers at any resolution
    HoldoutSet        — Fixed reference set for evaluation

Helper:
    subsample_deterministic — select n images reproducibly
"""

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from pathlib import Path
from typing import Optional


# =============================================================================
# Deterministic subsampling
# =============================================================================
# This is the most important function in this file. It guarantees that
# x-pred, ε-pred, and v-pred models see the EXACT SAME training images.
# If this breaks, the entire comparison is invalid.

def subsample_deterministic(dataset: Dataset, n: int, seed: int) -> Subset:
    """
    Select n samples from dataset using a fixed random seed.

    The key invariant: for a given (dataset, n, seed), the returned
    subset is ALWAYS the same, regardless of:
      - which parameterization we're training
      - which machine we're on
      - what time it is

    We use numpy's RandomState (not the global rng) to avoid interference
    from other random operations.

    Args:
        dataset: full dataset
        n: number of samples to select
        seed: deterministic seed for selection

    Returns:
        Subset containing exactly n samples
    """
    total = len(dataset)
    if n >= total:
        return dataset  # use full dataset

    rng = np.random.RandomState(seed)
    indices = rng.choice(total, size=n, replace=False)
    indices.sort()  # sort for deterministic iteration order
    return Subset(dataset, indices.tolist())


# =============================================================================
# CelebA 32×32 Greyscale (Bonnaire comparison)
# =============================================================================

class CelebA32Grey(Dataset):
    """
    CelebA preprocessed to 32×32 greyscale, following Bonnaire et al. (2025).

    Preprocessing pipeline:
        1. Center-crop to 140×140 (removes background, focuses on face)
        2. Resize to 32×32 (bilinear interpolation)
        3. Convert to greyscale (single channel)
        4. Normalize to [-1, 1]

    This gives D = 32 × 32 × 1 = 1,024.
    Estimated intrinsic dimension d ≈ 30-50 for face images.
    So ψ = D/d ≈ 20-34. The manifold gap exists but is moderate.

    This dataset exists solely for comparison with Bonnaire et al.'s
    published τ_mem results. Their paper uses ε-prediction with this
    exact preprocessing. We reproduce their setup, then add x-pred
    and v-pred to test whether parameterization shifts τ_mem.

    Args:
        root: path to CelebA dataset (parent dir containing 'celeba/' folder)
        split: 'train' or 'valid' or 'test'
        n: if given, subsample to n images
        seed: controls which n images are selected
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        n: Optional[int] = None,
        seed: int = 0,
    ):
        # Bonnaire's exact preprocessing
        self.transform = transforms.Compose([
            transforms.CenterCrop(140),
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),              # [0, 1], shape (1, 32, 32)
            transforms.Normalize([0.5], [0.5]), # [-1, 1]
        ])

        # torchvision CelebA
        # NOTE: CelebA must be pre-downloaded. On Berzelius, download on
        # the login node (which has internet) before submitting jobs.
        self.dataset = datasets.CelebA(
            root=root,
            split=split,
            transform=self.transform,
            download=False,  # don't try to download in a SLURM job
        )

        if n is not None and n < len(self.dataset):
            self.dataset = subsample_deterministic(self.dataset, n, seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self.dataset[idx]  # discard attributes/labels
        return img  # (1, 32, 32) in [-1, 1]


# =============================================================================
# Generic Image Folder (AFHQ / Flowers)
# =============================================================================

class ImageFolderSubset(Dataset):
    """
    Generic image folder loader for AFHQ, Flowers, or any folder of images.

    Supports two preprocessing modes (NO augmentation in either):

    'resize' mode (for AFHQ — images are already roughly square):
        Resize(resolution, resolution) → ToTensor → Normalize[-1,1]

    'shortest_side' mode (for Flowers — varying aspect ratios):
        Resize(resolution on shortest side) → CenterCrop(resolution) → etc.
        This preserves aspect ratio and avoids distortion.

    Alternative modes NOT used (documented for transparency):
        'center_crop_first': CenterCrop(min(H,W)) then Resize. Simpler but
            loses more content for non-square images.
        'random_crop': FORBIDDEN — this is data augmentation.
        'direct_resize_ignore_ratio': Distorts aspect ratio. Humans notice
            on Flowers (petals look wrong). Acceptable for AFHQ.

    Args:
        root: path to image folder (e.g., 'data/afhq/train/cat')
        resolution: target image size (64, 128, or 256)
        crop_mode: 'resize' or 'shortest_side'
        n: subsample size (None = use all)
        seed: controls subsampling
    """

    def __init__(
        self,
        root: str,
        resolution: int = 64,
        crop_mode: str = "resize",
        n: Optional[int] = None,
        seed: int = 0,
    ):
        # Build transform based on mode
        if crop_mode == "resize":
            # Direct resize — works for AFHQ (already ~square)
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif crop_mode == "shortest_side":
            # Resize shortest side, then center crop — for Flowers
            self.transform = transforms.Compose([
                transforms.Resize(resolution),        # shortest side = resolution
                transforms.CenterCrop(resolution),    # square crop from center
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        else:
            raise ValueError(
                f"Unknown crop_mode: '{crop_mode}'. "
                f"Use 'resize' (AFHQ) or 'shortest_side' (Flowers)."
            )

        # ImageFolder expects: root/class_name/image.jpg
        # For AFHQ single-class: root = 'data/afhq/train/cat' works directly
        # since ImageFolder will treat the parent as having one "class"
        root_path = Path(root)
        if root_path.is_dir() and not any(p.is_dir() for p in root_path.iterdir()):
            # No subdirectories — wrap in a dummy class folder
            # This handles the case where images are directly in the folder
            # without class subdirectories.
            #
            # Alternative: use a custom dataset that reads images directly.
            # ImageFolder's class requirement is a minor annoyance but not
            # worth a custom implementation.
            raise ValueError(
                f"ImageFolder requires class subdirectories. "
                f"If all images are in {root}, create a dummy subfolder:\n"
                f"  mkdir -p {root}/all && mv {root}/*.jpg {root}/all/"
            )

        self.dataset = datasets.ImageFolder(root=root, transform=self.transform)

        if n is not None and n < len(self.dataset):
            self.dataset = subsample_deterministic(self.dataset, n, seed)

        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self.dataset[idx]  # discard class label
        return img  # (3, resolution, resolution) in [-1, 1]


# =============================================================================
# Holdout reference set
# =============================================================================

class HoldoutSet:
    """
    Fixed reference set for FID and baseline NN statistics.

    The holdout set is NEVER subsampled and NEVER changes between runs.
    It provides:
      1. Reference distribution for FID computation
      2. Baseline NN statistics: what do "real" held-out samples look like
         when compared to the training set? If generated samples look similar
         to this baseline, the model is generalizing.

    Usage:
        holdout = HoldoutSet('data/afhq/val/cat', resolution=64)
        ref_images = holdout.as_tensor()  # (N, C, H, W)
    """

    def __init__(
        self,
        root: str,
        resolution: int = 64,
        crop_mode: str = "resize",
        max_samples: int = 10000,
    ):
        dataset = ImageFolderSubset(
            root=root,
            resolution=resolution,
            crop_mode=crop_mode,
            n=None,  # use all
            seed=0,
        )
        # Limit to max_samples
        if len(dataset) > max_samples:
            dataset.dataset = subsample_deterministic(
                dataset.dataset, max_samples, seed=999
            )
        self.dataset = dataset

    def as_tensor(self) -> torch.Tensor:
        """Load all images into a single tensor. May use a lot of memory."""
        images = [self.dataset[i] for i in range(len(self.dataset))]
        return torch.stack(images)

    def __len__(self) -> int:
        return len(self.dataset)
