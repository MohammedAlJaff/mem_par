"""
LPIPS-based nearest-neighbor computation for image memorization metrics.

WHY LPIPS INSTEAD OF PIXEL DISTANCE?
    At 32×32, pixel-space L2 is a reasonable proxy for perceptual distance.
    At 64×64 and above, it breaks down: two images can be perceptually
    similar but have large pixel L2 (e.g., a 1-pixel shift doubles the
    per-pixel error). LPIPS (Learned Perceptual Image Patch Similarity)
    uses features from a pretrained network (AlexNet) to compute distances
    that correlate better with human judgments.

    For memorization detection, this matters: a model that memorizes a
    training image but shifts it by 1 pixel would look "novel" in pixel
    space but "memorized" in LPIPS space.

ALTERNATIVES:
    AlexNet backbone (our choice): fastest, standard, well-validated
    VGG backbone: slightly better perceptual correlation, ~3× slower
    SqueezeNet backbone: fastest option, least accurate
    DINO / CLIP features: more semantically aware, but oversmooth for
        pixel-level memorization detection. Better for semantic similarity.
    SSCD (Self-Supervised Copy Detection): specifically designed for copy
        detection. Overkill for our synthetic experiments but appropriate
        for large-scale image experiments. Consider for AFHQ 256.

COMPUTATIONAL NOTE:
    Pairwise LPIPS between M generated and N training images requires
    M × N forward passes through AlexNet. For M=10k, N=5k, that's 50M
    evaluations — too expensive to do naively. We batch carefully and
    process in chunks. Still, expect ~10-30 minutes per evaluation on
    a single GPU. This is why we separate training from evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class LPIPSNNComputer:
    """
    Compute nearest-neighbor statistics in LPIPS perceptual space.

    Memory-efficient: processes in batches to avoid OOM.
    All computation is in fp32 for numerical stability.

    Usage:
        computer = LPIPSNNComputer(device="cuda")
        stats = computer.compute(generated_images, training_images)
        # stats has same format as memorization.compute_nn_stats:
        # {'d1': ..., 'd2': ..., 'r': ..., 'nn_idx': ...}
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 32,      # HYPERPARAMETER: LPIPS batch size
                                    # Smaller = less memory, slower
                                    # 32 works on A100 80GB for 64×64 images
                                    # Reduce to 16 for 256×256
    ):
        try:
            import lpips
        except ImportError:
            raise ImportError("lpips is required. Install with: pip install lpips")

        self.device = device
        self.batch_size = batch_size

        # Load pretrained LPIPS with AlexNet backbone
        # verbose=False suppresses the download message
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
        self.lpips_fn.eval()

        # Freeze — we never train the perceptual network
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def compute(
        self,
        generated: torch.Tensor,
        training: torch.Tensor,
    ) -> dict:
        """
        Compute NN stats in LPIPS space.

        Args:
            generated: (M, C, H, W) images in [-1, 1]
            training:  (N, C, H, W) images in [-1, 1]

        Returns dict:
            d1: (M,) LPIPS distance to nearest training neighbor
            d2: (M,) LPIPS distance to second nearest
            r:  (M,) d1/d2 ratio
            nn_idx: (M,) index of nearest training point
        """
        M = generated.shape[0]
        N = training.shape[0]

        # We compute pairwise distances in chunks to manage memory.
        # Strategy: for each chunk of generated images, compute LPIPS
        # distance to ALL training images, find top-2 nearest.

        d1 = torch.zeros(M)
        d2 = torch.zeros(M)
        nn_idx = torch.zeros(M, dtype=torch.long)

        for i in range(0, M, self.batch_size):
            i_end = min(i + self.batch_size, M)
            gen_chunk = generated[i:i_end].to(self.device)  # (chunk, C, H, W)
            chunk_size = gen_chunk.shape[0]

            # Compute LPIPS to all training images
            all_dists = []
            for j in range(0, N, self.batch_size):
                j_end = min(j + self.batch_size, N)
                train_chunk = training[j:j_end].to(self.device)  # (tchunk, C, H, W)
                tchunk_size = train_chunk.shape[0]

                # LPIPS expects matching batch dims, so we need to tile.
                # gen_chunk: (chunk, C, H, W) → (chunk, tchunk, C, H, W) → (chunk*tchunk, C, H, W)
                # train_chunk: (tchunk, C, H, W) → (chunk, tchunk, C, H, W) → (chunk*tchunk, C, H, W)
                gen_rep = gen_chunk.unsqueeze(1).expand(-1, tchunk_size, -1, -1, -1)
                train_rep = train_chunk.unsqueeze(0).expand(chunk_size, -1, -1, -1, -1)

                gen_flat = gen_rep.reshape(-1, *gen_chunk.shape[1:])
                train_flat = train_rep.reshape(-1, *train_chunk.shape[1:])

                # LPIPS returns (B, 1, 1, 1) — squeeze to (B,)
                d = self.lpips_fn(gen_flat, train_flat).squeeze()
                d = d.reshape(chunk_size, tchunk_size)  # (chunk, tchunk)
                all_dists.append(d.cpu())

            # Concatenate distances to all training images: (chunk, N)
            all_dists = torch.cat(all_dists, dim=1)

            # Find top-2 nearest
            top2_dists, top2_idxs = all_dists.topk(2, dim=1, largest=False)
            d1[i:i_end] = top2_dists[:, 0]
            d2[i:i_end] = top2_dists[:, 1]
            nn_idx[i:i_end] = top2_idxs[:, 0]

        r = d1 / (d2 + 1e-10)

        return {
            "d1": d1,
            "d2": d2,
            "r": r,
            "nn_idx": nn_idx,
        }
