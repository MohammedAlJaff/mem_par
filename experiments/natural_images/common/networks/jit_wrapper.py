"""
Thin wrapper around the official JiT (Joint image Transformer) implementation.

Setup:
    The vendored baseline now lives under ``baselines/jit/vendor``.

This wrapper exists solely to adapt JiT's interface to ours:
    Our interface:  model(x, t) → prediction,  x: (B,C,H,W), t: (B,)
    Their interface: may differ in time convention, class conditioning, etc.

The wrapper does NOT handle parameterization. FlowMatching does that.
The wrapper just makes JiT look like any other network.

IMPORTANT: Before using this, you MUST:
    1. Clone the JiT repo into external/JiT/
    2. Inspect their forward() signature
    3. Adapt the TODOs below to match their actual interface

The TODOs in this file are not laziness — they're genuine unknowns that
require inspecting their code. I've written the scaffolding so you know
exactly what to check and where to plug it in.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path


# Path to the vendored JiT baseline repository
JIT_PATH = Path(__file__).resolve().parents[4] / "baselines" / "jit" / "vendor"


class JiTWrapper(nn.Module):
    """
    Wrapper around Li & He's JiT-B Vision Transformer.

    The JiT architecture processes raw pixel patches through a ViT with:
        1. A 2-layer bottleneck patch embedding: (p²·C) → bottleneck_dim → hidden_dim
        2. Standard ViT blocks with adaLN-Zero conditioning on timestep
        3. Per-patch output projection: hidden_dim → (p²·C)

    The per-patch output projection is the CRITICAL bottleneck for the
    parameterization story. When hidden_dim ≈ p²·C (e.g., both 768 at p=16),
    the Jacobian rank is constrained, and only x-prediction survives.
    When hidden_dim ≫ p²·C (e.g., 768 vs 48 at p=4), no bottleneck.

    Configurations studied (paper §4.2, Table 4):
        JiT-B/16: patch_dim=768,  hidden=768, seq_len=(H/16)²
        JiT-B/8:  patch_dim=192,  hidden=768, seq_len=(H/8)²
        JiT-B/4:  patch_dim=48,   hidden=768, seq_len=(H/4)²

    Args:
        image_size: input image size (64, 128, or 256)
        patch_size: size of each patch (4, 8, or 16)
        in_channels: 3 for RGB
        hidden_dim: transformer hidden dimension (768 for JiT-B)
        depth: number of transformer blocks (12 for JiT-B)
        num_heads: attention heads (12 for JiT-B)
        bottleneck_dim: intermediate dim in patch embedding (128 for JiT-B)
        dropout: dropout rate (0 for JiT-B, 0.2 for JiT-H/G)
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        bottleneck_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self._hidden_dim = hidden_dim

        # ---- Import from JiT repository ----
        if not JIT_PATH.exists():
            raise FileNotFoundError(
                f"JiT repository not found at {JIT_PATH}.\n"
                f"Expected the vendored baseline under baselines/jit/vendor.\n"
                f"Restore or move the JiT baseline into that location."
            )

        # TODO: Inspect the JiT repo to find the correct import.
        # Expected locations:
        #   from models import JiT
        #   from models.jit import JiT
        #   from jit import JiT
        # Uncomment the correct one after inspecting:
        #
        # from models import JiT as JiTModel
        #
        # For now, raise an error with instructions:
        raise NotImplementedError(
            "JiT wrapper requires manual setup.\n\n"
            "Steps:\n"
            "  1. cd baselines/jit/vendor && ls *.py models/\n"
            "  2. Find the main model class (likely 'JiT' or 'JointImageTransformer')\n"
            "  3. Check its __init__ signature for:\n"
            "     - image_size, patch_size, in_channels\n"
            "     - hidden_size / embed_dim / d_model\n"
            "     - depth / num_layers\n"
            "     - num_heads\n"
            "     - Whether class label conditioning is required\n"
            "  4. Check its forward() signature for:\n"
            "     - Time convention: continuous t ∈ [0,1]? sigma? discrete 0-1000?\n"
            "     - Whether class labels are required (y argument)\n"
            "     - Output shape: (B, C, H, W) or something else?\n"
            "  5. Update the __init__ and forward below accordingly.\n"
            "  6. Remove this NotImplementedError.\n\n"
            "Example adaptation (PLACEHOLDER — adapt to actual code):\n"
            "  self.model = JiTModel(\n"
            "      image_size=image_size,\n"
            "      patch_size=patch_size,\n"
            "      in_channels=in_channels,\n"
            "      hidden_size=hidden_dim,\n"
            "      depth=depth,\n"
            "      num_heads=num_heads,\n"
            "  )\n"
        )

        # ---- Placeholder model construction (uncomment + adapt after inspection) ----
        #
        # self.model = JiTModel(
        #     image_size=image_size,
        #     patch_size=patch_size,
        #     in_channels=in_channels,
        #     hidden_size=hidden_dim,
        #     depth=depth,
        #     num_heads=num_heads,
        #     # bottleneck_dim=bottleneck_dim,  # check if exposed
        # )
        #
        # # For unconditional generation, we need a null class label.
        # # If their model requires a class label (for adaLN-Zero), we have options:
        # #
        # # Option A: Pass a dummy label (e.g., y=0)
        # # self.register_buffer('null_label', torch.zeros(1, dtype=torch.long))
        # #
        # # Option B: If they support cfg_scale=0 or null_label internally
        # # Check their code for "uncond" or "null" or "cfg"
        # #
        # # Option C: Modify their code to make class conditioning optional
        # # (last resort — try to avoid modifying external code)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy image in [-1, 1]
            t: (B,) continuous timestep in [0, 1]

        Returns:
            (B, C, H, W) prediction (interpretation depends on FlowMatching)
        """
        # TODO: Adapt time convention.
        # Their code may use:
        #   - Continuous t ∈ [0, 1]  → pass directly
        #   - Sigma (noise level)    → t_sigma = 1 - t (for FM interpolant)
        #   - Discrete 0-1000        → t_discrete = (t * 1000).long()
        # Check their training script or Algorithm 1 implementation.

        # TODO: Handle class labels.
        # If forward(x, t, y) is required:
        #   B = x.shape[0]
        #   y = self.null_label.expand(B)
        #   out = self.model(x, t, y)
        # If forward(x, t) works:
        #   out = self.model(x, t)

        # TODO: Check output shape.
        # Expected: (B, C, H, W). If they return patches or a flattened vector,
        # reshape here.

        raise NotImplementedError("See TODOs above")

    @property
    def patch_dim(self) -> int:
        """Dimension of each patch: p² × C. This is D_patch in the paper."""
        return self.in_channels * self.patch_size ** 2

    @property
    def seq_length(self) -> int:
        """Number of patches: (H/p)²."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def bottleneck_ratio(self) -> float:
        """
        hidden_dim / patch_dim. The Li & He capacity ratio.
        When ≈ 1.0: tight bottleneck, only x-prediction survives.
        When ≫ 1.0: no bottleneck, all parameterizations work.
        """
        return self._hidden_dim / self.patch_dim
