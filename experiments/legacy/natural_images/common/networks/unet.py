"""
Small UNet for flow matching on images.

This follows the architecture used by Bonnaire et al. (NeurIPS 2025) for
their memorization/generalization experiments on CelebA 32×32. If their
code is publicly available, prefer wrapping theirs directly — subtle
differences in normalization, attention, or time conditioning can shift
τ_mem, which is the quantity we're comparing against.

Architecture overview:
    Encoder:  [ResBlock, ResBlock, Downsample] × num_levels
    Middle:   ResBlock, Attention, ResBlock
    Decoder:  [ResBlock, ResBlock, Upsample] × num_levels
    Skip connections: concatenation (encoder → decoder at each level)

Each ResBlock:
    GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Dropout → Conv3×3 + skip

Time conditioning:
    t → sinusoidal embed → 2-layer MLP → (scale, shift) per ResBlock
    Applied via Adaptive Group Normalization (AdaGN):
        AdaGN(h, t) = scale(t) · GroupNorm(h) + shift(t)
    This is the standard mechanism from DDPM / ADM.

Self-attention:
    Applied only at specified resolutions (default: 8×8) to save memory.
    Single-head attention over spatial positions, after GroupNorm.

Key design choices (alternatives documented):
    Downsampling: stride-2 convolution (not pooling)
        Alt: average pooling + 1×1 conv (slightly less info loss)
        Alt: learned strided conv with 4×4 kernel (more standard in some codebases)
    Upsampling: nearest-neighbor interpolation + 3×3 conv
        Alt: transposed convolution (can cause checkerboard artifacts)
        Alt: bilinear interpolation + conv (smoother but slower)
    Normalization: GroupNorm with 32 groups
        Alt: 8 groups (better for small channel counts like 64)
        Alt: LayerNorm (simpler, non-standard for UNets)
    Skip connections: concatenation
        Alt: addition (saves memory, loses some information)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Time embedding (shared with MLP — same sinusoidal scheme)
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding: t ∈ [0,1] → R^{dim}. See mlp.py for details."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        out = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2:
            out = torch.cat([out, torch.zeros_like(t.unsqueeze(1))], dim=1)
        return out


class TimeMLPEmbedding(nn.Module):
    """
    Full time embedding: sinusoidal → MLP → time_embed_dim.

    The MLP transforms the sinusoidal features into a richer representation
    that the network can use for adaptive normalization. Two layers with
    SiLU activation is standard (DDPM, ADM, Bonnaire).
    """

    def __init__(self, time_embed_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(time_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → (B, time_embed_dim)"""
        if t.dim() == 2:
            t = t.squeeze(1)
        return self.mlp(self.sinusoidal(t))


# =============================================================================
# Building blocks
# =============================================================================

class ResBlock(nn.Module):
    """
    Residual block with adaptive group normalization for time conditioning.

    Architecture:
        GroupNorm → SiLU → Conv3×3 → GroupNorm(adaptive) → SiLU → Dropout → Conv3×3
        + skip connection (1×1 conv if channel counts differ, identity otherwise)

    The adaptive normalization (AdaGN) is where time information enters:
        h = GroupNorm(h)
        h = scale(t_emb) * h + shift(t_emb)
    where scale and shift are linear projections of the time embedding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        num_groups: int = 32,
    ):
        super().__init__()

        # Ensure num_groups divides channel count
        # If channels < 32, fall back to fewer groups
        ng1 = min(num_groups, in_channels)
        ng2 = min(num_groups, out_channels)

        # First conv
        self.norm1 = nn.GroupNorm(ng1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Time projection → (scale, shift) for AdaGN
        self.time_proj = nn.Linear(time_embed_dim, out_channels * 2)

        # Second conv
        self.norm2 = nn.GroupNorm(ng2, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        # Initialize conv2 to zero for residual learning
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
            t_emb: (B, time_embed_dim)
        Returns:
            (B, C_out, H, W)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Adaptive group normalization
        # Project time embedding to (scale, shift) pairs
        t_proj = self.time_proj(F.silu(t_emb))  # (B, 2 * C_out)
        scale, shift = t_proj.chunk(2, dim=1)     # each (B, C_out)
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # (B, C_out, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = h * (1 + scale) + shift  # AdaGN: scale around 1, shift around 0
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention(nn.Module):
    """
    Single-head self-attention over spatial positions.

    Applied after GroupNorm. The attention operates on flattened spatial
    dimensions: (B, C, H, W) → (B, H*W, C) → attention → (B, C, H, W).

    Single head is sufficient at the resolutions we use (8×8, 16×16).
    Multi-head would be more expressive but also more expensive.

    This is a lightweight attention block. For transformer-scale attention,
    use JiT instead.
    """

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        ng = min(num_groups, channels)
        self.norm = nn.GroupNorm(ng, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

        # Initialize projection to zero (residual connection)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, C, H*W)

        # Scaled dot-product attention
        # (B, C, H*W) × (B, H*W, C) → (B, C, C) ... no, we want spatial attention
        # Transpose: (B, H*W, C) for queries/keys
        q = q.permute(0, 2, 1)  # (B, H*W, C)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        scale = C ** -0.5
        attn = torch.bmm(q, k.transpose(1, 2)) * scale  # (B, H*W, H*W)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)  # (B, H*W, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return x + self.proj(out)


class Downsample(nn.Module):
    """Spatial downsampling via stride-2 convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling via nearest-neighbor interpolation + convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# =============================================================================
# Full UNet
# =============================================================================

class SmallUNet(nn.Module):
    """
    Small UNet for flow matching, following Bonnaire et al. (2025).

    Args:
        in_channels: input channels (1 for greyscale, 3 for RGB)
        out_channels: output channels (defaults to in_channels)
        base_channels: channel width at first level (capacity knob)
            32  → ~8M params (low capacity, faster)
            48  → ~18M params
            64  → ~35M params (Bonnaire default)
            96  → ~80M params (high capacity)
        channel_mults: multipliers at each level
            (1,2,4,4) → 4 levels, standard for 32×32
            (1,2,4,4,8) → 5 levels, for 64-128
            (1,2,4,4,8,8) → 6 levels, for 256
        attention_resolutions: spatial sizes at which to apply self-attention
            (8,) → standard for 32×32
            (16,) → for 64+
            (8, 16) → multi-scale attention (more expensive)
        num_res_blocks: ResBlocks per level (2 is standard)
        dropout: dropout rate in ResBlocks (0 for small datasets)
        num_groups: groups for GroupNorm (32 is standard, reduce if channels < 32)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = None,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 4),
        attention_resolutions: tuple = (8,),
        num_res_blocks: int = 2,
        time_embed_dim: int = None,
        dropout: float = 0.0,
        num_groups: int = 32,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels
        if time_embed_dim is None:
            time_embed_dim = base_channels * 4

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embedding
        self.time_embed = TimeMLPEmbedding(time_embed_dim)

        # Compute channel counts at each level
        channels = [base_channels * m for m in channel_mults]
        num_levels = len(channel_mults)

        # ---- Input projection ----
        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # ---- Encoder ----
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()

        ch_in = channels[0]
        self._encoder_channels = [ch_in]  # track for skip connections

        for level in range(num_levels):
            ch_out = channels[level]
            blocks_at_level = nn.ModuleList()

            for _ in range(num_res_blocks):
                blocks_at_level.append(
                    ResBlock(ch_in, ch_out, time_embed_dim, dropout, num_groups)
                )
                ch_in = ch_out
                self._encoder_channels.append(ch_in)

            self.encoder_blocks.append(blocks_at_level)

            # Attention at this level?
            # We'll check resolution during forward pass
            self.encoder_attns.append(
                SelfAttention(ch_out, num_groups) if level < num_levels - 1 else nn.Identity()
            )

            # Downsample (except at last level)
            if level < num_levels - 1:
                self.encoder_downsamples.append(Downsample(ch_out))
                self._encoder_channels.append(ch_out)
            else:
                self.encoder_downsamples.append(nn.Identity())

        # ---- Middle ----
        self.middle_block1 = ResBlock(ch_in, ch_in, time_embed_dim, dropout, num_groups)
        self.middle_attn = SelfAttention(ch_in, num_groups)
        self.middle_block2 = ResBlock(ch_in, ch_in, time_embed_dim, dropout, num_groups)

        # ---- Decoder ----
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()

        for level in reversed(range(num_levels)):
            ch_out = channels[level]
            blocks_at_level = nn.ModuleList()

            for i in range(num_res_blocks + 1):  # +1 for the skip connection block
                # Input channels: current + skip from encoder
                skip_ch = self._encoder_channels.pop()
                blocks_at_level.append(
                    ResBlock(ch_in + skip_ch, ch_out, time_embed_dim, dropout, num_groups)
                )
                ch_in = ch_out

            self.decoder_blocks.append(blocks_at_level)

            # Upsample (except at first decoder level = last encoder level)
            if level > 0:
                self.decoder_upsamples.append(Upsample(ch_out))
            else:
                self.decoder_upsamples.append(nn.Identity())

        # ---- Output projection ----
        ng_out = min(num_groups, ch_in)
        self.output_norm = nn.GroupNorm(ng_out, ch_in)
        self.output_conv = nn.Conv2d(ch_in, out_channels, 3, padding=1)

        # Initialize output to zero
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

        # Store attention resolutions for runtime check
        self._attention_resolutions = set(attention_resolutions)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) noisy image in [-1, 1]
            t: (B,) timestep in [0, 1]
        Returns:
            (B, C_out, H, W) prediction
        """
        if t.dim() == 2:
            t = t.squeeze(1)

        t_emb = self.time_embed(t)  # (B, time_embed_dim)

        # Input
        h = self.input_conv(x)

        # Encoder (save skip connections)
        skips = [h]
        for level, (blocks, attn, down) in enumerate(
            zip(self.encoder_blocks, self.encoder_attns, self.encoder_downsamples)
        ):
            for block in blocks:
                h = block(h, t_emb)
                # Apply attention if at the right resolution
                if h.shape[-1] in self._attention_resolutions:
                    if not isinstance(attn, nn.Identity):
                        h = attn(h)
                skips.append(h)

            if level < len(self.encoder_blocks) - 1:
                h = down(h)
                skips.append(h)

        # Middle
        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)

        # Decoder (consume skip connections in reverse)
        for blocks, up in zip(self.decoder_blocks, self.decoder_upsamples):
            for block in blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            h = up(h)

        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        return h


# =============================================================================
# Factory function for resolution-appropriate UNet
# =============================================================================

def make_unet_for_resolution(
    resolution: int,
    in_channels: int = 3,
    base_channels: int = 64,
) -> SmallUNet:
    """
    Create a UNet with level count and attention placement appropriate
    for the given image resolution.

    The idea: we want the bottleneck to be at ~4×4 spatial resolution.
    Each level halves the resolution, so:
        32  → 4 levels: 32→16→8→4    (Bonnaire default)
        64  → 4 levels: 64→32→16→8   (attention at 16)
        128 → 5 levels: 128→64→32→16→8
        256 → 6 levels: 256→128→64→32→16→8

    Alternative strategy: keep the same number of levels and let the
    bottleneck size change. This changes the information bottleneck
    (larger bottleneck = easier to pass information through).
    We use the fixed-bottleneck strategy for consistency.

    Args:
        resolution: input image size (32, 64, 128, or 256)
        in_channels: 1 for greyscale, 3 for RGB
        base_channels: width multiplier (64 is standard)
    """
    configs = {
        32: dict(
            channel_mults=(1, 2, 4, 4),
            attention_resolutions=(8,),
        ),
        64: dict(
            channel_mults=(1, 2, 4, 4),
            attention_resolutions=(16,),
        ),
        128: dict(
            channel_mults=(1, 2, 4, 4, 8),
            attention_resolutions=(16,),
        ),
        256: dict(
            channel_mults=(1, 2, 4, 4, 8, 8),
            attention_resolutions=(16,),
        ),
    }

    if resolution not in configs:
        raise ValueError(
            f"No UNet config for resolution {resolution}. "
            f"Available: {list(configs.keys())}"
        )

    return SmallUNet(
        in_channels=in_channels,
        base_channels=base_channels,
        **configs[resolution],
    )
