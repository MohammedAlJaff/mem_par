import torch
import torch.nn as nn

from gdfm.networks.common import ensure_time_batch


# ============================================================
# Bonnaire-style UNet adapter
# ============================================================


class BonnaireUNet(nn.Module):
    """
    Thin adapter over the repo's Bonnaire-style UNet implementation.

    The canonical gdfm-facing surface is:
        image_size
        image_channels
        base_channels
        channel_mult
        attn_resolutions
        num_res_blocks
        dropout
        time_emb_dim

    Internally, this maps to the current SmallUNet implementation used by the
    image experiments. Keeping that mapping here lets model setup converge on a
    single public constructor without rewriting the experiment stack yet.
    """

    def __init__(
        self,
        image_size,
        image_channels,
        base_channels,
        channel_mult,
        num_res_blocks,
        attn_resolutions,
        dropout,
        time_emb_dim=None,
    ):
        super().__init__()
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if image_channels <= 0:
            raise ValueError(f"image_channels must be positive, got {image_channels}")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if num_res_blocks < 1:
            raise ValueError(f"num_res_blocks must be >= 1, got {num_res_blocks}")

        from experiments.natural_images.common.networks.unet import SmallUNet

        self.image_size = image_size
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.channel_mult = tuple(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = tuple(attn_resolutions)
        self.dropout = dropout
        self.time_emb_dim = time_emb_dim

        self.model = SmallUNet(
            in_channels=image_channels,
            out_channels=image_channels,
            base_channels=base_channels,
            channel_mults=self.channel_mult,
            attention_resolutions=self.attn_resolutions,
            num_res_blocks=num_res_blocks,
            time_embed_dim=time_emb_dim,
            dropout=dropout,
        )

    def forward(self, z, t):
        if z.dim() != 4:
            raise ValueError(
                f"BonnaireUNet expects z shape [B, C, H, W], got {tuple(z.shape)}"
            )
        ensure_time_batch(t, z, "BonnaireUNet")
        if z.shape[1] != self.image_channels:
            raise ValueError(
                f"BonnaireUNet expects {self.image_channels} channels, got {z.shape[1]}"
            )
        if z.shape[2] != self.image_size or z.shape[3] != self.image_size:
            raise ValueError(
                f"BonnaireUNet expects spatial size {self.image_size}x{self.image_size}, "
                f"got {z.shape[2]}x{z.shape[3]}"
            )

        out = self.model(z, t)
        if out.shape != z.shape:
            raise ValueError(
                f"BonnaireUNet returned shape {tuple(out.shape)}, expected {tuple(z.shape)}"
            )
        return out
