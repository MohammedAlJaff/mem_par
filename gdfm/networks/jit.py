import importlib
import math
import sys
from math import pi
from pathlib import Path

import torch
import torch.nn as nn
from einops import repeat

from gdfm.networks.common import ensure_time_batch


VENDOR_ROOT = Path(__file__).resolve().parents[2] / "baselines" / "jit" / "vendor"


def _safe_scaled_dot_product_attention(query, key, value, dropout_p=0.0):
    """
    CPU-safe attention fallback for the vendored JiT implementation.

    The vendor code assumes a CUDA-oriented execution path. For local sanity
    checks and notebook inspection we want the wrapper to work on CPU too, so
    we replace that attention kernel with a plain PyTorch version that respects
    the device of the incoming tensors.
    """
    length_q = query.size(-2)
    length_k = key.size(-2)
    scale_factor = 1.0 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(
        query.size(0),
        1,
        length_q,
        length_k,
        dtype=query.dtype,
        device=query.device,
    )

    attn_weight = query.float() @ key.float().transpose(-2, -1)
    attn_weight = attn_weight * scale_factor
    attn_weight = torch.softmax(attn_weight + attn_bias, dim=-1).to(value.dtype)

    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value


def _make_device_safe_rope_class():
    """
    Rebuild the vendor rotary-embedding class without hard-coded `.cuda()`
    assumptions.

    The returned class has the same role as the original vendor class, but it
    stores buffers in a device-agnostic way so the wrapper can be exercised on
    CPU during tests and notebook walkthroughs.
    """
    util_module = importlib.import_module("util.model_util")
    broadcat = util_module.broadcat
    rotate_half = util_module.rotate_half

    class DeviceSafeVisionRotaryEmbeddingFast(nn.Module):
        def __init__(
            self,
            dim,
            pt_seq_len=16,
            ft_seq_len=None,
            custom_freqs=None,
            freqs_for="lang",
            theta=10000,
            max_freq=10,
            num_freqs=1,
            num_cls_token=0,
        ):
            """
            Precompute the cosine and sine tables used by rotary attention.

            The vendor implementation builds these once and reuses them in every
            forward pass. We keep that behavior, but store the result in buffers
            that follow the module's device automatically.
            """
            super().__init__()
            if custom_freqs is not None:
                freqs = custom_freqs
            elif freqs_for == "lang":
                freqs = 1.0 / (
                    theta
                    ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
                )
            elif freqs_for == "pixel":
                freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
            elif freqs_for == "constant":
                freqs = torch.ones(num_freqs).float()
            else:
                raise ValueError(f"unknown modality {freqs_for}")

            if ft_seq_len is None:
                ft_seq_len = pt_seq_len
            t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

            freqs = torch.einsum("..., f -> ... f", t, freqs)
            freqs = repeat(freqs, "... n -> ... (n r)", r=2)
            freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

            freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
            freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

            if num_cls_token > 0:
                num_img, dim_out = freqs_cos.shape
                cos_pad = torch.ones(num_cls_token, dim_out, dtype=freqs_cos.dtype)
                sin_pad = torch.zeros(num_cls_token, dim_out, dtype=freqs_sin.dtype)
                freqs_cos = torch.cat([cos_pad, freqs_cos], dim=0)
                freqs_sin = torch.cat([sin_pad, freqs_sin], dim=0)

            self.register_buffer("freqs_cos", freqs_cos)
            self.register_buffer("freqs_sin", freqs_sin)

        def forward(self, tensor):
            """
            Apply rotary mixing to the last feature dimension of `tensor`.
            """
            return tensor * self.freqs_cos + rotate_half(tensor) * self.freqs_sin

    return DeviceSafeVisionRotaryEmbeddingFast


def _load_vendor_module():
    """
    Import the vendored JiT module and patch it for CPU-safe local use.

    We patch exactly once per process. After that the wrapper can instantiate
    JiT models without every caller needing to know about the vendor quirks.
    """
    if not VENDOR_ROOT.exists():
        raise FileNotFoundError(
            f"Vendored JiT implementation not found at {VENDOR_ROOT}"
        )

    vendor_root = str(VENDOR_ROOT)
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)

    try:
        module = importlib.import_module("model_jit")
    except Exception as exc:
        raise ImportError(
            "Failed to import vendored JiT implementation from "
            f"{VENDOR_ROOT / 'model_jit.py'}"
        ) from exc

    if not getattr(module, "_gdfm_cpu_safe_patch", False):
        # The wrapper deliberately centralizes these patches so notebooks and
        # tests can use the same code path as real model setup.
        module.VisionRotaryEmbeddingFast = _make_device_safe_rope_class()
        module.scaled_dot_product_attention = _safe_scaled_dot_product_attention
        module._gdfm_cpu_safe_patch = True

    return module


# ============================================================
# Vendored JiT wrapper
# ============================================================


class JiT(nn.Module):
    """
    Wrapper over the vendored Li & He JiT implementation.

    Public interface:
        model(x, t)

    Vendor interface:
        model(x, t, y)

    We hide class conditioning here by always supplying a single dummy label.
    That keeps model setup focused on backbone construction rather than
    training-time label logic.
    """

    def __init__(
        self,
        image_size,
        image_channels,
        patch_size,
        hidden_dim,
        depth,
        heads,
        bottleneck_dim,
        proj_dropout=0.0,
        attn_dropout=0.0,
        in_context_len=32,
        in_context_start=4,
    ):
        """
        Build the `gdfm` wrapper around the vendored Li & He JiT backbone.

        This constructor validates the image/patch geometry, loads the vendor
        module, and hides away the unconditional-label convention that we want
        for GDFM-side model setup.
        """
        super().__init__()
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if image_channels <= 0:
            raise ValueError(f"image_channels must be positive, got {image_channels}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if image_size % patch_size != 0:
            raise ValueError(
                f"patch_size={patch_size} must divide image_size={image_size}"
            )
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        if bottleneck_dim <= 0:
            raise ValueError(
                f"bottleneck_dim must be positive, got {bottleneck_dim}"
            )
        if in_context_len < 0:
            raise ValueError(f"in_context_len must be >= 0, got {in_context_len}")
        if in_context_start < 0:
            raise ValueError(
                f"in_context_start must be >= 0, got {in_context_start}"
            )

        module = _load_vendor_module()

        self.image_size = image_size
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.heads = heads
        self.bottleneck_dim = bottleneck_dim
        self.proj_dropout = proj_dropout
        self.attn_dropout = attn_dropout
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = 1

        # The wrapper surface is unconditional, but the vendor implementation
        # still expects class labels. We therefore instantiate it with a single
        # null class and always feed that label in `forward`.
        self.model = module.JiT(
            input_size=image_size,
            patch_size=patch_size,
            in_channels=image_channels,
            hidden_size=hidden_dim,
            depth=depth,
            num_heads=heads,
            bottleneck_dim=bottleneck_dim,
            attn_drop=attn_dropout,
            proj_drop=proj_dropout,
            num_classes=self.num_classes,
            in_context_len=in_context_len,
            in_context_start=in_context_start,
        )
        self.register_buffer("null_label", torch.zeros(1, dtype=torch.long))

    def forward(self, z, t):
        """
        Run the wrapped JiT under the canonical `gdfm` image contract.

        Steps:
            1. validate image rank, channels, spatial size, and time shape
            2. expand the stored null label to match the batch
            3. call the underlying vendor model as `model(z, t, labels)`
            4. verify that the result stays in the same image space
        """
        if z.dim() != 4:
            raise ValueError(f"JiT expects z shape [B, C, H, W], got {tuple(z.shape)}")
        ensure_time_batch(t, z, "JiT")
        if z.shape[1] != self.image_channels:
            raise ValueError(
                f"JiT expects {self.image_channels} channels, got {z.shape[1]}"
            )
        if z.shape[2] != self.image_size or z.shape[3] != self.image_size:
            raise ValueError(
                f"JiT expects spatial size {self.image_size}x{self.image_size}, "
                f"got {z.shape[2]}x{z.shape[3]}"
            )

        # The public wrapper is unconditional; the dummy label keeps the vendor
        # signature satisfied without leaking class-conditioning into callers.
        labels = self.null_label.expand(z.shape[0]).to(z.device)
        out = self.model(z, t, labels)
        if out.shape != z.shape:
            raise ValueError(
                f"JiT returned shape {tuple(out.shape)}, expected {tuple(z.shape)}"
            )
        return out

    @property
    def patch_dim(self):
        """
        Number of scalar input values that live inside one image patch.
        """
        return self.image_channels * self.patch_size**2

    @property
    def seq_length(self):
        """
        Number of spatial patches processed by the transformer.
        """
        return (self.image_size // self.patch_size) ** 2

    @property
    def bottleneck_ratio(self):
        """
        Width expansion from raw patch dimension to transformer hidden size.

        This is a compact way to reason about how aggressively the model lifts
        local image patches into its latent representation.
        """
        return self.hidden_dim / self.patch_dim
