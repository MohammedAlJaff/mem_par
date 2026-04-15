import math

import torch
import torch.nn as nn


def ensure_time_batch(t, ref, name):
    if t.dim() != 1:
        raise ValueError(f"{name} expects t shape [B], got {tuple(t.shape)}")
    if ref.shape[0] != t.shape[0]:
        raise ValueError(
            f"{name} batch mismatch: input has {ref.shape[0]} samples, "
            f"t has {t.shape[0]} samples"
        )


def sinusoidal_time_embedding(t, time_emb_dim):
    """
    Standard sinusoidal embedding of scalar time values in [0,1].
    """
    half = time_emb_dim // 2
    idx = torch.arange(half, device=t.device, dtype=t.dtype)
    freqs = torch.exp(-math.log(10000.0) * idx / max(half - 1, 1))
    args = t[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmbeddingMLP(nn.Module):
    """
    Small learned projection on top of the sinusoidal embedding.
    """

    def __init__(self, time_emb_dim, out_dim):
        super().__init__()
        if time_emb_dim % 2 != 0:
            raise ValueError(f"time_emb_dim must be even, got {time_emb_dim}")
        self.time_emb_dim = time_emb_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(time_emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t):
        emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        return self.net(emb)
