import torch
import torch.nn as nn

from gdfm.networks.common import TimeEmbeddingMLP, ensure_time_batch


# ============================================================
# Time-conditioned MLP
# ============================================================


class TimeCondMLP(nn.Module):
    """
    Flat-vector MLP with sinusoidal time embedding.

    Input:
        z: [B, D]
        t: [B]

    Output:
        y: [B, D]
    """

    def __init__(self, data_dim, hidden_dim, n_layers, time_emb_dim):
        super().__init__()
        self.data_dim = data_dim
        self.time_emb_dim = time_emb_dim

        self.time_mlp = TimeEmbeddingMLP(time_emb_dim, time_emb_dim)

        layers = [nn.Linear(data_dim + time_emb_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, data_dim)]
        self.net = nn.Sequential(*layers)

    def time_embedding(self, t):
        """
        Sinusoidal embedding of scalar t in [0,1].
        """
        return self.time_mlp(t)

    def forward(self, z, t):
        if z.dim() != 2:
            raise ValueError(
                f"TimeCondMLP expects z shape [B, D], got {tuple(z.shape)}"
            )
        ensure_time_batch(t, z, "TimeCondMLP")

        t_emb = self.time_embedding(t)
        return self.net(torch.cat([z, t_emb], dim=-1))
