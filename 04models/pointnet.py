"""PointNet til punktsky-regression.

Samme idé som MinkowskiPointNet i DPCR-AGB, men i ren PyTorch:
  delt MLP pr. punkt -> pooling over punkter -> MLP -> lineært hoved.

Input:   pos   [B, N, 3]   koordinater (normaliser dem i dit dataset)
         feats [B, N, F]   valgfrie ekstra features pr. punkt (F = extra_features)
Output:  [B, num_outputs]  ét tal pr. mål (standardiserede mål anbefales)
"""
from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


def pool_points(x: torch.Tensor, mode: str) -> torch.Tensor:
    """x: [B, C, N] -> [B, C]"""
    if mode == "max":
        return x.max(dim=2).values
    if mode == "mean":
        return x.mean(dim=2)
    if mode == "sum":
        return x.sum(dim=2)
    raise ValueError(f"pool skal være 'max', 'mean' eller 'sum', ikke {mode!r}")


class PointNetRegressor(nn.Module):
    """
    Args:
        num_outputs:    antal mål der forudsiges (fx 1 eller 2).
        extra_features: antal ekstra features ud over x, y, z (0 = kun koordinater).
        pool:           'max' (original PointNet), 'sum' (brugt i DPCR-AGB) eller 'mean'.
        activation:     'relu' eller 'gelu' (DPCR-AGB bruger gelu).
        embedding_dim:  bredden før pooling (1024 i originalen).
        dropout:        dropout i hovedet (0 = fra).
    """

    def __init__(
        self,
        num_outputs: int = 1,
        extra_features: int = 0,
        pool: str = "max",
        activation: str = "relu",
        embedding_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        if pool not in ("max", "mean", "sum"):
            raise ValueError(f"pool skal være 'max', 'mean' eller 'sum', ikke {pool!r}")
        act = ACTIVATIONS[activation]
        self.pool = pool
        self.extra_features = extra_features

        def conv(i, o):
            return [nn.Conv1d(i, o, 1, bias=False), nn.BatchNorm1d(o), act()]

        def fc(i, o):
            return [nn.Linear(i, o, bias=False), nn.BatchNorm1d(o), act()]

        in_ch = 3 + extra_features
        # Delt MLP: samme vægte på hvert punkt, punkterne ser ikke hinanden her.
        self.point_mlp = nn.Sequential(*conv(in_ch, 64), *conv(64, 128), *conv(128, embedding_dim))
        # MLP på den samlede plot-vektor.
        self.head = nn.Sequential(
            *fc(embedding_dim, 512),
            *fc(512, 256),
            nn.Dropout(dropout),
            nn.Linear(256, num_outputs),
        )

    def forward(self, pos: torch.Tensor, feats: torch.Tensor | None = None) -> torch.Tensor:
        if pos.dim() != 3 or pos.shape[-1] != 3:
            raise ValueError(f"pos skal have formen [B, N, 3], fik {tuple(pos.shape)}")
        if self.extra_features > 0:
            if feats is None or feats.shape[-1] != self.extra_features:
                raise ValueError(f"modellen forventer feats [B, N, {self.extra_features}]")
            x = torch.cat([pos, feats], dim=-1)
        else:
            x = pos
        x = x.transpose(1, 2)            # [B, C, N]
        x = self.point_mlp(x)            # [B, embedding_dim, N]
        x = pool_points(x, self.pool)    # [B, embedding_dim]
        return self.head(x)              # [B, num_outputs]
