"""PointNet++ (SSG) til punktsky-regression, i ren PyTorch.

Opbygning (standardvariant fra PointNet++-artiklen):
  Set Abstraction 1: 512 centre, radius 0.2, 32 naboer
  Set Abstraction 2: 128 centre, radius 0.4, 64 naboer
  Global Set Abstraction: alle punkter -> én vektor
  MLP -> lineært hoved

Radierne er i samme enhed som dine koordinater. Standardværdierne passer til punktskyer,
der er normaliseret til ca. en enhedskugle/-terning (fx centreret og skaleret til [-1, 1]).
Ændr `radii` hvis dit dataset er skaleret anderledes.

Input:   pos   [B, N, 3]
         feats [B, N, F]   valgfrie ekstra features (F = extra_features)
Output:  [B, num_outputs]
"""
from __future__ import annotations

import torch
from torch import nn

from .pointnet import ACTIVATIONS


# ---------------------------------------------------------------- punkt-operationer
def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """src [B, N, 3], dst [B, M, 3] -> kvadrerede afstande [B, N, M]."""
    d = -2 * torch.matmul(src, dst.transpose(1, 2))
    d = d + (src ** 2).sum(-1, keepdim=True) + (dst ** 2).sum(-1).unsqueeze(1)
    return d.clamp_min(0)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """points [B, N, C], idx [B, S] eller [B, S, K] -> [B, S, C] eller [B, S, K, C]."""
    batch = torch.arange(points.shape[0], device=points.device)
    batch = batch.view([-1] + [1] * (idx.dim() - 1)).expand_as(idx)
    return points[batch, idx]


def farthest_point_sample(xyz: torch.Tensor, npoint: int, random_start: bool) -> torch.Tensor:
    """Vælger npoint punkter, der dækker punktskyen jævnt. xyz [B, N, 3] -> indeks [B, npoint]."""
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    dist = torch.full((B, N), float("inf"), device=device)
    if random_start:
        farthest = torch.randint(0, N, (B,), device=device)
    else:
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch = torch.arange(B, device=device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)             # [B, 1, 3]
        dist = torch.minimum(dist, ((xyz - centroid) ** 2).sum(-1))
        farthest = dist.argmax(-1)
    return idx


def ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """For hvert centrum: indeks på op til nsample punkter inden for radius.
    Hvis der er færre, fyldes op med det første gyldige punkt (centret selv er altid med)."""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    sqd = square_distance(new_xyz, xyz)                          # [B, S, N]
    idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, S, 1)
    idx[sqd > radius ** 2] = N                                   # N = "udenfor"
    idx = idx.sort(dim=-1)[0][:, :, :nsample]
    first = idx[:, :, :1].expand(-1, -1, idx.shape[-1])
    mask = idx == N
    idx[mask] = first[mask]
    return idx


# ---------------------------------------------------------------- Set Abstraction
class SetAbstraction(nn.Module):
    """Vælger centre (FPS), samler naboer (ball query), kører en lille PointNet på hver gruppe."""

    def __init__(self, npoint, radius, nsample, in_channels, mlp, activation="relu",
                 group_all=False, pool="max"):
        super().__init__()
        if pool not in ("max", "mean", "sum"):
            raise ValueError(f"pool skal være 'max', 'mean' eller 'sum', ikke {pool!r}")
        act = ACTIVATIONS[activation]
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        self.group_all, self.pool = group_all, pool
        layers, last = [], in_channels + 3                        # +3 for relative xyz
        for out in mlp:
            layers += [nn.Conv2d(last, out, 1, bias=False), nn.BatchNorm2d(out), act()]
            last = out
        self.mlp = nn.Sequential(*layers)
        self.out_channels = last

    def forward(self, xyz, feats):
        """xyz [B, N, 3], feats [B, N, C] eller None -> (new_xyz [B, S, 3], new_feats [B, S, D])."""
        B, N, _ = xyz.shape
        if self.group_all:
            new_xyz = xyz.new_zeros(B, 1, 3)
            grouped_xyz = xyz.unsqueeze(1)                       # [B, 1, N, 3]
            grouped_feats = None if feats is None else feats.unsqueeze(1)
        else:
            npoint, nsample = min(self.npoint, N), min(self.nsample, N)
            with torch.no_grad():
                fps_idx = farthest_point_sample(xyz.float(), npoint, random_start=self.training)
                new_xyz = index_points(xyz, fps_idx)             # [B, S, 3]
                idx = ball_query(self.radius, nsample, xyz.float(), new_xyz.float())  # [B, S, K]
            grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)   # relative koordinater
            grouped_feats = None if feats is None else index_points(feats, idx)
        g = grouped_xyz if grouped_feats is None else torch.cat([grouped_xyz, grouped_feats], dim=-1)
        g = g.permute(0, 3, 2, 1)                                # [B, 3+C, K, S]
        g = self.mlp(g)                                          # [B, D, K, S]
        if self.pool == "max":
            g = g.max(dim=2).values
        elif self.pool == "mean":
            g = g.mean(dim=2)
        else:
            g = g.sum(dim=2)
        return new_xyz, g.transpose(1, 2)                        # [B, S, D]


# ---------------------------------------------------------------- hele modellen
class PointNet2Regressor(nn.Module):
    """
    Args:
        num_outputs:    antal mål der forudsiges.
        extra_features: antal ekstra features ud over x, y, z (0 = kun koordinater).
        npoints:        antal centre i hvert Set Abstraction-lag.
        radii:          gruppe-radius i hvert lag (i dine koordinatenheder).
        nsamples:       max antal naboer pr. gruppe.
        mlps:           MLP-bredder i hvert lag.
        global_mlp:     MLP-bredder i det globale lag.
        global_pool:    pooling i det globale lag: 'max' (original), 'sum' eller 'mean'.
        activation:     'relu' eller 'gelu'.
        dropout:        dropout i hovedet (0 = fra).
    """

    def __init__(
        self,
        num_outputs: int = 1,
        extra_features: int = 0,
        npoints=(512, 128),
        radii=(0.2, 0.4),
        nsamples=(32, 64),
        mlps=((64, 64, 128), (128, 128, 256)),
        global_mlp=(256, 512, 1024),
        global_pool: str = "max",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if not (len(npoints) == len(radii) == len(nsamples) == len(mlps)):
            raise ValueError("npoints, radii, nsamples og mlps skal have samme længde")
        act = ACTIVATIONS[activation]
        self.extra_features = extra_features

        self.sa = nn.ModuleList()
        ch = extra_features
        for np_, r, ns, mlp in zip(npoints, radii, nsamples, mlps):
            layer = SetAbstraction(np_, r, ns, ch, mlp, activation)
            self.sa.append(layer)
            ch = layer.out_channels
        self.global_sa = SetAbstraction(None, None, None, ch, global_mlp, activation,
                                        group_all=True, pool=global_pool)
        d = self.global_sa.out_channels
        self.head = nn.Sequential(
            nn.Linear(d, 512, bias=False), nn.BatchNorm1d(512), act(),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), act(),
            nn.Dropout(dropout),
            nn.Linear(256, num_outputs),
        )

    def forward(self, pos: torch.Tensor, feats: torch.Tensor | None = None) -> torch.Tensor:
        if pos.dim() != 3 or pos.shape[-1] != 3:
            raise ValueError(f"pos skal have formen [B, N, 3], fik {tuple(pos.shape)}")
        if self.extra_features > 0:
            if feats is None or feats.shape[-1] != self.extra_features:
                raise ValueError(f"modellen forventer feats [B, N, {self.extra_features}]")
        else:
            feats = None
        xyz = pos
        for sa in self.sa:
            xyz, feats = sa(xyz, feats)
        _, feats = self.global_sa(xyz, feats)                    # [B, 1, D]
        return self.head(feats.squeeze(1))                       # [B, num_outputs]
