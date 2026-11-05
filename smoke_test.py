"""Hurtig test af modellerne. Kør fra projektets rod: python smoke_test.py"""
import time

import torch

from src.models import build_model, count_parameters
from src.models.pointnet2 import ball_query, farthest_point_sample, index_points, square_distance

torch.manual_seed(0)
B, N = 8, 1024
pos = torch.randn(B, N, 3)
pos = pos / pos.norm(dim=-1).max()          # ca. enhedskugle
feats = torch.randn(B, N, 2)

# Forward, backward og gradienter for begge modeller (med og uden ekstra features)
for name in ("pointnet", "pointnet2"):
    for kw in ({}, {"num_outputs": 2, "extra_features": 2}):
        model = build_model(name, **kw)
        f = feats if kw.get("extra_features") else None
        model.train()
        t = time.time()
        out = model(pos, f)
        assert out.shape == (B, kw.get("num_outputs", 1)), out.shape
        out.sum().backward()
        bad = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not bad, f"manglende/ugyldige gradienter: {bad[:3]}"
        print(f"{name:10s} {str(kw):45s} out={tuple(out.shape)} "
              f"params={count_parameters(model):>9,d}  {time.time() - t:.2f}s")

# PointNet skal ikke afhænge af punkternes rækkefølge
m = build_model("pointnet", pool="sum", activation="gelu").eval()
perm = torch.randperm(N)
assert torch.allclose(m(pos), m(pos[:, perm]), atol=1e-4), "PointNet er afhængig af rækkefølgen"

# Færre punkter end npoint, og batch på 1 i eval
assert build_model("pointnet2").eval()(pos[:1, :200]).shape == (1, 1)

# FPS og ball query mod brute force
xyz = torch.rand(3, 500, 3)
idx = farthest_point_sample(xyz, 64, random_start=True)
assert all(len(set(r.tolist())) == 64 for r in idx), "FPS gav dubletter"
c = index_points(xyz, idx)
g = ball_query(0.25, 16, xyz, c)
d = (index_points(xyz, g) - c.unsqueeze(2)).norm(dim=-1)
assert (d <= 0.25 + 1e-5).all(), "ball query gav punkter uden for radius"
sq = square_distance(c, xyz)
for b in range(3):
    for s in range(64):
        assert len(set(g[b, s].tolist())) == min(16, int((sq[b, s] <= 0.25 ** 2).sum()))

# Pooling-varianter og fejlbesked
for p in ("max", "mean", "sum"):
    build_model("pointnet", pool=p)(pos)
    build_model("pointnet2", global_pool=p)(pos)
try:
    build_model("pointnet", extra_features=2)(pos)
except ValueError as e:
    print("forventet fejl:", e)
print("OK")
