"""Modeller til punktsky-regression.

    from models import build_model
    model = build_model("pointnet2", num_outputs=2)
    pred = model(pos)            # pos: [B, N, 3]  ->  pred: [B, 2]
"""
from .pointnet import PointNetRegressor
from .pointnet2 import PointNet2Regressor

MODELS = {
    "pointnet": PointNetRegressor,
    "pointnet2": PointNet2Regressor,
}


def build_model(name: str, **kwargs):
    """Bygger en model ud fra navn. Ekstra argumenter går direkte til modellen."""
    key = name.lower().replace("-", "").replace("+", "").replace("_", "")
    key = {"pointnetpp": "pointnet2"}.get(key, key)
    if key not in MODELS:
        raise ValueError(f"Ukendt model {name!r}. Vælg mellem: {', '.join(MODELS)}")
    return MODELS[key](**kwargs)


def count_parameters(model) -> int:
    """Antal trænbare vægte."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
