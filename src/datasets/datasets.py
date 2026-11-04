import yaml
from pathlib import Path

ROOT = Path("02data")

def area_dirs(split: str, splits_file="01configs/splits.yaml") -> list[Path]:
    cfg = yaml.safe_load(Path(splits_file).read_text())
    seen = [a for areas in cfg.values() for a in areas]
    assert len(seen) == len(set(seen)), "et område står i flere splits"
    dirs = [ROOT / "raw" / a for a in cfg[split]]      # 02data/raw/skov_nomi
    for d in dirs:
        assert d.is_dir(), f"mappen findes ikke: {d}"
    return dirs