"""Small, dependency-free loader for the repository's JSON-compatible YAML files."""
from __future__ import annotations

import json
import math
from pathlib import Path
from functools import lru_cache


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@lru_cache(maxsize=None)
def load_environment_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name.lower()}.yml"
    if not path.is_file():
        raise FileNotFoundError(f"Environment contract is missing: {path}")
    # The files use JSON's YAML-compatible subset so the runtime has no new dependency.
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    for field in ("fps", "control_hz", "physics_hz"):
        if type(config.get(field)) is not int or config[field] <= 0:
            raise ValueError(f"{path.name}: {field} must be a positive integer.")
    if (config["control_hz"] % config["fps"]
            or config["physics_hz"] % config["control_hz"]):
        raise ValueError(f"{path.name}: control_hz must be a multiple of fps, and physics_hz of control_hz.")
    if name.lower() == "pusht" and config["control_hz"] != config["fps"]:
        raise ValueError("PushT requires control_hz == fps; interpolation is supported by AlignBox3D.")
    if name.lower() == "alignbox3d":
        for field in ("translation_speed_m_s", "rotation_speed_deg_s"):
            value = config.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{path.name}: {field} must be finite and positive.")
    return config
