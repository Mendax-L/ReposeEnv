"""Small, dependency-free loader for the repository's JSON-compatible YAML files."""
from __future__ import annotations

import json
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
        return json.load(stream)
