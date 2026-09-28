"""Shared utility helpers used across the project: config loading, RNG
creation and small filesystem helpers.

Every module in this project reads its parameters through
:func:`load_config` so nothing is hard-coded in the source files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml

# Project root = parent of the `src/` directory that contains this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """Load the YAML configuration file and return it as a dict.

    Parameters
    ----------
    config_path:
        Optional path to a YAML config file. Defaults to
        ``config/settings.yaml`` under the project root.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file did not parse to a dict: {path}")
    return config


def get_rng(seed: Optional[int] = None, config: Optional[dict] = None) -> np.random.Generator:
    """Build a NumPy random Generator for reproducible randomness.

    If ``seed`` is None, falls back to ``config['project']['seed']`` when a
    config dict is supplied, otherwise uses NumPy's default entropy-based
    seeding (non-reproducible).
    """
    if seed is None and config is not None:
        seed = config.get("project", {}).get("seed")
    return np.random.default_rng(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create ``path`` (and parents) if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(relative_path: Union[str, Path]) -> Path:
    """Resolve a path relative to the project root (absolute paths pass through)."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p
