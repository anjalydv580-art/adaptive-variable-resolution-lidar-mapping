"""
Top-level entry point for the Adaptive Variable Resolution 2.5D LiDAR
Mapping project.

Phase 1 status: this script only verifies that the project scaffold and
configuration file are valid and prints a summary. The full simulation /
detection / mapping / dashboard pipeline is wired in here as later phases
are implemented (see README.md "Development Phases").
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load and return the YAML configuration as a dict."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_structure(config: dict) -> list[str]:
    """Check that the folders referenced in config['paths'] exist.

    Returns a list of any missing paths (empty list == all good).
    """
    missing = []
    for key, rel_path in config.get("paths", {}).items():
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            missing.append(f"{key} -> {rel_path}")
    return missing


def main() -> int:
    print("=" * 70)
    print("Adaptive Variable Resolution 2.5D LiDAR Mapping")
    print("=" * 70)

    try:
        config = load_config()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load config: {exc}")
        return 1

    print(f"Project name : {config['project']['name']}")
    print(f"Random seed  : {config['project']['seed']}")

    missing = verify_structure(config)
    if missing:
        print("[WARNING] Missing expected paths:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("All configured paths exist. Project scaffold is valid.")

    print("-" * 70)
    print("Phase 1 (scaffold + config) is complete.")
    print("Run 'pytest -v' once test modules are implemented (later phases).")
    print("Run 'streamlit run dashboard/app.py' once the dashboard is implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
