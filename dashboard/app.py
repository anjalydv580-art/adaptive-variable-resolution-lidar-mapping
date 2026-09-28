"""Interactive Streamlit dashboard (Phase 15)."""
from __future__ import annotations
from src.dashboard_utils import render_dashboard
from src.pipeline import run_pipeline
from src.utils import load_config
def main() -> None:
    cfg = load_config()
    n = int(cfg.get("dashboard", {}).get("default_num_frames", 3))
    res = run_pipeline(num_frames=n, config=cfg)
    render_dashboard(res, cfg)
if __name__ == "__main__":
    main()
