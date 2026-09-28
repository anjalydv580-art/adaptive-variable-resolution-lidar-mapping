"""Generate simulated LiDAR scenario data (Phase 15 CLI)."""
from __future__ import annotations
import argparse
import numpy as np
from src.lidar_simulator import LidarSimulator
from src.utils import ensure_dir, load_config, resolve_path
def main() -> int:
    ap = argparse.ArgumentParser(description="Generate LiDAR frames")
    ap.add_argument("--num-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=str, default="data/raw/scenario.npz")
    args = ap.parse_args()
    cfg = load_config()
    n = int(args.num_frames) if args.num_frames is not None else int(
        cfg.get("simulation", {}).get("num_frames", 3))
    sim = LidarSimulator(config=cfg,
                         seed=args.seed if args.seed is not None else None)
    frames = [sim.simulate_frame(i) for i in range(max(0, n))]
    pts = [np.asarray(f.points, dtype=float) for f in frames]
    out = resolve_path(args.out)
    ensure_dir(out.parent)
    np.savez_compressed(out, **{f"frame_{i}": p for i, p in enumerate(pts)},
                        seed=np.asarray([int(sim.seed or 0)]))
    print(f"Wrote {out} ({len(frames)} frames)")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
