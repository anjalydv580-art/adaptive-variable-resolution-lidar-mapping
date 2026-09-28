"""Run the full detection + mapping pipeline (Phase 15 CLI)."""
from __future__ import annotations
import argparse
import json
from src.pipeline import run_pipeline
from src.utils import ensure_dir, load_config, resolve_path
def main() -> int:
    ap = argparse.ArgumentParser(description="Run adaptive mapping pipeline")
    ap.add_argument("--num-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=str, default="outputs/reports/pipeline.json")
    args = ap.parse_args()
    res = run_pipeline(num_frames=args.num_frames, seed=args.seed)
    out = resolve_path(args.out)
    ensure_dir(out.parent)
    payload = {"num_frames": res.num_frames, "seed": res.seed,
               "summary": res.summary,
               "metrics": {k: float(v) for k, v in res.metrics.items()},
               "ml_evaluation": res.ml_evaluation,
               "frames": [f.to_dict() for f in res.frames]}
    out.write_text(json.dumps(payload, indent=2, default=float),
                   encoding="utf-8")
    print(f"Wrote {out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
