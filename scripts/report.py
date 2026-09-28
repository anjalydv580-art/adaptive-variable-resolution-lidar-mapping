"""Research report CLI (Phase 18).

Reads the real Phase-16 benchmark CSV, optionally runs one small
deterministic pipeline + benchmark pass when requested, and writes the
Markdown report plus research-quality PNG figures. Importable from the
dashboard via :func:`run_and_save`.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Optional
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.reporting import (DEFAULT_BENCHMARK_CSV, DEFAULT_FIGURES_DIR,
                           DEFAULT_REPORT_MD, generate_report,
                           load_benchmark_rows)
from src.benchmark import benchmark_configs


def run_and_save(csv_path: Optional[str] = None,
                 out_md: str = DEFAULT_REPORT_MD,
                 out_dir: str = DEFAULT_FIGURES_DIR,
                 frames: Optional[List[int]] = None,
                 scenarios: Optional[List[str]] = None,
                 seed: Optional[int] = None) -> dict:
    """Build the report; optionally measure fresh rows when requested."""
    rows = load_benchmark_rows(csv_path or DEFAULT_BENCHMARK_CSV)
    pipe = None
    if not rows and (frames is not None or scenarios is not None):
        rows = benchmark_configs(frame_counts=frames or [1, 2],
                                 scenarios=scenarios or ["default"],
                                 seed=seed)
        from src.pipeline import run_pipeline
        pipe = run_pipeline(num_frames=int((frames or [1])[0]),
                            seed=seed if seed is not None else 42)
    elif not rows:
        rows = benchmark_configs(frame_counts=[1, 2],
                                 scenarios=["default"], seed=seed)
        from src.pipeline import run_pipeline
        pipe = run_pipeline(num_frames=1,
                            seed=seed if seed is not None else 42)
    else:
        from src.pipeline import run_pipeline
        try:
            first = rows[0]
            pipe = run_pipeline(num_frames=int(first.get("num_frames", 1)),
                                seed=seed if seed is not None else int(
                                    first.get("seed", 42)))
        except (TypeError, ValueError):
            pipe = None
    return generate_report(rows=rows, pipeline_result=pipe,
                           out_md=out_md, out_dir=out_dir,
                           seed=seed if seed is not None else 42)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build research report")
    ap.add_argument("--csv", type=str, default=DEFAULT_BENCHMARK_CSV)
    ap.add_argument("--out", type=str, default=DEFAULT_REPORT_MD)
    ap.add_argument("--figures-dir", type=str, default=DEFAULT_FIGURES_DIR)
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--scenarios", type=str, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    res = run_and_save(csv_path=args.csv, out_md=args.out,
                       out_dir=args.figures_dir, frames=args.frames,
                       scenarios=args.scenarios, seed=args.seed)
    print(f"Wrote {res['report']}")
    for p in res["figures"]:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
