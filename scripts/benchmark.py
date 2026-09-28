"""Adaptive-vs-fixed benchmark CLI (Phase 16).

Runs the real Phase-14 pipeline across frame counts x scenarios, saves a
structured CSV under ``outputs/benchmarks/`` and PNG comparison charts under
``outputs/figures/``. Also importable from the dashboard via
:func:`run_and_save`.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Optional
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.benchmark import (AVAILABLE_SCENARIOS, benchmark_configs,
                           generate_benchmark_charts, rows_to_dataframe,
                           save_benchmark_csv)
from src.utils import ensure_dir, resolve_path
def run_and_save(frame_counts: Optional[List[int]] = None,
                 scenarios: Optional[List[str]] = None,
                 seed: Optional[int] = None,
                 out: str = "outputs/benchmarks/benchmark.csv",
                 figures_dir: str = "outputs/figures",
                 make_charts: bool = True) -> dict:
    """Run the benchmark matrix, save CSV (+charts); return paths and rows.

    Usable both as a script backend and imported from the dashboard.
    """
    rows = benchmark_configs(frame_counts=frame_counts, scenarios=scenarios,
                             seed=seed)
    csv_path = save_benchmark_csv(rows, out)
    chart_paths = generate_benchmark_charts(rows, figures_dir) if make_charts else []
    return {"rows": rows, "csv": csv_path, "charts": chart_paths}
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark adaptive vs fixed")
    ap.add_argument("--frames", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--scenarios", type=str, nargs="+",
                    default=["default"], choices=list(AVAILABLE_SCENARIOS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=str,
                    default="outputs/benchmarks/benchmark.csv")
    ap.add_argument("--figures-dir", type=str, default="outputs/figures")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args(argv)
    result = run_and_save(frame_counts=list(args.frames),
                          scenarios=list(args.scenarios), seed=args.seed,
                          out=args.out, figures_dir=args.figures_dir,
                          make_charts=not args.no_charts)
    df = rows_to_dataframe(result["rows"])
    print(df.to_string(index=False))
    print(f"Wrote {result['csv']}")
    for p in result["charts"]:
        print(f"Wrote {p}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
