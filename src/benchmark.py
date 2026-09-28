"""Deterministic adaptive-vs-fixed benchmark system (Phase 16).

Runs the real Phase-14 pipeline (:func:`src.pipeline.run_pipeline`) over a
matrix of frame counts x scenarios and collects the real Phase-12 comparison
metrics (:class:`src.metrics.ComparisonMetrics`) — no fake numbers, no
duplicated mapping logic, no hard-coded results.

Metric definitions (all measured, deterministic except wall-clock timing)
------------------------------------------------------------------------
- ``processing_time_adaptive`` / ``processing_time_fixed``: wall-clock
  seconds spent building each map (caller-measured, non-deterministic).
- ``speedup``: ``processing_time_fixed / processing_time_adaptive``.
- ``num_cells_adaptive`` / ``num_cells_fixed``: occupied finest-grid cells.
- ``coverage_adaptive`` / ``coverage_fixed``: occupied / total finest cells.
- ``compression_ratio``: ``num_cells_adaptive / num_cells_fixed``.
- ``cell_reduction``: ``1 - compression_ratio``.
- ``elevation_mae`` / ``elevation_rmse`` / ``elevation_bias``: adaptive vs
  fixed mean-height error over cells where both maps are finite.
- ``occupied_cell_agreement``: fraction of union-occupied cells where both
  maps agree (1.0 on empty union).
- ``total_points_processed`` / ``total_objects`` / ``total_dynamic``:
  pipeline totals for the run.
- ``ml_accuracy_vs_rule`` / ``ml_model_source``: classifier evaluation vs
  the controller rule and where the model came from.

Phase-15 backwards compatibility is preserved: :func:`benchmark_frame_counts`
keeps its signature (plus optional scenario arguments) and its row schema.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

from .pipeline import run_pipeline


def benchmark_frame_counts(
    frame_counts: Optional[List[int]] = None,
    seed: Optional[int] = None,
    config: Optional[dict] = None,
    config_path: Optional[str] = None,
    scenario: str = "default",
    scenario_overrides: Optional[dict] = None,
) -> List[Dict[str, object]]:
    """Run the pipeline for each frame count; return one row dict per run.

    ``scenario`` selects a deterministic config overlay from
    :data:`SCENARIOS` (explicit ``scenario_overrides`` are merged on top).
    Every row carries a ``"scenario"`` key plus the Phase-15 columns.
    """
    counts = [2, 3] if frame_counts is None else [int(c) for c in frame_counts]
    overrides = _scenario_overrides(scenario, scenario_overrides)
    rows: List[Dict[str, object]] = []
    for n in counts:
        if int(n) < 0:
            raise ValueError(f"frame count must be >= 0, got {n}")
        eff_config = _effective_config(config, config_path, overrides)
        if eff_config is None:
            res = run_pipeline(num_frames=int(n), seed=seed)
        else:
            res = run_pipeline(num_frames=int(n), config=eff_config,
                               seed=seed)
        m = dict(res.metrics)
        row: Dict[str, object] = {
            "scenario": str(scenario),
            "num_frames": int(n),
            "seed": res.seed,
            "total_points_processed": int(
                sum(f.num_points_processed for f in res.frames)),
            "total_objects": int(sum(len(f.objects) for f in res.frames)),
            "total_dynamic": int(sum(f.num_dynamic for f in res.frames)),
            "ml_model_source": str(res.ml_model_source),
            "ml_accuracy_vs_rule": float(
                res.ml_evaluation.get("accuracy", 0.0)),
        }
        for k in ("occupied_cell_agreement", "elevation_mae",
                  "elevation_rmse", "elevation_bias", "coverage_adaptive",
                  "coverage_fixed", "num_cells_adaptive", "num_cells_fixed",
                  "compression_ratio", "cell_reduction",
                  "processing_time_adaptive", "processing_time_fixed",
                  "speedup"):
            row[k] = float(m.get(k, float("nan")))
        rows.append(row)
    return rows


def rows_to_dataframe(rows: List[Dict[str, object]]):
    """Convert benchmark rows to a pandas DataFrame (pandas required)."""
    import pandas as pd

    return pd.DataFrame(list(rows))


TIMING_KEYS = frozenset({"processing_time_adaptive",
                         "processing_time_fixed", "speedup"})
REQUIRED_METRICS = ("occupied_cell_agreement", "elevation_mae",
                    "elevation_rmse", "elevation_bias", "coverage_adaptive",
                    "coverage_fixed", "num_cells_adaptive", "num_cells_fixed",
                    "compression_ratio", "cell_reduction",
                    "processing_time_adaptive", "processing_time_fixed",
                    "speedup")
SCENARIOS: Dict[str, Dict[str, object]] = {
    "default": {},
    "sparse": {"simulation": {"obstacles": {"num_buildings": 2},
                              "vehicles": {"num_vehicles": 1},
                              "pedestrians": {"num_pedestrians": 2}}},
    "dense_urban": {"simulation": {"obstacles": {"num_buildings": 8},
                                   "vehicles": {"num_vehicles": 6},
                                   "pedestrians": {"num_pedestrians": 8}}},
}
AVAILABLE_SCENARIOS = tuple(SCENARIOS)
DEFAULT_BENCHMARK_CSV = "outputs/benchmarks/benchmark.csv"
DEFAULT_FIGURES_DIR = "outputs/figures"


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in dict(overrides or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _scenario_overrides(scenario: str, extra: Optional[dict] = None) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    return _deep_merge(SCENARIOS[scenario], extra or {})


def _effective_config(config, config_path, overrides):
    if config is None and config_path is None and not overrides:
        return None
    from .utils import load_config
    base = copy.deepcopy(config) if config is not None else load_config(
        config_path)
    return _deep_merge(base, overrides) if overrides else base


def deterministic_row(row):
    """Return row without non-deterministic wall-clock timing keys."""
    return {k: v for k, v in row.items() if k not in TIMING_KEYS}


def rows_equal_deterministic(a, b):
    """Compare row lists on deterministic fields only."""
    return [deterministic_row(r) for r in a] == [
        deterministic_row(r) for r in b]


def benchmark_configs(frame_counts=None, scenarios=None, seed=None,
                      config=None, config_path=None):
    """Run pipeline over frame-count x scenario matrix; one row per combo."""
    counts = [2, 3] if frame_counts is None else [int(c) for c in frame_counts]
    names = ["default"] if scenarios is None else [str(s) for s in scenarios]
    rows: List[Dict[str, object]] = []
    for name in names:
        rows.extend(benchmark_frame_counts(frame_counts=counts, seed=seed,
                                           config=config,
                                           config_path=config_path,
                                           scenario=name))
    return rows


def run_benchmark(frame_counts=None, scenarios=None, seed=None,
                  config=None, config_path=None):
    """Dashboard-friendly alias of benchmark_configs."""
    return benchmark_configs(frame_counts=frame_counts, scenarios=scenarios,
                             seed=seed, config=config,
                             config_path=config_path)


def save_benchmark_csv(rows, path=None):
    """Save rows as CSV, creating parent dirs; return path string."""
    from .utils import ensure_dir, resolve_path
    out = resolve_path(path or DEFAULT_BENCHMARK_CSV)
    ensure_dir(out.parent)
    rows_to_dataframe(rows).to_csv(out, index=False)
    return str(out)


def _finite(values):
    import numpy as np
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = float("nan")
        out.append(f if np.isfinite(f) else float("nan"))
    return out


def generate_benchmark_charts(rows, out_dir=None):
    """Generate PNG adaptive-vs-fixed charts; return file paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .utils import ensure_dir, resolve_path
    if not rows:
        return []
    out = resolve_path(out_dir or DEFAULT_FIGURES_DIR)
    ensure_dir(out)
    by_scen: Dict[str, list] = {}
    for r in rows:
        by_scen.setdefault(str(r.get("scenario", "default")), []).append(r)
    for v in by_scen.values():
        v.sort(key=lambda r: int(r.get("num_frames", 0)))
    paths: List[str] = []

    def _save(fig, name):
        p = out / name
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(str(p))

    fig, ax = plt.subplots()
    for scen, rs in sorted(by_scen.items()):
        x = np.array([int(r["num_frames"]) for r in rs], dtype=float)
        ya = np.array(_finite(r.get("processing_time_adaptive") for r in rs))
        yf = np.array(_finite(r.get("processing_time_fixed") for r in rs))
        ax.plot(x, ya, "o-", label=f"{scen}/adaptive")
        ax.plot(x, yf, "s--", label=f"{scen}/fixed")
    ax.set_xlabel("num_frames")
    ax.set_ylabel("processing time (s)")
    ax.set_title("Adaptive vs fixed processing time")
    ax.legend(fontsize="small")
    _save(fig, "benchmark_timing.png")
    fig, ax = plt.subplots()
    labels, ca, cf = [], [], []
    for scen, rs in sorted(by_scen.items()):
        for r in rs:
            labels.append(f"{scen}/{int(r['num_frames'])}")
            fa = float(r.get("num_cells_adaptive", 0.0))
            ff = float(r.get("num_cells_fixed", 0.0))
            ca.append(fa if np.isfinite(fa) else 0.0)
            cf.append(ff if np.isfinite(ff) else 0.0)
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, ca, w, label="adaptive")
    ax.bar(x + w / 2, cf, w, label="fixed")
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize="small")
    ax.set_ylabel("occupied cells")
    ax.set_title("Occupied cells: adaptive vs fixed")
    ax.legend()
    _save(fig, "benchmark_cells.png")
    fig, ax = plt.subplots()
    for scen, rs in sorted(by_scen.items()):
        x = np.array([int(r["num_frames"]) for r in rs], dtype=float)
        mae = np.array(_finite(r.get("elevation_mae") for r in rs))
        rmse = np.array(_finite(r.get("elevation_rmse") for r in rs))
        ax.plot(x, mae, "o-", label=f"{scen}/MAE")
        ax.plot(x, rmse, "s--", label=f"{scen}/RMSE")
    ax.set_xlabel("num_frames")
    ax.set_ylabel("elevation error (m)")
    ax.set_title("Elevation MAE / RMSE (adaptive vs fixed)")
    ax.legend(fontsize="small")
    _save(fig, "benchmark_error.png")
    fig, ax = plt.subplots()
    labels, ag, rd, cva = [], [], [], []
    for scen, rs in sorted(by_scen.items()):
        for r in rs:
            labels.append(f"{scen}/{int(r['num_frames'])}")
            for lst, key in ((ag, "occupied_cell_agreement"),
                             (rd, "cell_reduction"),
                             (cva, "coverage_adaptive")):
                try:
                    f = float(r.get(key, float("nan")))
                except (TypeError, ValueError):
                    f = float("nan")
                lst.append(f if np.isfinite(f) else 0.0)
    x = np.arange(len(labels))
    w = 0.24
    ax.bar(x - w, ag, w, label="agreement")
    ax.bar(x, rd, w, label="cell reduction")
    ax.bar(x + w, cva, w, label="coverage (adapt)")
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize="small")
    ax.set_ylabel("ratio")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Agreement / reduction / coverage")
    ax.legend(fontsize="small")
    _save(fig, "benchmark_agreement.png")
    return paths
