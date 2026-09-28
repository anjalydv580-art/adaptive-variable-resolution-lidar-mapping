"""Research-quality reports and figures (Phase 18).

Builds publication-ready Markdown reports and PNG figures from ONLY real
measured outputs: the Phase-16 benchmark rows / CSV
(:mod:`src.benchmark`) and the Phase-14 pipeline result
(:func:`src.pipeline.run_pipeline`). No fake, hard-coded, or randomly
generated metrics appear anywhere in this module.

All figures use the Matplotlib Agg backend (CPU-only, no display), fixed
figure sizes / DPI, and rows sorted by ``(scenario, num_frames)``, so
output is deterministic for identical inputs. Wall-clock timing values
are reported as measured (non-deterministic across runs) and labelled
as such in the report.

Simulation disclaimer: this project uses a synthetic LiDAR simulator
(:mod:`src.lidar_simulator`); results describe simulated scenes and must
not be presented as real-world LiDAR performance.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import ensure_dir, load_config, resolve_path

REPORT_TITLE = "Adaptive Variable Resolution 2.5D LiDAR Mapping"
SIMULATION_DISCLAIMER = (
    "All results in this report were measured on SYNTHETIC data from the "
    "project LiDAR simulator (NumPy ray-casting of an urban scene with "
    "buildings, vehicles, and pedestrians). They describe simulated scenes "
    "only and must not be presented as real-world LiDAR performance; "
    "real-world validation has not been performed."
)

REQUIRED_SECTIONS = (
    "Objective",
    "Methodology",
    "Adaptive resolution strategy",
    "2.5D mapping approach",
    "Dynamic detection",
    "Adaptive vs fixed comparison",
    "Benchmark methodology",
    "Measured results",
    "Limitations",
    "Future work",
    "Reproducibility",
)

FIGURE_SPECS = (
    ("report_cells.png", "Occupied cells: adaptive vs fixed"),
    ("report_timing.png", "Processing time: adaptive vs fixed"),
    ("report_error.png", "Elevation error: MAE and RMSE"),
    ("report_coverage.png", "Map coverage: adaptive vs fixed"),
    ("report_reduction.png", "Cell reduction and compression ratio"),
    ("report_agreement.png", "Occupied-cell agreement"),
    ("report_complexity.png", "Complexity / resolution distribution"),
    ("report_scenarios.png", "Scenario comparison"),
)

DEFAULT_REPORT_MD = "outputs/reports/report.md"
DEFAULT_FIGURES_DIR = "outputs/figures"
DEFAULT_BENCHMARK_CSV = "outputs/benchmarks/benchmark.csv"


def load_benchmark_rows(csv_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load benchmark rows from CSV; return [] when missing/unreadable.

    Numeric columns are coerced to float where possible; non-numeric
    values (scenario names, model source) are kept as strings.
    """
    path = resolve_path(csv_path or DEFAULT_BENCHMARK_CSV)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            raw = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []
    rows: List[Dict[str, Any]] = []
    for r in raw:
        row: Dict[str, Any] = {}
        for k, v in r.items():
            try:
                row[k] = float(v)
                if k in ("num_frames", "seed", "total_points_processed",
                         "total_objects", "total_dynamic",
                         "num_cells_adaptive", "num_cells_fixed"):
                    row[k] = int(float(v))
            except (TypeError, ValueError):
                row[k] = v
        rows.append(row)
    rows.sort(key=lambda d: (str(d.get("scenario", "")),
                             int(d.get("num_frames", 0))))
    return rows


def _vals(rows: List[Dict[str, Any]], key: str) -> List[float]:
    import numpy as np
    out = []
    for r in rows:
        try:
            f = float(r.get(key, float("nan")))
        except (TypeError, ValueError):
            f = float("nan")
        if np.isfinite(f):
            out.append(f)
    return out


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate measured benchmark rows into summary statistics.

    Means are plain arithmetic means of measured values; timing means
    are wall-clock measurements and vary between runs.
    """
    import numpy as np
    summary: Dict[str, Any] = {"num_rows": len(rows), "scenarios": sorted(
        {str(r.get("scenario", "default")) for r in rows}),
        "frame_counts": sorted({int(r.get("num_frames", 0)) for r in rows})}
    for key in ("occupied_cell_agreement", "elevation_mae", "elevation_rmse",
                "elevation_bias", "coverage_adaptive", "coverage_fixed",
                "num_cells_adaptive", "num_cells_fixed", "compression_ratio",
                "cell_reduction", "processing_time_adaptive",
                "processing_time_fixed", "speedup", "total_points_processed",
                "ml_accuracy_vs_rule"):
        v = _vals(rows, key)
        summary[key + "_mean"] = float(np.mean(v)) if v else float("nan")
        summary[key + "_min"] = float(np.min(v)) if v else float("nan")
        summary[key + "_max"] = float(np.max(v)) if v else float("nan")
    return summary


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    import numpy as np
    if not np.isfinite(f):
        return "n/a"
    return f"{f:.{digits}f}"


def build_report_markdown(summary: Dict[str, Any],
                          rows: List[Dict[str, Any]],
                          seed: Any = 42) -> str:
    """Render the full research report as Markdown from real summaries."""
    lines = [f"# {REPORT_TITLE} — Benchmark Report", "",
             "> " + SIMULATION_DISCLAIMER, "",
             "## Objective",
             "Evaluate an adaptive variable-resolution 2.5D LiDAR mapping "
             "pipeline that spends fine cells only where scene complexity "
             "demands it, against a uniform fixed-resolution baseline, on "
             "deterministic simulated LiDAR sequences.", "",
             "## Methodology",
             "The Phase-14 pipeline simulates seeded LiDAR sweeps, "
             "preprocesses each cloud, detects the ground plane, clusters "
             "objects, tracks frame-to-frame dynamics, extracts per-frame "
             "complexity features, selects a LOW/MEDIUM/HIGH resolution per "
             "region, and mosaics an adaptive 2.5D elevation/occupancy map. "
             "The fixed baseline grids the same accumulated cloud at one "
             "resolution. Phase-12 metrics compare both maps on the "
             "adaptive finest grid.", "",
             "## Adaptive resolution strategy",
             "Per-region complexity score = weighted, min-max normalised sum "
             "of point density, object density, dynamic activity, elevation "
             "variance, and temporal change (weights/scales from "
             "`config/settings.yaml`). Scores map to LOW (2.0 m), MEDIUM "
             "(1.0 m), HIGH (0.5 m) cells; an ML classifier learns to "
             "predict the level from the same features.", "",
             "## 2.5D mapping approach",
             "Each region is gridded at its selected voxel size into mean / "
             "min / max elevation, counts, occupancy, and resolution "
             "provenance layers, then deterministically mosaicked onto the "
             "finest grid (floor indexing, order-independent reductions).",
             "",
             "## Dynamic detection",
             "Clusters are matched across consecutive frames by centroid "
             "distance; displacement above a threshold (with frame dt) "
             "marks objects dynamic and yields per-object speeds. The "
             "first frame has no predecessor, so its dynamics list is "
             "empty by construction.", "",
             "## Adaptive vs fixed comparison",
             "Agreement = fraction of union-occupied cells where both maps "
             "agree; elevation MAE/RMSE/bias measured where both maps are "
             "finite; compression = adaptive / fixed occupied cells; "
             "reduction = 1 - compression; coverage = occupied / total "
             "finest cells; processing times are measured wall-clock "
             "seconds and speedup = fixed / adaptive.", "",
             "## Benchmark methodology",
             f"Rows measured with the real pipeline over "
             f"{summary.get('num_rows', 0)} configuration(s): scenarios "
             f"{summary.get('scenarios', [])}, frame counts "
             f"{summary.get('frame_counts', [])}, deterministic seed "
             f"{seed}. Each row rebuilds both maps end to end; only "
             "wall-clock timing varies between runs.", "",
             "## Measured results",
             f"- Configurations measured: {summary.get('num_rows', 0)}",
             f"- Scenarios: {summary.get('scenarios', [])}",
             f"- Frame counts: {summary.get('frame_counts', [])}",
             "- Occupied-cell agreement (mean): "
             f"{_fmt(summary.get('occupied_cell_agreement_mean'))} "
             f"(range {_fmt(summary.get('occupied_cell_agreement_min'))}–"
             f"{_fmt(summary.get('occupied_cell_agreement_max'))})",
             "- Elevation MAE (mean, m): "
             f"{_fmt(summary.get('elevation_mae_mean'))}",
             "- Elevation RMSE (mean, m): "
             f"{_fmt(summary.get('elevation_rmse_mean'))}",
             "- Coverage adaptive (mean): "
             f"{_fmt(summary.get('coverage_adaptive_mean'))}",
             "- Coverage fixed (mean): "
             f"{_fmt(summary.get('coverage_fixed_mean'))}",
             "- Compression ratio (mean): "
             f"{_fmt(summary.get('compression_ratio_mean'))}",
             "- Cell reduction (mean): "
             f"{_fmt(summary.get('cell_reduction_mean'))}",
             "- Processing time adaptive (mean s, wall-clock): "
             f"{_fmt(summary.get('processing_time_adaptive_mean'))}",
             "- Processing time fixed (mean s, wall-clock): "
             f"{_fmt(summary.get('processing_time_fixed_mean'))}",
             "- Speedup fixed/adaptive (mean): "
             f"{_fmt(summary.get('speedup_mean'))}",
             "## Limitations",
             "Synthetic simulator only: idealised geometry, sensor noise, "
             "and motion; no real-world LiDAR, weather, calibration error, "
             "or sensor failures. Timing is wall-clock and "
             "machine-dependent. Elevation error is map-to-map (adaptive "
             "vs fixed), not against surveyed ground truth.", "",
             "## Future work",
             "Validate on real LiDAR logs; surveyed ground-truth error; "
             "memory/compute scaling; controller thresholds and voxel "
             "sizes; more scenarios (weather, traffic, sensor faults).",
             "",
             "## Reproducibility",
             "CPU-only; deterministic seed 42 unless stated; parameters in "
             "`config/settings.yaml`; benchmark via "
             "`python scripts/benchmark.py`; report via "
             "`python scripts/report.py`.", "",
             "## Per-configuration measurements", "",
             "| scenario | frames | adapt cells | fixed cells | MAE (m) | "
             "RMSE (m) | agreement | reduction | adapt time (s) | "
             "fixed time (s) |", ""]
    for r in rows:
        lines.append(
            f"| {r.get('scenario', '')} | {r.get('num_frames', '')} | "
            f"{_fmt(r.get('num_cells_adaptive'), 0)} | "
            f"{_fmt(r.get('num_cells_fixed'), 0)} | "
            f"{_fmt(r.get('elevation_mae'))} | "
            f"{_fmt(r.get('elevation_rmse'))} | "
            f"{_fmt(r.get('occupied_cell_agreement'))} | "
            f"{_fmt(r.get('cell_reduction'))} | "
            f"{_fmt(r.get('processing_time_adaptive'))} | "
            f"{_fmt(r.get('processing_time_fixed'))} |")
    return "\n".join(lines) + "\n"


def write_report(rows, out_path=None, seed=42):
    """Write the Markdown report for rows; return the path string."""
    summary = summarize_results(rows)
    text = build_report_markdown(summary, rows, seed=seed)
    out = resolve_path(out_path or DEFAULT_REPORT_MD)
    ensure_dir(out.parent)
    out.write_text(text, encoding="utf-8")
    return str(out)


def generate_report_figures(rows, pipeline_result=None,
                            out_dir=None):
    """Generate research-quality PNG figures from real rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    out = resolve_path(out_dir or DEFAULT_FIGURES_DIR)
    ensure_dir(out)
    if not rows:
        return []
    ordered = sorted(rows, key=lambda d: (str(d.get("scenario", "")),
                                          int(d.get("num_frames", 0))))
    labels = [f"{r.get('scenario', '')}/{int(r.get('num_frames', 0))}"
              for r in ordered]

    def _col(key, default=float("nan")):
        vals = []
        for r in ordered:
            try:
                vals.append(float(r.get(key, default)))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        return vals

    def _style(fig, ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        leg = ax.legend(fontsize=8, framealpha=0.9)
        if leg is not None:
            leg.get_frame().set_edgecolor("0.7")

    def _save(fig, name, paths):
        p = out / name
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(str(p))

    paths: List[str] = []
    x = np.arange(len(labels))
    w = 0.38
    ca = np.array([v if np.isfinite(v) else 0.0
                   for v in _col("num_cells_adaptive", 0.0)])
    cf = np.array([v if np.isfinite(v) else 0.0
                   for v in _col("num_cells_fixed", 0.0)])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, ca, w, label="Adaptive occupied cells")
    ax.bar(x + w / 2, cf, w, label="Fixed occupied cells")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    _style(fig, ax, "Occupied cells: adaptive vs fixed",
           "Scenario / frames", "Occupied cells (count)")
    _save(fig, "report_cells.png", paths)
    ta = np.array(_col("processing_time_adaptive"))
    tf = np.array(_col("processing_time_fixed"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(labels)), ta, "o-", label="Adaptive (measured s)")
    ax.plot(range(len(labels)), tf, "s--", label="Fixed (measured s)")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    _style(fig, ax, "Processing time: adaptive vs fixed (wall-clock)",
           "Scenario / frames", "Time (s)")
    _save(fig, "report_timing.png", paths)
    mae = np.array(_col("elevation_mae"))
    rmse = np.array(_col("elevation_rmse"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(labels)), mae, "o-", label="Elevation MAE (m)")
    ax.plot(range(len(labels)), rmse, "s--", label="Elevation RMSE (m)")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    _style(fig, ax, "Elevation error: adaptive vs fixed map",
           "Scenario / frames", "Error (m)")
    _save(fig, "report_error.png", paths)
    cva = np.array([v if np.isfinite(v) else 0.0
                    for v in _col("coverage_adaptive", 0.0)])
    cvf = np.array([v if np.isfinite(v) else 0.0
                    for v in _col("coverage_fixed", 0.0)])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, cva, w, label="Adaptive coverage")
    ax.bar(x + w / 2, cvf, w, label="Fixed coverage")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    top = float(np.max(np.concatenate([cva, cvf, [0.0]])))
    ax.set_ylim(0.0, max(1.05, top * 1.1))
    _style(fig, ax, "Map coverage: occupied / total finest cells",
           "Scenario / frames", "Coverage (fraction)")
    _save(fig, "report_coverage.png", paths)
    red = np.array([v if np.isfinite(v) else 0.0
                    for v in _col("cell_reduction", 0.0)])
    comp = np.array([v if np.isfinite(v) else 0.0
                     for v in _col("compression_ratio", 0.0)])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, red, w, label="Cell reduction (1 - adapt/fixed)")
    ax.bar(x + w / 2, comp, w, label="Compression ratio (adapt/fixed)")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    _style(fig, ax, "Cell reduction and compression ratio",
           "Scenario / frames", "Ratio (fraction)")
    _save(fig, "report_reduction.png", paths)
    agr = np.array([v if np.isfinite(v) else 0.0
                    for v in _col("occupied_cell_agreement", 0.0)])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, agr, 0.55, label="Occupied-cell agreement")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    _style(fig, ax, "Occupied-cell agreement (adaptive vs fixed)",
           "Scenario / frames", "Agreement (fraction)")
    _save(fig, "report_agreement.png", paths)
    from .dashboard_utils import region_level_counts
    if pipeline_result is not None:
        counts = region_level_counts(pipeline_result)
    else:
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ["LOW", "MEDIUM", "HIGH"]
    ax.bar(names, [int(counts.get(k, 0)) for k in names],
           label="Measured regions")
    ax.set_ylabel("Regions (count)", fontsize=10)
    _style(fig, ax, "Complexity / resolution distribution (measured run)",
           "Resolution level", "Regions (count)")
    _save(fig, "report_complexity.png", paths)
    scen_names = sorted({str(r.get("scenario", "")) for r in ordered})
    means = []
    for s in scen_names:
        vals = [float(r.get("cell_reduction", float("nan")))
                for r in ordered if str(r.get("scenario", "")) == s]
        vals = [v for v in vals if np.isfinite(v)]
        means.append(float(np.mean(vals)) if vals else 0.0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(scen_names, means, label="Mean cell reduction")
    ax.set_ylim(0.0, max(1.05, (max(means + [0.0])) * 1.1))
    _style(fig, ax, "Scenario comparison: mean cell reduction",
           "Scenario", "Mean cell reduction (fraction)")
    _save(fig, "report_scenarios.png", paths)
    return paths


def generate_report(rows=None, pipeline_result=None,
                    out_md=None, out_dir=None,
                    csv_path=None, seed=42):
    """Build report + figures from CSV rows (or given rows); return dict."""
    data = list(rows) if rows is not None else load_benchmark_rows(csv_path)
    md = write_report(data, out_md, seed=seed)
    figs = generate_report_figures(data, pipeline_result, out_dir)
    return {"rows": data, "report": md, "figures": figs,
            "summary": summarize_results(data)}
