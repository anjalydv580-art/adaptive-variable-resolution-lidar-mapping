# Adaptive Variable Resolution 2.5D LiDAR Mapping

CPU-only adaptive 2.5D LiDAR mapping prototype. A deterministic simulator
sweeps a synthetic urban scene (ground, buildings, moving vehicles and
pedestrians); the pipeline detects ground/objects/dynamics, scores regional
complexity, and builds a variable-resolution elevation/occupancy map that is
benchmarked against a fixed-resolution baseline with real measured metrics.

> **Simulation disclaimer:** all data are synthetic (NumPy ray-casting in
> `src/lidar_simulator.py`). Reported numbers are simulation/benchmark
> results only — **no real-world LiDAR performance is claimed**. Real-world
> validation would require real sensor datasets and field testing.

---

## Problem statement

Uniform high-resolution grids waste cells on flat/empty areas, while uniform
coarse grids blur buildings, vehicles, and pedestrians. The project answers:
how to spend fine cells only where measured scene complexity demands them,
while keeping a deterministic, reproducible CPU-only pipeline.

## Objective

Build and benchmark an adaptive variable-resolution 2.5D mapping pipeline:
simulate seeded LiDAR sweeps, detect ground/objects/dynamics, score regional
complexity, mosaic an adaptive elevation/occupancy map, and compare it
against a fixed-resolution baseline using measured metrics.

## Key features

- Deterministic 32-line LiDAR simulator with occlusion, noise, dropout
  (`src/lidar_simulator.py`)
- Preprocessing, RANSAC/grid ground detection, DBSCAN object detection
- Frame-to-frame dynamic detection with per-object speed
- Weighted regional complexity scoring → LOW (2.0 m) / MEDIUM (1.0 m) /
  HIGH (0.5 m) (`config/settings.yaml`)
- Adaptive mosaic mapper + fixed-resolution baseline
- Phase-12 measured comparison metrics (agreement, MAE/RMSE/bias,
  coverage, compression/reduction, wall-clock timing/speedup)
- RandomForest complexity classifier with rule-label evaluation
- Phase-14 pipeline, Phase-15 Streamlit dashboard, Phase-16 benchmarks,
  Phase-18 research reports/figures

## System architecture / pipeline

```text
simulate (seed 42) → preprocess → ground → objects → dynamics
→ features → LOW/MEDIUM/HIGH → adaptive map ─┐
                                             ├→ compare (Phase-12 metrics)
accumulated cloud → fixed baseline map ──────┘
```

`src/pipeline.py` (`AdaptiveMappingPipeline` / `run_pipeline`) orchestrates
all stages; `scripts/run_pipeline.py` writes `outputs/reports/pipeline.json`.

## Adaptive resolution methodology

Per-region score = weighted min-max normalised sum of `point_density`
(0.25), `object_density` (0.20), `dynamic_activity` (0.25),
`elevation_variance` (0.15), `temporal_change` (0.15); scales and
thresholds (`low_max` 0.33, `medium_max` 0.66) live in
`config/settings.yaml`. See `src/resolution_controller.py` and
`docs/METHODOLOGY.md`.

## 2.5D elevation mapping

Each region is gridded at its selected cell size into mean/min/max
elevation, counts, occupancy, and provenance layers, then mosaicked onto
the finest grid with floor indexing and order-independent reductions
(`src/map_2_5d.py`, `src/adaptive_mapper.py`).

## Dynamic object detection

Clusters are matched across consecutive frames by centroid distance
(`match_distance_threshold` 1.5 m); displacement above `min_displacement`
0.15 m over `frame_dt` 0.1 s marks objects dynamic with speed estimates
(`src/dynamic_detection.py`). Frame 0 has no predecessor, so no dynamics.

## ML complexity classification

RandomForest (`n_estimators` 200, `max_depth` 12, `random_state` 42)
trained on real pipeline features predicts LOW/MEDIUM/HIGH; evaluated
against controller rule labels (accuracy/precision/recall/F1, confusion
matrix). Model: `models/complexity_classifier.joblib`.

## Adaptive vs fixed comparison

Both maps are compared on the adaptive finest grid: occupied-cell
agreement, elevation MAE/RMSE/bias (finite-overlap cells only), occupied
cell counts, coverage, compression (`adapt/fixed`), reduction
(`1 − compression`), wall-clock build times, speedup (`fixed/adaptive`).

## Benchmark methodology

`src/benchmark.py` runs `run_pipeline` over frame counts × scenarios
(`default`, `sparse`, `dense_urban`; deterministic config overlays, seed
42). Each row rebuilds both maps end to end; only wall-clock timing varies
between runs. CSV via `save_benchmark_csv` → `outputs/benchmarks/`; charts
via `generate_benchmark_charts` → `outputs/figures/`.

## Measured results

From `outputs/reports/report.md` (4 configs: scenarios `default`,
`sparse`; frames 1–2; seed 42; machine-dependent timing):

| scenario | frames | adapt cells | fixed cells | MAE (m) | RMSE (m) |
|---|---|---|---|---|---|
| default | 1 | 13 | 734 | 0.0000 | 0.0000 |
| default | 2 | 13 | 772 | 0.0000 | 0.0000 |
| sparse | 1 | 0 | 685 | 0.0000 | 0.0000 |
| sparse | 2 | 0 | 713 | 0.0000 | 0.0000 |

Means: agreement 0.0000, coverage adapt 0.0006 / fixed 0.0726,
compression 0.0086, reduction 0.9914, speedup 1.6856 (wall-clock;
adaptive ~0.0856 s, fixed ~0.1444 s). MAE/RMSE 0.0 because no finest
cells overlap in these runs (sparse scenes; disjoint adaptive/fixed
footprints) — map-to-map difference, not ground-truth error. Rerun
`scripts/benchmark.py` to reproduce on this machine.

## Figures

Research PNGs in `outputs/figures/` (Phase 18, Matplotlib Agg):

- `report_cells.png` — occupied cells adaptive vs fixed
- `report_timing.png` — processing time (wall-clock s)
- `report_error.png` — elevation MAE/RMSE (m)
- `report_coverage.png` — map coverage
- `report_reduction.png` — cell reduction / compression
- `report_agreement.png` — occupied-cell agreement
- `report_complexity.png` — LOW/MEDIUM/HIGH distribution
- `report_scenarios.png` — scenario comparison

## Project Structure

```
adaptive-lidar-mapping/
├── README.md
├── requirements.txt
├── .gitignore
├── run.py
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── scenarios/
├── models/
├── outputs/
│   ├── figures/
│   ├── maps/
│   ├── reports/
│   └── benchmarks/
├── src/
│   ├── lidar_simulator.py       # LiDAR ray-casting + point cloud generation
│   ├── preprocessing.py         # voxel downsampling, outlier removal
│   ├── ground_detection.py      # RANSAC / grid based ground segmentation
│   ├── object_detection.py      # DBSCAN clustering + bounding boxes
│   ├── dynamic_detection.py     # frame-to-frame object tracking/matching
│   ├── feature_extraction.py    # per-cell complexity features
│   ├── resolution_controller.py # LOW/MEDIUM/HIGH resolution decision logic
│   ├── map_2_5d.py              # elevation & occupancy grid construction
│   ├── adaptive_mapper.py       # variable-resolution map builder
│   ├── fixed_mapper.py          # fixed-resolution baseline map builder
│   ├── metrics.py               # error/timing/coverage comparison metrics
│   ├── pipeline.py              # end-to-end orchestration
│   ├── reporting.py             # research reports + figures
│   ├── benchmark.py             # frame×scenario benchmark matrix
│   ├── dashboard_utils.py       # dashboard data helpers
│   ├── complexity_classifier.py # RandomForest complexity classifier
│   └── utils.py                 # shared helpers (config loading, paths...)
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── scripts/
│   ├── generate_data.py
│   ├── run_pipeline.py
│   ├── benchmark.py
│   └── report.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── METHODOLOGY.md
└── tests/
    ├── test_lidar.py
    ├── test_preprocessing.py
    ├── test_ground_detection.py
    ├── test_detection.py
    ├── test_dynamic_detection.py
    ├── test_features.py
    ├── test_resolution.py
    ├── test_fixed_mapper.py
    ├── test_mapping.py
    ├── test_adaptive_mapper.py
    ├── test_metrics.py
    ├── test_classifier.py
    ├── test_pipeline.py
    ├── test_dashboard.py
    ├── test_benchmark.py
    └── test_reports.py
```

## Technology stack

- Python 3.11+, CPU-only; NumPy, SciPy, scikit-learn, Pandas, PyYAML
- Matplotlib (Agg PNG figures), Streamlit + Plotly (dashboard)
- pytest; see `requirements.txt`

## Installation

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import numpy, scipy, sklearn, pandas, matplotlib, yaml; print('OK')"
```

## How to run the pipeline

```powershell
python scripts\generate_data.py --num-frames 3
python scripts\run_pipeline.py --num-frames 3 --out outputs\reports\pipeline.json
```

## How to run the dashboard

```powershell
streamlit run dashboard\app.py
```

## How to run benchmarks

```powershell
python scripts\benchmark.py --frames 1 2 --scenarios default sparse --out outputs\benchmarks\benchmark.csv --figures-dir outputs\figures
```

## How to generate the report

```powershell
python scripts\report.py --csv outputs\benchmarks\benchmark.csv --out outputs\reports\report.md --figures-dir outputs\figures
```

## How to run tests

```powershell
python -m pytest tests/test_reports.py -v -p no:cacheprovider --tb=short
python -m pytest tests -v -p no:cacheprovider --tb=short
```

## Reproducibility / deterministic execution

Seed 42 default (`config/settings.yaml` → `project.seed`); simulator,
features, controller, mappers, and metrics are deterministic — reruns give
identical non-timing results (timing/speedup are wall-clock and vary).
Config overrides: `run_pipeline(..., config=dict)` or scenario overlays in
`src/benchmark.py`. Empty inputs (`num_frames=0`) yield honest zero rows.

## Configuration

All tunable parameters live in `config/settings.yaml` (world, sensor,
preprocessing, detection, mapping, controller, ML, paths, dashboard).

## Limitations

- Synthetic data only: idealised geometry/noise/motion; no real LiDAR,
  weather, calibration error, or sensor failures.
- Elevation MAE/RMSE is map-to-map (adaptive vs fixed), not surveyed
  ground-truth error.
- Wall-clock timing is machine-dependent.
- Full test suite: 89 passed — no real-world performance implied.

## Future work

Real LiDAR logs, surveyed ground-truth error, memory/compute scaling,
controller/threshold studies, richer scenarios, sensor-fault robustness.

## Simulation disclaimer

Simulated scenes only (`src/lidar_simulator.py`); do not claim real-world
LiDAR performance. Real-world validation needs real sensor datasets and
field testing.

## License

Educational / academic (B.Tech CSE-AI) project prototype.
