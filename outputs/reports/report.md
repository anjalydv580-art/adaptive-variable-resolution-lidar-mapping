# Adaptive Variable Resolution 2.5D LiDAR Mapping — Benchmark Report

> All results in this report were measured on SYNTHETIC data from the project LiDAR simulator (NumPy ray-casting of an urban scene with buildings, vehicles, and pedestrians). They describe simulated scenes only and must not be presented as real-world LiDAR performance; real-world validation has not been performed.

## Objective
Evaluate an adaptive variable-resolution 2.5D LiDAR mapping pipeline that spends fine cells only where scene complexity demands it, against a uniform fixed-resolution baseline, on deterministic simulated LiDAR sequences.

## Methodology
The Phase-14 pipeline simulates seeded LiDAR sweeps, preprocesses each cloud, detects the ground plane, clusters objects, tracks frame-to-frame dynamics, extracts per-frame complexity features, selects a LOW/MEDIUM/HIGH resolution per region, and mosaics an adaptive 2.5D elevation/occupancy map. The fixed baseline grids the same accumulated cloud at one resolution. Phase-12 metrics compare both maps on the adaptive finest grid.

## Adaptive resolution strategy
Per-region complexity score = weighted, min-max normalised sum of point density, object density, dynamic activity, elevation variance, and temporal change (weights/scales from `config/settings.yaml`). Scores map to LOW (2.0 m), MEDIUM (1.0 m), HIGH (0.5 m) cells; an ML classifier learns to predict the level from the same features.

## 2.5D mapping approach
Each region is gridded at its selected voxel size into mean / min / max elevation, counts, occupancy, and resolution provenance layers, then deterministically mosaicked onto the finest grid (floor indexing, order-independent reductions).

## Dynamic detection
Clusters are matched across consecutive frames by centroid distance; displacement above a threshold (with frame dt) marks objects dynamic and yields per-object speeds. The first frame has no predecessor, so its dynamics list is empty by construction.

## Adaptive vs fixed comparison
Agreement = fraction of union-occupied cells where both maps agree; elevation MAE/RMSE/bias measured where both maps are finite; compression = adaptive / fixed occupied cells; reduction = 1 - compression; coverage = occupied / total finest cells; processing times are measured wall-clock seconds and speedup = fixed / adaptive.

## Benchmark methodology
Rows measured with the real pipeline over 4 configuration(s): scenarios ['default', 'sparse'], frame counts [1, 2], deterministic seed 42. Each row rebuilds both maps end to end; only wall-clock timing varies between runs.

## Measured results
- Configurations measured: 4
- Scenarios: ['default', 'sparse']
- Frame counts: [1, 2]
- Occupied-cell agreement (mean): 0.0000 (range 0.0000–0.0000)
- Elevation MAE (mean, m): 0.0000
- Elevation RMSE (mean, m): 0.0000
- Coverage adaptive (mean): 0.0006
- Coverage fixed (mean): 0.0726
- Compression ratio (mean): 0.0086
- Cell reduction (mean): 0.9914
- Processing time adaptive (mean s, wall-clock): 0.0856
- Processing time fixed (mean s, wall-clock): 0.1444
- Speedup fixed/adaptive (mean): 1.6856
## Limitations
Synthetic simulator only: idealised geometry, sensor noise, and motion; no real-world LiDAR, weather, calibration error, or sensor failures. Timing is wall-clock and machine-dependent. Elevation error is map-to-map (adaptive vs fixed), not against surveyed ground truth.

## Future work
Validate on real LiDAR logs; surveyed ground-truth error; memory/compute scaling; controller thresholds and voxel sizes; more scenarios (weather, traffic, sensor faults).

## Reproducibility
CPU-only; deterministic seed 42 unless stated; parameters in `config/settings.yaml`; benchmark via `python scripts/benchmark.py`; report via `python scripts/report.py`.

## Per-configuration measurements

| scenario | frames | adapt cells | fixed cells | MAE (m) | RMSE (m) | agreement | reduction | adapt time (s) | fixed time (s) |

| default | 1 | 13 | 734 | 0.0000 | 0.0000 | 0.0000 | 0.9823 | 0.0852 | 0.1610 |
| default | 2 | 13 | 772 | 0.0000 | 0.0000 | 0.0000 | 0.9832 | 0.0957 | 0.1552 |
| sparse | 1 | 0 | 685 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0835 | 0.1418 |
| sparse | 2 | 0 | 713 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0780 | 0.1195 |
