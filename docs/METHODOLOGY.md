# Methodology

Facts only — parameters from `config/settings.yaml`, no new claims.

- Complexity score: weighted min-max normalised sum of point_density
  (w 0.25, scale 5.0), object_density (0.20, 0.05), dynamic_activity
  (0.25, 1.0), elevation_variance (0.15, 2.0), temporal_change (0.15,
  1.0); thresholds low_max 0.33, medium_max 0.66.
- 2.5D mapping: per-region mean/min/max elevation, counts, occupancy,
  provenance; floor indexing; deterministic mosaic.
- Comparison (adaptive finest grid): agreement, MAE/RMSE/bias on
  finite-overlap cells, counts, coverage, compression, reduction,
  wall-clock times, speedup.
- Benchmark: frame counts × scenarios (`default`, `sparse`,
  `dense_urban`), seed 42; CSV `outputs/benchmarks/`, PNG
  `outputs/figures/`; report `outputs/reports/report.md`.
- Simulation only: synthetic scenes; map-to-map error, not surveyed
  truth; timing machine-dependent.
