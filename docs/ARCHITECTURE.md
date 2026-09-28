# Architecture

Facts only — mirrors the implemented code, no new claims.

```text
LidarSimulator (seed 42, 32-line, occlusion/noise/dropout)
→ Preprocessor (voxel 0.25 m, outlier k=8) → GroundDetector (RANSAC)
→ ObjectDetector (DBSCAN eps 0.6) → DynamicDetector (match 1.5 m)
→ FeatureExtractor → ResolutionController (LOW 2.0 / MEDIUM 1.0 / HIGH 0.5 m)
→ AdaptiveMapper (mosaic) vs FixedMapper (0.5 m) → MapComparator
→ AdaptiveMappingPipeline → benchmark / reporting / dashboard
```

Key modules: `src/lidar_simulator.py`, `src/preprocessing.py`,
`src/ground_detection.py`, `src/object_detection.py`,
`src/dynamic_detection.py`, `src/feature_extraction.py`,
`src/resolution_controller.py`, `src/map_2_5d.py`,
`src/adaptive_mapper.py`, `src/fixed_mapper.py`, `src/metrics.py`,
`src/complexity_classifier.py`, `src/pipeline.py`, `src/benchmark.py`,
`src/reporting.py`, `src/dashboard_utils.py`. Scripts:
`scripts/generate_data.py`, `scripts/run_pipeline.py`,
`scripts/benchmark.py`, `scripts/report.py`; dashboard
`dashboard/app.py`; config `config/settings.yaml`.
