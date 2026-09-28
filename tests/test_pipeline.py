"""Phase 14 tests: end-to-end adaptive mapping pipeline."""

import numpy as np

from src.complexity_classifier import CLASS_LABELS
from src.pipeline import AdaptiveMappingPipeline, run_pipeline


def test_end_to_end_pipeline_execution():
    res = run_pipeline(num_frames=2)
    assert res.num_frames == 2
    assert len(res.frames) == 2
    assert res.adaptive_map is not None
    assert res.fixed_map is not None
    assert isinstance(res.metrics, dict) and res.metrics
    assert isinstance(res.ml_predictions, list)
    assert len(res.ml_predictions) == 2
    d = res.to_dict()
    assert d["summary"]["num_frames"] == 2
    assert len(d["frames"]) == 2


def test_deterministic_output():
    a = run_pipeline(num_frames=2)
    b = run_pipeline(num_frames=2)
    assert [f.num_points_processed for f in a.frames] == [
        f.num_points_processed for f in b.frames]
    assert [f.complexity_score for f in a.frames] == [
        f.complexity_score for f in b.frames]
    assert a.ml_predictions == b.ml_predictions
    assert a.metrics["elevation_mae"] == b.metrics["elevation_mae"]
    assert np.array_equal(np.asarray(a.adaptive_map.elevation),
                          np.asarray(b.adaptive_map.elevation),
                          equal_nan=True)


def test_multi_frame_processing_for_dynamics():
    res = run_pipeline(num_frames=3)
    assert len(res.frames) == 3
    # Per-frame dynamics entries exist from the second frame on.
    assert isinstance(res.frames[0].dynamics, list)
    assert isinstance(res.frames[1].dynamics, list)
    total_dyn = sum(f.num_dynamic for f in res.frames)
    assert total_dyn >= 0
    # Temporal feature uses the previous frame (0.0 on the first frame).
    assert res.frames[0].features["temporal_change"] == 0.0


def test_adaptive_vs_fixed_results():
    res = run_pipeline(num_frames=2)
    assert res.adaptive_stats["num_occupied_cells"] > 0
    assert res.fixed_stats["num_output_cells"] > 0
    # Adaptive finest resolution is the min per-region voxel size from the
    # controller; it must be finite and positive (it may be coarser than
    # the fixed baseline when every region is LOW complexity).
    assert np.isfinite(res.adaptive_map.fine_resolution)
    assert res.adaptive_map.fine_resolution > 0
    assert len(res.region_decisions) > 0
    assert {r["resolution_level"] for r in res.region_decisions} <= {
        "LOW", "MEDIUM", "HIGH"}


def test_metrics_availability():
    res = run_pipeline(num_frames=2)
    for key in ("occupied_cell_agreement", "elevation_mae", "elevation_rmse",
                "elevation_bias", "coverage_adaptive", "coverage_fixed",
                "num_cells_adaptive", "num_cells_fixed", "compression_ratio",
                "cell_reduction"):
        assert key in res.metrics, key
        assert np.isfinite(res.metrics[key]) or np.isnan(res.metrics[key])
    assert 0.0 <= res.metrics["occupied_cell_agreement"] <= 1.0
    assert res.metrics["elevation_mae"] >= 0.0
    assert res.metrics["num_overlap_cells"] >= 0


def test_classifier_output_and_evaluation():
    res = run_pipeline(num_frames=2)
    assert set(res.ml_predictions).issubset(set(CLASS_LABELS))
    assert set(res.ml_rule_labels).issubset(set(CLASS_LABELS))
    ev = res.ml_evaluation
    assert ev["num_samples"] == 2
    assert 0.0 <= ev["accuracy"] <= 1.0
    assert 0.0 <= ev["precision_macro"] <= 1.0
    assert 0.0 <= ev["recall_macro"] <= 1.0
    assert 0.0 <= ev["f1_macro"] <= 1.0
    assert np.asarray(ev["confusion_matrix"]).shape == (3, 3)
    assert res.ml_model_source in ("loaded", "trained", "rule_fallback")


def test_empty_and_small_input_safety():
    empty = run_pipeline(num_frames=0)
    assert empty.num_frames == 0
    assert empty.frames == []
    assert empty.ml_predictions == []
    assert empty.ml_evaluation["num_samples"] == 0
    assert "elevation_mae" in empty.metrics

    single = run_pipeline(num_frames=1)
    assert len(single.frames) == 1
    assert single.frames[0].dynamics == []
    assert single.frames[0].num_dynamic == 0
    assert len(single.ml_predictions) == 1

    try:
        run_pipeline(num_frames=-1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
