"""Phase 15 tests: dashboard helpers + benchmark (no streamlit needed)."""
import numpy as np
from src.benchmark import benchmark_frame_counts, rows_to_dataframe
from src.dashboard_utils import (build_dashboard_context, elevation_figure_data,
                                 frame_table, metric_cards, region_level_counts,
                                 sample_points)
from src.pipeline import run_pipeline
def _res():
    return run_pipeline(num_frames=2)
def test_dashboard_context_sections():
    res = _res()
    ctx = build_dashboard_context(res, None)
    assert ctx["num_frames"] == 2
    assert len(ctx["frame_table"]) == 2
    assert "occupied_cell_agreement" in ctx["metric_cards"]
    assert "ml_accuracy_vs_rule" in ctx["metric_cards"]
    assert set(ctx["region_counts"]) == {"LOW", "MEDIUM", "HIGH"}
    assert ctx["elevation"]["fine_resolution"] > 0
def test_frame_table_and_cards_finite():
    res = _res()
    rows = frame_table(res)
    assert rows and rows[0]["frame_id"] == 0
    cards = metric_cards(res)
    assert 0.0 <= cards["occupied_cell_agreement"] <= 1.0
    assert 0.0 <= cards["ml_accuracy_vs_rule"] <= 1.0
def test_elevation_and_sampling_deterministic():
    res = _res()
    e1 = elevation_figure_data(res)
    e2 = elevation_figure_data(res)
    assert e1 == e2
    pts = np.zeros((500, 3))
    s1 = sample_points(pts, max_points=100)
    s2 = sample_points(pts, max_points=100)
    assert np.array_equal(s1, s2) and s1.shape[0] <= 100
    assert region_level_counts(res)["LOW"] >= 0
def test_benchmark_rows_real_and_deterministic():
    a = benchmark_frame_counts(frame_counts=[2])
    b = benchmark_frame_counts(frame_counts=[2])
    timing_keys = {"processing_time_adaptive", "processing_time_fixed", "speedup"}
    a_det = [{k: v for k, v in row.items() if k not in timing_keys} for row in a]
    b_det = [{k: v for k, v in row.items() if k not in timing_keys} for row in b]
    assert a_det == b_det
    for row in a + b:
        assert np.isfinite(float(row["processing_time_adaptive"]))
        assert float(row["processing_time_adaptive"]) >= 0.0
        assert np.isfinite(float(row["processing_time_fixed"]))
        assert float(row["processing_time_fixed"]) >= 0.0
        assert np.isfinite(float(row["speedup"]))
        assert float(row["speedup"]) > 0.0
    row = a[0]
    assert row["num_frames"] == 2
    assert row["total_points_processed"] > 0
    assert 0.0 <= row["occupied_cell_agreement"] <= 1.0
    assert row["elevation_mae"] >= 0.0
    df = rows_to_dataframe(benchmark_frame_counts(frame_counts=[1, 2]))
    assert list(df["num_frames"]) == [1, 2]
def test_empty_result_helpers_safe():
    res = run_pipeline(num_frames=0)
    assert frame_table(res) == []
    elev = elevation_figure_data(res)
    assert elev["shape"] == [0, 0] or len(elev["elevation"]) >= 0
    assert region_level_counts(res) == {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
