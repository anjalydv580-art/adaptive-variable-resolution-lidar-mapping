"""Phase 16 tests: benchmark system (real pipeline + Phase-12 metrics)."""
import numpy as np
import pandas as pd
import pytest
from src.benchmark import (AVAILABLE_SCENARIOS, REQUIRED_METRICS, TIMING_KEYS,
                           benchmark_configs, benchmark_frame_counts,
                           deterministic_row, generate_benchmark_charts,
                           rows_equal_deterministic, rows_to_dataframe,
                           run_benchmark, save_benchmark_csv)


def test_output_structure():
    rows = benchmark_configs(frame_counts=[1], scenarios=["default"])
    assert len(rows) == 1
    row = rows[0]
    assert row["scenario"] == "default"
    assert row["num_frames"] == 1
    assert row["seed"] == 42
    for key in REQUIRED_METRICS:
        assert key in row, key
    assert row["total_points_processed"] > 0
    assert set(AVAILABLE_SCENARIOS) >= {"default"}


def test_deterministic_non_timing_results():
    a = benchmark_configs(frame_counts=[1], scenarios=["default"])
    b = benchmark_configs(frame_counts=[1], scenarios=["default"])
    assert rows_equal_deterministic(a, b)
    assert deterministic_row(a[0]) == deterministic_row(b[0])
    assert len(a) == len(b) == 1


def test_required_metrics_finite():
    row = benchmark_configs(frame_counts=[2], scenarios=["default"])[0]
    assert 0.0 <= float(row["occupied_cell_agreement"]) <= 1.0
    assert float(row["elevation_mae"]) >= 0.0
    assert float(row["elevation_rmse"]) >= 0.0
    assert np.isfinite(float(row["elevation_bias"]))
    assert 0.0 <= float(row["coverage_adaptive"]) <= 1.0
    assert 0.0 <= float(row["coverage_fixed"]) <= 1.0
    assert float(row["num_cells_adaptive"]) > 0
    assert float(row["num_cells_fixed"]) > 0
    for key in TIMING_KEYS:
        assert np.isfinite(float(row[key])) and float(row[key]) >= 0.0
    assert np.isfinite(float(row["speedup"])) and float(row["speedup"]) > 0.0


def test_csv_generation(tmp_path):
    rows = benchmark_configs(frame_counts=[1, 2], scenarios=["default"])
    out = save_benchmark_csv(rows, str(tmp_path / "bench.csv"))
    df = pd.read_csv(out)
    assert len(df) == 2
    assert list(df["num_frames"]) == [1, 2]
    for key in REQUIRED_METRICS:
        assert key in df.columns, key
    assert (df["total_points_processed"] > 0).all()


def test_chart_generation(tmp_path):
    rows = benchmark_configs(frame_counts=[1, 2], scenarios=["default"])
    paths = generate_benchmark_charts(rows, str(tmp_path / "figs"))
    assert len(paths) == 4
    for p in paths:
        assert p.endswith(".png")
        with open(p, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG"
    assert generate_benchmark_charts([], str(tmp_path / "empty")) == []


def test_adaptive_fixed_comparison():
    row = run_benchmark(frame_counts=[1], scenarios=["default"])[0]
    ca = float(row["num_cells_adaptive"])
    cf = float(row["num_cells_fixed"])
    assert ca > 0 and cf > 0
    assert float(row["compression_ratio"]) == pytest.approx(ca / cf)
    assert float(row["cell_reduction"]) == pytest.approx(1.0 - ca / cf)
    assert 0.0 <= float(row["coverage_adaptive"]) <= 1.0
    assert 0.0 <= float(row["coverage_fixed"]) <= 1.0
    assert 0.0 <= float(row["occupied_cell_agreement"]) <= 1.0


def test_empty_small_configuration_safety(tmp_path):
    rows = benchmark_configs(frame_counts=[0], scenarios=["default"])
    assert len(rows) == 1
    row = rows[0]
    assert row["num_frames"] == 0
    assert row["total_points_processed"] == 0
    assert float(row["num_cells_adaptive"]) == 0.0
    assert float(row["occupied_cell_agreement"]) == 1.0
    small = benchmark_frame_counts(frame_counts=[1])
    assert small[0]["total_points_processed"] > 0
    df = rows_to_dataframe(benchmark_configs(frame_counts=[0, 1],
                                             scenarios=["default"]))
    assert list(df["num_frames"]) == [0, 1]
    out = save_benchmark_csv(benchmark_configs(frame_counts=[0],
                                               scenarios=["default"]),
                             str(tmp_path / "empty.csv"))
    assert pd.read_csv(out).shape[0] == 1
    assert generate_benchmark_charts(
        benchmark_configs(frame_counts=[0], scenarios=["default"]),
        str(tmp_path / "figs0"))
    with pytest.raises(ValueError):
        benchmark_configs(frame_counts=[-1], scenarios=["default"])
    with pytest.raises(ValueError):
        benchmark_configs(frame_counts=[1], scenarios=["nope"])
