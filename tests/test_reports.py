"""Phase 18 tests: research reports and figures from real outputs only."""
import numpy as np
from src.benchmark import benchmark_configs
from src.pipeline import run_pipeline
from src.reporting import (FIGURE_SPECS, REQUIRED_SECTIONS,
                           SIMULATION_DISCLAIMER, build_report_markdown,
                           generate_report, generate_report_figures,
                           load_benchmark_rows, summarize_results,
                           write_report)

ROWS = [
    {"scenario": "default", "num_frames": 1, "seed": 42,
     "total_points_processed": 8000, "total_objects": 10,
     "total_dynamic": 2, "ml_model_source": "trained",
     "ml_accuracy_vs_rule": 0.8, "occupied_cell_agreement": 0.9,
     "elevation_mae": 0.05, "elevation_rmse": 0.08, "elevation_bias": 0.01,
     "coverage_adaptive": 0.02, "coverage_fixed": 0.05,
     "num_cells_adaptive": 20, "num_cells_fixed": 100,
     "compression_ratio": 0.2, "cell_reduction": 0.8,
     "processing_time_adaptive": 0.10, "processing_time_fixed": 0.20,
     "speedup": 2.0},
    {"scenario": "default", "num_frames": 2, "seed": 42,
     "total_points_processed": 16000, "total_objects": 20,
     "total_dynamic": 4, "ml_model_source": "trained",
     "ml_accuracy_vs_rule": 0.7, "occupied_cell_agreement": 0.85,
     "elevation_mae": 0.06, "elevation_rmse": 0.09, "elevation_bias": 0.02,
     "coverage_adaptive": 0.03, "coverage_fixed": 0.06,
     "num_cells_adaptive": 30, "num_cells_fixed": 120,
     "compression_ratio": 0.25, "cell_reduction": 0.75,
     "processing_time_adaptive": 0.12, "processing_time_fixed": 0.22,
     "speedup": 1.83},
]


def test_report_files_created(tmp_path):
    res = generate_report(rows=ROWS, pipeline_result=None,
                          out_md=str(tmp_path / "report.md"),
                          out_dir=str(tmp_path / "figs"), seed=42)
    text = open(res["report"], encoding="utf-8").read()
    for section in REQUIRED_SECTIONS:
        assert section in text
    assert SIMULATION_DISCLAIMER.split()[0] in text
    assert "real-world" in text.lower() or "real-world" in text
    for key in ("Occupied-cell agreement", "Elevation MAE",
                "Cell reduction"):
        assert key in text or key.lower() in text.lower()


def test_figures_created_and_valid_png(tmp_path):
    paths = generate_report_figures(ROWS, None, str(tmp_path / "figs"))
    assert [p.split("/")[-1].split("\\")[-1] for p in paths] == [
        s for s, _ in FIGURE_SPECS]
    for p in paths:
        assert p.endswith(".png")
        with open(p, "rb") as f:
            magic = f.read(8)
        assert magic[:4] == b"\x89PNG"
        assert len(magic) == 8


def test_outputs_deterministic(tmp_path):
    a = generate_report_figures(ROWS, None, str(tmp_path / "a"))
    b = generate_report_figures(ROWS, None, str(tmp_path / "b"))
    assert len(a) == len(b) == len(FIGURE_SPECS)
    for pa, pb in zip(a, b):
        assert open(pa, "rb").read() == open(pb, "rb").read()
    ra = write_report(ROWS, str(tmp_path / "ra.md"), seed=42)
    rb = write_report(ROWS, str(tmp_path / "rb.md"), seed=42)
    assert open(ra, encoding="utf-8").read() == open(
        rb, encoding="utf-8").read()
    s1 = summarize_results(ROWS)
    s2 = summarize_results(list(reversed(ROWS)))
    assert s1["num_cells_adaptive_mean"] == s2["num_cells_adaptive_mean"]


def test_empty_missing_data_safe(tmp_path):
    assert generate_report_figures([], None, str(tmp_path / "e")) == []
    assert load_benchmark_rows(str(tmp_path / "nope.csv")) == []
    empty_md = write_report([], str(tmp_path / "empty.md"), seed=42)
    text = open(empty_md, encoding="utf-8").read()
    assert "Measured results" in text
    md = build_report_markdown(summarize_results([]), [], seed=42)
    assert "Limitations" in md and "Future work" in md


def test_real_pipeline_and_benchmark_integration(tmp_path):
    rows = benchmark_configs(frame_counts=[1], scenarios=["default"])
    assert rows[0]["total_points_processed"] > 0
    pipe = run_pipeline(num_frames=1)
    paths = generate_report_figures(rows, pipe, str(tmp_path / "real"))
    assert len(paths) == len(FIGURE_SPECS)
    for p in paths:
        with open(p, "rb") as f:
            assert f.read(4) == b"\x89PNG"
    md_path = write_report(rows, str(tmp_path / "real.md"), seed=42)
    text = open(md_path, encoding="utf-8").read()
    assert str(rows[0]["num_frames"]) in text
    assert np.isfinite(summarize_results(rows)["elevation_mae_mean"])
