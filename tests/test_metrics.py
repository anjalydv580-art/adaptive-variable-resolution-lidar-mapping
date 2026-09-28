"""Phase 12 tests: adaptive vs fixed evaluation metrics."""
import time

import numpy as np

from src.adaptive_mapper import AdaptiveMapper
from src.fixed_mapper import FixedMapper
from src.lidar_simulator import LidarSimulator
from src.map_2_5d import ElevationMapper
from src.metrics import MapComparator, compare_maps
from src.preprocessing import Preprocessor


def _proc(frame):
    proc, _ = Preprocessor().preprocess_frame(frame)
    return proc


def test_identical_maps_give_perfect_scores():
    sim = LidarSimulator(seed=42)
    proc = _proc(sim.simulate_frame(0))
    pts = np.asarray(proc.points, dtype=float)
    fixed, _ = ElevationMapper().build(pts, resolution=1.0)
    m = MapComparator().compare(fixed, fixed)
    assert m.occupied_cell_agreement == 1.0
    assert m.elevation_mae == 0.0
    assert m.elevation_rmse == 0.0
    assert m.elevation_bias == 0.0
    assert m.num_cells_adaptive == m.num_cells_fixed > 0
    assert m.compression_ratio == 1.0
    assert m.cell_reduction == 0.0
    d = m.to_dict()
    assert d["occupied_cell_agreement"] == 1.0
    assert np.isnan(m.speedup)


def test_known_elevation_offset_metrics():
    cfg = {"map": {"x_min": 0.0, "x_max": 2.0, "y_min": 0.0,
                   "y_max": 2.0, "fixed_resolution": 1.0}}
    pts = np.array([[0.5, 0.5, 2.0], [1.5, 1.5, 4.0]])
    a, _ = ElevationMapper(config=cfg).build(pts, resolution=1.0)
    b, _ = ElevationMapper(config=cfg).build(pts + np.array([0.0, 0.0, 1.0]),
                                             resolution=1.0)
    m = compare_maps(a, b)
    assert m.num_overlap_cells == 2
    assert abs(m.elevation_mae - 1.0) < 1e-9
    assert abs(m.elevation_rmse - 1.0) < 1e-9
    assert abs(m.elevation_bias + 1.0) < 1e-9
    assert m.occupied_cell_agreement == 1.0


def test_empty_maps_safe_no_division_errors():
    a, _ = ElevationMapper().build(np.zeros((0, 3)))
    b, _ = ElevationMapper().build(np.zeros((0, 3)))
    m = MapComparator().compare(a, b)
    assert m.occupied_cell_agreement == 1.0
    assert m.elevation_mae == 0.0 and m.elevation_rmse == 0.0
    assert m.coverage_adaptive == 0.0 and m.coverage_fixed == 0.0
    assert np.isnan(m.compression_ratio) and np.isnan(m.cell_reduction)


def test_real_adaptive_vs_fixed_with_timing():
    sim = LidarSimulator(seed=42)
    proc = _proc(sim.simulate_frame(1))
    t0 = time.perf_counter()
    amap, _, _ = AdaptiveMapper().build_frame(proc)
    t_a = time.perf_counter() - t0
    t0 = time.perf_counter()
    fmap, _ = FixedMapper().build_frame(proc)
    t_f = time.perf_counter() - t0
    m = compare_maps(amap, fmap, processing_time_adaptive=t_a,
                     processing_time_fixed=t_f)
    assert m.num_cells_adaptive > 0 and m.num_cells_fixed > 0
    assert 0.0 <= m.occupied_cell_agreement <= 1.0
    assert m.elevation_mae >= 0.0 and m.elevation_rmse >= m.elevation_mae - 1e-9
    assert 0.0 <= m.coverage_adaptive <= 1.0
    assert 0.0 <= m.coverage_fixed <= 1.0
    assert np.isfinite(m.compression_ratio) and m.compression_ratio >= 0.0
    assert m.processing_time_adaptive == t_a
    assert m.processing_time_fixed == t_f
    assert np.isfinite(m.speedup)
    batch = MapComparator().compare_batch(
        [(amap, fmap)], processing_time_adaptive=t_a,
        processing_time_fixed=t_f)
    assert batch[0].to_dict() == m.to_dict()
