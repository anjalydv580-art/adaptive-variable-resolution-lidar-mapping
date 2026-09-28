"""Phase 10 tests: 2.5D elevation/occupancy map."""
import numpy as np

from src.lidar_simulator import LidarSimulator
from src.map_2_5d import ElevationMapper, build_elevation_map
from src.preprocessing import Preprocessor


def _proc(frame):
    proc, _ = Preprocessor().preprocess_frame(frame)
    return proc


def test_resolution_comes_from_settings_yaml():
    from src.utils import load_config
    cfg = load_config()
    mapper = ElevationMapper(config=cfg)
    assert mapper.resolution == float(cfg["map"]["fixed_resolution"]) == 0.5


def test_cell_statistics_on_handcrafted_cloud():
    pts = np.array([[0.1, 0.1, 1.0],
                    [0.2, 0.1, 3.0],
                    [1.1, 0.1, 5.0]])
    cfg = {"map": {"x_min": 0.0, "x_max": 2.0, "y_min": 0.0,
                   "y_max": 2.0, "fixed_resolution": 1.0}}
    emap, stats = ElevationMapper(config=cfg).build(pts)
    assert emap.shape == (2, 2)
    assert emap.counts[0, 0] == 2
    assert emap.counts[0, 1] == 1
    assert emap.counts[1, 0] == 0 and emap.counts[1, 1] == 0
    assert bool(emap.occupancy[0, 0]) and bool(emap.occupancy[0, 1])
    assert not bool(emap.occupancy[1, 0])
    assert emap.elevation_min[0, 0] == 1.0
    assert emap.elevation_max[0, 0] == 3.0
    assert emap.elevation[0, 0] == 2.0
    assert emap.elevation[0, 1] == 5.0
    assert np.isnan(emap.elevation[1, 0])
    assert stats.num_input_points == 3
    assert stats.num_occupied_cells == 2


def test_empty_cells_safe_and_shapes_consistent():
    mapper = ElevationMapper()
    emap, stats = mapper.build(np.zeros((0, 3)))
    ny, nx = emap.shape
    assert emap.elevation.shape == (ny, nx)
    assert emap.counts.shape == (ny, nx)
    assert emap.occupancy.shape == (ny, nx)
    assert emap.density.shape == (ny, nx)
    assert emap.centroid.shape == (ny, nx, 3)
    assert stats.num_occupied_cells == 0
    assert stats.num_input_points == 0


def test_configurable_resolution_changes_grid():
    cfg = {"map": {"x_min": -10.0, "x_max": 10.0, "y_min": -10.0,
                   "y_max": 10.0, "fixed_resolution": 0.5}}
    mapper = ElevationMapper(config=cfg)
    assert mapper.grid_shape(1.0) == (20, 20)
    assert mapper.grid_shape(2.0) == (10, 10)
    pts = np.array([[0.0, 0.0, 1.0]])
    fine, _ = mapper.build(pts, resolution=1.0)
    coarse, _ = mapper.build(pts, resolution=2.0)
    assert fine.shape == (20, 20) and coarse.shape == (10, 10)


def test_deterministic_repeated_builds():
    sim = LidarSimulator(seed=11)
    proc = _proc(sim.simulate_frame(2))
    mapper = ElevationMapper()
    a, sa = mapper.build_frame(proc)
    b, sb = mapper.build_frame(proc)
    assert sa.to_dict() == sb.to_dict()
    assert np.array_equal(a.counts, b.counts)
    assert np.allclose(a.elevation, b.elevation, equal_nan=True)


def test_real_preprocessed_frame_statistics():
    sim = LidarSimulator(seed=42)
    proc = _proc(sim.simulate_frame(0))
    pts = np.asarray(proc.points, dtype=float)
    assert pts.shape[0] > 0
    emap, stats = ElevationMapper().build_frame(proc)
    assert stats.num_input_points == pts.shape[0]
    assert stats.num_occupied_cells > 0
    occ = emap.occupancy
    assert np.all(np.isfinite(emap.elevation[occ]))
    assert np.all(emap.elevation_min[occ] <= emap.elevation[occ])
    assert np.all(emap.elevation[occ] <= emap.elevation_max[occ])
    assert np.all(np.isfinite(emap.centroid[occ]))
    assert np.all(emap.density[occ] > 0.0)
    c, _ = build_elevation_map(pts)
    assert np.array_equal(c.counts, emap.counts)
