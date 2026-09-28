"""Phase 9 tests: fixed-resolution baseline mapper."""
import numpy as np

from src.fixed_mapper import FixedMapper, build_fixed_map
from src.lidar_simulator import LidarSimulator
from src.preprocessing import Preprocessor


def _proc(frame):
    proc, _ = Preprocessor().preprocess_frame(frame)
    return proc


def test_voxel_size_comes_from_settings_yaml():
    from src.utils import load_config
    cfg = load_config()
    mapper = FixedMapper(config=cfg)
    assert mapper.voxel_size == float(cfg["map"]["fixed_resolution"]) == 0.5


def test_single_fixed_resolution_for_whole_frame():
    sim = LidarSimulator(seed=42)
    proc = _proc(sim.simulate_frame(0))
    mapper = FixedMapper()
    fmap, stats = mapper.build_frame(proc)
    assert fmap.voxel_size == mapper.voxel_size == 0.5
    assert stats.voxel_size == 0.5
    sizes = {fmap.voxel_size}
    assert len(sizes) == 1
    assert fmap.elevation.shape == (fmap.shape[0], fmap.shape[1])


def test_stats_counts_and_height_preserved():
    sim = LidarSimulator(seed=42)
    proc = _proc(sim.simulate_frame(3))
    pts = np.asarray(proc.points, dtype=float)
    mapper = FixedMapper()
    fmap, stats = mapper.build(pts)
    assert stats.num_input_points == pts.shape[0] == fmap.num_input_points
    assert stats.num_output_cells == fmap.num_output_cells > 0
    assert int((fmap.counts > 0).sum()) == fmap.num_output_cells
    occ = fmap.occupied
    assert bool(occ.any())
    assert np.all(np.isfinite(fmap.elevation[occ]))
    assert np.all(fmap.elevation_min[occ] <= fmap.elevation[occ])
    assert np.all(fmap.elevation[occ] <= fmap.elevation_max[occ])
    assert np.all(fmap.elevation_var[occ] >= 0.0)
    assert np.all(np.isfinite(fmap.centroid[occ]))
    d = fmap.to_dict()
    assert d["voxel_size"] == 0.5
    assert d["num_input_points"] == pts.shape[0]
    assert d["num_output_cells"] == fmap.num_output_cells


def test_deterministic_repeated_builds():
    sim = LidarSimulator(seed=7)
    proc = _proc(sim.simulate_frame(1))
    mapper = FixedMapper()
    a, sa = mapper.build_frame(proc)
    b, sb = mapper.build_frame(proc)
    assert sa.to_dict() == sb.to_dict()
    assert np.array_equal(a.counts, b.counts)
    assert np.array_equal(a.occupied, b.occupied)
    assert np.allclose(a.elevation, b.elevation, equal_nan=True)


def test_empty_input_edge_case():
    mapper = FixedMapper()
    fmap, stats = mapper.build(np.zeros((0, 3)))
    assert stats.num_input_points == 0
    assert stats.num_output_cells == 0
    assert fmap.num_output_cells == 0


def test_functional_shortcut_matches():
    sim = LidarSimulator(seed=42)
    proc = _proc(sim.simulate_frame(0))
    a, _ = FixedMapper().build_frame(proc)
    b, _ = build_fixed_map(np.asarray(proc.points, dtype=float))
    assert np.array_equal(a.counts, b.counts)
