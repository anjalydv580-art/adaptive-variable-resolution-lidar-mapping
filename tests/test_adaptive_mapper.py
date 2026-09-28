"""Phase 11 tests: adaptive 2.5D mapper."""
import numpy as np

from src.adaptive_mapper import AdaptiveMapper, build_adaptive_map
from src.lidar_simulator import LidarSimulator
from src.preprocessing import Preprocessor


def _proc(frame):
    proc, _ = Preprocessor().preprocess_frame(frame)
    return proc


def _small_cfg():
    return {"map": {"x_min": 0.0, "x_max": 20.0, "y_min": 0.0,
                    "y_max": 20.0, "fixed_resolution": 0.5},
            "feature_extraction": {"grid_cell_size": 1.0,
                                   "max_points_temporal": 2000,
                                   "region_size": 10.0},
            "ground_detection": {"grid_cell_size": 1.0},
            "resolution_controller": {
                "weights": {"point_density": 0.25, "object_density": 0.20,
                            "dynamic_activity": 0.25,
                            "elevation_variance": 0.15,
                            "temporal_change": 0.15},
                "normalization": {"point_density": 5.0, "object_density": 0.05,
                                  "dynamic_activity": 1.0,
                                  "elevation_variance": 2.0,
                                  "temporal_change": 1.0},
                "complexity_thresholds": {"low_max": 0.33,
                                          "medium_max": 0.66},
                "levels": {"low": {"cell_size": 2.0},
                           "medium": {"cell_size": 1.0},
                           "high": {"cell_size": 0.5}}}}



def test_per_region_levels_scores_and_voxels():
    rng = np.random.default_rng(0)
    flat = np.column_stack([rng.random(50) * 9 + 0.5,
                            rng.random(50) * 9 + 0.5,
                            np.full(50, 0.5)])
    tower = np.column_stack([rng.random(300) * 9 + 10.5,
                             rng.random(300) * 9 + 10.5,
                             rng.random(300) * 8 + 1.0])
    pts = np.vstack([flat, tower])
    mapper = AdaptiveMapper(config=_small_cfg())
    amap, stats, regions = mapper.build(pts)
    assert len(regions) == 4
    for r in regions:
        assert r.resolution_level in ("LOW", "MEDIUM", "HIGH")
        assert 0.0 <= r.complexity_score <= 1.0
        assert r.voxel_size in (2.0, 1.0, 0.5)
        assert r.num_points >= 0
    by_id = {r.region_id: r for r in regions}
    assert by_id[3].complexity_score >= by_id[0].complexity_score
    assert stats.num_input_points == pts.shape[0]
    assert stats.num_regions == 4
    occ = amap.occupancy
    assert occ.any()
    assert np.all(np.isfinite(amap.elevation[occ]))
    assert set(np.unique(amap.level_map[occ])).issubset({0, 1, 2})


def test_deterministic_repeated_builds():
    sim = LidarSimulator(seed=5)
    proc = _proc(sim.simulate_frame(1))
    mapper = AdaptiveMapper()
    a, sa, ra = mapper.build_frame(proc)
    b, sb, rb = mapper.build_frame(proc)
    assert sa.to_dict() == sb.to_dict()
    assert [r.to_dict() for r in ra] == [r.to_dict() for r in rb]
    assert np.array_equal(a.counts, b.counts)
    assert np.allclose(a.elevation, b.elevation, equal_nan=True)


def test_empty_input_safe():
    mapper = AdaptiveMapper(config=_small_cfg())
    amap, stats, regions = mapper.build(np.zeros((0, 3)))
    assert stats.num_input_points == 0
    assert stats.num_occupied_cells == 0
    assert amap.num_occupied_cells == 0
    assert len(regions) == 4
    for r in regions:
        assert r.num_points == 0
        assert r.resolution_level in ("LOW", "MEDIUM", "HIGH")


def test_real_preprocessed_frame_end_to_end():
    sim = LidarSimulator(seed=42)
    prev = _proc(sim.simulate_frame(0))
    curr = _proc(sim.simulate_frame(1))
    mapper = AdaptiveMapper()
    amap, stats, regions = mapper.build_frame(curr, prev_frame=prev)
    assert stats.num_input_points == int(np.asarray(curr.points).shape[0])
    assert stats.num_occupied_cells > 0
    assert stats.num_regions > 0
    d = amap.to_dict()
    assert d["num_input_points"] == stats.num_input_points
    c, _, _ = build_adaptive_map(np.asarray(curr.points, dtype=float),
                                 prev_points=np.asarray(prev.points,
                                                        dtype=float))
    assert np.array_equal(c.counts, amap.counts)
