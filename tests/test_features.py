"""Phase 7 tests: deterministic LiDAR feature extraction."""
import numpy as np

from src.dynamic_detection import DynamicDetector
from src.feature_extraction import (
    FEATURE_NAMES,
    FeatureExtractor,
    extract_frame_features,
    extract_grid_features,
)
from src.ground_detection import GroundDetector
from src.lidar_simulator import LidarSimulator
from src.object_detection import DetectedObject, ObjectDetector
from src.preprocessing import Preprocessor


def _mk(oid, x, category="vehicle"):
    c = np.array([x, 0.0, 1.0])
    return DetectedObject(oid, c, c - 0.5, c + 0.5, 20, category, np.arange(5))


def _mk_dyn(x0, x1, dynamic=True):
    from src.dynamic_detection import DynamicObject
    p = np.array([x0, 0.0, 1.0])
    c = np.array([x1, 0.0, 1.0])
    disp = float(abs(x1 - x0))
    return DynamicObject(0, 0, p, c, disp, disp / 0.1, dynamic,
                         "dynamic" if dynamic else "static", "vehicle")


def _pipe(frame):
    proc, _ = Preprocessor().preprocess_frame(frame)
    _, gres = GroundDetector().detect_frame(proc)
    objs = ObjectDetector().detect_frame(proc, ground_mask=gres.ground_mask)
    return proc, objs


def test_required_feature_names_present():
    for name in ("point_density", "object_density", "dynamic_activity",
                 "elevation_variance", "temporal_change", "occupancy"):
        assert name in FEATURE_NAMES


def test_config_defaults_match_settings_yaml():
    from src.utils import load_config
    cfg = load_config()
    fe = FeatureExtractor(config=cfg)
    fcfg = cfg["feature_extraction"]
    assert fe.grid_cell_size == float(fcfg["grid_cell_size"])
    assert fe.max_points_temporal == int(fcfg["max_points_temporal"])


def test_frame_features_formulas_and_ranges():
    xs = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    ys = np.array([0.0, 0.0, 0.0, 3.0, 3.0, 3.0])
    zs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    pts = np.stack([xs, ys, zs], axis=1)
    objs = [_mk(0, 0.5), _mk(1, 1.5)]
    dyns = [_mk_dyn(0.5, 1.0, True)]
    fe = FeatureExtractor()
    f = fe.extract_frame_features(pts, objects=objs, dynamics=dyns,
                                  prev_points=pts + 10.0, grid_cell_size=1.0)
    area = 2.0 * 3.0
    assert f.num_points == 6 and f.num_objects == 2 and f.num_dynamic == 1
    assert abs(f.area - area) < 1e-9
    assert abs(f.point_density - 6.0 / area) < 1e-9
    assert abs(f.object_density - 2.0 / area) < 1e-9
    assert abs(f.dynamic_activity - 0.5) < 1e-9
    assert abs(f.elevation_variance - float(zs.var())) < 1e-9
    assert f.temporal_change > 0.0 and np.isfinite(f.temporal_change)
    assert 0.0 <= f.occupancy <= 1.0
    d = f.to_dict()
    for name in FEATURE_NAMES:
        assert name in d and np.isfinite(d[name])
    assert np.allclose(f.to_vector(), extract_frame_features(
        pts, objects=objs, dynamics=dyns, prev_points=pts + 10.0,
        grid_cell_size=1.0).to_vector())


def test_static_pair_low_temporal_change_and_activity():
    fe = FeatureExtractor()
    rng = np.random.default_rng(0)
    base = rng.random((200, 3)) * np.array([10.0, 10.0, 2.0])
    objs = [_mk(0, 2.0), _mk(1, 7.0)]
    f_same = fe.extract_frame_features(base, objects=objs, dynamics=[],
                                       prev_points=base.copy())
    assert f_same.temporal_change == 0.0
    assert f_same.dynamic_activity == 0.0
    shifted = base + np.array([0.05, 0.0, 0.0])
    f_small = fe.extract_frame_features(shifted, objects=objs, dynamics=[],
                                        prev_points=base)
    assert 0.0 < f_small.temporal_change < 0.2


def test_moved_objects_raise_temporal_and_dynamic():
    fe = FeatureExtractor()
    prev = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [5.0, 5.0, 0.5]])
    curr = np.array([[2.0, 0.0, 1.0], [3.0, 0.0, 1.0], [5.0, 5.0, 0.5]])
    dyns = [_mk_dyn(0.0, 2.0, True)]
    f = fe.extract_frame_features(curr, objects=[_mk(0, 2.0)],
                                  dynamics=dyns, prev_points=prev,
                                  grid_cell_size=1.0)
    assert f.dynamic_activity == 1.0
    assert f.temporal_change > 0.5


def test_empty_inputs_all_zero():
    fe = FeatureExtractor()
    f = fe.extract_frame_features(np.zeros((0, 3)))
    for name in FEATURE_NAMES:
        assert float(getattr(f, name)) == 0.0
    g = fe.extract_grid_features(np.zeros((0, 3)))
    assert g["features"].shape == (0, 0, len(FEATURE_NAMES))


def test_determinism_repeated_calls():
    sim = LidarSimulator(seed=7)
    proc, objs = _pipe(sim.simulate_frame(0))
    fe = FeatureExtractor()
    a = fe.extract_frame_features(np.asarray(proc.points), objects=objs)
    b = fe.extract_frame_features(np.asarray(proc.points), objects=objs)
    assert np.allclose(a.to_vector(), b.to_vector())


def test_real_consecutive_frames_pipeline():
    sim = LidarSimulator(seed=42)
    prev_proc, prev_objs = _pipe(sim.simulate_frame(0))
    curr_proc, curr_objs = _pipe(sim.simulate_frame(1))
    assert len(prev_objs) > 0 and len(curr_objs) > 0
    tracks = DynamicDetector().match_and_classify(prev_objs, curr_objs)
    fe = FeatureExtractor()
    f = fe.extract_from_frames(curr_proc, objects=curr_objs, dynamics=tracks,
                               prev_frame=prev_proc)
    assert f.num_points == int(np.asarray(curr_proc.points).shape[0])
    assert f.num_objects == len(curr_objs)
    assert f.num_dynamic == sum(1 for t in tracks if t.is_dynamic)
    assert f.point_density >= 0.0 and f.temporal_change >= 0.0
    assert 0.0 <= f.occupancy <= 1.0
    assert 0.0 <= f.dynamic_activity <= 1.0 + 1e-9
    g = extract_grid_features(np.asarray(curr_proc.points), objects=curr_objs,
                              dynamics=tracks,
                              prev_points=np.asarray(prev_proc.points))
    assert g["features"].ndim == 3
    assert g["features"].shape[2] == len(FEATURE_NAMES)
    ny, nx = g["shape"]
    assert g["features"].shape[:2] == (ny, nx)
    assert np.all(g["features"][:, :, 0] >= 0.0)
