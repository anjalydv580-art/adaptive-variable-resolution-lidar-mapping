"""Phase 3 tests: preprocessing on real LiDAR simulator output."""
import numpy as np

from src.lidar_simulator import LidarSimulator
from src.preprocessing import Preprocessor


def _frame():
    return LidarSimulator(seed=42).simulate_frame(0)


def test_config_defaults_match_settings_yaml():
    from src.utils import load_config
    cfg = load_config()
    prep = Preprocessor(config=cfg)
    assert prep.voxel_size == float(cfg["preprocessing"]["voxel_size"])
    assert prep.k_neighbors == int(cfg["preprocessing"]["statistical_outlier"]["k_neighbors"])
    assert prep.std_ratio == float(cfg["preprocessing"]["statistical_outlier"]["std_ratio"])
    assert prep.max_range == float(cfg["simulation"]["sensor"]["max_range"])


def test_invalid_point_removal():
    prep = Preprocessor()
    pts = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 2.0],
                    [1.0, np.inf, 2.0], [5.0, 5.0, 1.0]])
    mask = Preprocessor.remove_invalid_points(pts)
    assert mask.tolist() == [True, False, False, True]


def test_range_filtering():
    prep = Preprocessor()
    origin = np.array([0.0, 0.0, 1.8])
    pts = np.array([[0.0, 0.0, 1.8],  # range 0 -> too close
                    [10.0, 0.0, 1.8],  # in range
                    [200.0, 0.0, 1.8]])  # beyond max_range
    mask = prep.range_filter_mask(pts, sensor_origin=origin)
    assert mask.tolist() == [False, True, False]


def test_noise_filtering_removes_sparse_outlier():
    prep = Preprocessor()
    rng = np.random.default_rng(0)
    cluster = rng.normal(0.0, 0.05, size=(60, 3)) + np.array([20.0, 0.0, 1.0])
    outlier = np.array([[20.0, 30.0, 1.0]])
    pts = np.vstack([cluster, outlier])
    mask = prep.statistical_outlier_mask(pts)
    assert mask.shape == (61,)
    assert bool(mask[-1]) is False  # far sparse point rejected
    assert int(mask.sum()) >= 55


def test_preprocess_frame_keeps_arrays_aligned():
    prep = Preprocessor()
    frame = _frame()
    new_frame, stats = prep.preprocess_frame(frame)
    n = new_frame.points.shape[0]
    assert n > 100
    assert new_frame.labels.shape == (n,)
    assert new_frame.object_ids.shape == (n,)
    assert new_frame.intensities.shape == (n,)
    assert np.all(np.isfinite(new_frame.points))
    assert stats.n_input == frame.points.shape[0]
    assert stats.n_output == n
    assert n <= frame.points.shape[0]


def test_preprocess_frame_without_downsampling_keeps_more_points():
    prep = Preprocessor()
    frame = _frame()
    full, _ = prep.preprocess_frame(frame, apply_voxel=False, apply_outlier=False)
    lite, _ = prep.preprocess_frame(frame, apply_voxel=True, apply_outlier=True)
    assert full.points.shape[0] >= lite.points.shape[0]
