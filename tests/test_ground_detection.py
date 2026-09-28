"""Phase 4 tests: ground detection on preprocessed simulator output."""
import numpy as np

from src.ground_detection import GroundDetector
from src.lidar_simulator import LABEL_GROUND, LidarSimulator
from src.preprocessing import Preprocessor


def _preprocessed_frame(seed=42):
    sim = LidarSimulator(seed=seed)
    frame = sim.simulate_frame(0)
    prep = Preprocessor()
    new_frame, _ = prep.preprocess_frame(frame)
    return frame, new_frame


def test_config_defaults_match_settings_yaml():
    from src.utils import load_config
    cfg = load_config()
    det = GroundDetector(config=cfg)
    assert det.method == str(cfg["ground_detection"]["method"])
    assert det.distance_threshold == float(
        cfg["ground_detection"]["ransac"]["distance_threshold"])
    assert det.max_iterations == int(
        cfg["ground_detection"]["ransac"]["max_iterations"])
    assert det.grid_cell_size == float(cfg["ground_detection"]["grid_cell_size"])


def test_ransac_on_synthetic_plane():
    det = GroundDetector()
    rng = np.random.default_rng(0)
    xs = rng.uniform(-10, 10, 300)
    ys = rng.uniform(-10, 10, 300)
    ground = np.stack([xs, ys, rng.normal(0.0, 0.02, 300)], axis=1)
    walls = np.stack([rng.uniform(-10, 10, 60), rng.uniform(-10, 10, 60),
                      rng.uniform(1.0, 5.0, 60)], axis=1)
    pts = np.vstack([ground, walls])
    res = det.detect(pts, method="ransac")
    assert res.ground_mask.shape == (360,)
    assert res.ground_points.shape[1] == 3
    assert res.non_ground_points.shape[1] == 3
    # Most true ground recovered, elevated points rejected.
    assert res.n_ground >= 240
    assert res.n_non_ground >= 40
    assert res.plane is not None and res.plane.shape == (4,)


def test_grid_method_height_based():
    det = GroundDetector()
    rng = np.random.default_rng(1)
    xs = rng.uniform(0, 4, 200)
    ys = rng.uniform(0, 4, 200)
    ground = np.stack([xs, ys, rng.normal(0.0, 0.02, 200)], axis=1)
    boxes = np.stack([rng.uniform(0, 4, 60), rng.uniform(0, 4, 60),
                      rng.uniform(1.0, 3.0, 60)], axis=1)
    res = det.detect(np.vstack([ground, boxes]), method="grid")
    assert res.n_ground >= 150
    assert res.n_non_ground >= 40


def test_preprocessed_frame_ground_recall():
    _, proc = _preprocessed_frame()
    det = GroundDetector()
    res = det.detect(proc.points, method=det.method)
    total = res.n_ground + res.n_non_ground
    assert total == proc.points.shape[0]
    assert res.ground_points.shape[0] == res.n_ground
    assert res.non_ground_points.shape[0] == res.n_non_ground
    # Simulator labels give ground truth: most true ground must be flagged.
    true_ground = (np.asarray(proc.labels) == LABEL_GROUND)
    assert true_ground.sum() > 0
    recall = float((res.ground_mask & true_ground).sum()) / float(true_ground.sum())
    assert recall >= 0.80
    # Detected ground must be low elevation on average.
    assert float(res.ground_points[:, 2].mean()) < float(
        res.non_ground_points[:, 2].mean())


def test_empty_and_small_inputs():
    det = GroundDetector()
    res = det.detect(np.zeros((0, 3)))
    assert res.n_ground == 0 and res.n_non_ground == 0
    res = det.detect(np.array([[1.0, 2.0, 0.0]]))
    assert res.ground_mask.shape == (1,)
