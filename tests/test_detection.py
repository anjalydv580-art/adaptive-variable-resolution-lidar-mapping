"""Phase 5 tests: object detection on preprocessed + ground-separated clouds."""
import numpy as np

from src.ground_detection import GroundDetector
from src.lidar_simulator import LidarSimulator
from src.object_detection import ALLOWED_CATEGORIES, ObjectDetector
from src.preprocessing import Preprocessor


def _pipeline(seed=42):
    sim = LidarSimulator(seed=seed)
    frame = sim.simulate_frame(0)
    proc, _ = Preprocessor().preprocess_frame(frame)
    _, gres = GroundDetector().detect_frame(proc)
    return proc, gres


def test_config_defaults_match_settings_yaml():
    from src.utils import load_config
    cfg = load_config()
    det = ObjectDetector(config=cfg)
    assert det.eps == float(cfg["object_detection"]["clustering"]["eps"])
    assert det.min_samples == int(cfg["object_detection"]["clustering"]["min_samples"])
    assert det.min_points == int(cfg["object_detection"]["size_filters"]["min_points"])
    assert det.max_bbox_volume == float(cfg["object_detection"]["size_filters"]["max_bbox_volume"])


def test_synthetic_clusters_found():
    det = ObjectDetector()
    rng = np.random.default_rng(0)
    c1 = rng.normal(0.0, 0.1, size=(40, 3)) + np.array([0.0, 0.0, 1.0])
    c2 = rng.normal(0.0, 0.1, size=(40, 3)) + np.array([10.0, 0.0, 1.0])
    objs = det.detect(np.vstack([c1, c2]))
    assert len(objs) == 2
    for o in objs:
        assert o.point_count >= det.min_points
        assert o.bounding_box.shape == (2, 3)
        assert o.centroid.shape == (3,)
        assert o.category in ALLOWED_CATEGORIES
    centers = sorted([float(o.centroid[0]) for o in objs])
    assert centers[0] < 2.0 and centers[1] > 8.0


def test_pipeline_detects_objects_with_required_fields():
    proc, gres = _pipeline()
    det = ObjectDetector()
    objs = det.detect(gres.non_ground_points)
    assert len(objs) >= 3  # buildings + vehicles + pedestrians present
    for o in objs:
        assert isinstance(o.object_id, int)
        assert o.centroid.shape == (3,)
        assert o.bounding_box.shape == (2, 3)
        assert np.all(o.bbox_max >= o.bbox_min)
        assert o.point_count >= det.min_points
        assert o.bbox_volume <= det.max_bbox_volume
        assert o.category in ALLOWED_CATEGORIES
    cats = {o.category for o in objs}
    assert len(cats) >= 2  # size heuristic separates at least two types


def test_detect_frame_indices_map_to_frame():
    proc, gres = _pipeline()
    det = ObjectDetector()
    objs = det.detect_frame(proc, ground_mask=gres.ground_mask)
    for o in objs:
        idx = np.asarray(o.point_indices, dtype=int)
        assert idx.size == o.point_count
        assert bool(((idx >= 0) & (idx < proc.points.shape[0])).all())
        assert bool(gres.ground_mask[idx].any() == False)  # all non-ground
        assert np.allclose(proc.points[idx].mean(axis=0), o.centroid)


def test_empty_and_size_filter():
    det = ObjectDetector()
    assert det.detect(np.zeros((0, 3))) == []
    tiny = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]])  # < min_points
    assert det.detect(tiny) == []
