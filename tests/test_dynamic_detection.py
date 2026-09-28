"""Phase 6 tests: dynamic detection across consecutive real frames."""
import numpy as np

from src.dynamic_detection import DynamicDetector
from src.ground_detection import GroundDetector
from src.lidar_simulator import LidarSimulator
from src.object_detection import DetectedObject, ObjectDetector
from src.preprocessing import Preprocessor


def _detect(frame):
    proc, _ = Preprocessor().preprocess_frame(frame)
    _, gres = GroundDetector().detect_frame(proc)
    return ObjectDetector().detect_frame(proc, ground_mask=gres.ground_mask)


def _mk(oid, x, category="vehicle"):
    c = np.array([x, 0.0, 1.0])
    return DetectedObject(oid, c, c - 0.5, c + 0.5, 20, category,
                          np.arange(5))


def test_config_defaults_match_settings_yaml():
    from src.utils import load_config
    cfg = load_config()
    det = DynamicDetector(config=cfg)
    dd = cfg["dynamic_detection"]
    assert det.match_distance_threshold == float(dd["match_distance_threshold"])
    assert det.min_displacement == float(dd["min_displacement"])
    assert det.frame_dt == float(dd["frame_dt"])


def test_static_pair_stays_static():
    det = DynamicDetector()
    prev = [_mk(0, 0.0), _mk(1, 5.0)]
    curr = [_mk(0, 0.02), _mk(1, 5.01)]  # below min_displacement (0.15)
    tracks = det.match_and_classify(prev, curr)
    assert len(tracks) == 2
    for t in tracks:
        assert t.prev_centroid.shape == (3,)
        assert t.curr_centroid.shape == (3,)
        assert t.displacement < det.min_displacement
        assert t.is_dynamic is False
        assert t.status == "static"
        assert t.speed == t.displacement / det.frame_dt


def test_moved_object_is_dynamic_with_speed():
    det = DynamicDetector()
    prev = [_mk(0, 0.0)]
    curr = [_mk(0, 1.0)]  # 1 m shift
    (t,) = [x for x in det.match_and_classify(prev, curr) if x.object_id == 0]
    assert t.object_id == 0 and t.prev_object_id == 0
    assert abs(t.displacement - 1.0) < 1e-9
    assert abs(t.speed - 1.0 / det.frame_dt) < 1e-9
    assert t.is_dynamic is True and t.status == "dynamic"
    assert np.allclose(t.prev_centroid, [0.0, 0.0, 1.0])
    assert np.allclose(t.curr_centroid, [1.0, 0.0, 1.0])


def test_unmatched_become_new_and_lost():
    det = DynamicDetector()
    tracks = det.match_and_classify([_mk(0, 0.0)], [_mk(7, 50.0)])
    by_status = {t.status for t in tracks}
    assert by_status == {"new", "lost"}
    assert all(t.is_dynamic is False for t in tracks)


def test_real_consecutive_frames_no_random_labels():
    sim = LidarSimulator(seed=42)
    prev_objs = _detect(sim.simulate_frame(0))
    curr_objs = _detect(sim.simulate_frame(1))
    assert len(prev_objs) > 0 and len(curr_objs) > 0
    det = DynamicDetector()
    tracks = det.match_and_classify(prev_objs, curr_objs)
    matched = [t for t in tracks if t.status in ("static", "dynamic")]
    assert len(matched) > 0
    for t in tracks:
        assert isinstance(t.object_id, int)
        assert t.displacement >= 0.0 and t.speed >= 0.0
        # Status/displacement consistency: no random dynamic flags.
        assert t.is_dynamic == (t.displacement >= det.min_displacement
                                if t.status in ("static", "dynamic") else False)
        assert abs(t.speed - t.displacement / det.frame_dt) < 1e-9


def test_empty_inputs():
    det = DynamicDetector()
    assert det.match_and_classify([], []) == []
    tracks = det.match_and_classify([], [_mk(3, 1.0)])
    assert len(tracks) == 1 and tracks[0].status == "new"
