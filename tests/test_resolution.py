"""Phase 8 tests: adaptive variable-resolution controller."""
import numpy as np

from src.dynamic_detection import DynamicDetector
from src.feature_extraction import FeatureExtractor, FrameFeatures
from src.ground_detection import GroundDetector
from src.lidar_simulator import LidarSimulator
from src.object_detection import ObjectDetector
from src.preprocessing import Preprocessor
from src.resolution_controller import (
    LEVELS,
    WEIGHTED_FEATURES,
    ResolutionController,
    select_resolution,
)


def _pipe(frame):
    proc, _ = Preprocessor().preprocess_frame(frame)
    _, gres = GroundDetector().detect_frame(proc)
    objs = ObjectDetector().detect_frame(proc, ground_mask=gres.ground_mask)
    return proc, objs


def _feats(**kw):
    base = {"point_density": 0.0, "object_density": 0.0,
            "dynamic_activity": 0.0, "elevation_variance": 0.0,
            "temporal_change": 0.0, "occupancy": 0.0}
    base.update(kw)
    return FrameFeatures(**base)


def test_config_defaults_match_settings_yaml():
    from src.utils import load_config
    cfg = load_config()
    rc = ResolutionController(config=cfg)
    y = cfg["resolution_controller"]
    for f in WEIGHTED_FEATURES:
        assert rc.weights[f] == float(y["weights"][f])
        assert rc.scales[f] == float(y["normalization"][f])
    assert rc.low_max == float(y["complexity_thresholds"]["low_max"])
    assert rc.medium_max == float(y["complexity_thresholds"]["medium_max"])
    assert rc.voxel_sizes["LOW"] == float(y["levels"]["low"]["cell_size"]) == 2.0
    assert rc.voxel_sizes["MEDIUM"] == float(y["levels"]["medium"]["cell_size"]) == 1.0
    assert rc.voxel_sizes["HIGH"] == float(y["levels"]["high"]["cell_size"]) == 0.5


def test_weighted_features_cover_spec():
    for name in ("point_density", "object_density", "dynamic_activity",
                 "elevation_variance", "temporal_change"):
        assert name in WEIGHTED_FEATURES
    assert set(LEVELS) == {"LOW", "MEDIUM", "HIGH"}


def test_complexity_score_is_weighted_and_bounded():
    rc = ResolutionController()
    scales = dict(rc.scales)
    f = _feats(point_density=scales["point_density"],
               object_density=scales["object_density"],
               dynamic_activity=scales["dynamic_activity"],
               elevation_variance=scales["elevation_variance"],
               temporal_change=scales["temporal_change"])
    score, norm = rc.complexity_score(f)
    assert all(v == 1.0 for v in norm.values())
    assert abs(score - 1.0) < 1e-9
    f2 = _feats(point_density=scales["point_density"] / 2,
                object_density=scales["object_density"] / 2,
                dynamic_activity=0.5,
                elevation_variance=scales["elevation_variance"] / 2,
                temporal_change=scales["temporal_change"] / 2)
    s2, _ = rc.complexity_score(f2)
    assert abs(s2 - 0.5) < 1e-9
    w = {"point_density": 1.0, "object_density": 0.0, "dynamic_activity": 0.0,
         "elevation_variance": 0.0, "temporal_change": 0.0}
    s3, _ = rc.complexity_score(
        _feats(point_density=scales["point_density"]), weights=w)
    assert abs(s3 - 1.0) < 1e-9


def test_level_mapping_low_medium_high():
    rc = ResolutionController()
    assert rc.level_for_score(0.0) == "LOW"
    assert rc.level_for_score(rc.low_max) == "LOW"
    assert rc.level_for_score(rc.low_max + 1e-6) == "MEDIUM"
    assert rc.level_for_score(rc.medium_max) == "MEDIUM"
    assert rc.level_for_score(rc.medium_max + 1e-6) == "HIGH"
    assert rc.level_for_score(1.0) == "HIGH"


def test_decide_returns_score_level_voxel_size():
    rc = ResolutionController()
    low = rc.decide(_feats())
    assert low.complexity_score == 0.0
    assert low.resolution_level == "LOW"
    assert low.selected_voxel_size == 2.0
    d = select_resolution(_feats(dynamic_activity=1.0,
                                 temporal_change=rc.scales["temporal_change"]))
    assert d.resolution_level in LEVELS
    assert 0.0 <= d.complexity_score <= 1.0
    assert d.to_dict()["resolution_level"] == d.resolution_level
    busy = _feats(point_density=99.0, object_density=99.0, dynamic_activity=1.0,
                  elevation_variance=99.0, temporal_change=99.0)
    assert rc.decide(busy).resolution_level == "HIGH"
    assert rc.decide(busy).complexity_score >= low.complexity_score



def test_edge_cases_deterministic():
    rc = ResolutionController()
    for bad in ({"point_density": np.nan}, {"point_density": np.inf},
                {"point_density": -5.0}, {}):
        a = rc.decide(dict(bad))
        b = rc.decide(dict(bad))
        assert a.to_dict() == b.to_dict()
        assert 0.0 <= a.complexity_score <= 1.0
        assert a.resolution_level in LEVELS
    zero = rc.decide({})
    assert (zero.complexity_score, zero.resolution_level,
            zero.selected_voxel_size) == (0.0, "LOW", 2.0)
    batch = rc.decide_batch([_feats(), _feats(dynamic_activity=1.0)])
    assert [d.resolution_level for d in batch] == [
        rc.decide(_feats()).resolution_level,
        rc.decide(_feats(dynamic_activity=1.0)).resolution_level]


def test_real_phase7_features_drive_controller():
    sim = LidarSimulator(seed=42)
    prev_proc, prev_objs = _pipe(sim.simulate_frame(0))
    curr_proc, curr_objs = _pipe(sim.simulate_frame(1))
    assert len(prev_objs) > 0 and len(curr_objs) > 0
    tracks = DynamicDetector().match_and_classify(prev_objs, curr_objs)
    fe = FeatureExtractor()
    feats = fe.extract_from_frames(curr_proc, objects=curr_objs,
                                   dynamics=tracks, prev_frame=prev_proc)
    rc = ResolutionController()
    a = rc.decide(feats)
    b = rc.decide(feats)
    assert a.to_dict() == b.to_dict()
    assert 0.0 <= a.complexity_score <= 1.0
    assert a.resolution_level in LEVELS
    assert a.selected_voxel_size in (2.0, 1.0, 0.5)
    busy = dict(feats.to_dict())
    for k in WEIGHTED_FEATURES:
        busy[k] = float(busy.get(k, 0.0)) * 10.0 + 1.0
    assert rc.decide(busy).complexity_score >= a.complexity_score
