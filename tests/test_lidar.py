"""Phase 2 tests: LiDAR simulator smoke + determinism checks."""
import numpy as np
from src.lidar_simulator import LidarSimulator


def test_frame_shapes_and_labels():
    sim = LidarSimulator(seed=42)
    f = sim.simulate_frame(0)
    assert f.points.shape[0] > 100
    assert f.points.shape[1] == 3
    assert f.labels.shape[0] == f.points.shape[0]
    assert f.object_ids.shape[0] == f.points.shape[0]
    assert f.intensities.shape[0] == f.points.shape[0]
    assert set(np.unique(f.labels)).issubset({0, 1, 2, 3})
    assert float(f.intensities.min()) >= 0.0
    assert float(f.intensities.max()) <= 1.0


def test_determinism_same_seed():
    a = LidarSimulator(seed=123).simulate_frame(0)
    b = LidarSimulator(seed=123).simulate_frame(0)
    assert np.allclose(a.points, b.points)


def test_objects_and_sequence():
    sim = LidarSimulator(seed=42)
    objs = sim.get_objects(0)
    n_dyn = sum(1 for o in objs if o.is_dynamic)
    assert n_dyn == sim.num_vehicles + sim.num_pedestrians
    seq = sim.simulate_sequence(2)
    assert len(seq) == 2
    assert seq[1].frame_id == 1
