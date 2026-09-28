"""Realistic synthetic 3D LiDAR point-cloud simulator (NumPy, CPU-only).

A static urban scene (undulating ground, building boxes, moving vehicle
and pedestrian boxes) is ray-cast by a rotating multi-channel LiDAR.
Nearest-hit-wins gives true occlusion; range gating, dropout and
Gaussian noise are applied. All randomness uses one seeded Generator,
so identical seeds give identical clouds. Parameters come from
config/settings.yaml -- nothing is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import get_rng, load_config

LABEL_GROUND = 0
LABEL_BUILDING = 1
LABEL_VEHICLE = 2
LABEL_PEDESTRIAN = 3

LABEL_NAMES = {
    LABEL_GROUND: "ground",
    LABEL_BUILDING: "building",
    LABEL_VEHICLE: "vehicle",
    LABEL_PEDESTRIAN: "pedestrian",
}

_BASE_INTENSITY = {
    LABEL_GROUND: 0.30,
    LABEL_BUILDING: 0.60,
    LABEL_VEHICLE: 0.85,
    LABEL_PEDESTRIAN: 0.70,
}


@dataclass
class ObjectState:
    """Ground-truth state of one scene object at a given frame."""

    object_id: int
    category: str  # 'building' | 'vehicle' | 'pedestrian'
    center: np.ndarray  # (3,) full 3D box center
    size: np.ndarray  # (3,) full extents (lx, ly, lz)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    is_dynamic: bool = False

    @property
    def box_min(self) -> np.ndarray:
        return self.center - self.size * 0.5

    @property
    def box_max(self) -> np.ndarray:
        return self.center + self.size * 0.5

    def to_dict(self) -> dict:
        return {
            "object_id": int(self.object_id),
            "category": self.category,
            "center": [float(v) for v in self.center],
            "size": [float(v) for v in self.size],
            "velocity": [float(v) for v in self.velocity],
            "is_dynamic": bool(self.is_dynamic),
        }


@dataclass
class LidarFrame:
    """One simulated LiDAR sweep with labels and ground-truth objects."""

    frame_id: int
    timestamp: float
    points: np.ndarray  # (M, 3)
    labels: np.ndarray  # (M,) int LABEL_* constants
    object_ids: np.ndarray  # (M,) int, -1 for ground
    intensities: np.ndarray  # (M,) float in [0, 1]
    objects: List[ObjectState]
    sensor_origin: np.ndarray  # (3,)

    def summary(self) -> dict:
        counts = {n: int(np.sum(self.labels == i)) for i, n in LABEL_NAMES.items()}
        if len(self.points):
            mn = [float(v) for v in self.points.min(axis=0)]
            mx = [float(v) for v in self.points.max(axis=0)]
        else:
            mn, mx = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        return {
            "frame_id": int(self.frame_id),
            "timestamp": float(self.timestamp),
            "num_points": int(self.points.shape[0]),
            "counts": counts,
            "xyz_min": mn,
            "xyz_max": mx,
            "num_objects": len(self.objects),
            "num_dynamic_objects": int(sum(1 for o in self.objects if o.is_dynamic)),
        }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ground_height(x, y, base=0.0, amplitude=0.05, scale=8.0):
    """Deterministic undulating ground height h(x, y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return base + amplitude * np.sin(x / scale) * np.cos(y / scale)


def _ray_box_intersection(rays_origin, rays_dir, box_min, box_max):
    """Vectorised slab-method ray vs single AABB -> (N,) t_hit or inf."""
    safe_dir = np.where(np.abs(rays_dir) < 1e-9, 1e-9, rays_dir)
    t1 = (box_min - rays_origin) / safe_dir
    t2 = (box_max - rays_origin) / safe_dir
    t_near = np.minimum(t1, t2).max(axis=1)
    t_far = np.maximum(t1, t2).min(axis=1)
    parallel = np.abs(rays_dir) < 1e-9
    outside = (rays_origin < box_min) | (rays_origin > box_max)
    miss = (parallel & outside).any(axis=1) | (t_near > t_far) | (t_far <= 0.0)
    t_hit = np.where(miss, np.inf, np.where(t_near > 0.0, t_near, t_far))
    return t_hit


def _reflect_coordinate(raw, lo, hi):
    """Reflecting (ping-pong) wrap so objects bounce inside [lo, hi]."""
    span = hi - lo
    span = np.where(span < 1e-6, np.ones_like(span), span)
    period = 2.0 * span
    m = np.mod(raw - lo, period)
    return np.where(m <= span, lo + m, hi - (m - span))

# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------


class LidarSimulator:
    """Rotating multi-channel LiDAR ray-caster over a synthetic urban scene."""

    def __init__(self, config=None, config_path=None, seed=None):
        cfg = config if config is not None else load_config(config_path)
        self.config = cfg
        sim = cfg.get("simulation", {})
        proj_seed = cfg.get("project", {}).get("seed")
        self.seed = seed if seed is not None else proj_seed
        self.rng = get_rng(self.seed, cfg)
        world = sim.get("world", {})
        self.x_min = float(world.get("x_min", -50.0))
        self.x_max = float(world.get("x_max", 50.0))
        self.y_min = float(world.get("y_min", -50.0))
        self.y_max = float(world.get("y_max", 50.0))
        self.z_min = float(world.get("z_min", -2.0))
        self.z_max = float(world.get("z_max", 15.0))
        sens = sim.get("sensor", {})
        origin = sens.get("origin", [0.0, 0.0, 1.8])
        self.sensor_origin = np.asarray(origin, dtype=float).reshape(3)
        self.num_channels = int(sens.get("num_channels", 32))
        vmin, vmax = sens.get("vertical_fov_deg", [-20.0, 15.0])
        self.v_fov_min = float(vmin)
        self.v_fov_max = float(vmax)
        self.h_res_deg = float(sens.get("horizontal_resolution_deg", 0.4))
        self.max_range = float(sens.get("max_range", 60.0))
        self.range_noise_std = float(sens.get("range_noise_std", 0.02))
        self.angular_noise_std_deg = float(sens.get("angular_noise_std_deg", 0.05))
        self.dropout_prob = float(sens.get("dropout_prob", 0.01))
        g = sim.get("ground", {})
        self.ground_base = float(g.get("base_height", 0.0))
        self.ground_amp = float(g.get("roughness_amplitude", 0.05))
        self.ground_scale = float(g.get("roughness_scale", 8.0))
        obs = sim.get("obstacles", {})
        self.num_buildings = int(obs.get("num_buildings", 5))
        bs = obs.get("building_size_range", [3.0, 10.0])
        bh = obs.get("building_height_range", [3.0, 20.0])
        self.building_size_range = (float(bs[0]), float(bs[1]))
        self.building_height_range = (float(bh[0]), float(bh[1]))
        veh = sim.get("vehicles", {})
        self.num_vehicles = int(veh.get("num_vehicles", 4))
        self.vehicle_size = np.asarray(veh.get("size", [4.5, 1.9, 1.6]), dtype=float)
        vs = veh.get("speed_range", [1.0, 6.0])
        self.vehicle_speed_range = (float(vs[0]), float(vs[1]))
        ped = sim.get("pedestrians", {})
        self.num_pedestrians = int(ped.get("num_pedestrians", 6))
        self.pedestrian_size = np.asarray(ped.get("size", [0.5, 0.5, 1.7]), dtype=float)
        ps = ped.get("speed_range", [0.5, 1.8])
        self.pedestrian_speed_range = (float(ps[0]), float(ps[1]))
        dyn = cfg.get("dynamic_detection", {})
        self.frame_dt = float(dyn.get("frame_dt", 0.1))
        self._init_scene()
    def _init_scene(self):
        rng = self.rng
        oid = 0
        self._buildings = []
        for _ in range(self.num_buildings):
            lx = float(rng.uniform(*self.building_size_range))
            ly = float(rng.uniform(*self.building_size_range))
            lz = float(rng.uniform(*self.building_height_range))
            cx, cy = 30.0, 30.0
            for _try in range(50):
                cx = float(rng.uniform(self.x_min + lx / 2, self.x_max - lx / 2))
                cy = float(rng.uniform(self.y_min + ly / 2, self.y_max - ly / 2))
                if (cx ** 2 + cy ** 2) ** 0.5 > 12.0:
                    break
            cz = self.ground_base + lz / 2.0
            self._buildings.append(ObjectState(oid, "building",
                np.array([cx, cy, cz]), np.array([lx, ly, lz]),
                np.zeros(3), False))
            oid += 1
        self._vehicles_init = []
        for _ in range(self.num_vehicles):
            cx = float(rng.uniform(self.x_min + 5, self.x_max - 5))
            cy = float(rng.uniform(self.y_min + 5, self.y_max - 5))
            heading = float(rng.uniform(0, 2 * np.pi))
            speed = float(rng.uniform(*self.vehicle_speed_range))
            vel = np.array([np.cos(heading) * speed, np.sin(heading) * speed, 0.0])
            self._vehicles_init.append({"object_id": oid,
                "center0": np.array([cx, cy, 0.0]), "velocity": vel,
                "size": self.vehicle_size.copy()})
            oid += 1
        self._peds_init = []
        for _ in range(self.num_pedestrians):
            cx = float(rng.uniform(self.x_min + 3, self.x_max - 3))
            cy = float(rng.uniform(self.y_min + 3, self.y_max - 3))
            heading = float(rng.uniform(0, 2 * np.pi))
            speed = float(rng.uniform(*self.pedestrian_speed_range))
            vel = np.array([np.cos(heading) * speed, np.sin(heading) * speed, 0.0])
            self._peds_init.append({"object_id": oid,
                "center0": np.array([cx, cy, 0.0]), "velocity": vel,
                "size": self.pedestrian_size.copy()})
            oid += 1

    def get_objects(self, frame_id=0):
        t = float(frame_id) * self.frame_dt
        objects = list(self._buildings)
        for v in self._vehicles_init:
            raw = v["center0"][:2] + v["velocity"][:2] * t
            lo = np.array([self.x_min + v["size"][0] / 2, self.y_min + v["size"][1] / 2])
            hi = np.array([self.x_max - v["size"][0] / 2, self.y_max - v["size"][1] / 2])
            xy = _reflect_coordinate(raw, lo, hi)
            gz = float(ground_height(xy[0], xy[1], self.ground_base,
                                     self.ground_amp, self.ground_scale))
            center = np.array([xy[0], xy[1], gz + v["size"][2] / 2.0])
            objects.append(ObjectState(v["object_id"], "vehicle", center,
                                       v["size"].copy(), v["velocity"].copy(), True))
        for p in self._peds_init:
            raw = p["center0"][:2] + p["velocity"][:2] * t
            lo = np.array([self.x_min + p["size"][0] / 2, self.y_min + p["size"][1] / 2])
            hi = np.array([self.x_max - p["size"][0] / 2, self.y_max - p["size"][1] / 2])
            xy = _reflect_coordinate(raw, lo, hi)
            gz = float(ground_height(xy[0], xy[1], self.ground_base,
                                     self.ground_amp, self.ground_scale))
            center = np.array([xy[0], xy[1], gz + p["size"][2] / 2.0])
            objects.append(ObjectState(p["object_id"], "pedestrian", center,
                                       p["size"].copy(), p["velocity"].copy(), True))
        return objects
    def _ray_directions(self):
        elev = np.deg2rad(np.linspace(self.v_fov_min, self.v_fov_max, self.num_channels))
        n_az = max(1, int(round(360.0 / self.h_res_deg)))
        azim = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
        EL, AZ = np.meshgrid(elev, azim, indexing="ij")
        EL = EL.ravel()
        AZ = AZ.ravel()
        return np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ),
                         np.sin(EL)], axis=1)

    def simulate_frame(self, frame_id=0):
        objects = self.get_objects(frame_id)
        origin = self.sensor_origin
        dirs = self._ray_directions()
        n = dirs.shape[0]
        base = int(self.seed) if self.seed is not None else 0
        frng = np.random.default_rng((base + int(frame_id) * 1000003) % (2 ** 31 - 1))
        dz = dirs[:, 2]
        down = dz < -1e-6
        t_plane = (self.ground_base - origin[2]) / np.where(down, dz, -1.0)
        valid = down & (t_plane > 0)
        hx = origin[0] + dirs[:, 0] * t_plane
        hy = origin[1] + dirs[:, 1] * t_plane
        gh = ground_height(hx, hy, self.ground_base, self.ground_amp, self.ground_scale)
        t_ref = (gh - origin[2]) / np.where(down, dz, -1.0)
        t_ref = np.where(valid & (t_ref > 0), t_ref, np.inf)
        with np.errstate(invalid="ignore"):
            gx = origin[0] + dirs[:, 0] * t_ref
            gy = origin[1] + dirs[:, 1] * t_ref
        in_w = ((gx >= self.x_min) & (gx <= self.x_max)
                & (gy >= self.y_min) & (gy <= self.y_max))
        t_best = np.where(in_w, t_ref, np.inf)
        best_label = np.where(np.isfinite(t_best), LABEL_GROUND, -1)
        best_oid = np.where(np.isfinite(t_best), -1, -1)
        origins = np.broadcast_to(origin, dirs.shape)
        label_of = {"building": LABEL_BUILDING, "vehicle": LABEL_VEHICLE,
                    "pedestrian": LABEL_PEDESTRIAN}
        for obj in objects:
            t_hit = _ray_box_intersection(origins, dirs, obj.box_min, obj.box_max)
            closer = t_hit < t_best
            if np.any(closer):
                t_best = np.where(closer, t_hit, t_best)
                best_label = np.where(closer, label_of[obj.category], best_label)
                best_oid = np.where(closer, obj.object_id, best_oid)
        hit = np.isfinite(t_best) & (t_best > 0) & (t_best <= self.max_range)
        hit = hit & (frng.random(n) >= self.dropout_prob)
        idx = np.where(hit)[0]
        t_hit = t_best[idx]
        dirs_hit = dirs[idx]
        labels = best_label[idx].astype(int)
        oids = best_oid[idx].astype(int)
        t_noisy = t_hit + frng.normal(0.0, self.range_noise_std, size=t_hit.shape)
        t_noisy = np.clip(t_noisy, 0.5, self.max_range)
        ang_std = np.deg2rad(self.angular_noise_std_deg)
        lateral = frng.normal(0.0, 1.0, size=(idx.size, 3)) * (t_noisy[:, None] * ang_std)
        points = origin[None, :] + dirs_hit * t_noisy[:, None] + lateral * 0.1
        base_int = np.array([_BASE_INTENSITY.get(int(x), 0.5) for x in labels])
        intensities = np.clip(base_int + frng.normal(0.0, 0.05, size=base_int.shape),
                              0.0, 1.0).astype(float)
        return LidarFrame(int(frame_id), float(frame_id) * self.frame_dt,
            points.astype(float), labels, oids, intensities, objects, origin.copy())

    def generate_frame(self, frame_id=0):
        return self.simulate_frame(frame_id)

    def simulate_sequence(self, num_frames):
        return [self.simulate_frame(i) for i in range(int(num_frames))]

    def generate_sequence(self, num_frames):
        return self.simulate_sequence(num_frames)
