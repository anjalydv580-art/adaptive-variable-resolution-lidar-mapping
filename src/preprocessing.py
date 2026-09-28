"""Point-cloud preprocessing (NumPy/SciPy, CPU-only).

Operates on the output of :mod:`src.lidar_simulator` (``LidarFrame`` or raw
``(N, 3)`` arrays) and applies, in order:

1. invalid-point removal (NaN / +/-inf),
2. range filtering (radial distance from the sensor origin),
3. world / z bounds filtering,
4. voxel-grid downsampling,
5. statistical outlier removal (kNN distance test via SciPy cKDTree).

All parameters come from ``config/settings.yaml`` (``preprocessing`` and
``simulation`` sections) so nothing is hard-coded. Any parameter can be
overridden per-call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .lidar_simulator import LidarFrame
from .utils import load_config

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore
    _HAS_SCIPY = False


@dataclass
class PreprocessStats:
    """Counts collected during one :meth:`Preprocessor.preprocess_frame` run."""

    n_input: int = 0
    n_invalid_removed: int = 0
    n_range_removed: int = 0
    n_bounds_removed: int = 0
    n_voxel_removed: int = 0
    n_outlier_removed: int = 0
    n_output: int = 0

    def to_dict(self) -> dict:
        return {k: int(v) for k, v in self.__dict__.items()}


class Preprocessor:
    """Configurable preprocessing pipeline for LiDAR point clouds."""

    def __init__(self, config: Optional[dict] = None,
                 config_path: Optional[str] = None) -> None:
        cfg = config if config is not None else load_config(config_path)
        self.config = cfg
        pre = cfg.get("preprocessing", {})
        sim = cfg.get("simulation", {})
        sens = sim.get("sensor", {})
        world = sim.get("world", {})

        self.voxel_size = float(pre.get("voxel_size", 0.25))
        sor = pre.get("statistical_outlier", {})
        self.k_neighbors = int(sor.get("k_neighbors", 8))
        self.std_ratio = float(sor.get("std_ratio", 2.0))

        # Optional overrides in the preprocessing section; fall back to the
        # sensor/world description so the default settings.yaml just works.
        self.min_range = float(pre.get("min_range", 0.5))
        self.max_range = float(pre.get("max_range",
                                       sens.get("max_range", 60.0)))
        self.x_min = float(pre.get("x_min", world.get("x_min", -50.0)))
        self.x_max = float(pre.get("x_max", world.get("x_max", 50.0)))
        self.y_min = float(pre.get("y_min", world.get("y_min", -50.0)))
        self.y_max = float(pre.get("y_max", world.get("y_max", 50.0)))
        self.z_min = float(pre.get("z_min", world.get("z_min", -2.0)))
        self.z_max = float(pre.get("z_max", world.get("z_max", 15.0)))
        default_origin = sens.get("origin", [0.0, 0.0, 1.8])
        self.sensor_origin = np.asarray(
            pre.get("sensor_origin", default_origin), dtype=float).reshape(3)

    # -- individual stages (pure functions of arrays) -------------------
    @staticmethod
    def remove_invalid_points(points: np.ndarray) -> np.ndarray:
        """Return boolean mask of finite points (no NaN / +/-inf)."""
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return np.zeros((0,), dtype=bool)
        return np.all(np.isfinite(points), axis=1)

    def range_filter_mask(self, points: np.ndarray,
                          sensor_origin: Optional[np.ndarray] = None,
                          min_range: Optional[float] = None,
                          max_range: Optional[float] = None) -> np.ndarray:
        """Mask of points whose radial range from the sensor is in bounds."""
        points = np.asarray(points, dtype=float)
        if points.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        origin = (self.sensor_origin if sensor_origin is None
                  else np.asarray(sensor_origin, dtype=float).reshape(3))
        lo = self.min_range if min_range is None else float(min_range)
        hi = self.max_range if max_range is None else float(max_range)
        dist = np.linalg.norm(points - origin[None, :], axis=1)
        return (dist >= lo) & (dist <= hi)

    def bounds_filter_mask(self, points: np.ndarray) -> np.ndarray:
        """Mask of points inside the configured world / z bounds."""
        points = np.asarray(points, dtype=float)
        if points.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        return ((points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max)
                & (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max)
                & (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max))

    def voxel_downsample_mask(self, points: np.ndarray,
                              voxel_size: Optional[float] = None) -> np.ndarray:
        """Mask keeping the first point of every occupied voxel cell."""
        points = np.asarray(points, dtype=float)
        if points.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        vs = self.voxel_size if voxel_size is None else float(voxel_size)
        if vs <= 0:
            return np.ones((points.shape[0],), dtype=bool)
        keys = np.floor(points / vs).astype(np.int64)
        _, first_idx = np.unique(keys, axis=0, return_index=True)
        mask = np.zeros((points.shape[0],), dtype=bool)
        mask[np.sort(first_idx)] = True
        return mask

    def statistical_outlier_mask(self, points: np.ndarray,
                                 k_neighbors: Optional[int] = None,
                                 std_ratio: Optional[float] = None) -> np.ndarray:
        """Mask of inliers: mean-kNN-distance within mean + std_ratio * std."""
        points = np.asarray(points, dtype=float)
        n = points.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=bool)
        k = self.k_neighbors if k_neighbors is None else int(k_neighbors)
        sr = self.std_ratio if std_ratio is None else float(std_ratio)
        k = max(1, min(k, n - 1)) if n > 1 else 0
        if n <= 2 or k <= 0 or not _HAS_SCIPY:
            return np.ones((n,), dtype=bool)
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=k + 1)  # includes self distance 0
        mean_d = np.mean(dists[:, 1:], axis=1)
        mu = float(np.mean(mean_d))
        sigma = float(np.std(mean_d))
        return mean_d <= mu + sr * sigma

    # -- full pipeline -------------------------------------------------
    def preprocess_points(self, points: np.ndarray,
                          sensor_origin: Optional[np.ndarray] = None,
                          apply_voxel: bool = True,
                          apply_outlier: bool = True):
        """Run the full pipeline on an (N, 3) array.

        Returns ``(filtered_points, keep_indices, stats)`` where
        ``keep_indices`` index into the original input array.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        stats = PreprocessStats(n_input=int(points.shape[0]))
        if points.shape[0] == 0:
            return points.copy(), np.zeros((0,), dtype=int), stats

        origin = (self.sensor_origin if sensor_origin is None
                  else np.asarray(sensor_origin, dtype=float).reshape(3))

        keep = np.arange(points.shape[0])
        m = self.remove_invalid_points(points)
        stats.n_invalid_removed = int((~m).sum())
        points, keep = points[m], keep[m]
        if points.shape[0] == 0:
            return points, keep, stats

        m = self.range_filter_mask(points, origin)
        stats.n_range_removed = int((~m).sum())
        points, keep = points[m], keep[m]
        if points.shape[0] == 0:
            return points, keep, stats

        m = self.bounds_filter_mask(points)
        stats.n_bounds_removed = int((~m).sum())
        points, keep = points[m], keep[m]
        if points.shape[0] == 0:
            return points, keep, stats

        if apply_voxel:
            m = self.voxel_downsample_mask(points)
            stats.n_voxel_removed = int((~m).sum())
            points, keep = points[m], keep[m]
            if points.shape[0] == 0:
                return points, keep, stats

        if apply_outlier:
            m = self.statistical_outlier_mask(points)
            stats.n_outlier_removed = int((~m).sum())
            points, keep = points[m], keep[m]

        stats.n_output = int(points.shape[0])
        return points, keep, stats

    def preprocess_frame(self, frame: LidarFrame,
                         apply_voxel: bool = True,
                         apply_outlier: bool = True):
        """Preprocess a :class:`LidarFrame`, carrying labels along.

        Returns ``(new_frame, stats)``; ``new_frame`` is a filtered copy of
        ``frame`` with matching ``points`` / ``labels`` / ``object_ids`` /
        ``intensities`` arrays.
        """
        origin = getattr(frame, "sensor_origin", self.sensor_origin)
        pts, keep, stats = self.preprocess_points(
            np.asarray(frame.points, dtype=float),
            sensor_origin=origin, apply_voxel=apply_voxel,
            apply_outlier=apply_outlier)
        labels = np.asarray(frame.labels)[keep] if len(keep) else np.zeros((0,), int)
        oids = (np.asarray(frame.object_ids)[keep] if len(keep)
                else np.zeros((0,), int))
        intens = (np.asarray(frame.intensities)[keep] if len(keep)
                  else np.zeros((0,), float))
        new_frame = LidarFrame(
            frame_id=int(frame.frame_id),
            timestamp=float(frame.timestamp),
            points=np.asarray(pts, dtype=float),
            labels=np.asarray(labels, dtype=int),
            object_ids=np.asarray(oids, dtype=int),
            intensities=np.asarray(intens, dtype=float),
            objects=list(getattr(frame, "objects", [])),
            sensor_origin=np.asarray(origin, dtype=float).copy(),
        )
        return new_frame, stats


def preprocess_points(points: np.ndarray,
                      config: Optional[dict] = None,
                      config_path: Optional[str] = None,
                      **kwargs):
    """Functional shortcut: ``Preprocessor(...).preprocess_points(...)``."""
    return Preprocessor(config=config, config_path=config_path).preprocess_points(
        points, **kwargs)


def preprocess_frame(frame: LidarFrame,
                     config: Optional[dict] = None,
                     config_path: Optional[str] = None,
                     **kwargs):
    """Functional shortcut: ``Preprocessor(...).preprocess_frame(...)``."""
    return Preprocessor(config=config, config_path=config_path).preprocess_frame(
        frame, **kwargs)


