"""LiDAR feature extraction for adaptive resolution (NumPy, CPU-only).

Consumes existing outputs from preprocessing, ground detection, object
detection and dynamic detection and computes real, deterministic region
features. No randomness is used anywhere: ties are broken by sorted order
and temporal_change uses a deterministic stride subsample.

Frame-level features (``extract_frame_features``)
-------------------------------------------------
- ``point_density``: N / XY footprint area of the current points.
- ``object_density``: n_objects / XY footprint area.
- ``dynamic_activity``: n_dynamic / max(n_objects, 1).
- ``elevation_variance``: variance of z over current points.
- ``temporal_change``: mean nearest-neighbour distance from current points
  to previous-frame points (0.0 when no previous frame is given).
- ``occupancy``: fraction of occupied XY grid cells (``grid_cell_size``).

Grid-level features (``extract_grid_features``) compute the same quantities
per XY cell and return an ``(ny, nx, n_features)`` array plus metadata.

All parameters come from the ``feature_extraction`` section of
``config/settings.yaml``; any parameter may be overridden per call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dynamic_detection import DynamicObject
from .lidar_simulator import LidarFrame
from .object_detection import DetectedObject
from .utils import load_config

FEATURE_NAMES: Tuple[str, ...] = (
    "point_density",
    "object_density",
    "dynamic_activity",
    "elevation_variance",
    "temporal_change",
    "occupancy",
)


@dataclass
class FrameFeatures:
    """Deterministic feature vector for one frame / region."""

    point_density: float = 0.0
    object_density: float = 0.0
    dynamic_activity: float = 0.0
    elevation_variance: float = 0.0
    temporal_change: float = 0.0
    occupancy: float = 0.0
    num_points: int = 0
    num_objects: int = 0
    num_dynamic: int = 0
    area: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "point_density": float(self.point_density),
            "object_density": float(self.object_density),
            "dynamic_activity": float(self.dynamic_activity),
            "elevation_variance": float(self.elevation_variance),
            "temporal_change": float(self.temporal_change),
            "occupancy": float(self.occupancy),
            "num_points": int(self.num_points),
            "num_objects": int(self.num_objects),
            "num_dynamic": int(self.num_dynamic),
            "area": float(self.area),
        }
        d.update({k: float(v) for k, v in self.extra.items()})
        return d

    def to_vector(self, order: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
        return np.array([float(getattr(self, n)) for n in order], dtype=float)


def _footprint_area(points: np.ndarray) -> float:
    """XY bounding-box area of points; 0.0 when degenerate/empty."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return 0.0
    ex = float(pts[:, 0].max() - pts[:, 0].min())
    ey = float(pts[:, 1].max() - pts[:, 1].min())
    if ex <= 0.0 or ey <= 0.0:
        return 0.0
    return ex * ey


def _occupancy(points: np.ndarray, grid_cell_size: float) -> float:
    """Fraction of occupied XY cells over the points' bounding box."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return 0.0
    cs = max(float(grid_cell_size), 1e-9)
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    ix = np.floor((pts[:, 0] - x0) / cs).astype(np.int64)
    iy = np.floor((pts[:, 1] - y0) / cs).astype(np.int64)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    if nx <= 0 or ny <= 0:
        return 0.0
    occupied = len(np.unique(ix * ny + iy))
    return float(occupied) / float(nx * ny)


def _mean_nn_distance(cur: np.ndarray, prev: np.ndarray,
                      max_points: int) -> float:
    """Mean nearest-neighbour distance cur -> prev (deterministic)."""
    a = np.asarray(cur, dtype=float).reshape(-1, 3)
    b = np.asarray(prev, dtype=float).reshape(-1, 3)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    cap = max(int(max_points), 1)
    if a.shape[0] > cap:
        order = np.lexsort((a[:, 2], a[:, 1], a[:, 0]))
        a = a[order[:: max(1, a.shape[0] // cap)][:cap]]
    if b.shape[0] > cap:
        order = np.lexsort((b[:, 2], b[:, 1], b[:, 0]))
        b = b[order[:: max(1, b.shape[0] // cap)][:cap]]
    best = np.full(a.shape[0], np.inf)
    chunk = 512
    for s in range(0, a.shape[0], chunk):
        e = min(s + chunk, a.shape[0])
        d2 = ((a[s:e, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        best[s:e] = d2.min(axis=1)
    return float(np.sqrt(best).mean())


class FeatureExtractor:
    """Deterministic region-feature extractor driven by settings.yaml."""

    def __init__(self, config: Optional[dict] = None,
                 config_path: Optional[str] = None) -> None:
        cfg = config if config is not None else load_config(config_path)
        self.config = cfg
        fe = cfg.get("feature_extraction", {})
        gd = cfg.get("ground_detection", {})
        self.grid_cell_size = float(
            fe.get("grid_cell_size", gd.get("grid_cell_size", 1.0)))
        self.max_points_temporal = int(fe.get("max_points_temporal", 2000))

    def extract_frame_features(
        self,
        points: np.ndarray,
        objects: Sequence[DetectedObject] | None = None,
        dynamics: Sequence[DynamicObject] | None = None,
        prev_points: Optional[np.ndarray] = None,
        grid_cell_size: Optional[float] = None,
        max_points_temporal: Optional[int] = None,
    ) -> FrameFeatures:
        """Compute the six adaptive-resolution features for one point set."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        objs: List[DetectedObject] = list(objects) if objects else []
        dyns: List[DynamicObject] = list(dynamics) if dynamics else []
        cs = self.grid_cell_size if grid_cell_size is None else float(grid_cell_size)
        cap = (self.max_points_temporal if max_points_temporal is None
               else int(max_points_temporal))
        n = int(pts.shape[0])
        area = _footprint_area(pts)
        n_obj = len(objs)
        n_dyn = sum(1 for d in dyns if bool(getattr(d, "is_dynamic", False)))
        if prev_points is None:
            tchange = 0.0
        else:
            tchange = _mean_nn_distance(pts, prev_points, cap)
        elev_var = float(pts[:, 2].var()) if n > 0 else 0.0
        occ = _occupancy(pts, cs)
        return FrameFeatures(
            point_density=float(n / area) if area > 0 else 0.0,
            object_density=float(n_obj / area) if area > 0 else 0.0,
            dynamic_activity=float(n_dyn / max(n_obj, 1)),
            elevation_variance=elev_var,
            temporal_change=float(tchange),
            occupancy=float(occ),
            num_points=n,
            num_objects=n_obj,
            num_dynamic=int(n_dyn),
            area=float(area),
        )

    def extract_from_frames(
        self,
        curr_frame: LidarFrame,
        objects: Sequence[DetectedObject] | None = None,
        dynamics: Sequence[DynamicObject] | None = None,
        prev_frame: Optional[LidarFrame] = None,
        **kwargs,
    ) -> FrameFeatures:
        """Convenience wrapper accepting LidarFrame objects."""
        prev_pts = (np.asarray(prev_frame.points, dtype=float)
                    if prev_frame is not None else None)
        return self.extract_frame_features(
            np.asarray(curr_frame.points, dtype=float),
            objects=objects, dynamics=dynamics,
            prev_points=prev_pts, **kwargs)

    def extract_grid_features(
        self,
        points: np.ndarray,
        objects: Sequence[DetectedObject] | None = None,
        dynamics: Sequence[DynamicObject] | None = None,
        prev_points: Optional[np.ndarray] = None,
        grid_cell_size: Optional[float] = None,
        max_points_temporal: Optional[int] = None,
    ) -> dict:
        """Per-cell features over a regular XY grid."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        objs: List[DetectedObject] = list(objects) if objects else []
        dyns: List[DynamicObject] = list(dynamics) if dynamics else []
        cs = self.grid_cell_size if grid_cell_size is None else float(grid_cell_size)
        cs = max(cs, 1e-9)
        cap = (self.max_points_temporal if max_points_temporal is None
               else int(max_points_temporal))
        prev = (np.asarray(prev_points, dtype=float).reshape(-1, 3)
                if prev_points is not None else None)
        if pts.shape[0] == 0:
            empty = np.zeros((0, 0, len(FEATURE_NAMES)), dtype=float)
            return {"features": empty, "feature_names": list(FEATURE_NAMES),
                    "origin": [0.0, 0.0], "cell_size": float(cs), "shape": (0, 0)}
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        ix = np.floor((pts[:, 0] - x0) / cs).astype(np.int64)
        iy = np.floor((pts[:, 1] - y0) / cs).astype(np.int64)
        nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
        obj_cell = []
        for o in objs:
            c = np.asarray(o.centroid, dtype=float).reshape(3)
            obj_cell.append((int(np.floor((c[1] - y0) / cs)),
                             int(np.floor((c[0] - x0) / cs))))
        dyn_centroids = [np.asarray(getattr(d, "curr_centroid", [np.nan] * 3),
                                    dtype=float).reshape(3) for d in dyns
                         if bool(getattr(d, "is_dynamic", False))]
        feats = np.zeros((ny, nx, len(FEATURE_NAMES)), dtype=float)
        cell_area = cs * cs
        for j in range(ny):
            for i in range(nx):
                sel = (ix == i) & (iy == j)
                cell_pts = pts[sel]
                k = int(sel.sum())
                if k == 0:
                    continue
                n_obj = sum(1 for (cj, ci) in obj_cell if cj == j and ci == i)
                n_dyn = 0
                for c in dyn_centroids:
                    if np.all(np.isfinite(c)):
                        ci = int(np.floor((c[0] - x0) / cs))
                        cj = int(np.floor((c[1] - y0) / cs))
                        if ci == i and cj == j:
                            n_dyn += 1
                tc = _mean_nn_distance(cell_pts, prev, cap) if prev is not None else 0.0
                feats[j, i, 0] = k / cell_area
                feats[j, i, 1] = n_obj / cell_area
                if n_obj:
                    feats[j, i, 2] = n_dyn / max(n_obj, 1)
                else:
                    feats[j, i, 2] = float(min(n_dyn, 1))
                feats[j, i, 3] = float(cell_pts[:, 2].var())
                feats[j, i, 4] = float(tc)
                feats[j, i, 5] = 1.0
        return {"features": feats, "feature_names": list(FEATURE_NAMES),
                "origin": [float(x0), float(y0)], "cell_size": float(cs),
                "shape": (int(ny), int(nx))}

    def feature_matrix(self, feats: Sequence[FrameFeatures],
                       order: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
        """Stack FrameFeatures into an (len(feats), len(order)) matrix."""
        return np.array([f.to_vector(order) for f in feats], dtype=float)


def extract_frame_features(points: np.ndarray,
                           config: Optional[dict] = None,
                           config_path: Optional[str] = None,
                           **kwargs) -> FrameFeatures:
    """Functional shortcut for extract_frame_features."""
    return FeatureExtractor(config=config, config_path=config_path
                            ).extract_frame_features(points, **kwargs)


def extract_grid_features(points: np.ndarray,
                          config: Optional[dict] = None,
                          config_path: Optional[str] = None,
                          **kwargs) -> dict:
    """Functional shortcut for extract_grid_features."""
    return FeatureExtractor(config=config, config_path=config_path
                            ).extract_grid_features(points, **kwargs)

