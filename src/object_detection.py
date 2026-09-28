"""LiDAR object detection on non-ground points (NumPy/SciPy/sklearn, CPU-only).

Consumes preprocessed + ground-separated clouds (see :mod:`src.preprocessing`
and :mod:`src.ground_detection`, ultimately from :mod:`src.lidar_simulator`)
and clusters the non-ground points into objects via DBSCAN (``method:
dbscan`` in ``config/settings.yaml``) with a SciPy ``cKDTree`` connected
components fallback when scikit-learn is unavailable.

All parameters come from the ``object_detection`` section of
``config/settings.yaml``; any parameter may be overridden per call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .lidar_simulator import LidarFrame
from .utils import load_config

try:
    from sklearn.cluster import DBSCAN as _SkDBSCAN
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _SkDBSCAN = None  # type: ignore
    _HAS_SKLEARN = False

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore
    _HAS_SCIPY = False

ALLOWED_CATEGORIES = ("building", "vehicle", "pedestrian", "unknown")


@dataclass
class DetectedObject:
    """One clustered non-ground object."""

    object_id: int
    centroid: np.ndarray            # (3,)
    bbox_min: np.ndarray            # (3,)
    bbox_max: np.ndarray            # (3,)
    point_count: int
    category: str = "unknown"       # building | vehicle | pedestrian | unknown
    point_indices: np.ndarray = field(default_factory=lambda: np.zeros((0,), int))

    @property
    def bbox_size(self) -> np.ndarray:
        return np.asarray(self.bbox_max, dtype=float) - np.asarray(
            self.bbox_min, dtype=float)

    @property
    def bbox_volume(self) -> float:
        return float(np.prod(np.maximum(self.bbox_size, 0.0)))

    @property
    def bounding_box(self) -> np.ndarray:
        """(2, 3) array: row 0 = min corner, row 1 = max corner."""
        return np.stack([np.asarray(self.bbox_min, dtype=float),
                         np.asarray(self.bbox_max, dtype=float)], axis=0)

    def to_dict(self) -> dict:
        return {"object_id": int(self.object_id),
                "centroid": [float(v) for v in self.centroid],
                "bbox_min": [float(v) for v in self.bbox_min],
                "bbox_max": [float(v) for v in self.bbox_max],
                "bbox_volume": self.bbox_volume,
                "point_count": int(self.point_count),
                "category": str(self.category)}
def _classify_by_size(size: np.ndarray, height: float) -> str:
    """Heuristic category from AABB extents using simulator size priors."""
    lx, ly, lz = (float(size[0]), float(size[1]), float(size[2]))
    footprint = max(lx, ly)
    # Pedestrian: small footprint, human height.
    if footprint <= 1.2 and 1.0 <= height <= 2.4:
        return "pedestrian"
    # Vehicle: car-sized footprint, low height.
    if footprint <= 6.5 and max(lx, ly) >= 2.0 and height <= 2.6:
        return "vehicle"
    # Building: tall or very large footprint.
    if height > 2.6 or footprint > 6.5:
        return "building"
    return "unknown"


def _cc_labels(points: np.ndarray, eps: float) -> np.ndarray:
    """Connected-components clustering via cKDTree (fallback, no sklearn)."""
    n = points.shape[0]
    labels = np.full(n, -1, dtype=int)
    if n == 0 or not _HAS_SCIPY:
        return labels
    tree = cKDTree(points)
    pairs = tree.query_ball_point(points, r=float(eps))
    cur = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = cur
        while stack:
            j = stack.pop()
            for k in pairs[j]:
                if labels[k] == -1:
                    labels[k] = cur
                    stack.append(k)
        cur += 1
    return labels


class ObjectDetector:
    """Configurable DBSCAN/connected-components detector for non-ground clouds."""

    def __init__(self, config: Optional[dict] = None,
                 config_path: Optional[str] = None) -> None:
        cfg = config if config is not None else load_config(config_path)
        self.config = cfg
        od = cfg.get("object_detection", {})
        cl = od.get("clustering", {})
        self.method = str(cl.get("method", "dbscan")).lower()
        self.eps = float(cl.get("eps", 0.6))
        self.min_samples = int(cl.get("min_samples", 6))
        sf = od.get("size_filters", {})
        self.min_points = int(sf.get("min_points", 6))
        self.max_bbox_volume = float(sf.get("max_bbox_volume", 400.0))

    def cluster(self, points: np.ndarray,
                eps: Optional[float] = None,
                min_samples: Optional[int] = None,
                method: Optional[str] = None) -> np.ndarray:
        """Return integer cluster labels (-1 == noise) for ``(N, 3)`` points."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        n = pts.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=int)
        e = self.eps if eps is None else float(eps)
        ms = self.min_samples if min_samples is None else int(min_samples)
        meth = (self.method if method is None else str(method)).lower()
        if meth == "dbscan" and _HAS_SKLEARN:
            return np.asarray(_SkDBSCAN(eps=e, min_samples=ms).fit_predict(pts),
                              dtype=int)
        # Fallback: radius connected components + min_samples noise rule.
        raw = _cc_labels(pts, e)
        if ms > 1:
            out = np.full(n, -1, dtype=int)
            nxt = 0
            for lab in np.unique(raw):
                if lab < 0:
                    continue
                idx = np.where(raw == lab)[0]
                if idx.size >= ms:
                    out[idx] = nxt
                    nxt += 1
            return out
        return raw

    def detect(self, non_ground_points: np.ndarray,
               eps: Optional[float] = None,
               min_samples: Optional[int] = None,
               min_points: Optional[int] = None,
               max_bbox_volume: Optional[float] = None,
               method: Optional[str] = None) -> List[DetectedObject]:
        """Cluster non-ground points and build :class:`DetectedObject`s."""
        pts = np.asarray(non_ground_points, dtype=float).reshape(-1, 3)
        mp = self.min_points if min_points is None else int(min_points)
        mv = self.max_bbox_volume if max_bbox_volume is None \
            else float(max_bbox_volume)
        labels = self.cluster(pts, eps=eps, min_samples=min_samples, method=method)
        objects: List[DetectedObject] = []
        oid = 0
        for lab in sorted(l for l in np.unique(labels) if l >= 0):
            idx = np.where(labels == lab)[0]
            if idx.size < mp:
                continue
            cloud = pts[idx]
            bmin = cloud.min(axis=0)
            bmax = cloud.max(axis=0)
            vol = float(np.prod(np.maximum(bmax - bmin, 0.0)))
            if vol > mv:
                continue
            centroid = cloud.mean(axis=0)
            size = bmax - bmin
            height = float(size[2])
            objects.append(DetectedObject(
                object_id=oid, centroid=centroid, bbox_min=bmin, bbox_max=bmax,
                point_count=int(idx.size),
                category=_classify_by_size(size, height),
                point_indices=idx.astype(int)))
            oid += 1
        return objects
    def detect_frame(self, frame: LidarFrame,
                     ground_mask: Optional[np.ndarray] = None,
                     **kwargs) -> List[DetectedObject]:
        """Detect objects in a preprocessed frame's non-ground points.

        If ``ground_mask`` is None, the whole frame is treated as non-ground.
        ``point_indices`` in the results index the frame's point array when a
        mask is supplied, else the non-ground array.
        """
        pts = np.asarray(frame.points, dtype=float).reshape(-1, 3)
        if ground_mask is None:
            objs = self.detect(pts, **kwargs)
            return objs
        mask = np.asarray(ground_mask, dtype=bool)
        ng_idx = np.where(~mask)[0]
        objs = self.detect(pts[ng_idx] if ng_idx.size else pts[:0], **kwargs)
        for o in objs:
            o.point_indices = ng_idx[np.asarray(o.point_indices, dtype=int)] \
                if ng_idx.size else np.zeros((0,), int)
        return objs


def detect_objects(non_ground_points: np.ndarray,
                   config: Optional[dict] = None,
                   config_path: Optional[str] = None,
                   **kwargs) -> List[DetectedObject]:
    """Functional shortcut: ``ObjectDetector(...).detect(...)``."""
    return ObjectDetector(config=config, config_path=config_path).detect(
        non_ground_points, **kwargs)


