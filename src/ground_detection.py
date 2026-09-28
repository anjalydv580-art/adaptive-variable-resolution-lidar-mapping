"""Ground / non-ground segmentation (NumPy, CPU-only).

Consumes preprocessed clouds (see :mod:`src.preprocessing`, ultimately from
:mod:`src.lidar_simulator`) and splits them into ground vs. non-ground using
height/elevation-based processing:

* ``ransac`` (default per ``config/settings.yaml``): RANSAC plane fit; points
  within ``distance_threshold`` of the dominant plane are ground.
* ``grid``: per-cell minimum-elevation estimate on an XY grid of
  ``grid_cell_size``; points within ``distance_threshold`` above their cell
  minimum are ground (handles gently undulating terrain).

All parameters come from the ``ground_detection`` section of
``config/settings.yaml``; any parameter may be overridden per call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .lidar_simulator import LidarFrame
from .utils import get_rng, load_config


@dataclass
class GroundResult:
    """Container for one ground-segmentation run."""

    ground_points: np.ndarray      # (G, 3)
    non_ground_points: np.ndarray  # (N-G, 3)
    ground_mask: np.ndarray        # (N,) bool, True == ground
    plane: Optional[np.ndarray] = None  # (4,) [a, b, c, d] if fitted
    method: str = "ransac"

    @property
    def n_ground(self) -> int:
        return int(self.ground_mask.sum())

    @property
    def n_non_ground(self) -> int:
        return int((~self.ground_mask).sum())

class GroundDetector:
    """Configurable ground detector driven by ``settings.yaml``."""

    def __init__(self, config: Optional[dict] = None,
                 config_path: Optional[str] = None,
                 seed: Optional[int] = None) -> None:
        cfg = config if config is not None else load_config(config_path)
        self.config = cfg
        gd = cfg.get("ground_detection", {})
        self.method = str(gd.get("method", "ransac")).lower()
        ransac = gd.get("ransac", {})
        self.distance_threshold = float(ransac.get("distance_threshold", 0.15))
        self.max_iterations = int(ransac.get("max_iterations", 200))
        self.grid_cell_size = float(gd.get("grid_cell_size", 1.0))
        proj_seed = cfg.get("project", {}).get("seed")
        self.seed = seed if seed is not None else proj_seed
        self.rng = get_rng(self.seed, cfg)

    # -- RANSAC plane -------------------------------------------------
    @staticmethod
    def _plane_from_points(p0, p1, p2) -> Optional[np.ndarray]:
        n = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            return None
        n = n / norm
        d = -float(np.dot(n, p0))
        return np.array([n[0], n[1], n[2], d], dtype=float)

    def _ransac_plane(self, points: np.ndarray,
                      distance_threshold: Optional[float] = None,
                      max_iterations: Optional[int] = None,
                      seed: Optional[int] = None):
        pts = np.asarray(points, dtype=float)
        n = pts.shape[0]
        thr = self.distance_threshold if distance_threshold is None \
            else float(distance_threshold)
        iters = self.max_iterations if max_iterations is None \
            else int(max_iterations)
        if n < 3:
            return None, np.zeros((n,), dtype=bool)
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        best_plane, best_mask, best_count = None, None, -1
        for _ in range(max(1, iters)):
            idx = rng.choice(n, size=3, replace=False)
            plane = self._plane_from_points(pts[idx[0]], pts[idx[1]], pts[idx[2]])
            if plane is None:
                continue
            dist = np.abs(pts @ plane[:3] + plane[3])
            mask = dist <= thr
            # Reject near-vertical planes (ground must face mostly upward).
            if abs(float(plane[2])) < 0.5:
                continue
            count = int(mask.sum())
            if count > best_count:
                best_count = count
                best_plane, best_mask = plane, mask
        if best_plane is None:
            return None, np.zeros((n,), dtype=bool)
        return best_plane, best_mask

    def grid_ground_mask(self, points: np.ndarray,
                         grid_cell_size: Optional[float] = None,
                         distance_threshold: Optional[float] = None) -> np.ndarray:
        """Height/elevation grid method: near cell-minimum => ground."""
        pts = np.asarray(points, dtype=float)
        n = pts.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=bool)
        cs = self.grid_cell_size if grid_cell_size is None \
            else float(grid_cell_size)
        thr = self.distance_threshold if distance_threshold is None \
            else float(distance_threshold)
        cs = max(cs, 1e-6)
        ix = np.floor((pts[:, 0] - pts[:, 0].min()) / cs).astype(np.int64)
        iy = np.floor((pts[:, 1] - pts[:, 1].min()) / cs).astype(np.int64)
        cells = np.stack([ix, iy], axis=1)
        _, inv = np.unique(cells, axis=0, return_inverse=True)
        zmin = np.full(int(inv.max()) + 1, np.inf)
        np.minimum.at(zmin, inv, pts[:, 2])
        return (pts[:, 2] - zmin[inv]) <= thr

    def detect(self, points: np.ndarray,
               method: Optional[str] = None,
               distance_threshold: Optional[float] = None,
               max_iterations: Optional[int] = None,
               grid_cell_size: Optional[float] = None,
               seed: Optional[int] = None) -> GroundResult:
        """Split ``(N, 3)`` points into ground / non-ground + mask."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        meth = (self.method if method is None else str(method)).lower()
        if pts.shape[0] == 0:
            empty = pts.copy()
            return GroundResult(empty, empty,
                                np.zeros((0,), dtype=bool), None, meth)
        if meth == "grid":
            mask = self.grid_ground_mask(pts, grid_cell_size, distance_threshold)
            plane = None
        else:  # ransac (default)
            plane, mask = self._ransac_plane(pts, distance_threshold,
                                             max_iterations, seed)
            mask = np.asarray(mask, dtype=bool)
        return GroundResult(pts[mask], pts[~mask], mask, plane, meth)

    def detect_frame(self, frame: LidarFrame, **kwargs) -> Tuple[LidarFrame, GroundResult]:
        """Segment a preprocessed :class:`LidarFrame`; labels ride along."""
        res = self.detect(np.asarray(frame.points, dtype=float), **kwargs)
        return frame, res


def detect_ground(points: np.ndarray,
                  config: Optional[dict] = None,
                  config_path: Optional[str] = None,
                  **kwargs) -> GroundResult:
    """Functional shortcut: ``GroundDetector(...).detect(...)``."""
    return GroundDetector(config=config, config_path=config_path).detect(
        points, **kwargs)
