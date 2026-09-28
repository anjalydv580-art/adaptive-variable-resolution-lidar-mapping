"""Fixed-resolution baseline mapper (NumPy, CPU-only).

Phase 9 baseline: builds a uniform 2.5D grid map from ONE preprocessed
LiDAR point cloud using ONE fixed voxel/cell size for the entire frame.

- The fixed voxel size comes from ``config/settings.yaml`` (``map.fixed_resolution``).
- The adaptive resolution controller is intentionally NOT used here.
- Deterministic: points are assigned to cells with ``floor`` indexing on a
  fixed origin; per-cell statistics use order-independent reductions
  (count, min, max, mean, var). Ties are impossible by construction.
- Point/height information is preserved per cell: point count, min/max/mean
  height, height variance, mean intensity (when provided) and the centroid
  of points inside the cell, so later 2.5D mapping can consume this output.

Returns a :class:`FixedMap` plus basic statistics (input point count,
output cell count, voxel size).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .lidar_simulator import LidarFrame
from .utils import load_config


@dataclass
class FixedMap:
    """Uniform fixed-resolution 2.5D grid map for one frame."""

    voxel_size: float
    origin: np.ndarray            # (2,) [x0, y0] of the grid
    shape: tuple                  # (ny, nx)
    elevation: np.ndarray         # (ny, nx) mean height, NaN where empty
    elevation_min: np.ndarray     # (ny, nx) min height, NaN where empty
    elevation_max: np.ndarray     # (ny, nx) max height, NaN where empty
    elevation_var: np.ndarray     # (ny, nx) height variance, NaN where empty
    counts: np.ndarray            # (ny, nx) int points per cell
    occupied: np.ndarray          # (ny, nx) bool
    centroid: np.ndarray          # (ny, nx, 3) mean xyz per cell, NaN where empty
    intensity: np.ndarray         # (ny, nx) mean intensity, NaN where empty
    num_input_points: int = 0
    num_output_cells: int = 0
    frame_id: int = -1

    def to_dict(self) -> dict:
        occ = self.occupied
        return {
            "voxel_size": float(self.voxel_size),
            "origin": [float(v) for v in np.asarray(self.origin).ravel()],
            "shape": [int(v) for v in self.shape],
            "num_input_points": int(self.num_input_points),
            "num_output_cells": int(self.num_output_cells),
            "occupancy_ratio": float(np.mean(occ)) if occ.size else 0.0,
            "frame_id": int(self.frame_id),
        }


@dataclass
class FixedMapStats:
    """Basic statistics for one fixed-resolution mapping run."""

    num_input_points: int = 0
    num_output_cells: int = 0
    voxel_size: float = 0.0
    occupancy_ratio: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "num_input_points": int(self.num_input_points),
            "num_output_cells": int(self.num_output_cells),
            "voxel_size": float(self.voxel_size),
            "occupancy_ratio": float(self.occupancy_ratio),
        }
        d.update({k: float(v) for k, v in self.extra.items()})
        return d



class FixedMapper:
    """Uniform voxel-grid mapper using ONE fixed resolution."""

    def __init__(self, config: Optional[dict] = None,
                 config_path: Optional[str] = None) -> None:
        cfg = config if config is not None else load_config(config_path)
        self.config = cfg
        m = cfg.get("map", {})
        self.voxel_size = float(m.get("fixed_resolution", 0.5))
        self.x_min = float(m.get("x_min", -50.0))
        self.x_max = float(m.get("x_max", 50.0))
        self.y_min = float(m.get("y_min", -50.0))
        self.y_max = float(m.get("y_max", 50.0))

    def build(self, points: np.ndarray,
              intensities: Optional[np.ndarray] = None,
              voxel_size: Optional[float] = None,
              frame_id: int = -1) -> tuple:
        """Build a fixed-resolution map from preprocessed (N, 3) points."""
        vs = self.voxel_size if voxel_size is None else float(voxel_size)
        if vs <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        n = int(pts.shape[0])
        if intensities is None:
            inten = np.full(n, np.nan)
        else:
            inten = np.asarray(intensities, dtype=float).ravel()
            if inten.shape[0] != n:
                raise ValueError("intensities length must match points")
        if n == 0:
            empty2 = np.zeros((0, 0), dtype=float)
            fmap = FixedMap(vs, np.array([self.x_min, self.y_min]),
                            (0, 0), empty2.copy(), empty2.copy(),
                            empty2.copy(), empty2.copy(),
                            np.zeros((0, 0), dtype=int),
                            np.zeros((0, 0), dtype=bool),
                            np.zeros((0, 0, 3)), empty2.copy(), 0, 0, frame_id)
            return fmap, FixedMapStats(0, 0, vs, 0.0)
        nx = max(1, int(np.ceil((self.x_max - self.x_min) / vs)))
        ny = max(1, int(np.ceil((self.y_max - self.y_min) / vs)))
        ix = np.floor((pts[:, 0] - self.x_min) / vs).astype(np.int64)
        iy = np.floor((pts[:, 1] - self.y_min) / vs).astype(np.int64)
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy = ix[valid], iy[valid]
        cell = iy * nx + ix
        order = np.argsort(cell, kind="stable")
        cell_s = cell[order]
        idx_s = np.where(valid)[0][order]
        uniq, start, cnt = np.unique(cell_s, return_index=True,
                                     return_counts=True)
        ncell = int(uniq.shape[0])
        jy = (uniq // nx).astype(int)
        jx = (uniq % nx).astype(int)
        elevation = np.full((ny, nx), np.nan)
        emin = np.full((ny, nx), np.nan)
        emax = np.full((ny, nx), np.nan)
        evar = np.full((ny, nx), np.nan)
        counts = np.zeros((ny, nx), dtype=int)
        centroid = np.full((ny, nx, 3), np.nan)
        intens = np.full((ny, nx), np.nan)
        for k in range(ncell):
            s = int(start[k])
            e = s + int(cnt[k])
            members = idx_s[s:e]
            cloud = pts[members]
            j, i = int(jy[k]), int(jx[k])
            z = cloud[:, 2]
            counts[j, i] = int(members.shape[0])
            emin[j, i] = float(z.min())
            emax[j, i] = float(z.max())
            elevation[j, i] = float(z.mean())
            evar[j, i] = float(z.var())
            centroid[j, i] = cloud.mean(axis=0)
            ii = inten[members]
            ii = ii[np.isfinite(ii)]
            if ii.size:
                intens[j, i] = float(ii.mean())
        occupied = counts > 0
        occ_ratio = float(occupied.mean()) if occupied.size else 0.0
        fmap = FixedMap(vs, np.array([self.x_min, self.y_min]), (ny, nx),
                        elevation, emin, emax, evar, counts, occupied,
                        centroid, intens, n, ncell, int(frame_id))
        stats = FixedMapStats(n, ncell, vs, occ_ratio,
                              {"num_occupied": float(ncell),
                               "grid_ny": float(ny), "grid_nx": float(nx)})
        return fmap, stats

    def build_frame(self, frame: LidarFrame,
                    voxel_size: Optional[float] = None) -> tuple:
        """Build a fixed-resolution map from a preprocessed LidarFrame."""
        return self.build(np.asarray(frame.points, dtype=float),
                          intensities=np.asarray(frame.intensities,
                                                 dtype=float),
                          voxel_size=voxel_size,
                          frame_id=int(getattr(frame, "frame_id", -1)))


def build_fixed_map(points: np.ndarray,
                    config: Optional[dict] = None,
                    config_path: Optional[str] = None,
                    **kwargs) -> tuple:
    """Functional shortcut: ``FixedMapper(...).build(...)``."""
    return FixedMapper(config=config,
                       config_path=config_path).build(points, **kwargs)
