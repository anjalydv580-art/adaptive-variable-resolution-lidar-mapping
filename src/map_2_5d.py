"""2.5D elevation / occupancy grid map (NumPy, CPU-only).

Phase 10: builds a deterministic 2.5D grid map from ONE preprocessed
LiDAR point cloud. X/Y space is divided into configurable square grid
cells anchored at a fixed origin. Per cell the builder computes real
statistics from the points that fall inside it:

- elevation (mean height), minimum height, maximum height
- height variance / std / range
- point count, occupancy flag, occupancy ratio (global)
- mean intensity and mean XYZ centroid (when available)

Empty cells are handled safely: counts are 0, occupancy is False, and
all height/intensity/centroid layers hold NaN there. No randomness and
no placeholder metrics are used: every value is an order-independent
reduction (count, min, max, mean, var) over the cell members, and cell
assignment uses ``floor`` indexing with a stable sort.

The layer stack preserves everything later adaptive mapping and
visualization need (elevation + occupancy + density + variance).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .lidar_simulator import LidarFrame
from .utils import load_config


@dataclass
class ElevationMap:
    """Dense 2.5D grid map for one frame."""

    resolution: float
    origin: np.ndarray              # (2,) [x0, y0]
    shape: tuple                    # (ny, nx)
    elevation: np.ndarray           # (ny, nx) mean height, NaN where empty
    elevation_min: np.ndarray       # (ny, nx) min height, NaN where empty
    elevation_max: np.ndarray       # (ny, nx) max height, NaN where empty
    elevation_var: np.ndarray       # (ny, nx) height variance, NaN where empty
    elevation_std: np.ndarray       # (ny, nx) height std, NaN where empty
    elevation_range: np.ndarray     # (ny, nx) max-min, NaN where empty
    counts: np.ndarray              # (ny, nx) int points per cell
    occupancy: np.ndarray           # (ny, nx) bool
    density: np.ndarray             # (ny, nx) points per m^2
    centroid: np.ndarray            # (ny, nx, 3) mean xyz, NaN where empty
    intensity: np.ndarray           # (ny, nx) mean intensity, NaN where empty
    num_input_points: int = 0
    num_occupied_cells: int = 0
    frame_id: int = -1

    def to_dict(self) -> dict:
        occ = np.asarray(self.occupancy)
        return {
            "resolution": float(self.resolution),
            "origin": [float(v) for v in np.asarray(self.origin).ravel()],
            "shape": [int(v) for v in self.shape],
            "num_input_points": int(self.num_input_points),
            "num_occupied_cells": int(self.num_occupied_cells),
            "occupancy_ratio": float(np.mean(occ)) if occ.size else 0.0,
            "frame_id": int(self.frame_id),
        }



@dataclass
class MapStats:
    """Basic statistics for one 2.5D mapping run."""

    num_input_points: int = 0
    num_occupied_cells: int = 0
    resolution: float = 0.0
    occupancy_ratio: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "num_input_points": int(self.num_input_points),
            "num_occupied_cells": int(self.num_occupied_cells),
            "resolution": float(self.resolution),
            "occupancy_ratio": float(self.occupancy_ratio),
        }
        d.update({k: float(v) for k, v in self.extra.items()})
        return d


class ElevationMapper:
    """Configurable-resolution 2.5D elevation/occupancy mapper."""

    def __init__(self, config: Optional[dict] = None,
                 config_path: Optional[str] = None) -> None:
        cfg = config if config is not None else load_config(config_path)
        self.config = cfg
        m = cfg.get("map", {})
        self.resolution = float(m.get("fixed_resolution", 0.5))
        self.x_min = float(m.get("x_min", -50.0))
        self.x_max = float(m.get("x_max", 50.0))
        self.y_min = float(m.get("y_min", -50.0))
        self.y_max = float(m.get("y_max", 50.0))

    def grid_shape(self, resolution: Optional[float] = None) -> tuple:
        """Return (ny, nx) for a resolution (config default if None)."""
        res = self.resolution if resolution is None else float(resolution)
        if res <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        nx = max(1, int(np.ceil((self.x_max - self.x_min) / res)))
        ny = max(1, int(np.ceil((self.y_max - self.y_min) / res)))
        return (ny, nx)
        return (ny, nx)

    def build(self, points: np.ndarray,
              intensities: Optional[np.ndarray] = None,
              resolution: Optional[float] = None,
              frame_id: int = -1) -> tuple:
        """Build the 2.5D map from preprocessed (N, 3) points."""
        res = self.resolution if resolution is None else float(resolution)
        if res <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        n = int(pts.shape[0])
        if intensities is None:
            inten = np.full(n, np.nan)
        else:
            inten = np.asarray(intensities, dtype=float).ravel()
            if inten.shape[0] != n:
                raise ValueError("intensities length must match points")
        ny, nx = self.grid_shape(res)
        if n == 0:
            empty = np.full((ny, nx), np.nan)
            emap = ElevationMap(res, np.array([self.x_min, self.y_min]),
                                (ny, nx), empty.copy(), empty.copy(),
                                empty.copy(), empty.copy(), empty.copy(),
                                empty.copy(),
                                np.zeros((ny, nx), dtype=int),
                                np.zeros((ny, nx), dtype=bool),
                                np.zeros((ny, nx), dtype=float),
                                np.full((ny, nx, 3), np.nan), empty.copy(),
                                0, 0, int(frame_id))
            return emap, MapStats(0, 0, res, 0.0)
        ix = np.floor((pts[:, 0] - self.x_min) / res).astype(np.int64)
        iy = np.floor((pts[:, 1] - self.y_min) / res).astype(np.int64)
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        cell = iy * nx + ix
        order = np.argsort(np.where(valid, cell, -1), kind="stable")
        valid_s = valid[order]
        idx_s = order[valid_s]
        cell_s = cell[idx_s]
        uniq, start, cnt = np.unique(cell_s, return_index=True,
                                     return_counts=True)
        ncell = int(uniq.shape[0])
        jy = (uniq // nx).astype(int)
        jx = (uniq % nx).astype(int)
        elevation = np.full((ny, nx), np.nan)
        emin = np.full((ny, nx), np.nan)
        emax = np.full((ny, nx), np.nan)
        evar = np.full((ny, nx), np.nan)
        estd = np.full((ny, nx), np.nan)
        erange = np.full((ny, nx), np.nan)
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
            estd[j, i] = float(z.std())
            erange[j, i] = float(z.max() - z.min())
            centroid[j, i] = cloud.mean(axis=0)
            ii = inten[members]
            ii = ii[np.isfinite(ii)]
            if ii.size:
                intens[j, i] = float(ii.mean())
        occupancy = counts > 0
        density = counts.astype(float) / (res * res)
        occ_ratio = float(occupancy.mean()) if occupancy.size else 0.0
        emap = ElevationMap(res, np.array([self.x_min, self.y_min]),
                            (ny, nx), elevation, emin, emax, evar, estd,
                            erange, counts, occupancy, density, centroid,
                            intens, n, ncell, int(frame_id))
        n_valid = int(valid.sum())
        stats = MapStats(n, ncell, res, occ_ratio,
                         {"num_occupied": float(ncell),
                          "grid_ny": float(ny), "grid_nx": float(nx),
                          "mean_points_per_cell": float(n_valid / ncell)
                          if ncell else 0.0})
        return emap, stats
        return emap, stats

    def build_frame(self, frame: LidarFrame,
                    resolution: Optional[float] = None) -> tuple:
        """Build the 2.5D map from a preprocessed LidarFrame."""
        return self.build(np.asarray(frame.points, dtype=float),
                          intensities=np.asarray(frame.intensities,
                                                 dtype=float),
                          resolution=resolution,
                          frame_id=int(getattr(frame, "frame_id", -1)))


def build_elevation_map(points: np.ndarray,
                        config: Optional[dict] = None,
                        config_path: Optional[str] = None,
                        **kwargs) -> tuple:
    """Functional shortcut: ``ElevationMapper(...).build(...)``."""
    return ElevationMapper(config=config,
                           config_path=config_path).build(points, **kwargs)


