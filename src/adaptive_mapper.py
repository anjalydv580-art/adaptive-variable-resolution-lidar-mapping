"""Adaptive 2.5D mapper (NumPy, CPU-only).

Phase 11: builds a variable-resolution 2.5D elevation/occupancy map from
ONE preprocessed LiDAR point cloud.

Pipeline (all deterministic, no randomness / placeholders)
----------------------------------------------------------
1. Tile X/Y space into square regions of ``feature_extraction.region_size``.
2. Per region, extract real Phase 7 features
   (:class:`src.feature_extraction.FeatureExtractor`) from the region's
   points (plus optional same-region previous-frame points for
   ``temporal_change``).
3. Score each region with the Phase 8
   (:class:`src.resolution_controller.ResolutionController`) and select a
   LOW / MEDIUM / HIGH voxel size per region.
4. Build the final adaptive 2.5D map with the Phase 10
   (:class:`src.map_2_5d.ElevationMapper`) at each region's selected
   resolution and mosaic the per-region cells into one adaptive map.

Returns the adaptive map, per-region resolution level / complexity score /
selected voxel size, and basic mapping statistics. Empty input is handled
safely (empty region grid + zero statistics).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class RegionDecision:
    """Per-region adaptive resolution decision."""

    region_id: int
    bounds: List[float]          # [x0, x1, y0, y1]
    num_points: int = 0
    complexity_score: float = 0.0
    resolution_level: str = "LOW"
    voxel_size: float = 2.0

    def to_dict(self) -> dict:
        return {
            "region_id": int(self.region_id),
            "bounds": [float(v) for v in self.bounds],
            "num_points": int(self.num_points),
            "complexity_score": float(self.complexity_score),
            "resolution_level": str(self.resolution_level),
            "voxel_size": float(self.voxel_size),
        }


@dataclass
class AdaptiveMap:
    """Mosaic adaptive 2.5D map with per-region provenance."""

    elevation: np.ndarray        # (ny, nx) mean height at finest grid
    elevation_min: np.ndarray
    elevation_max: np.ndarray
    counts: np.ndarray           # (ny, nx) int
    occupancy: np.ndarray        # (ny, nx) bool
    resolution_map: np.ndarray   # (ny, nx) voxel size used per fine cell
    level_map: np.ndarray        # (ny, nx) 0=LOW 1=MEDIUM 2=HIGH (-1 empty)
    score_map: np.ndarray        # (ny, nx) region complexity, NaN where empty
    origin: np.ndarray           # (2,) finest-grid origin
    fine_resolution: float
    shape: tuple
    regions: List[RegionDecision] = field(default_factory=list)
    num_input_points: int = 0
    num_occupied_cells: int = 0
    frame_id: int = -1

    def to_dict(self) -> dict:
        occ = np.asarray(self.occupancy)
        return {
            "fine_resolution": float(self.fine_resolution),
            "shape": [int(v) for v in self.shape],
            "num_input_points": int(self.num_input_points),
            "num_occupied_cells": int(self.num_occupied_cells),
            "occupancy_ratio": float(np.mean(occ)) if occ.size else 0.0,
            "num_regions": len(self.regions),
            "frame_id": int(self.frame_id),
        }


@dataclass
class AdaptiveMapStats:
    """Basic statistics for one adaptive mapping run."""

    num_input_points: int = 0
    num_occupied_cells: int = 0
    num_regions: int = 0
    occupancy_ratio: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "num_input_points": int(self.num_input_points),
            "num_occupied_cells": int(self.num_occupied_cells),
            "num_regions": int(self.num_regions),
            "occupancy_ratio": float(self.occupancy_ratio),
        }
        d.update({k: float(v) for k, v in self.extra.items()})
        return d


from .feature_extraction import FeatureExtractor
from .lidar_simulator import LidarFrame
from .map_2_5d import ElevationMap, ElevationMapper
from .resolution_controller import ResolutionController
from .utils import load_config



class AdaptiveMapper:
    """Region-adaptive 2.5D mapper (features -> controller -> 2.5D maps)."""

    def __init__(self, config: Optional[dict] = None,
                 config_path: Optional[str] = None,
                 extractor: Optional[FeatureExtractor] = None,
                 controller: Optional[ResolutionController] = None,
                 mapper: Optional[ElevationMapper] = None) -> None:
        cfg = config if config is not None else load_config(config_path)
        self.config = cfg
        m = cfg.get("map", {})
        self.x_min = float(m.get("x_min", -50.0))
        self.x_max = float(m.get("x_max", 50.0))
        self.y_min = float(m.get("y_min", -50.0))
        self.y_max = float(m.get("y_max", 50.0))
        fe = cfg.get("feature_extraction", {})
        self.region_size = float(fe.get("region_size", 10.0))
        self.extractor = extractor or FeatureExtractor(config=cfg)
        self.controller = controller or ResolutionController(config=cfg)
        self.mapper = mapper or ElevationMapper(config=cfg)
        self._level_code = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    def region_grid(self, region_size: Optional[float] = None) -> tuple:
        """Return (nry, nrx) region tile counts."""
        rs = self.region_size if region_size is None else float(region_size)
        if rs <= 0:
            raise ValueError(f"region_size must be positive, got {region_size}")
        nrx = max(1, int(np.ceil((self.x_max - self.x_min) / rs)))
        nry = max(1, int(np.ceil((self.y_max - self.y_min) / rs)))
        return (nry, nrx)

    def decide_regions(self, points: np.ndarray,
                       prev_points: Optional[np.ndarray] = None,
                       region_size: Optional[float] = None) -> List[RegionDecision]:
        """Extract Phase 7 features per region and decide LOW/MED/HIGH."""
        rs = self.region_size if region_size is None else float(region_size)
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        prev = (np.asarray(prev_points, dtype=float).reshape(-1, 3)
                if prev_points is not None else None)
        nry, nrx = self.region_grid(rs)
        if pts.shape[0] == 0:
            out: List[RegionDecision] = []
            rid = 0
            for jy in range(nry):
                for jx in range(nrx):
                    x0 = self.x_min + jx * rs
                    y0 = self.y_min + jy * rs
                    dec = self.controller.decide({})
                    out.append(RegionDecision(rid, [x0, x0 + rs, y0, y0 + rs],
                                              0, dec.complexity_score,
                                              dec.resolution_level,
                                              dec.selected_voxel_size))
                    rid += 1
            return out
        rix = np.clip(np.floor((pts[:, 0] - self.x_min) / rs).astype(np.int64),
                      0, nrx - 1)
        riy = np.clip(np.floor((pts[:, 1] - self.y_min) / rs).astype(np.int64),
                      0, nry - 1)
        pix = piy = None
        if prev is not None and prev.shape[0]:
            pix = np.clip(np.floor((prev[:, 0] - self.x_min) / rs).astype(np.int64),
                          0, nrx - 1)
            piy = np.clip(np.floor((prev[:, 1] - self.y_min) / rs).astype(np.int64),
                          0, nry - 1)
        out = []
        rid = 0
        for jy in range(nry):
            for jx in range(nrx):
                x0 = self.x_min + jx * rs
                y0 = self.y_min + jy * rs
                sel = (rix == jx) & (riy == jy)
                cell_pts = pts[sel]
                psel = None
                if pix is not None:
                    psel = prev[(pix == jx) & (piy == jy)]
                    if psel.shape[0] == 0:
                        psel = None
                feats = self.extractor.extract_frame_features(
                    cell_pts, prev_points=psel)
                dec = self.controller.decide(feats)
                out.append(RegionDecision(rid, [x0, x0 + rs, y0, y0 + rs],
                                          int(sel.sum()), dec.complexity_score,
                                          dec.resolution_level,
                                          dec.selected_voxel_size))
                rid += 1
        return out
        return out

    def build(self, points: np.ndarray,
              intensities: Optional[np.ndarray] = None,
              prev_points: Optional[np.ndarray] = None,
              region_size: Optional[float] = None,
              frame_id: int = -1) -> tuple:
        """Build the adaptive 2.5D map at per-region resolutions."""
        rs = self.region_size if region_size is None else float(region_size)
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        n = int(pts.shape[0])
        if intensities is None:
            inten = np.full(n, np.nan)
        else:
            inten = np.asarray(intensities, dtype=float).ravel()
            if inten.shape[0] != n:
                raise ValueError("intensities length must match points")
        regions = self.decide_regions(pts, prev_points=prev_points,
                                      region_size=rs)
        fine = min(r.voxel_size for r in regions)
        nry, nrx = self.region_grid(rs)
        finest = ElevationMapper(config={
            "map": {"x_min": self.x_min, "x_max": self.x_max,
                    "y_min": self.y_min, "y_max": self.y_max,
                    "fixed_resolution": fine}})
        ny, nx = finest.grid_shape(fine)
        elevation = np.full((ny, nx), np.nan)
        emin = np.full((ny, nx), np.nan)
        emax = np.full((ny, nx), np.nan)
        counts = np.zeros((ny, nx), dtype=int)
        occupancy = np.zeros((ny, nx), dtype=bool)
        res_map = np.full((ny, nx), np.nan)
        level_map = np.full((ny, nx), -1, dtype=int)
        score_map = np.full((ny, nx), np.nan)
        if n > 0:
            rix = np.clip(np.floor((pts[:, 0] - self.x_min) / rs).astype(np.int64),
                          0, nrx - 1)
            riy = np.clip(np.floor((pts[:, 1] - self.y_min) / rs).astype(np.int64),
                          0, nry - 1)
            for r in regions:
                jy, jx = divmod(r.region_id, nrx)
                sel = (rix == jx) & (riy == jy)
                if not bool(sel.any()):
                    continue
                sub, _ = self.mapper.build(pts[sel], intensities=inten[sel],
                                           resolution=r.voxel_size)
                x0, x1, y0, y1 = r.bounds
                gx0 = int(np.floor((x0 - self.x_min) / fine))
                gx1 = int(np.ceil((x1 - self.x_min) / fine))
                gy0 = int(np.floor((y0 - self.y_min) / fine))
                gy1 = int(np.ceil((y1 - self.y_min) / fine))
                gx0, gy0 = max(gx0, 0), max(gy0, 0)
                gx1, gy1 = min(gx1, nx), min(gy1, ny)
                if gx1 <= gx0 or gy1 <= gy0:
                    continue
                sy, sx = sub.elevation.shape
                wy, wx = gy1 - gy0, gx1 - gx0
                rep_y = max(1, int(np.ceil(wy / max(sy, 1))))
                rep_x = max(1, int(np.ceil(wx / max(sx, 1))))
                e_up = np.repeat(np.repeat(sub.elevation, rep_y, axis=0),
                                 rep_x, axis=1)[:wy, :wx]
                mn_up = np.repeat(np.repeat(sub.elevation_min, rep_y, axis=0),
                                  rep_x, axis=1)[:wy, :wx]
                mx_up = np.repeat(np.repeat(sub.elevation_max, rep_y, axis=0),
                                  rep_x, axis=1)[:wy, :wx]
                c_up = np.repeat(np.repeat(sub.counts, rep_y, axis=0),
                                 rep_x, axis=1)[:wy, :wx]
                win_e = elevation[gy0:gy1, gx0:gx1]
                elevation[gy0:gy1, gx0:gx1] = np.where(
                    np.isfinite(e_up), e_up, win_e)
                win_mn = emin[gy0:gy1, gx0:gx1]
                emin[gy0:gy1, gx0:gx1] = np.where(
                    np.isfinite(mn_up), mn_up, win_mn)
                win_mx = emax[gy0:gy1, gx0:gx1]
                emax[gy0:gy1, gx0:gx1] = np.where(
                    np.isfinite(mx_up), mx_up, win_mx)
                occ_up = c_up > 0
                win_c = counts[gy0:gy1, gx0:gx1]
                counts[gy0:gy1, gx0:gx1] = np.where(occ_up, c_up, win_c)
                occupancy[gy0:gy1, gx0:gx1] |= occ_up
                win_r = res_map[gy0:gy1, gx0:gx1]
                res_map[gy0:gy1, gx0:gx1] = np.where(
                    occ_up | ~np.isfinite(win_r), r.voxel_size, win_r)
                win_l = level_map[gy0:gy1, gx0:gx1]
                level_map[gy0:gy1, gx0:gx1] = np.where(
                    occ_up | (win_l < 0),
                    self._level_code[r.resolution_level], win_l)
                win_s = score_map[gy0:gy1, gx0:gx1]
                score_map[gy0:gy1, gx0:gx1] = np.where(
                    occ_up | ~np.isfinite(win_s), r.complexity_score, win_s)
        ncell = int(occupancy.sum())
        amap = AdaptiveMap(elevation, emin, emax, counts, occupancy, res_map,
                           level_map, score_map,
                           np.array([self.x_min, self.y_min]), fine, (ny, nx),
                           regions, n, ncell, int(frame_id))
        return amap, self._stats(amap, regions, occupancy), regions
        return amap, self._stats(amap, regions, occupancy), regions

    def _stats(self, amap: AdaptiveMap, regions: list,
               occupancy: np.ndarray) -> AdaptiveMapStats:
        counts_lvl = {lv: sum(1 for r in regions if r.resolution_level == lv)
                      for lv in ("LOW", "MEDIUM", "HIGH")}
        occ_ratio = float(np.mean(occupancy)) if occupancy.size else 0.0
        return AdaptiveMapStats(
            amap.num_input_points, amap.num_occupied_cells, len(regions),
            occ_ratio,
            {"fine_resolution": float(amap.fine_resolution),
             "regions_low": float(counts_lvl["LOW"]),
             "regions_medium": float(counts_lvl["MEDIUM"]),
             "regions_high": float(counts_lvl["HIGH"]),
             "mean_complexity": float(np.mean(
                 [r.complexity_score for r in regions])) if regions else 0.0})

    def build_frame(self, frame: LidarFrame,
                    prev_frame: Optional[LidarFrame] = None,
                    **kwargs) -> tuple:
        """Build the adaptive map from a preprocessed LidarFrame."""
        prev_pts = (np.asarray(prev_frame.points, dtype=float)
                    if prev_frame is not None else None)
        return self.build(np.asarray(frame.points, dtype=float),
                          intensities=np.asarray(frame.intensities,
                                                 dtype=float),
                          prev_points=prev_pts,
                          frame_id=int(getattr(frame, "frame_id", -1)),
                          **kwargs)


def build_adaptive_map(points: np.ndarray,
                       config: Optional[dict] = None,
                       config_path: Optional[str] = None,
                       **kwargs) -> tuple:
    """Functional shortcut: ``AdaptiveMapper(...).build(...)``."""
    return AdaptiveMapper(config=config,
                          config_path=config_path).build(points, **kwargs)


