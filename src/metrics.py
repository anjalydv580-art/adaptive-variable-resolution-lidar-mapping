"""Adaptive vs fixed map evaluation metrics (NumPy, CPU-only).

Phase 12: compares real map outputs from the adaptive mapper
(:class:`src.adaptive_mapper.AdaptiveMap`) and the fixed-resolution
baseline (:class:`src.fixed_mapper.FixedMap` / Phase 10
:class:`src.map_2_5d.ElevationMap`).

Metric definitions (all deterministic, real measurements)
---------------------------------------------------------
- ``occupied_cell_agreement``: fraction of union-occupied finest cells
  where both maps agree on occupied/empty (1.0 when the union is empty).
- ``elevation_mae`` / ``elevation_rmse``: mean / root-mean-square absolute
  error of the adaptive mean-height layer vs the fixed mean-height layer,
  over finest cells where BOTH maps have finite elevation.
- ``elevation_bias``: mean signed error (adaptive - fixed) on the same cells.
- ``map_coverage``: per map, occupied finest cells / total finest cells.
- ``num_cells_*``: occupied cell counts per map.
- ``compression_ratio``: ``num_cells_adaptive / num_cells_fixed``
  (NaN when the fixed map has zero occupied cells).
- ``cell_reduction``: ``1 - compression_ratio`` (NaN when undefined).
- ``processing_time_*``: caller-supplied wall-clock build seconds per map
  (NaN when not provided); ``speedup`` = fixed / adaptive.

The adaptive map already lives on its finest grid, so every comparison is
done there: the fixed map is resampled onto the adaptive finest grid with
deterministic nearest-neighbour indexing (no interpolation, no randomness).
Empty maps and zero denominators are handled safely (agreement = 1.0 on
empty union, errors = 0.0 with zero overlapping cells, ratios = NaN when
their denominator is zero).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np



@dataclass
class ComparisonMetrics:
    """Structured adaptive-vs-fixed comparison result."""

    occupied_cell_agreement: float = 1.0
    elevation_mae: float = 0.0
    elevation_rmse: float = 0.0
    elevation_bias: float = 0.0
    num_overlap_cells: int = 0
    coverage_adaptive: float = 0.0
    coverage_fixed: float = 0.0
    num_cells_adaptive: int = 0
    num_cells_fixed: int = 0
    compression_ratio: float = float("nan")
    cell_reduction: float = float("nan")
    processing_time_adaptive: float = float("nan")
    processing_time_fixed: float = float("nan")
    speedup: float = float("nan")
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "occupied_cell_agreement": float(self.occupied_cell_agreement),
            "elevation_mae": float(self.elevation_mae),
            "elevation_rmse": float(self.elevation_rmse),
            "elevation_bias": float(self.elevation_bias),
            "num_overlap_cells": int(self.num_overlap_cells),
            "coverage_adaptive": float(self.coverage_adaptive),
            "coverage_fixed": float(self.coverage_fixed),
            "num_cells_adaptive": int(self.num_cells_adaptive),
            "num_cells_fixed": int(self.num_cells_fixed),
            "compression_ratio": float(self.compression_ratio),
            "cell_reduction": float(self.cell_reduction),
            "processing_time_adaptive": float(self.processing_time_adaptive),
            "processing_time_fixed": float(self.processing_time_fixed),
            "speedup": float(self.speedup),
        }
        d.update({k: float(v) for k, v in self.extra.items()})
        return d


def _alayers(amap) -> tuple:
    """Extract (elevation, occupancy, origin, resolution, (ny, nx))."""
    elev = np.asarray(amap.elevation, dtype=float)
    occ_raw = getattr(amap, "occupancy", None)
    if occ_raw is None:
        occ_raw = getattr(amap, "occupied", np.zeros((0, 0)))
    occ = np.asarray(occ_raw, dtype=bool)
    origin = np.asarray(amap.origin, dtype=float).ravel()
    res = getattr(amap, "fine_resolution", None)
    if res is None:
        res = getattr(amap, "resolution", None)
    if res is None:
        res = getattr(amap, "voxel_size", 0.5)
    shape = tuple(getattr(amap, "shape", elev.shape))
    return elev, occ, origin, float(res), (int(shape[0]), int(shape[1]))


def _layers(fixed) -> tuple:
    """Extract (elevation, occupied, origin, resolution, shape)."""
    elev = np.asarray(fixed.elevation, dtype=float)
    occ_raw = getattr(fixed, "occupied", None)
    if occ_raw is None:
        occ_raw = getattr(fixed, "occupancy", np.zeros((0, 0)))
    occ = np.asarray(occ_raw, dtype=bool)
    origin = np.asarray(fixed.origin, dtype=float).ravel()
    vs = getattr(fixed, "voxel_size", None)
    if vs is None:
        vs = getattr(fixed, "resolution", 0.5)
    shape = tuple(getattr(fixed, "shape", elev.shape))
    ny_f, nx_f = (int(shape[0]), int(shape[1]))
    return elev, occ, origin, float(vs), (ny_f, nx_f)


def _resample_to_adaptive(fixed, amap) -> tuple:
    """Nearest-neighbour resample of a fixed map onto the adaptive grid."""
    fe, fo, fo_origin, fres, fshape = _layers(fixed)
    _, _, a_origin, a_res, (ny, nx) = _alayers(amap)
    if fe.size == 0 or ny == 0 or nx == 0 or fres <= 0 or a_res <= 0:
        return (np.full((ny, nx), np.nan), np.zeros((ny, nx), dtype=bool))
    fy, fx = int(fshape[0]), int(fshape[1])
    cx = a_origin[0] + (np.arange(nx) + 0.5) * a_res
    cy = a_origin[1] + (np.arange(ny) + 0.5) * a_res
    jx = np.floor((cx - float(fo_origin[0])) / fres).astype(np.int64)
    jy = np.floor((cy - float(fo_origin[1])) / fres).astype(np.int64)
    okx = (jx >= 0) & (jx < fx)
    oky = (jy >= 0) & (jy < fy)
    e_out = np.full((ny, nx), np.nan)
    o_out = np.zeros((ny, nx), dtype=bool)
    if okx.any() and oky.any():
        e_out[np.ix_(oky, okx)] = fe[np.ix_(jy[oky], jx[okx])]
        o_out[np.ix_(oky, okx)] = fo[np.ix_(jy[oky], jx[okx])]
    return e_out, o_out



class MapComparator:
    """Deterministic adaptive-vs-fixed map comparator."""

    def compare(self, adaptive, fixed,
                processing_time_adaptive: Optional[float] = None,
                processing_time_fixed: Optional[float] = None) -> ComparisonMetrics:
        """Compare one adaptive map against one fixed map."""
        a_elev, a_occ, a_origin, a_res, (ny, nx) = _alayers(adaptive)
        if a_elev.shape != (ny, nx):
            a_elev = np.full((ny, nx), np.nan)
        if a_occ.shape != (ny, nx):
            a_occ = np.zeros((ny, nx), dtype=bool)
        f_elev, f_occ = _resample_to_adaptive(fixed, adaptive)
        union = a_occ | f_occ
        n_union = int(union.sum())
        if n_union == 0:
            agreement = 1.0
        else:
            agreement = float((a_occ[union] == f_occ[union]).mean())
        both = np.isfinite(a_elev) & np.isfinite(f_elev)
        n_both = int(both.sum())
        if n_both == 0:
            mae = rmse = bias = 0.0
        else:
            err = a_elev[both] - f_elev[both]
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            bias = float(np.mean(err))
        total = ny * nx
        n_a = int(a_occ.sum())
        n_f = int(f_occ.sum())
        cov_a = float(n_a / total) if total else 0.0
        cov_f = float(n_f / total) if total else 0.0
        if n_f == 0:
            cr = float("nan")
            red = float("nan")
        else:
            cr = float(n_a / n_f)
            red = float(1.0 - cr)
        t_a = (float(processing_time_adaptive)
               if processing_time_adaptive is not None else float("nan"))
        t_f = (float(processing_time_fixed)
               if processing_time_fixed is not None else float("nan"))
        if np.isfinite(t_a) and np.isfinite(t_f) and t_a > 0:
            speedup = float(t_f / t_a)
        else:
            speedup = float("nan")
        extra = {"num_union_cells": float(n_union),
                 "num_total_cells": float(total),
                 "fine_resolution": float(a_res)}
        return ComparisonMetrics(agreement, mae, rmse, bias, n_both,
                                 cov_a, cov_f, n_a, n_f, cr, red,
                                 t_a, t_f, speedup, extra)

    def compare_batch(self, pairs, **kwargs) -> list:
        """Compare a sequence of (adaptive, fixed) pairs in order."""
        return [self.compare(a, f, **kwargs) for a, f in pairs]


def compare_maps(adaptive, fixed, **kwargs) -> ComparisonMetrics:
    """Functional shortcut: ``MapComparator().compare(adaptive, fixed)``."""
    return MapComparator().compare(adaptive, fixed, **kwargs)

