"""Adaptive variable-resolution controller (NumPy, CPU-only).

Consumes Phase 7 region features (:class:`src.feature_extraction.FrameFeatures`
or plain dicts) and computes a deterministic complexity score in [0, 1] as a
weighted sum of min-max normalised features:

    score = sum(w_f * clip(x_f / scale_f, 0, 1)) / sum(w)

with weights/scales from the ``resolution_controller`` section of
``config/settings.yaml`` (any parameter may be overridden per call).
The score maps to LOW / MEDIUM / HIGH via ``complexity_thresholds`` and each
level maps to a configurable voxel size (``levels``).

No randomness is used: identical inputs always give identical outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .feature_extraction import FEATURE_NAMES, FrameFeatures
from .utils import load_config

LEVELS: Tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")

# The five weighted features required by the Phase 8 spec. ``occupancy`` from
# Phase 7 is intentionally unweighted (kept for diagnostics/Phase 9+).
WEIGHTED_FEATURES: Tuple[str, ...] = (
    "point_density",
    "object_density",
    "dynamic_activity",
    "elevation_variance",
    "temporal_change",
)


@dataclass
class ResolutionDecision:
    """Deterministic controller output for one region / frame."""

    complexity_score: float
    resolution_level: str  # "LOW" | "MEDIUM" | "HIGH"
    selected_voxel_size: float
    normalized_features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "complexity_score": float(self.complexity_score),
            "resolution_level": str(self.resolution_level),
            "selected_voxel_size": float(self.selected_voxel_size),
            "normalized_features": {k: float(v)
                                    for k, v in self.normalized_features.items()},
        }


def _as_feature_dict(features: Union[FrameFeatures, Mapping[str, float]]) -> Dict[str, float]:
    if isinstance(features, FrameFeatures):
        return features.to_dict()
    return {str(k): float(v) for k, v in dict(features).items()}


def _clean(value: float) -> float:
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        return 0.0
    return v



class ResolutionController:
    """Deterministic complexity -> LOW/MEDIUM/HIGH voxel-size controller."""

    def __init__(self, config: Optional[dict] = None,
                 config_path: Optional[str] = None) -> None:
        cfg = config if config is not None else load_config(config_path)
        self.config = cfg
        rc = cfg.get("resolution_controller", {})
        w = rc.get("weights", {})
        n = rc.get("normalization", {})
        self.weights: Dict[str, float] = {
            f: float(w.get(f, 0.2)) for f in WEIGHTED_FEATURES}
        self.scales: Dict[str, float] = {
            f: float(n.get(f, 1.0)) for f in WEIGHTED_FEATURES}
        th = rc.get("complexity_thresholds", {})
        self.low_max = float(th.get("low_max", 0.33))
        self.medium_max = float(th.get("medium_max", 0.66))
        lv = rc.get("levels", {})
        self.voxel_sizes: Dict[str, float] = {
            "LOW": float(lv.get("low", {}).get("cell_size", 2.0)),
            "MEDIUM": float(lv.get("medium", {}).get("cell_size", 1.0)),
            "HIGH": float(lv.get("high", {}).get("cell_size", 0.5)),
        }

    def complexity_score(
        self,
        features: Union[FrameFeatures, Mapping[str, float]],
        weights: Optional[Mapping[str, float]] = None,
        scales: Optional[Mapping[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Weighted normalised complexity score in [0, 1] + normalised feats."""
        fdict = _as_feature_dict(features)
        w = {f: float(weights[f]) if weights and f in weights
             else self.weights[f] for f in WEIGHTED_FEATURES}
        s = {f: float(scales[f]) if scales and f in scales
             else self.scales[f] for f in WEIGHTED_FEATURES}
        norm: Dict[str, float] = {}
        for f in WEIGHTED_FEATURES:
            scale = s[f] if s[f] > 0 else 1.0
            norm[f] = min(max(_clean(fdict.get(f, 0.0)) / scale, 0.0), 1.0)
        wsum = sum(max(v, 0.0) for v in w.values())
        if wsum <= 0.0:
            return 0.0, norm
        score = sum(max(w[f], 0.0) * norm[f] for f in WEIGHTED_FEATURES) / wsum
        return float(min(max(score, 0.0), 1.0)), norm

    def level_for_score(self, score: float) -> str:
        """Map a score in [0, 1] to LOW / MEDIUM / HIGH (deterministic)."""
        v = min(max(float(score), 0.0), 1.0)
        if v <= self.low_max:
            return "LOW"
        if v <= self.medium_max:
            return "MEDIUM"
        return "HIGH"

    def decide(
        self,
        features: Union[FrameFeatures, Mapping[str, float]],
        weights: Optional[Mapping[str, float]] = None,
        scales: Optional[Mapping[str, float]] = None,
        low_max: Optional[float] = None,
        medium_max: Optional[float] = None,
    ) -> ResolutionDecision:
        """Full pipeline: features -> (score, level, voxel size)."""
        lo = self.low_max if low_max is None else float(low_max)
        hi = self.medium_max if medium_max is None else float(medium_max)
        score, norm = self.complexity_score(features, weights, scales)
        v = min(max(score, 0.0), 1.0)
        if v <= lo:
            level = "LOW"
        elif v <= hi:
            level = "MEDIUM"
        else:
            level = "HIGH"
        return ResolutionDecision(
            complexity_score=v, resolution_level=level,
            selected_voxel_size=float(self.voxel_sizes[level]),
            normalized_features=dict(norm))

    def decide_batch(
        self, feats: Sequence[Union[FrameFeatures, Mapping[str, float]]],
        **kwargs,
    ) -> list:
        """Deterministic per-region decisions preserving input order."""
        return [self.decide(f, **kwargs) for f in feats]


def select_resolution(features: Union[FrameFeatures, Mapping[str, float]],
                      config: Optional[dict] = None,
                      config_path: Optional[str] = None,
                      **kwargs) -> ResolutionDecision:
    """Functional shortcut: ``ResolutionController(...).decide(...)``."""
    return ResolutionController(config=config, config_path=config_path
                                ).decide(features, **kwargs)
