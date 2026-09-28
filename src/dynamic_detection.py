"""Multi-frame dynamic object detection (NumPy, CPU-only).

Consumes per-frame :class:`src.object_detection.DetectedObject` lists from
consecutive LiDAR frames (see :mod:`src.preprocessing`,
:mod:`src.ground_detection`, :mod:`src.object_detection`) and detects actual
movement by comparing object centroids across frames with greedy nearest
neighbour matching. No random labels are used: an object is dynamic only if
its measured centroid displacement exceeds ``min_displacement``.

All thresholds come from the ``dynamic_detection`` section of
``config/settings.yaml``; any parameter may be overridden per call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

@dataclass
class DynamicObject:
    """One cross-frame object association with measured motion."""

    object_id: int            # current-frame object_id (-1 if unmatched/new)
    prev_object_id: int       # previous-frame object_id (-1 if unmatched)
    prev_centroid: np.ndarray  # (3,)
    curr_centroid: np.ndarray  # (3,)
    displacement: float        # meters, Euclidean centroid distance
    speed: float               # m/s = displacement / frame_dt
    is_dynamic: bool           # displacement >= min_displacement
    status: str = "static"     # "dynamic" | "static" | "new" | "lost"
    category: str = "unknown"

    @property
    def is_static(self) -> bool:
        return not self.is_dynamic

    def to_dict(self) -> dict:
        return {"object_id": int(self.object_id),
                "prev_object_id": int(self.prev_object_id),
                "prev_centroid": [float(v) for v in self.prev_centroid],
                "curr_centroid": [float(v) for v in self.curr_centroid],
                "displacement": float(self.displacement),
                "speed": float(self.speed),
                "is_dynamic": bool(self.is_dynamic),
                "status": str(self.status),
                "category": str(self.category)}


from .object_detection import DetectedObject  # noqa: E402
from .utils import load_config  # noqa: E402


class DynamicDetector:
    """Greedy centroid matcher + displacement threshold classifier."""

    def __init__(self, config: Optional[dict] = None,
                 config_path: Optional[str] = None) -> None:
        cfg = config if config is not None else load_config(config_path)
        self.config = cfg
        dd = cfg.get("dynamic_detection", {})
        self.match_distance_threshold = float(
            dd.get("match_distance_threshold", 1.5))
        self.min_displacement = float(dd.get("min_displacement", 0.15))
        self.frame_dt = float(dd.get("frame_dt", 0.1))

    def match_and_classify(
            self, prev_objects: Sequence[DetectedObject],
            curr_objects: Sequence[DetectedObject],
            match_distance_threshold: Optional[float] = None,
            min_displacement: Optional[float] = None,
            frame_dt: Optional[float] = None) -> List[DynamicObject]:
        """Match current objects to previous ones; measure real motion.

        Greedy global-nearest matching on 3D centroid distance: pairs are
        taken in ascending distance order while <= match threshold. Matched
        displacement = centroid distance; unmatched current => new object
        (displacement 0, static); unmatched previous => lost entry
        (object_id -1, displacement 0, static).
        """
        prev_list = list(prev_objects)
        curr_list = list(curr_objects)
        mth = self.match_distance_threshold if match_distance_threshold is None \
            else float(match_distance_threshold)
        mdp = self.min_displacement if min_displacement is None \
            else float(min_displacement)
        dt = self.frame_dt if frame_dt is None else float(frame_dt)
        dt = dt if dt > 0 else 1.0

        pairs = []  # (dist, prev_idx, curr_idx)
        for i, p in enumerate(prev_list):
            pc = np.asarray(p.centroid, dtype=float)
            for j, c in enumerate(curr_list):
                cc = np.asarray(c.centroid, dtype=float)
                pairs.append((float(np.linalg.norm(cc - pc)), i, j))
        pairs.sort(key=lambda t: t[0])
        used_p: set = set()
        used_c: set = set()
        match_c2p: Dict[int, int] = {}
        for dist, i, j in pairs:
            if dist > mth or i in used_p or j in used_c:
                continue
            used_p.add(i)
            used_c.add(j)
            match_c2p[j] = i

        results: List[DynamicObject] = []
        for j, c in enumerate(curr_list):
            cc = np.asarray(c.centroid, dtype=float)
            if j in match_c2p:
                p = prev_list[match_c2p[j]]
                pc = np.asarray(p.centroid, dtype=float)
                disp = float(np.linalg.norm(cc - pc))
                dyn = bool(disp >= mdp)
                results.append(DynamicObject(
                    object_id=int(c.object_id), prev_object_id=int(p.object_id),
                    prev_centroid=pc.copy(), curr_centroid=cc.copy(),
                    displacement=disp, speed=disp / dt, is_dynamic=dyn,
                    status="dynamic" if dyn else "static",
                    category=str(getattr(c, "category", "unknown"))))
            else:
                results.append(DynamicObject(
                    object_id=int(c.object_id), prev_object_id=-1,
                    prev_centroid=cc.copy(), curr_centroid=cc.copy(),
                    displacement=0.0, speed=0.0, is_dynamic=False,
                    status="new",
                    category=str(getattr(c, "category", "unknown"))))
        for i, p in enumerate(prev_list):
            if i not in used_p:
                pc = np.asarray(p.centroid, dtype=float)
                results.append(DynamicObject(
                    object_id=-1, prev_object_id=int(p.object_id),
                    prev_centroid=pc.copy(), curr_centroid=pc.copy(),
                    displacement=0.0, speed=0.0, is_dynamic=False,
                    status="lost",
                    category=str(getattr(p, "category", "unknown"))))
        return results

    # Alias with the name some callers expect.
    def detect(self, prev_objects: Sequence[DetectedObject],
               curr_objects: Sequence[DetectedObject],
               **kwargs) -> List[DynamicObject]:
        return self.match_and_classify(prev_objects, curr_objects, **kwargs)

    def dynamic_objects(self, tracks: Sequence[DynamicObject]) -> List[DynamicObject]:
        return [t for t in tracks if t.is_dynamic]

    def static_objects(self, tracks: Sequence[DynamicObject]) -> List[DynamicObject]:
        return [t for t in tracks if not t.is_dynamic]


def detect_dynamics(prev_objects: Sequence[DetectedObject],
                    curr_objects: Sequence[DetectedObject],
                    config: Optional[dict] = None,
                    config_path: Optional[str] = None,
                    **kwargs) -> List[DynamicObject]:
    """Functional shortcut: ``DynamicDetector(...).match_and_classify(...)``."""
    return DynamicDetector(config=config, config_path=config_path
                           ).match_and_classify(prev_objects, curr_objects, **kwargs)



