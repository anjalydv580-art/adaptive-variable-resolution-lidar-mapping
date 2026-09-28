"""End-to-end adaptive-resolution 2.5D mapping pipeline (Phase 14)."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from .adaptive_mapper import AdaptiveMapper
from .complexity_classifier import CLASS_LABELS, ComplexityClassifier, build_training_dataset
from .dynamic_detection import DynamicDetector
from .feature_extraction import FeatureExtractor, FrameFeatures
from .fixed_mapper import FixedMapper
from .ground_detection import GroundDetector
from .lidar_simulator import LidarSimulator
from .metrics import ComparisonMetrics, MapComparator
from .object_detection import ObjectDetector
from .preprocessing import Preprocessor
from .resolution_controller import ResolutionController
from .utils import load_config, resolve_path
__all__ = ["FrameResult", "PipelineResult", "AdaptiveMappingPipeline", "run_pipeline"]


@dataclass
class FrameResult:
    """Structured per-frame output (dashboard / benchmark friendly)."""

    frame_id: int
    num_points_raw: int = 0
    num_points_processed: int = 0
    num_ground: int = 0
    num_non_ground: int = 0
    objects: List[dict] = field(default_factory=list)
    dynamics: List[dict] = field(default_factory=list)
    num_dynamic: int = 0
    features: Dict[str, float] = field(default_factory=dict)
    complexity_score: float = 0.0
    complexity_level: str = "LOW"
    voxel_size: float = 2.0
    ml_prediction: str = "SIMPLE"
    ml_rule_label: str = "SIMPLE"

    def to_dict(self) -> dict:
        return {"frame_id": int(self.frame_id),
                "num_points_raw": int(self.num_points_raw),
                "num_points_processed": int(self.num_points_processed),
                "num_ground": int(self.num_ground),
                "num_non_ground": int(self.num_non_ground),
                "num_objects": len(self.objects),
                "objects": [dict(o) for o in self.objects],
                "dynamics": [dict(d) for d in self.dynamics],
                "num_dynamic": int(self.num_dynamic),
                "features": {k: float(v) for k, v in self.features.items()},
                "complexity_score": float(self.complexity_score),
                "complexity_level": str(self.complexity_level),
                "voxel_size": float(self.voxel_size),
                "ml_prediction": str(self.ml_prediction),
                "ml_rule_label": str(self.ml_rule_label)}


@dataclass
class PipelineResult:
    """Structured whole-pipeline output for dashboard and benchmark."""

    num_frames: int = 0
    seed: Optional[int] = None
    frames: List[FrameResult] = field(default_factory=list)
    adaptive_map: object = None
    fixed_map: object = None
    adaptive_stats: Dict[str, float] = field(default_factory=dict)
    fixed_stats: Dict[str, float] = field(default_factory=dict)
    region_decisions: List[dict] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    ml_predictions: List[str] = field(default_factory=list)
    ml_rule_labels: List[str] = field(default_factory=list)
    ml_evaluation: Dict[str, object] = field(default_factory=dict)
    ml_model_source: str = "unknown"
    summary: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"num_frames": int(self.num_frames), "seed": self.seed,
                "frames": [f.to_dict() for f in self.frames],
                "adaptive_stats": {k: float(v) for k, v in self.adaptive_stats.items()},
                "fixed_stats": {k: float(v) for k, v in self.fixed_stats.items()},
                "region_decisions": [dict(r) for r in self.region_decisions],
                "metrics": {k: float(v) for k, v in self.metrics.items()},
                "ml_predictions": [str(v) for v in self.ml_predictions],
                "ml_rule_labels": [str(v) for v in self.ml_rule_labels],
                "ml_evaluation": dict(self.ml_evaluation),
                "ml_model_source": str(self.ml_model_source),
                "summary": dict(self.summary)}


def _rule_self_evaluation(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    """Honest finite evaluation when predictions follow the rule by design."""
    idx = {lab: i for i, lab in enumerate(CLASS_LABELS)}
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    precs, recs, f1s = [], [], []
    for i in range(3):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        precs.append(pr)
        recs.append(rc)
        f1s.append(f1)
    n = len(y_true)
    acc = float(sum(1 for t, p in zip(y_true, y_pred) if t == p) / n) if n else 0.0
    return {"accuracy": acc, "precision_macro": float(np.mean(precs)),
            "recall_macro": float(np.mean(recs)),
            "f1_macro": float(np.mean(f1s)),
            "confusion_matrix": cm.tolist(), "num_samples": n,
            "classes": list(CLASS_LABELS)}


def _empty_evaluation() -> Dict[str, object]:
    return {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0,
            "f1_macro": 0.0,
            "confusion_matrix": np.zeros((3, 3), dtype=int).tolist(),
            "num_samples": 0, "classes": list(CLASS_LABELS)}


class AdaptiveMappingPipeline:
    """End-to-end pipeline connecting all Phase 2-13 modules (no logic dup)."""

    def __init__(self, config: Optional[dict] = None,
                 config_path: Optional[str] = None,
                 seed: Optional[int] = None) -> None:
        cfg = config if config is not None else load_config(config_path)
        self.config = cfg
        proj_seed = cfg.get("project", {}).get("seed", 42)
        self.seed = seed if seed is not None else proj_seed
        self.simulator = LidarSimulator(config=cfg, seed=self.seed)
        self.preprocessor = Preprocessor(config=cfg)
        self.ground = GroundDetector(config=cfg)
        self.objects = ObjectDetector(config=cfg)
        self.dynamics = DynamicDetector(config=cfg)
        self.extractor = FeatureExtractor(config=cfg)
        self.controller = ResolutionController(config=cfg)
        self.adaptive = AdaptiveMapper(config=cfg)
        self.fixed = FixedMapper(config=cfg)
        self.comparator = MapComparator()
        self.classifier = ComplexityClassifier(config=cfg)

    def ensure_classifier(self) -> str:
        """Load saved model if present, else train small in-memory model."""
        cfg_path = self.config.get("ml_classifier", {}).get(
            "model_path", "models/complexity_classifier.joblib")
        try:
            if resolve_path(cfg_path).exists():
                self.classifier = ComplexityClassifier.load(config=self.config)
                return "loaded"
        except Exception:
            pass
        try:
            X, y, _ = build_training_dataset(config=self.config, num_frames=2)
            if X.shape[0] == 0 or len(y) == 0:
                raise ValueError("empty training dataset")
            # Lightweight but real training (deterministic seed from config);
            # keeps the end-to-end tests fast on CPU while the saved-test
            # model path still uses full configured hyperparameters.
            self.classifier = ComplexityClassifier(
                config=self.config, n_estimators=50)
            self.classifier.fit(X, y)
            return "trained"
        except Exception:
            self.classifier = ComplexityClassifier(config=self.config)
            return "rule_fallback"

    def run(self, num_frames: Optional[int] = None) -> PipelineResult:
        """Process ``num_frames`` sequential frames; build both maps."""
        if num_frames is None:
            num_frames = int(
                self.config.get("simulation", {}).get("num_frames", 3))
        try:
            n_frames = int(num_frames)
        except (TypeError, ValueError):
            raise ValueError(
                f"num_frames must be an integer, got {num_frames!r}")
        if n_frames < 0:
            raise ValueError(f"num_frames must be >= 0, got {n_frames}")
        model_source = self.ensure_classifier()
        frames: List[FrameResult] = []
        prev_objs: list = []
        prev_pts: Optional[np.ndarray] = None
        acc_pts: List[np.ndarray] = []
        acc_inten: List[np.ndarray] = []
        frame_feats: List[FrameFeatures] = []
        for fid in range(n_frames):
            raw = self.simulator.simulate_frame(fid)
            n_raw = int(np.asarray(raw.points, dtype=float).shape[0])
            proc, _ = self.preprocessor.preprocess_frame(raw)
            pts = np.asarray(proc.points, dtype=float).reshape(-1, 3)
            _, gres = self.ground.detect_frame(proc)
            mask = np.asarray(gres.ground_mask, dtype=bool)
            objs = self.objects.detect_frame(proc, ground_mask=mask)
            tracks = [] if fid == 0 else self.dynamics.match_and_classify(
                prev_objs, objs)
            feats = self.extractor.extract_frame_features(
                pts, objects=objs, dynamics=tracks, prev_points=prev_pts)
            decision = self.controller.decide(feats)
            rule_label = self.classifier.label_features(feats)
            try:
                ml_pred = self.classifier.predict_features([feats])[0]
            except RuntimeError:
                ml_pred = rule_label
            frames.append(FrameResult(
                frame_id=int(fid), num_points_raw=n_raw,
                num_points_processed=int(pts.shape[0]),
                num_ground=int(mask.sum()), num_non_ground=int((~mask).sum()),
                objects=[o.to_dict() for o in objs],
                dynamics=[t.to_dict() for t in tracks],
                num_dynamic=int(sum(1 for t in tracks
                                    if bool(getattr(t, "is_dynamic", False)))),
                features=feats.to_dict(),
                complexity_score=float(decision.complexity_score),
                complexity_level=str(decision.resolution_level),
                voxel_size=float(decision.selected_voxel_size),
                ml_prediction=str(ml_pred), ml_rule_label=str(rule_label)))
            frame_feats.append(feats)
            if pts.shape[0]:
                acc_pts.append(pts)
                inten = np.asarray(proc.intensities, dtype=float).ravel()
                acc_inten.append(inten if inten.shape[0] == pts.shape[0]
                                else np.full(pts.shape[0], np.nan))
            prev_objs = objs
            prev_pts = pts
        if acc_pts:
            all_pts = np.concatenate(acc_pts, axis=0)
            all_inten = np.concatenate(acc_inten, axis=0)
        else:
            all_pts = np.zeros((0, 3), dtype=float)
            all_inten = np.zeros((0,), dtype=float)
        last_fid = int(n_frames - 1) if n_frames else -1
        t0 = time.perf_counter()
        adaptive_map, adaptive_stats, regions = self.adaptive.build(
            all_pts, intensities=all_inten, frame_id=last_fid)
        t_adaptive = time.perf_counter() - t0
        t0 = time.perf_counter()
        fixed_map, fixed_stats = self.fixed.build(
            all_pts, intensities=all_inten, frame_id=last_fid)
        t_fixed = time.perf_counter() - t0
        cmp_metrics: ComparisonMetrics = self.comparator.compare(
            adaptive_map, fixed_map,
            processing_time_adaptive=t_adaptive,
            processing_time_fixed=t_fixed)
        ml_preds = [f.ml_prediction for f in frames]
        ml_rule = [f.ml_rule_label for f in frames]
        if frame_feats and self.classifier.is_fitted:
            X_all = np.array([f.to_vector() for f in frame_feats], dtype=float)
            ml_eval = self.classifier.evaluate(X_all, ml_rule).to_dict()
            ml_eval["confusion_matrix"] = np.asarray(
                ml_eval.get("confusion_matrix")).tolist()
        elif frame_feats:
            ml_eval = _rule_self_evaluation(ml_rule, ml_preds)
        else:
            ml_eval = _empty_evaluation()
        metrics_d = {k: float(v) for k, v in cmp_metrics.to_dict().items()}
        adaptive_d = {k: float(v) for k, v in adaptive_stats.to_dict().items()}
        fixed_d = {k: float(v) for k, v in fixed_stats.to_dict().items()}
        level_counts = {lv: sum(1 for f in frames if f.complexity_level == lv)
                        for lv in ("LOW", "MEDIUM", "HIGH")}
        summary = {"num_frames": int(n_frames), "seed": self.seed,
                   "total_points_raw": int(sum(f.num_points_raw for f in frames)),
                   "total_points_processed": int(sum(
                       f.num_points_processed for f in frames)),
                   "total_objects": int(sum(len(f.objects) for f in frames)),
                   "total_dynamic": int(sum(f.num_dynamic for f in frames)),
                   "complexity_levels": level_counts,
                   "ml_predictions": list(ml_preds),
                   "ml_accuracy_vs_rule": ml_eval.get("accuracy", 0.0),
                   "ml_model_source": str(model_source),
                   "adaptive_stats": dict(adaptive_d),
                   "fixed_stats": dict(fixed_d), "metrics": dict(metrics_d)}
        return PipelineResult(
            num_frames=int(n_frames), seed=self.seed, frames=frames,
            adaptive_map=adaptive_map, fixed_map=fixed_map,
            adaptive_stats=adaptive_d, fixed_stats=fixed_d,
            region_decisions=[] if int(n_frames) == 0 else [r.to_dict() for r in regions],
            metrics=metrics_d, ml_predictions=ml_preds, ml_rule_labels=ml_rule,
            ml_evaluation=ml_eval, ml_model_source=str(model_source),
            summary=summary)


def run_pipeline(num_frames: Optional[int] = 3,
                 config: Optional[dict] = None,
                 config_path: Optional[str] = None,
                 seed: Optional[int] = None) -> PipelineResult:
    """Functional shortcut: build the pipeline and run it."""
    return AdaptiveMappingPipeline(config=config, config_path=config_path,
                                   seed=seed).run(num_frames=num_frames)


def default_result_path(filename: str = "pipeline_result.npz") -> Path:
    """Resolve the default output path for saved pipeline artefacts."""
    return resolve_path(Path("outputs") / "reports" / filename)
