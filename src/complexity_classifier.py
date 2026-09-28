"""ML complexity classifier (scikit-learn, CPU-only).

Phase 13: classifies deterministic Phase 7 region-feature vectors into
``SIMPLE`` / ``MODERATE`` / ``COMPLEX`` with a fixed-seed
``RandomForestClassifier``.

Label-generation rule
---------------------
Training labels are NOT hand-made or random. Each training sample is a
real Phase 7 feature vector extracted from the deterministic LiDAR
simulator + preprocessing pipeline (see :func:`build_training_dataset`),
and its label comes from the Phase 8 :class:`ResolutionController`
complexity score via the configured thresholds:

- score <= ``resolution_controller.complexity_thresholds.low_max`` -> SIMPLE
- score <= ``medium_max`` -> MODERATE
- otherwise -> COMPLEX

So the classifier learns to mimic the controller's LOW/MEDIUM/HIGH bands
(renamed SIMPLE/MODERATE/COMPLEX for the ML API) directly from feature
space. The ``occupancy`` feature is included as a sixth model input even
though the controller does not weight it.

Limitations
-----------
- Labels are controller-derived, so the model cannot outperform the
  controller's rule; it generalises that rule to new feature vectors.
- Training data covers the simulator's scene distribution only; scenes
  far outside the simulated density/variance ranges may classify poorly.
- Small datasets (< ~10 samples or single-class) train a valid but weak
  model; empty datasets raise ``ValueError``.

API (for Phase 14 pipeline / dashboard)
---------------------------------------
- ``ComplexityClassifier``: ``fit`` / ``predict`` / ``predict_proba`` /
  ``evaluate`` (accuracy, precision, recall, F1, confusion matrix) /
  ``save`` / ``load``.
- ``build_training_dataset(...)``: deterministic real-pipeline dataset.
- ``train_default_classifier(...)``: one-call train + save under ``models/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .feature_extraction import FEATURE_NAMES, FeatureExtractor, FrameFeatures
from .resolution_controller import ResolutionController
from .utils import ensure_dir, load_config, resolve_path

CLASS_LABELS: Tuple[str, ...] = ("SIMPLE", "MODERATE", "COMPLEX")
MODEL_FEATURES: Tuple[str, ...] = tuple(FEATURE_NAMES)

_LEVEL_TO_CLASS = {"LOW": "SIMPLE", "MEDIUM": "MODERATE", "HIGH": "COMPLEX"}
_CLASS_TO_LEVEL = {v: k for k, v in _LEVEL_TO_CLASS.items()}
_LABEL_TO_INDEX = {lab: i for i, lab in enumerate(CLASS_LABELS)}



def _label_for_score(score: float, low_max: float, medium_max: float) -> str:
    v = min(max(float(score), 0.0), 1.0)
    if v <= low_max:
        return "SIMPLE"
    if v <= medium_max:
        return "MODERATE"
    return "COMPLEX"


def _row_to_vector(row: Union[FrameFeatures, Mapping[str, float]]) -> np.ndarray:
    if isinstance(row, FrameFeatures):
        d = row.to_dict()
    else:
        d = {str(k): float(v) for k, v in dict(row).items()}
    vals = []
    for name in MODEL_FEATURES:
        v = float(d.get(name, 0.0))
        vals.append(0.0 if not np.isfinite(v) or v < 0.0 else v)
    return np.array(vals, dtype=float)


@dataclass
class ClassificationReport:
    """Structured evaluation output (dashboard/benchmark friendly)."""

    accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    confusion_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=int))
    labels: Tuple[str, ...] = CLASS_LABELS
    num_samples: int = 0
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accuracy": float(self.accuracy),
            "precision_macro": float(self.precision_macro),
            "recall_macro": float(self.recall_macro),
            "f1_macro": float(self.f1_macro),
            "confusion_matrix": np.asarray(
                self.confusion_matrix, dtype=int).tolist(),
            "labels": list(self.labels),
            "num_samples": int(self.num_samples),
            "per_class": {k: {kk: float(vv) for kk, vv in v.items()}
                          for k, v in self.per_class.items()},
        }



class ComplexityClassifier:
    """Deterministic RandomForest SIMPLE/MODERATE/COMPLEX classifier."""

    # Class-level override hook (used by tests via
    # ``monkeypatch.setattr(ComplexityClassifier, "model_path", ...)``).
    # When set to a ``str``/``Path`` it takes precedence over the config
    # value for instances created afterwards. ``None`` means "use config".
    model_path: Optional[str] = None

    def __init__(self, config: Optional[dict] = None,
                 config_path: Optional[str] = None,
                 n_estimators: Optional[int] = None,
                 max_depth: Optional[int] = None,
                 min_samples_leaf: Optional[int] = None,
                 random_state: Optional[int] = None) -> None:
        cfg = config if config is not None else load_config(config_path)
        self.config = cfg
        ml = cfg.get("ml_classifier", {})
        rf = ml.get("random_forest", {})
        self.n_estimators = int(n_estimators if n_estimators is not None
                                else rf.get("n_estimators", 200))
        md = max_depth if max_depth is not None else rf.get("max_depth", 12)
        self.max_depth = int(md) if md is not None else None
        self.min_samples_leaf = int(min_samples_leaf if min_samples_leaf
                                    is not None
                                    else rf.get("min_samples_leaf", 2))
        self.random_state = int(random_state if random_state is not None
                                else rf.get("random_state", 42))
        _override = getattr(type(self), "model_path", None)
        # ``type(self).model_path`` is the class-level default (None) unless
        # a test monkeypatches it to a tmp path; instance dict must not mask
        # that check, so inspect the class attribute explicitly. An instance
        # attribute set on a previously created object must not leak either.
        if isinstance(_override, (str, Path)):
            self.model_path = str(_override)
        else:
            self.model_path = str(ml.get("model_path",
                                         "models/complexity_classifier.joblib"))
        rc_cfg = cfg.get("resolution_controller", {}).get(
            "complexity_thresholds", {})
        self.low_max = float(rc_cfg.get("low_max", 0.33))
        self.medium_max = float(rc_cfg.get("medium_max", 0.66))
        self.controller = ResolutionController(config=cfg)
        self.model = None
        self.classes_: Tuple[str, ...] = CLASS_LABELS

    def _make_model(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=self.n_estimators,
                                      max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf,
                                      random_state=self.random_state,
                                      n_jobs=1)

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def label_features(self, row: Union[FrameFeatures, Mapping[str, float]],
                       low_max: Optional[float] = None,
                       medium_max: Optional[float] = None) -> str:
        """Controller-derived label for one feature vector."""
        lo = self.low_max if low_max is None else float(low_max)
        hi = self.medium_max if medium_max is None else float(medium_max)
        score, _ = self.controller.complexity_score(row)
        return _label_for_score(score, lo, hi)

    def fit(self, X: np.ndarray, y: Sequence[str]):
        """Train on a feature matrix + label sequence."""
        X = np.asarray(X, dtype=float)
        labels = [str(v).upper() for v in list(y)]
        if X.shape[0] == 0 or len(labels) == 0:
            raise ValueError("Cannot train on an empty dataset.")
        if X.shape[0] != len(labels):
            raise ValueError("X and y lengths differ.")
        for lab in labels:
            if lab not in CLASS_LABELS:
                raise ValueError(f"Unknown label {lab!r}; "
                                 f"expected one of {CLASS_LABELS}.")
        self.model = self._make_model()
        self.model.fit(X, labels)
        self.classes_ = tuple(str(c) for c in self.model.classes_)
        return self

    def fit_features(self, rows: Sequence[Union[FrameFeatures, Mapping[str, float]]],
                     labels: Optional[Sequence[str]] = None):
        """Train from feature rows; labels via controller rule by default."""
        X = np.array([_row_to_vector(r) for r in rows], dtype=float)
        if labels is None:
            y = [self.label_features(r) for r in rows]
        else:
            y = list(labels)
        return self.fit(X, y)

    def predict(self, X: np.ndarray) -> List[str]:
        """Predict class labels for a feature matrix."""
        if self.model is None:
            raise RuntimeError("Model not trained/loaded; call fit() or load().")
        X = np.asarray(X, dtype=float).reshape(-1, len(MODEL_FEATURES))
        if X.shape[0] == 0:
            return []
        return [str(v) for v in self.model.predict(X)]

    def predict_features(self, rows: Sequence[Union[FrameFeatures, Mapping[str, float]]]) -> List[str]:
        """Predict from feature rows (FrameFeatures or dicts)."""
        if len(rows) == 0:
            return []
        X = np.array([_row_to_vector(r) for r in rows], dtype=float)
        return self.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Per-class probabilities (columns follow classes_)."""
        if self.model is None:
            raise RuntimeError("Model not trained/loaded; call fit() or load().")
        X = np.asarray(X, dtype=float).reshape(-1, len(MODEL_FEATURES))
        if X.shape[0] == 0:
            return np.zeros((0, len(self.classes_)), dtype=float)
        return np.asarray(self.model.predict_proba(X), dtype=float)

    def evaluate(self, X: np.ndarray, y_true: Sequence[str]) -> ClassificationReport:
        """Accuracy / macro precision / recall / F1 + confusion matrix."""
        y_true = [str(v).upper() for v in list(y_true)]
        y_pred = self.predict(X)
        n = len(y_true)
        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[_LABEL_TO_INDEX[t], _LABEL_TO_INDEX[p]] += 1
        per_class: Dict[str, Dict[str, float]] = {}
        precisions, recalls, f1s = [], [], []
        for lab, i in _LABEL_TO_INDEX.items():
            tp = float(cm[i, i])
            fp = float(cm[:, i].sum() - tp)
            fn = float(cm[i, :].sum() - tp)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class[lab] = {"precision": prec, "recall": rec, "f1": f1,
                              "support": float(cm[i, :].sum())}
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        if n:
            acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
        else:
            acc = 0.0
        return ClassificationReport(acc, float(np.mean(precisions)),
                                    float(np.mean(recalls)),
                                    float(np.mean(f1s)), cm, CLASS_LABELS, n,
                                    per_class)

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist the trained model (+ thresholds) with joblib."""
        if self.model is None:
            raise RuntimeError("Nothing to save; model is not trained.")
        import joblib
        dest = resolve_path(path or self.model_path)
        ensure_dir(dest.parent)
        joblib.dump({"model": self.model,
                     "classes": list(self.classes_),
                     "features": list(MODEL_FEATURES),
                     "low_max": float(self.low_max),
                     "medium_max": float(self.medium_max),
                     "params": {"n_estimators": self.n_estimators,
                                "max_depth": self.max_depth,
                                "min_samples_leaf": self.min_samples_leaf,
                                "random_state": self.random_state}}, dest)
        return dest

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None,
             config: Optional[dict] = None,
             config_path: Optional[str] = None) -> "ComplexityClassifier":
        """Load a model saved with :meth:`save`."""
        import joblib
        cfg = config if config is not None else load_config(config_path)
        ml = cfg.get("ml_classifier", {})
        src = resolve_path(path or ml.get("model_path",
                                          "models/complexity_classifier.joblib"))
        payload = joblib.load(src)
        params = payload.get("params", {})
        obj = cls(config=cfg,
                  n_estimators=params.get("n_estimators"),
                  max_depth=params.get("max_depth"),
                  min_samples_leaf=params.get("min_samples_leaf"),
                  random_state=params.get("random_state"))
        obj.model = payload["model"]
        obj.classes_ = tuple(str(c) for c in payload.get("classes", CLASS_LABELS))
        obj.low_max = float(payload.get("low_max", obj.low_max))
        obj.medium_max = float(payload.get("medium_max", obj.medium_max))
        return obj


def build_training_dataset(config: Optional[dict] = None,
                           config_path: Optional[str] = None,
                           num_frames: int = 6,
                           seed: Optional[int] = None,
                           region_size: Optional[float] = None) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Build a deterministic training set from the real pipeline.

    Simulates ``num_frames`` frames (seeded ``LidarSimulator``), preprocesses
    each frame, tiles it into ``region_size`` regions and extracts real
    Phase 7 features per region; labels come from the Phase 8 controller
    thresholds. Returns ``(X, y, meta)`` with ``X`` shaped
    ``(n_regions_total, 6)`` in :data:`MODEL_FEATURES` order.
    """
    from .lidar_simulator import LidarSimulator
    from .preprocessing import Preprocessor
    cfg = config if config is not None else load_config(config_path)
    sim_seed = seed if seed is not None else cfg.get("project", {}).get("seed", 42)
    sim = LidarSimulator(seed=sim_seed)
    pre = Preprocessor(config=cfg)
    ext = FeatureExtractor(config=cfg)
    clf = ComplexityClassifier(config=cfg)
    rs = float(region_size if region_size is not None
               else cfg.get("feature_extraction", {}).get("region_size", 10.0))
    m = cfg.get("map", {})
    x_min, x_max = float(m.get("x_min", -50.0)), float(m.get("x_max", 50.0))
    y_min, y_max = float(m.get("y_min", -50.0)), float(m.get("y_max", 50.0))
    nrx = max(1, int(np.ceil((x_max - x_min) / rs)))
    nry = max(1, int(np.ceil((y_max - y_min) / rs)))
    rows: List[np.ndarray] = []
    labels: List[str] = []
    meta: List[dict] = []
    prev_pts: Optional[np.ndarray] = None
    for fid in range(max(1, int(num_frames))):
        frame = sim.simulate_frame(fid)
        proc, _ = pre.preprocess_frame(frame)
        pts = np.asarray(proc.points, dtype=float)
        rix = np.clip(np.floor((pts[:, 0] - x_min) / rs).astype(np.int64), 0, nrx - 1)
        riy = np.clip(np.floor((pts[:, 1] - y_min) / rs).astype(np.int64), 0, nry - 1)
        for jy in range(nry):
            for jx in range(nrx):
                sel = (rix == jx) & (riy == jy)
                cell_pts = pts[sel]
                if cell_pts.shape[0] == 0:
                    continue
                feats = ext.extract_frame_features(cell_pts, prev_points=prev_pts)
                vec = _row_to_vector(feats)
                rows.append(vec)
                labels.append(clf.label_features(feats))
                meta.append({"frame_id": int(fid), "region": [int(jx), int(jy)],
                             "num_points": int(sel.sum())})
        prev_pts = pts
    if not rows:
        return np.zeros((0, len(MODEL_FEATURES))), [], []
    return np.array(rows, dtype=float), labels, meta


def train_default_classifier(config: Optional[dict] = None,
                             config_path: Optional[str] = None,
                             **kwargs) -> Tuple["ComplexityClassifier", ClassificationReport, Path]:
    """Train on the deterministic pipeline dataset and save under models/.

    Uses a deterministic stratified split (every k-th sample per class to
    test, with ``train_test_split`` ratio from config) so results reproduce
    exactly. Returns ``(classifier, test_report, model_path)``.
    """
    cfg = config if config is not None else load_config(config_path)
    X, y, _ = build_training_dataset(config=cfg, **kwargs)
    if X.shape[0] == 0:
        raise ValueError("Training dataset is empty; cannot train.")
    ratio = float(cfg.get("ml_classifier", {}).get("train_test_split", 0.25))
    ratio = min(max(ratio, 0.0), 0.5)
    test_idx: List[int] = []
    if ratio > 0.0:
        step = max(2, int(round(1.0 / ratio)))
        by_class: Dict[str, List[int]] = {}
        for i, lab in enumerate(y):
            by_class.setdefault(lab, []).append(i)
        for idxs in by_class.values():
            test_idx.extend(idxs[::step])
    test_set = set(test_idx)
    train_idx = [i for i in range(len(y)) if i not in test_set]
    if not train_idx or not test_idx:
        # Fall back to train/test on all data (deterministic, documented).
        train_idx, test_idx = list(range(len(y))), list(range(len(y)))
    clf = ComplexityClassifier(config=cfg)
    clf.fit(X[train_idx], [y[i] for i in train_idx])
    report = clf.evaluate(X[test_idx], [y[i] for i in test_idx])
    path = clf.save()
    return clf, report, path

