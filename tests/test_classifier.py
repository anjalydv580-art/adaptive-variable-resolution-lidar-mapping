"""Phase 13 tests: ML complexity classifier."""
import numpy as np

from src.complexity_classifier import (
    CLASS_LABELS,
    MODEL_FEATURES,
    ComplexityClassifier,
    build_training_dataset,
    train_default_classifier,
)


def test_feature_set_and_labels_cover_spec():
    assert tuple(MODEL_FEATURES) == ("point_density", "object_density",
                                     "dynamic_activity", "elevation_variance",
                                     "temporal_change", "occupancy")
    assert tuple(CLASS_LABELS) == ("SIMPLE", "MODERATE", "COMPLEX")


def test_config_matches_settings_yaml():
    from src.utils import load_config
    cfg = load_config()
    clf = ComplexityClassifier(config=cfg)
    rf = cfg["ml_classifier"]["random_forest"]
    assert clf.n_estimators == int(rf["n_estimators"])
    assert clf.max_depth == int(rf["max_depth"])
    assert clf.min_samples_leaf == int(rf["min_samples_leaf"])
    assert clf.random_state == int(rf["random_state"]) == 42
    rc = cfg["resolution_controller"]["complexity_thresholds"]
    assert clf.low_max == float(rc["low_max"])
    assert clf.medium_max == float(rc["medium_max"])
    assert clf.model_path == str(cfg["ml_classifier"]["model_path"])


def test_training_data_real_and_deterministic():
    X1, y1, _ = build_training_dataset(num_frames=3)
    X2, y2, _ = build_training_dataset(num_frames=3)
    assert X1.shape[1] == 6 and len(y1) == X1.shape[0] > 0
    assert np.array_equal(X1, X2) and list(y1) == list(y2)
    assert set(y1).issubset(set(CLASS_LABELS))
    assert np.all(np.isfinite(X1)) and np.all(X1 >= 0.0)


def test_train_predict_evaluate_save_load(tmp_path):
    X, y, _ = build_training_dataset(num_frames=4)
    clf = ComplexityClassifier(n_estimators=20, max_depth=6,
                               min_samples_leaf=1, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset(set(CLASS_LABELS))
    proba = clf.predict_proba(X[:5])
    assert proba.shape == (5, len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    rep = clf.evaluate(X, y)
    d = rep.to_dict()
    assert 0.0 <= d["accuracy"] <= 1.0
    assert 0.0 <= d["precision_macro"] <= 1.0
    assert 0.0 <= d["recall_macro"] <= 1.0
    assert 0.0 <= d["f1_macro"] <= 1.0
    assert np.asarray(d["confusion_matrix"]).shape == (3, 3)
    assert d["num_samples"] == len(y)
    assert clf.predict_features([]) == []
    dest = tmp_path / "clf.joblib"
    saved = clf.save(dest)
    assert saved.exists()
    back = ComplexityClassifier.load(saved)
    assert back.predict(X) == preds


def test_empty_dataset_raises_and_default_train_saves(tmp_path, monkeypatch):
    clf = ComplexityClassifier()
    try:
        clf.fit(np.zeros((0, 6)), [])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    import src.complexity_classifier as cc
    monkeypatch.setattr(cc.ComplexityClassifier, "model_path",
                        str(tmp_path / "cc.joblib"))
    trained, rep, path = train_default_classifier(num_frames=3)
    assert path.exists()
    assert trained.is_fitted
    assert rep.num_samples > 0
