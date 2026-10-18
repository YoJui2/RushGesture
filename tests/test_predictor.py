"""Pruebas del predictor y del suavizado temporal."""

from __future__ import annotations

import numpy as np
import pytest

from gesture_rush import config, features, synthetic_data
from gesture_rush import dataset as dataset_mod
from gesture_rush.predictor import NO_HAND, Predictor, TemporalSmoother


@pytest.fixture(scope="module")
def trained_model():
    from sklearn.ensemble import RandomForestClassifier

    df = synthetic_data.generate(
        per_class_per_participant=30,
        participants=3,
        sessions_per_participant=1,
        seed=11,
    )
    X, y, _ = dataset_mod.featurize(df)
    model = RandomForestClassifier(n_estimators=60, random_state=config.RANDOM_STATE)
    model.fit(X, y)
    return model


def _predictor(model) -> Predictor:
    metadata = {
        "feature_spec": features.feature_spec(),
        "classes": [str(c) for c in model.classes_],
        "best_model": "random_forest",
    }
    return Predictor(model, metadata)


def test_predict_returns_known_class(trained_model):
    pred = _predictor(trained_model)
    rng = np.random.default_rng(0)
    landmarks = rng.normal(0, 0.5, size=(21, 3))
    label, conf = pred.predict(landmarks)
    assert label in config.GESTURES
    assert 0.0 <= conf <= 1.0


def test_predict_none_is_no_hand(trained_model):
    pred = _predictor(trained_model)
    label, conf = pred.predict(None)
    assert label == NO_HAND
    assert conf == 0.0


def test_predictor_dimension_mismatch(trained_model):
    metadata = {"feature_spec": {**features.feature_spec(), "dim": 999},
                "classes": ["a"]}
    with pytest.raises(ValueError):
        Predictor(trained_model, metadata)


def test_smoother_reaches_stability():
    smoother = TemporalSmoother(window=5, min_votes=3, confidence_threshold=0.5)
    results = [smoother.update("fist", 0.9) for _ in range(3)]
    assert results[-1] == "fist"


def test_smoother_requires_min_votes():
    smoother = TemporalSmoother(window=5, min_votes=4, confidence_threshold=0.5)
    assert smoother.update("fist", 0.9) is None
    assert smoother.update("fist", 0.9) is None


def test_smoother_rejects_low_confidence():
    smoother = TemporalSmoother(window=5, min_votes=1, confidence_threshold=0.8)
    assert smoother.update("fist", 0.5) is None


def test_smoother_oscillation_is_unstable():
    smoother = TemporalSmoother(window=6, min_votes=4, confidence_threshold=0.5)
    labels = ["fist", "open_palm"] * 6
    outs = [smoother.update(lab, 0.9) for lab in labels]
    assert outs[-1] is None


def test_smoother_reset():
    smoother = TemporalSmoother(window=3, min_votes=2, confidence_threshold=0.5)
    smoother.update("fist", 0.9)
    smoother.update("fist", 0.9)
    smoother.reset()
    assert smoother.update("fist", 0.9) is None
