"""Pruebas de normalizacion y dimensiones de features."""

from __future__ import annotations

import numpy as np
import pytest

from gesture_rush import config, features


def _random_landmarks(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.normal(0.0, 1.0, size=(features.N_LANDMARKS, 3))
    return pts


def test_feature_dimensions():
    assert features.feature_dim(use_z=False) == 42
    assert features.feature_dim(use_z=True) == 63
    assert len(features.feature_names(use_z=False)) == 42
    assert len(features.feature_names(use_z=True)) == 63


def test_normalize_output_shape_and_dtype():
    vec = features.normalize_landmarks(_random_landmarks())
    assert vec.shape == (42,)
    assert vec.dtype == np.float64


def test_translation_invariance():
    pts = _random_landmarks(1)
    shifted = pts + np.array([10.0, -5.0, 3.0])
    assert np.allclose(
        features.normalize_landmarks(pts), features.normalize_landmarks(shifted)
    )


def test_scale_invariance():
    pts = _random_landmarks(2)
    scaled = pts * 4.0
    assert np.allclose(
        features.normalize_landmarks(pts), features.normalize_landmarks(scaled)
    )


def test_wrist_at_origin():
    pts = _random_landmarks(3)
    vec = features.normalize_landmarks(pts).reshape(21, 2)
    assert np.allclose(vec[config.WRIST_INDEX], 0.0, atol=1e-9)


def test_validate_rejects_nan_and_bad_shape():
    pts = _random_landmarks(4)
    pts[0, 0] = np.nan
    with pytest.raises(ValueError):
        features.validate_landmarks(pts)
    with pytest.raises(ValueError):
        features.validate_landmarks(np.zeros((5, 3)))


def test_accepts_2d_landmarks():
    pts = _random_landmarks(5)[:, :2]
    vec = features.normalize_landmarks(pts)
    assert vec.shape == (42,)


def test_degenerate_palm_raises():
    pts = np.zeros((21, 3))
    with pytest.raises(ValueError):
        features.normalize_landmarks(pts)
