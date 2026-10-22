"""Pruebas de dataset: generacion sintetica, validacion, featurizacion y split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gesture_rush import config, dataset, features, synthetic_data


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return synthetic_data.generate(
        per_class_per_participant=20,
        participants=4,
        sessions_per_participant=2,
        seed=7,
    )


def test_synthetic_dataset_is_valid(synthetic_df):
    report = dataset.validate_dataset(synthetic_df)
    assert report["n_samples"] == len(synthetic_df)
    assert set(report["class_counts"]) == set(config.GESTURES)
    assert report["n_participants"] == 4


def test_featurize_shapes(synthetic_df):
    X, y, groups = dataset.featurize(synthetic_df)
    assert X.shape == (len(synthetic_df), features.feature_dim())
    assert y.shape == (len(synthetic_df),)
    assert groups.shape == (len(synthetic_df),)
    assert np.all(np.isfinite(X))


def test_validate_rejects_unknown_label(synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, "label"] = "not_a_gesture"
    with pytest.raises(ValueError):
        dataset.validate_dataset(bad)


def test_validate_rejects_nonfinite(synthetic_df):
    bad = synthetic_df.copy()
    bad.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        dataset.validate_dataset(bad)


def test_group_split_is_disjoint(synthetic_df):
    X, y, groups = dataset.featurize(synthetic_df)
    split = dataset.group_split(X, y, groups, seed=1)
    g_train = set(groups[split["train"]])
    g_val = set(groups[split["val"]])
    g_test = set(groups[split["test"]])
    assert g_train.isdisjoint(g_val)
    assert g_train.isdisjoint(g_test)
    assert g_val.isdisjoint(g_test)
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == len(X)


def test_group_split_requires_enough_groups(synthetic_df):
    X, y, groups = dataset.featurize(synthetic_df)
    groups = np.array(["a", "b"] * (len(groups) // 2) + ["a"] * (len(groups) % 2))
    with pytest.raises(ValueError):
        dataset.group_split(X, y, groups)


def test_class_balance_report(synthetic_df):
    balance = dataset.check_class_balance(synthetic_df, min_per_class=10)
    assert balance["insufficient"] == {}


def test_synthetic_samples_are_separable(synthetic_df):
    """Sanity check: las clases sinteticas no son todas identicas."""
    X, y, _ = dataset.featurize(synthetic_df)
    means = {lab: X[y == lab].mean(axis=0) for lab in config.GESTURES}
    assert not np.allclose(means["fist"], means["open_palm"])
    assert not np.allclose(means["victory"], means["pointing_index"])


def test_synthetic_generator_covers_every_gesture():
    """Agregar una clase exige actualizarla en config y en el generador."""
    assert set(synthetic_data.GESTURE_EXTENSIONS) == set(config.GESTURES)


def test_every_gesture_has_spanish_label():
    assert set(config.GESTURE_LABELS_ES) == set(config.GESTURES)
