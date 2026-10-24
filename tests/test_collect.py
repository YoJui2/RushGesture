"""Pruebas del recolector: mapeo de teclas, conteo y persistencia."""

from __future__ import annotations

import numpy as np

from gesture_rush import config
from gesture_rush.collect_data import gesture_index_for_key


def test_digits_select_the_first_nine_gestures():
    assert gesture_index_for_key(ord("1"), 10) == 0
    assert gesture_index_for_key(ord("9"), 10) == 8


def test_zero_selects_the_tenth_gesture():
    assert gesture_index_for_key(ord("0"), 10) == 9


def test_zero_is_ignored_with_fewer_than_ten_gestures():
    assert gesture_index_for_key(ord("0"), 5) is None


def test_digit_beyond_the_vocabulary_is_ignored():
    assert gesture_index_for_key(ord("7"), 5) is None


def test_next_and_previous_cycle_around():
    assert gesture_index_for_key(ord("n"), 10, current=9) == 0
    assert gesture_index_for_key(ord("p"), 10, current=0) == 9


def test_unrelated_keys_are_ignored():
    assert gesture_index_for_key(255, 10) is None
    assert gesture_index_for_key(ord("q"), 10) is None


def test_keyboard_reaches_every_configured_gesture():
    """Toda clase de config.GESTURES debe ser seleccionable con el teclado."""
    reachable = {
        gesture_index_for_key(ord(k), len(config.GESTURES)) for k in "1234567890"
    }
    assert reachable == set(range(len(config.GESTURES)))


def _landmarks(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.4, size=(21, 3))


def test_counts_include_existing_rows(tmp_path):
    from gesture_rush.collect_data import DataCollector

    csv = tmp_path / "landmarks.csv"
    first = DataCollector(participant="participant_01", session="s01", output_path=csv)
    assert first.add("fist", _landmarks(1))
    first.save()

    second = DataCollector(participant="participant_02", session="s02", output_path=csv)
    counts = second.counts()
    assert counts["fist"] == 1
    assert counts["open_palm"] == 0
    assert second.total == 1
    assert second.session_samples == 0


def test_counts_update_without_rebuilding(tmp_path):
    from gesture_rush.collect_data import DataCollector

    collector = DataCollector(output_path=tmp_path / "landmarks.csv")
    assert collector.add("fist", _landmarks(1))
    assert collector.add("open_palm", _landmarks(2))
    assert collector.counts()["fist"] == 1
    assert collector.counts()["open_palm"] == 1
    assert collector.session_samples == 2


def test_save_appends_and_keeps_previous_rows(tmp_path):
    import pandas as pd

    from gesture_rush.collect_data import DataCollector

    csv = tmp_path / "landmarks.csv"
    first = DataCollector(participant="participant_01", session="s01", output_path=csv)
    for i in range(3):
        first.add("fist", _landmarks(10 + i))
    first.save()

    second = DataCollector(participant="participant_02", session="s02", output_path=csv)
    for i in range(2):
        second.add("victory", _landmarks(20 + i))
    second.save()

    df = pd.read_csv(csv)
    assert len(df) == 5
    assert df["label"].value_counts().to_dict() == {"fist": 3, "victory": 2}
    assert set(df["participant"]) == {"participant_01", "participant_02"}
    assert list(df.columns) == list(
        [f"lm_{i}_{a}" for i in range(21) for a in "xyz"]
        + ["label", "participant", "session", "handedness", "source"]
    )


def test_duplicate_samples_are_dropped(tmp_path):
    from gesture_rush.collect_data import DataCollector

    collector = DataCollector(output_path=tmp_path / "landmarks.csv")
    pts = _landmarks(5)
    assert collector.add("fist", pts)
    assert not collector.add("fist", pts + 1e-4)
    assert collector.counts()["fist"] == 1
