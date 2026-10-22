"""Pruebas del mapeo de teclas del recolector de datos."""

from __future__ import annotations

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
