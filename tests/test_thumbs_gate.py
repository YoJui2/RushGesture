"""Pruebas del inicio por pulgar arriba (logica pura, sin pantalla)."""

from __future__ import annotations

from gesture_rush.ui import ThumbsUpGate, start_hint


def test_gate_requires_every_player():
    gate = ThumbsUpGate(2)

    assert gate.update([None, None], active=True) is False  # primer frame: arma
    assert gate.update(["thumbs_up", None], active=True) is False
    assert gate.ready == [True, False]

    assert gate.update(["thumbs_up", "thumbs_up"], active=True) is True
    assert gate.ready == [True, True]


def test_gate_does_not_chain_if_the_thumb_was_already_up():
    gate = ThumbsUpGate(2)

    # Nadie puede aprobar con un pulgar que ya venia levantado.
    assert gate.update(["thumbs_up", "thumbs_up"], active=True) is False
    assert gate.is_locked(0) and gate.is_locked(1)
    assert gate.update(["thumbs_up", "thumbs_up"], active=True) is False

    # Bajan el pulgar y lo vuelven a levantar: recien ahi cuenta.
    gate.update([None, None], active=True)
    assert gate.update(["thumbs_up", "thumbs_up"], active=True) is True


def test_gate_resets_when_the_screen_is_not_ready():
    gate = ThumbsUpGate(2)
    gate.update([None, None], active=True)
    gate.update(["thumbs_up", None], active=True)
    assert gate.ready[0] is True

    assert gate.update(["thumbs_up", None], active=False) is False
    assert gate.ready == [False, False]


def test_solo_player_gate():
    gate = ThumbsUpGate(1)
    gate.update([None], active=True)
    assert gate.update(["thumbs_up"], active=True) is True


def test_other_gestures_do_not_approve():
    gate = ThumbsUpGate(1)
    gate.update([None], active=True)

    assert gate.update(["victory"], active=True) is False

    assert gate.update(["thumbs_up"], active=True) is True


def test_start_hint_names_the_missing_player():
    gate = ThumbsUpGate(2)
    gate.update([None, None], active=True)
    assert start_hint(gate) == "Los dos con el pulgar arriba para empezar"

    gate.update(["thumbs_up", None], active=True)
    assert "jugador 2" in start_hint(gate)

    gate.update(["thumbs_up", "thumbs_up"], active=True)
    assert start_hint(gate) == "¡Listo!"


def test_gate_always_has_at_least_one_player():
    assert ThumbsUpGate(0).num_players == 1
