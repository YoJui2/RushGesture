"""Pruebas de utilidades de la interfaz que no necesitan abrir una ventana."""

from __future__ import annotations

from gesture_rush.ui import format_ms


def test_format_ms_uses_milliseconds():
    assert format_ms(None) == "—"
    assert format_ms(0) == "0 ms"
    assert format_ms(67) == "67 ms"
    assert format_ms(1234) == "1234 ms"


def test_format_ms_rounds_decimals():
    assert format_ms(999.6) == "1000 ms"
