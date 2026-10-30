"""Pruebas del bucle musical sintetizado (no necesitan dispositivo de audio)."""

from __future__ import annotations

import numpy as np

from gesture_rush.audio import BEAT, MELODY, build_loop


def test_build_loop_returns_int16_stereo():
    sample_rate = 22050
    pcm = build_loop(sample_rate, 2)
    total_beats = sum(beats for _, beats in MELODY)
    assert pcm.dtype == np.int16
    assert pcm.shape == (int(round(total_beats * BEAT * sample_rate)), 2)
    assert int(np.abs(pcm).max()) < 32767


def test_mono_loop_is_one_dimensional():
    pcm = build_loop(8000, 1)
    assert pcm.ndim == 1
