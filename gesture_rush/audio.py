"""Musica de fondo sintetizada y control de volumen.

El juego no incluye archivos de audio: el bucle se genera con numpy al arrancar
y se reproduce en bucle infinito, asi que funciona igual en cualquier copia del
repositorio. Si la maquina no tiene dispositivo de audio, ``Music`` queda en
silencio sin romper la partida (igual que la interfaz funciona sin camara).
"""

from __future__ import annotations

import numpy as np
import pygame

SAMPLE_RATE = 44100
BPM = 120.0
BEAT = 60.0 / BPM  # 0,5 s por negra

# Melodia en La menor pentatonica: (nota MIDI, duracion en negras).
# Dos frases de dos compases; el bucle dura 16 negras (8 s a 120 BPM).
MELODY: tuple[tuple[int, float], ...] = (
    (69, 0.5), (72, 0.5), (76, 0.5), (79, 0.5),
    (76, 0.5), (72, 0.5), (74, 0.5), (76, 0.5),
    (72, 0.5), (69, 0.5), (72, 0.5), (74, 0.5),
    (72, 1.0), (69, 1.0),
    (67, 0.5), (69, 0.5), (72, 0.5), (76, 0.5),
    (74, 0.5), (72, 0.5), (69, 0.5), (72, 0.5),
    (74, 0.5), (76, 0.5), (79, 0.5), (76, 0.5),
    (72, 2.0),
)

# Bajo: una fundamental por compas (4 negras cada uno).
BASS: tuple[tuple[int, float], ...] = (
    (45, 4.0),  # La
    (41, 4.0),  # Fa
    (48, 4.0),  # Do
    (43, 4.0),  # Sol
)


def _midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _envelope(length: int, sample_rate: int, attack: float, release: float) -> np.ndarray:
    env = np.ones(length, dtype=np.float64)
    a = max(1, min(length, int(attack * sample_rate)))
    r = max(1, min(length, int(release * sample_rate)))
    env[:a] = np.linspace(0.0, 1.0, a)
    env[-r:] *= np.linspace(1.0, 0.0, r)
    return env


def _tone(note: int, beats: float, sample_rate: int, *, amp: float, wave: str) -> np.ndarray:
    """Nota con envolvente suave para que el bucle no chasquee."""
    length = max(1, int(round(beats * BEAT * sample_rate)))
    t = np.arange(length, dtype=np.float64) / sample_rate
    freq = _midi_to_hz(note)
    if wave == "square":
        signal = np.sign(np.sin(2.0 * np.pi * freq * t))
    else:
        signal = np.sin(2.0 * np.pi * freq * t)
    return amp * signal * _envelope(length, sample_rate, 0.012, 0.10)


def _kick(sample_rate: int) -> np.ndarray:
    """Golpe grave corto, tipo latido de juego arcade."""
    length = int(0.14 * sample_rate)
    t = np.arange(length, dtype=np.float64) / sample_rate
    freq = 115.0 * np.exp(-18.0 * t) + 42.0
    phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
    return 0.22 * np.sin(phase) * np.exp(-16.0 * t)


def _add(buffer: np.ndarray, samples: np.ndarray, start: int) -> None:
    end = min(len(buffer), start + len(samples))
    if end > start:
        buffer[start:end] += samples[: end - start]


def build_loop(sample_rate: int = SAMPLE_RATE, channels: int = 2) -> np.ndarray:
    """Bucle musical de 8 s en int16, listo para ``pygame.sndarray``."""
    total_beats = sum(beats for _, beats in MELODY)
    length = int(round(total_beats * BEAT * sample_rate))
    mix = np.zeros(length, dtype=np.float64)

    start = 0.0
    for note, beats in MELODY:
        _add(
            mix,
            _tone(note, beats, sample_rate, amp=0.20, wave="square"),
            int(round(start * BEAT * sample_rate)),
        )
        start += beats

    start = 0.0
    for note, beats in BASS:
        _add(
            mix,
            _tone(note, beats, sample_rate, amp=0.26, wave="sine"),
            int(round(start * BEAT * sample_rate)),
        )
        start += beats

    kick = _kick(sample_rate)
    for beat in range(int(total_beats)):
        if beat % 2 == 0:
            _add(mix, kick, int(round(beat * BEAT * sample_rate)))

    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix *= 0.62 / peak

    pcm = np.clip(mix * 32767.0, -32768, 32767).astype(np.int16)
    if channels > 1:
        pcm = np.column_stack([pcm] * channels)
    return np.ascontiguousarray(pcm)


class Music:
    """Bucle de musica de fondo con volumen y silencio controlables."""

    def __init__(self, volume: float = 0.30) -> None:
        self.volume = max(0.0, min(1.0, float(volume)))
        self.muted = False
        self.available = False
        self._sound: pygame.mixer.Sound | None = None
        self._channel: pygame.mixer.Channel | None = None
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
            info = pygame.mixer.get_init()
            if info is None:
                raise pygame.error("mezclador no disponible")
            sample_rate, _size, channels = info
            samples = build_loop(sample_rate, channels)
            self._sound = pygame.sndarray.make_sound(samples)
            self._channel = self._sound.play(loops=-1)
            self.available = True
            self._apply()
        except Exception:
            # Sin audio (o sin soporte de numpy en pygame): todo queda mudo.
            self._sound = None
            self._channel = None
            self.available = False

    # -- Control -------------------------------------------------------------
    def _apply(self) -> None:
        if self._channel is not None:
            self._channel.set_volume(0.0 if self.muted else self.volume)

    def set_volume(self, value: float) -> None:
        """Ajusta el volumen 0..1; mover el deslizador quita el silencio."""
        self.volume = max(0.0, min(1.0, float(value)))
        if self.volume > 0.0:
            self.muted = False
        self._apply()

    def change_volume(self, delta: float) -> None:
        self.set_volume(self.volume + float(delta))

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        self._apply()
        return self.muted

    def close(self) -> None:
        if self._channel is not None:
            self._channel.stop()
            self._channel = None
        self._sound = None
