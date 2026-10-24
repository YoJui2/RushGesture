"""Fuentes de datos en vivo para la interfaz del juego.

Dos implementaciones con la misma interfaz ``read() -> (frame, readings)``:

- ``CameraPipeline``: camara real + MediaPipe + modelo entrenado.
- ``DemoPipeline``: jugadores simulados, para trabajar la interfaz sin camara
  ni modelo (y para la defensa, si falla el hardware).

La interfaz solo conoce esa interfaz, de modo que cambiar de una a otra no
toca el codigo de dibujo.
"""

from __future__ import annotations

import random
import time

import numpy as np

from . import config
from .camera import Camera, CameraError
from .hand_detector import HandDetector
from .player_tracker import DuelTracker, PlayerReading
from .predictor import TemporalSmoother


class CameraPipeline:
    """Camara + deteccion de manos + clasificacion, un frame por ``read``."""

    def __init__(
        self,
        predictor,
        camera_index: int | None = None,
        num_players: int = 2,
        *,
        mirror: bool = True,
    ) -> None:
        self.predictor = predictor
        self.camera_index = config.CAMERA_INDEX if camera_index is None else camera_index
        self.num_players = num_players
        self.mirror = mirror
        self.tracker = DuelTracker(predictor, num_players)
        self._camera: Camera | None = None
        self._detector: HandDetector | None = None

    # -- Ciclo de vida -------------------------------------------------------
    def start(self) -> "CameraPipeline":
        self._camera = Camera(index=self.camera_index).open()
        self._detector = HandDetector()
        return self

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    @property
    def fps(self) -> float:
        return self._camera.fps if self._camera is not None else 0.0

    # -- Lectura -------------------------------------------------------------
    def bind(self, game) -> None:
        """Adapta el reparto de franjas al modo activo.

        En Prueba hay una sola franja (el encuadre completo); en Duelo y Cadena
        hay una por jugador. Cambiar de banda obliga a rehacer los suavizados.
        """
        self.current_game = game
        players = int(getattr(game, "num_players", self.num_players))
        if players != self.tracker.num_players:
            self.tracker = DuelTracker(self.predictor, players)

    def reset(self) -> None:
        self.tracker.reset()

    def read(
        self, now: float | None = None, *, probe: bool = False
    ) -> tuple[np.ndarray | None, list[PlayerReading]]:
        if self._camera is None:
            self.start()
        assert self._camera is not None and self._detector is not None

        ok, frame = self._camera.read()
        if not ok or frame is None:
            return None, self._empty_readings()
        if self.mirror:
            # Espejo para que la izquierda de la imagen coincida con la
            # izquierda de quien esta frente a la camara.
            frame = frame[:, ::-1].copy()

        hands = self._detector.detect(frame)
        return frame, self.tracker.update(hands, probe=probe)

    def _empty_readings(self) -> list[PlayerReading]:
        self.tracker.reset()
        return [PlayerReading(player=i) for i in range(self.tracker.num_players)]


class DemoPipeline:
    """Simula dos o tres jugadores sin camara, para desarrollar la interfaz.

    Cada jugador "reacciona" con un retraso aleatorio y a veces falla. Las
    teclas ``1`` y ``2`` (ver ``force``) hacen que ese jugador acierte al
    instante, util para probar el cierre de ronda y el marcador.
    """

    def __init__(
        self,
        game,
        *,
        rng: random.Random | None = None,
        min_delay: float = 0.9,
        max_delay: float = 2.8,
    ) -> None:
        self.game = game
        self._rng = rng or random.Random()
        self.min_delay = float(min_delay)
        self.max_delay = float(max_delay)
        # Por jugador: clave del objetivo actual y momento en que "acierta".
        self._plan: dict[int, tuple[tuple, float]] = {}
        self._labels: dict[int, tuple[str | None, float]] = {}
        self._forced: set[int] = set()
        # Modo Prueba: gesto sostenido por jugador y suavizado real.
        self._held: dict[int, tuple[str, float, float]] = {}
        self._practice_smoothers: dict[int, TemporalSmoother] = {}
        self._last_phase: str | None = None

    # -- API esperada por la interfaz ---------------------------------------
    def bind(self, game) -> None:
        """Cambia el minijuego simulado (duelo, cadena o prueba)."""
        if game is self.game:
            return
        self.game = game
        self.reset()

    def reset(self) -> None:
        self._plan.clear()
        self._labels.clear()
        self._forced.clear()
        self._held.clear()
        self._practice_smoothers.clear()
        self._last_phase = None

    def read(
        self, now: float | None = None, *, probe: bool = False
    ) -> tuple[np.ndarray | None, list[PlayerReading]]:
        now = time.perf_counter() if now is None else now
        phase = self.game.phase
        if phase == "playing":
            if self._last_phase == "countdown":
                # El pulgar con el que se aprobo no debe contar como acierto.
                self._forced.clear()
            if getattr(self.game, "practice", False):
                self._simulate_practice(now)
            else:
                self._simulate(now)
        else:
            self._simulate_waiting()
        self._last_phase = phase

        readings: list[PlayerReading] = []
        for player in range(self.game.num_players):
            label, confidence = self._labels.get(player, (None, 0.0))
            if probe:
                stable, vote_label, vote_count, vote_total = self._practice_vote(
                    player, label, confidence
                )
            else:
                stable = label if confidence >= config.CONFIDENCE_THRESHOLD else None
                vote_label, vote_count, vote_total = None, 0, 0
            readings.append(
                PlayerReading(
                    player=player,
                    label=label,
                    confidence=confidence,
                    stable=stable,
                    present=label is not None,
                    top=self._synthetic_top(label, confidence) if probe else None,
                    vote_label=vote_label,
                    vote_count=vote_count,
                    vote_total=vote_total,
                )
            )
        return None, readings

    def force(self, player: int) -> None:
        """Hace que ``player`` acierte su objetivo en el proximo ``read``."""
        self._forced.add(player)

    def close(self) -> None:
        pass

    @property
    def fps(self) -> float:
        return 0.0

    # -- Simulacion ----------------------------------------------------------
    def _simulate(self, now: float) -> None:
        for player in range(self.game.num_players):
            target = self.game.current_target(player)
            key = (target, self.game.phase)
            planned = self._plan.get(player)
            if planned is None:
                self._plan[player] = (
                    key,
                    now + self._rng.uniform(self.min_delay, self.max_delay),
                )
            elif planned[0] != key:
                # Objetivo nuevo (ronda nueva o paso siguiente de la cadena):
                # se reprograma el acierto y se consume el forzado anterior.
                self._plan[player] = (
                    key,
                    now + self._rng.uniform(self.min_delay, self.max_delay),
                )
                self._forced.discard(player)

            if target is None:
                self._labels[player] = (None, 0.0)
                continue

            due = self._plan[player][1]
            if player in self._forced or now >= due:
                self._labels[player] = (target, self._rng.uniform(0.86, 0.99))
            else:
                others = [g for g in self.game.gestures if g != target]
                self._labels[player] = (
                    self._rng.choice(others),
                    self._rng.uniform(0.30, 0.58),
                )

    def _simulate_waiting(self) -> None:
        """Fuera de ronda solo cambia el pulgar de los jugadores forzados.

        En ``--demo`` las teclas 1 y 2 hacen de "pulgar arriba" para aprobar el
        inicio sin camara; el resto conserva su ultima lectura.
        """
        for player in self._forced:
            if 0 <= player < self.game.num_players:
                self._labels[player] = ("thumbs_up", 0.96)

    def _simulate_practice(self, now: float) -> None:
        """Modo Prueba: cada jugador sostiene un gesto libre unos segundos.

        Se alterna el gesto para que se vea como cambia la prediccion, y el
        jugador 2 mantiene confianzas mas bajas para mostrar el caso en que el
        voto temporal no confirma.
        """
        for player in range(self.game.num_players):
            held = self._held.get(player)
            if held is None or now >= held[2]:
                label = self._rng.choice(self.game.gestures)
                base = (
                    self._rng.uniform(0.78, 0.98)
                    if player % 2 == 0
                    else self._rng.uniform(0.45, 0.85)
                )
                held = (label, base, now + self._rng.uniform(1.4, 2.8))
                self._held[player] = held
            label, base, _until = held
            confidence = min(0.999, max(0.05, base + self._rng.uniform(-0.03, 0.03)))
            self._labels[player] = (label, confidence)

    def _practice_vote(
        self, player: int, label: str | None, confidence: float
    ) -> tuple[str | None, str | None, int, int]:
        """Pasa la etiqueta por un suavizado real, como en la camara."""
        smoother = self._practice_smoothers.get(player)
        if smoother is None:
            smoother = self._practice_smoothers[player] = TemporalSmoother()
        stable = smoother.update(label, confidence)
        vote_label, vote_count = smoother.snapshot()
        return stable, vote_label, vote_count, smoother.filled

    def _synthetic_top(
        self, label: str | None, confidence: float
    ) -> list[tuple[str, float]]:
        """Top-3 verosimil: la clase ganadora y el resto repartido."""
        if label is None:
            return []
        others = [g for g in self.game.gestures if g != label]
        self._rng.shuffle(others)
        rest = max(0.0, 1.0 - confidence)
        top = [(label, confidence)]
        for other, weight in zip(others, (0.55, 0.30)):
            top.append((other, rest * weight))
        return top
