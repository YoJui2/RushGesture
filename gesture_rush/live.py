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
    def read(self, now: float | None = None) -> tuple[np.ndarray | None, list[PlayerReading]]:
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
        return frame, self.tracker.update(hands)

    def _empty_readings(self) -> list[PlayerReading]:
        self.tracker.reset()
        return [PlayerReading(player=i) for i in range(self.num_players)]


class DemoPipeline:
    """Simula dos o tres jugadores sin camara, para desarrollar la interfaz.

    Cada jugador "reacciona" con un retraso aleatorio y a veces falla. Las
    teclas ``1`` y ``2`` (ver ``force``) hacen que ese jugador acierte al
    instante, util para probar el cierre de ronda y el marcador.
    """

    def __init__(
        self,
        duel,
        *,
        rng: random.Random | None = None,
        min_delay: float = 0.9,
        max_delay: float = 2.8,
    ) -> None:
        self.duel = duel
        self._rng = rng or random.Random()
        self.min_delay = float(min_delay)
        self.max_delay = float(max_delay)
        self._round_key: tuple[int, str | None, str] | None = None
        self._due: dict[int, float] = {}
        self._labels: dict[int, tuple[str | None, float]] = {}
        self._forced: set[int] = set()

    # -- API esperada por la interfaz ---------------------------------------
    def read(self, now: float | None = None) -> tuple[np.ndarray | None, list[PlayerReading]]:
        now = time.perf_counter() if now is None else now
        self._sync(now)
        readings: list[PlayerReading] = []
        for player in range(self.duel.num_players):
            label, confidence = self._labels.get(player, (None, 0.0))
            stable = label if confidence >= config.CONFIDENCE_THRESHOLD else None
            readings.append(
                PlayerReading(
                    player=player,
                    label=label,
                    confidence=confidence,
                    stable=stable,
                    present=label is not None,
                )
            )
        return None, readings

    def force(self, player: int) -> None:
        """Hace que ``player`` acierte en el proximo ``read``."""
        self._forced.add(player)

    def close(self) -> None:
        pass

    @property
    def fps(self) -> float:
        return 0.0

    # -- Simulacion ----------------------------------------------------------
    def _sync(self, now: float) -> None:
        duel = self.duel
        key = (duel.round_number, duel.target, duel.phase)
        if key != self._round_key:
            self._round_key = key
            self._labels.clear()
            self._forced.clear()
            self._due = {
                player: now + self._rng.uniform(self.min_delay, self.max_delay)
                for player in range(duel.num_players)
            }

        if duel.phase != "playing" or duel.target is None:
            return

        for player in range(duel.num_players):
            if self._labels.get(player, (None, 0.0))[0] == duel.target:
                continue
            due = self._due.get(player, now)
            if player in self._forced or now >= due:
                self._labels[player] = (duel.target, self._rng.uniform(0.86, 0.99))
                self._due[player] = now + 1.0  # no repetir el aviso
            else:
                others = [g for g in self.duel.gestures if g != duel.target]
                self._labels[player] = (self._rng.choice(others), self._rng.uniform(0.30, 0.58))
