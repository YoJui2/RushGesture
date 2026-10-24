"""Asociacion de manos detectadas a jugadores y prediccion por jugador.

Con una sola camara cada jugador ocupa una franja vertical del encuadre: el
jugador 1 la franja izquierda, el jugador 2 la derecha (y el 3 el centro, si se
juega de a tres). Cada franja se queda con la mano mas centrada y la clasifica
por separado, con su propio suavizado temporal para no arrastrar predicciones
de un jugador al otro.

El modulo no importa MediaPipe: trabaja con cualquier objeto que exponga
``centroid``, ``landmarks`` y (opcionalmente) ``handedness``/``index``, como
``hand_detector.HandResult``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from . import config
from .predictor import NO_HAND, TemporalSmoother

if TYPE_CHECKING:  # pragma: no cover - solo para tipado
    from .hand_detector import HandResult


@dataclass
class PlayerReading:
    """Lectura de un jugador en un frame."""

    player: int
    label: str | None = None
    confidence: float = 0.0
    stable: str | None = None
    landmarks: np.ndarray | None = None
    handedness: str | None = None
    present: bool = False


def zone_center(player: int, num_players: int) -> float:
    """Centro horizontal normalizado de la franja de ``player``."""
    return (player + 0.5) / num_players


def assign_hands(
    hands: Sequence["HandResult"],
    num_players: int = 2,
) -> dict[int, "HandResult | None"]:
    """Reparte las manos detectadas entre las franjas de los jugadores.

    Se asignan por cercania a la franja, en orden global: la pareja
    mano/franja mas proxima gana primero. Asi, con una sola mano en el
    encuadre, cae en la franja que le corresponde y no en la primera libre.
    """
    assigned: dict[int, "HandResult | None"] = {i: None for i in range(num_players)}
    if not hands or num_players < 1:
        return assigned

    centers = [zone_center(i, num_players) for i in range(num_players)]
    candidates: list[tuple[float, int, int, "HandResult"]] = []
    for hand in hands:
        x = float(hand.centroid[0])
        # La mano solo puede pertenecer a la franja que ocupa: una mano en la
        # mitad izquierda nunca debe puntuar para el jugador de la derecha.
        band = min(num_players - 1, max(0, int(x * num_players)))
        candidates.append((abs(x - centers[band]), band, int(getattr(hand, "index", 0)), hand))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    used: set[int] = set()
    for _distance, player, _index, hand in candidates:
        if assigned[player] is not None or id(hand) in used:
            continue
        assigned[player] = hand
        used.add(id(hand))
    return assigned


class DuelTracker:
    """Traduce manos detectadas en una lectura estable por jugador."""

    def __init__(
        self,
        predictor,
        num_players: int = 2,
        *,
        window: int = config.SMOOTHING_WINDOW,
        min_votes: int = config.SMOOTHING_MIN_VOTES,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
    ) -> None:
        self.predictor = predictor
        self.num_players = num_players
        self._smoothers = [
            TemporalSmoother(
                window=window,
                min_votes=min_votes,
                confidence_threshold=confidence_threshold,
            )
            for _ in range(num_players)
        ]

    def update(self, hands: Sequence["HandResult"]) -> list[PlayerReading]:
        """Devuelve una lectura por jugador, tenga mano o no."""
        assignment = assign_hands(hands, self.num_players)
        readings: list[PlayerReading] = []
        for player in range(self.num_players):
            hand = assignment.get(player)
            smoother = self._smoothers[player]
            if hand is None:
                smoother.update(None, 0.0)
                readings.append(PlayerReading(player=player))
                continue

            label, confidence = self.predictor.predict(hand.landmarks)
            stable = smoother.update(None if label == NO_HAND else label, confidence)
            readings.append(
                PlayerReading(
                    player=player,
                    label=None if label == NO_HAND else label,
                    confidence=confidence,
                    stable=stable,
                    landmarks=hand.landmarks,
                    handedness=getattr(hand, "handedness", None),
                    present=True,
                )
            )
        return readings

    def reset(self) -> None:
        for smoother in self._smoothers:
            smoother.reset()
