"""Motor del minijuego "Duelo de gestos" (modo clasico).

No depende de pygame, OpenCV ni MediaPipe: recibe observaciones
``(jugador, etiqueta, confianza)`` junto con un reloj monotono y expone el
estado que la interfaz dibuja. Asi las reglas se pueden probar sin camara ni
ventana.

Reglas
------
- En cada ronda se sortea un gesto objetivo y se muestra en pantalla.
- El primero en hacerlo bien se lleva el punto.
- Tras el primer acierto la ronda sigue abierta ``grace_seconds`` para que el
  rival registre su tiempo: de ahi sale la "diferencia" que se muestra al
  cerrar la ronda.
- Si nadie acierta dentro de ``round_timeout`` la ronda queda desierta.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import config

# Fases del ciclo de una ronda.
PHASE_IDLE = "idle"
PHASE_COUNTDOWN = "countdown"
PHASE_PLAYING = "playing"
PHASE_RESULT = "result"
PHASE_FINISHED = "finished"


@dataclass
class PlayerScore:
    """Marcador y ultima lectura de un jugador."""

    index: int
    points: int = 0
    last_ms: int | None = None
    best_ms: int | None = None
    won_last: bool = False
    misses: int = 0
    live_label: str | None = None
    live_confidence: float = 0.0


@dataclass
class RoundRecord:
    """Resultado archivado de una ronda cerrada."""

    index: int
    gesture: str
    times_ms: dict[int, int | None]
    winner: int | None = None

    @property
    def margin_ms(self) -> int | None:
        """Diferencia entre el tiempo mas lento y el mas rapido, si hay dos."""
        times = [t for t in self.times_ms.values() if t is not None]
        if len(times) < 2:
            return None
        return abs(max(times) - min(times))


class Duel:
    """Estado completo de una partida de "Duelo de gestos"."""

    def __init__(
        self,
        num_players: int = 2,
        total_rounds: int = 5,
        *,
        rng: random.Random | None = None,
        countdown_seconds: float = 3.0,
        grace_seconds: float = 2.0,
        round_timeout: float = 8.0,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        gestures: tuple[str, ...] | None = None,
    ) -> None:
        if num_players < 1:
            raise ValueError("Hace falta al menos un jugador")
        self.num_players = num_players
        # total_rounds == 0 significa partida sin limite.
        self.total_rounds = max(0, int(total_rounds))
        self.countdown_seconds = float(countdown_seconds)
        self.grace_seconds = float(grace_seconds)
        self.round_timeout = float(round_timeout)
        self.confidence_threshold = float(confidence_threshold)
        self.gestures = tuple(config.GESTURES if gestures is None else gestures)
        if not self.gestures:
            raise ValueError("La lista de gestos no puede estar vacia")

        self._rng = rng or random.Random()
        self.players = [PlayerScore(index=i) for i in range(num_players)]
        self.history: list[RoundRecord] = []

        self.phase = PHASE_IDLE
        self.target: str | None = None
        self.winner: int | None = None
        self.times_ms: dict[int, int | None] = {}
        self.phase_started_at = 0.0

        self._last_target: str | None = None
        self._countdown_end = 0.0
        self._play_start = 0.0
        self._close_at: float | None = None

    # -- Consultas -----------------------------------------------------------
    @property
    def round_number(self) -> int:
        """Numero (1-based) de la ronda que se esta jugando o se jugara."""
        return len(self.history) + 1

    @property
    def rounds_played(self) -> int:
        return len(self.history)

    @property
    def is_match_over(self) -> bool:
        return self.total_rounds > 0 and len(self.history) >= self.total_rounds

    @property
    def current_round(self) -> RoundRecord | None:
        return self.history[-1] if self.history else None

    @property
    def match_winner(self) -> int | None:
        """Indice del ganador de la partida, o ``None`` si hay empate."""
        ranked = self.ranking()
        if not ranked or ranked[0].points == 0:
            return None
        if len(ranked) > 1 and ranked[0].points == ranked[1].points:
            return None
        return ranked[0].index

    def ranking(self) -> list[PlayerScore]:
        def key(score: PlayerScore) -> tuple[int, int]:
            best = score.best_ms if score.best_ms is not None else 10**9
            return (-score.points, best)

        return sorted(self.players, key=key)

    def elapsed_ms(self, player: int, now: float) -> int | None:
        """Tiempo del jugador en la ronda actual (congelado si ya acerto)."""
        recorded = self.times_ms.get(player)
        if recorded is not None:
            return recorded
        if self.phase == PHASE_PLAYING:
            return max(0, int(round((now - self._play_start) * 1000.0)))
        return None

    def current_target(self, player: int) -> str | None:
        """Gesto que debe hacer ``player`` ahora mismo."""
        if self.phase != PHASE_PLAYING:
            return None
        return self.target

    def countdown_value(self, now: float) -> int:
        remaining = max(0.0, self._countdown_end - now)
        return int(math.ceil(remaining))

    def play_elapsed(self, now: float) -> float:
        if self.phase in (PHASE_PLAYING, PHASE_RESULT, PHASE_FINISHED):
            return max(0.0, now - self._play_start)
        return 0.0

    # -- Transiciones --------------------------------------------------------
    def start(self, now: float) -> bool:
        """Arranca la cuenta atras de la ronda. ``False`` si no corresponde."""
        if self.phase == PHASE_FINISHED or (self.is_match_over and self.phase != PHASE_PLAYING):
            self.reset()
        if self.phase not in (PHASE_IDLE, PHASE_RESULT):
            return False
        self.target = self._pick_target()
        self.winner = None
        self.times_ms = {p.index: None for p in self.players}
        for player in self.players:
            player.won_last = False
            player.misses = 0
        self._close_at = None
        self._countdown_end = now + self.countdown_seconds
        self.phase = PHASE_COUNTDOWN
        self.phase_started_at = now
        return True

    def tick(self, now: float) -> None:
        """Avanza la fase segun el reloj (cuenta atras, timeout, cierre)."""
        if self.phase == PHASE_COUNTDOWN and now >= self._countdown_end:
            self.phase = PHASE_PLAYING
            self.phase_started_at = now
            self._play_start = now
        if self.phase != PHASE_PLAYING:
            return
        if self._close_at is not None and now >= self._close_at:
            self._close(now)
        elif (now - self._play_start) >= self.round_timeout:
            self._close(now)

    def observe(
        self,
        player: int,
        label: str | None,
        confidence: float,
        now: float,
    ) -> bool:
        """Registra una prediccion. ``True`` si acerto el gesto objetivo."""
        if self.phase != PHASE_PLAYING or self.target is None:
            return False
        if not 0 <= player < self.num_players:
            return False
        score = self.players[player]
        score.live_label = label
        score.live_confidence = confidence
        if self.times_ms.get(player) is not None:
            return False
        if label is None or label != self.target or confidence < self.confidence_threshold:
            if label is not None:
                score.misses += 1
            return False

        elapsed = max(0, int(round((now - self._play_start) * 1000.0)))
        self.times_ms[player] = elapsed
        if self._close_at is None:
            self._close_at = now + self.grace_seconds
        if all(t is not None for t in self.times_ms.values()):
            self._close(now)
        return True

    def reset(self) -> None:
        """Devuelve la partida al estado inicial."""
        self.history.clear()
        self.phase = PHASE_IDLE
        self.target = None
        self.winner = None
        self.times_ms = {}
        self.phase_started_at = 0.0
        self._last_target = None
        self._close_at = None
        self._countdown_end = 0.0
        self._play_start = 0.0
        for player in self.players:
            player.points = 0
            player.last_ms = None
            player.best_ms = None
            player.won_last = False
            player.misses = 0
            player.live_label = None
            player.live_confidence = 0.0

    # -- Internos ------------------------------------------------------------
    def _close(self, now: float) -> None:
        valid = {p: t for p, t in self.times_ms.items() if t is not None}
        winner: int | None = None
        if valid:
            best = min(valid.values())
            leaders = [p for p, t in valid.items() if t == best]
            # Dos tiempos identicos son un empate: no se lleva el punto nadie.
            winner = leaders[0] if len(leaders) == 1 else None
        self.winner = winner

        record = RoundRecord(
            index=self.round_number,
            gesture=self.target or "",
            times_ms=dict(self.times_ms),
            winner=winner,
        )
        self.history.append(record)

        for player in self.players:
            ms = self.times_ms.get(player.index)
            player.last_ms = ms
            player.won_last = player.index == winner
            if ms is not None and (player.best_ms is None or ms < player.best_ms):
                player.best_ms = ms
            if player.index == winner:
                player.points += 1

        self.phase = PHASE_FINISHED if self.is_match_over else PHASE_RESULT
        self.phase_started_at = now
        self._close_at = None

    def _pick_target(self) -> str:
        options = [g for g in self.gestures if g != self._last_target]
        target = self._rng.choice(options or list(self.gestures))
        self._last_target = target
        return target


# ---------------------------------------------------------------------------
# Modo cadena
# ---------------------------------------------------------------------------
@dataclass
class ChainPlayer:
    """Estado de un jugador dentro de la carrera."""

    index: int
    progress: int = 0
    finished_ms: int | None = None
    step_times_ms: list[int] = field(default_factory=list)
    misses: int = 0
    live_label: str | None = None
    live_confidence: float = 0.0

    @property
    def done(self) -> bool:
        return self.finished_ms is not None


class ChainRace:
    """Carrera por una cadena de gestos.

    Los dos jugadores reciben **la misma cadena** y avanzan una posicion por
    cada acierto, en orden. Gana quien la completa antes; si los dos llegan,
    gana el tiempo total mas bajo. Fallar no penaliza: solo cuesta tiempo,
    porque el objetivo de cada jugador no cambia hasta que lo acierta.
    """

    def __init__(
        self,
        num_players: int = 2,
        chain_length: int = 5,
        *,
        rng: random.Random | None = None,
        countdown_seconds: float = 3.0,
        grace_seconds: float = 2.0,
        race_timeout: float = 45.0,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        gestures: tuple[str, ...] | None = None,
    ) -> None:
        if num_players < 1:
            raise ValueError("Hace falta al menos un jugador")
        self.num_players = num_players
        self.chain_length = max(1, int(chain_length))
        self.countdown_seconds = float(countdown_seconds)
        self.grace_seconds = float(grace_seconds)
        self.race_timeout = float(race_timeout)
        self.confidence_threshold = float(confidence_threshold)
        self.gestures = tuple(config.GESTURES if gestures is None else gestures)
        if not self.gestures:
            raise ValueError("La lista de gestos no puede estar vacia")
        if len(self.gestures) < 2:
            raise ValueError("La cadena necesita al menos dos gestos distintos")

        self._rng = rng or random.Random()
        self.players = [ChainPlayer(index=i) for i in range(num_players)]
        self.chain: tuple[str, ...] = ()
        self.phase = PHASE_IDLE
        self.winner: int | None = None
        self.phase_started_at = 0.0

        self._last_chain: tuple[str, ...] | None = None
        self._countdown_end = 0.0
        self._play_start = 0.0
        self._close_at: float | None = None

    # -- Consultas -----------------------------------------------------------
    @property
    def num_steps(self) -> int:
        return len(self.chain)

    def current_target(self, player: int) -> str | None:
        """Gesto que ``player`` tiene que hacer para avanzar."""
        if self.phase != PHASE_PLAYING:
            return None
        if not 0 <= player < self.num_players:
            return None
        score = self.players[player]
        if score.progress >= len(self.chain):
            return None
        return self.chain[score.progress]

    def elapsed_ms(self, player: int, now: float) -> int | None:
        score = self.players[player]
        if score.finished_ms is not None:
            return score.finished_ms
        if self.phase == PHASE_PLAYING:
            return max(0, int(round((now - self._play_start) * 1000.0)))
        return None

    def average_step_ms(self, player: int) -> int | None:
        steps = self.players[player].step_times_ms
        if not steps:
            return None
        return int(round(sum(steps) / len(steps)))

    def countdown_value(self, now: float) -> int:
        remaining = max(0.0, self._countdown_end - now)
        return int(math.ceil(remaining))

    def play_elapsed(self, now: float) -> float:
        if self.phase in (PHASE_PLAYING, PHASE_FINISHED):
            return max(0.0, now - self._play_start)
        return 0.0

    def standings(self) -> list[ChainPlayer]:
        def key(score: ChainPlayer) -> tuple[int, int, int]:
            best = score.finished_ms if score.finished_ms is not None else 10**9
            return (0 if score.done else 1, -score.progress, best)

        return sorted(self.players, key=key)

    # -- Transiciones --------------------------------------------------------
    def start(self, now: float) -> bool:
        if self.phase == PHASE_FINISHED:
            self.reset()
        if self.phase not in (PHASE_IDLE, PHASE_RESULT):
            return False
        self.chain = self._pick_chain()
        self.winner = None
        for score in self.players:
            score.progress = 0
            score.finished_ms = None
            score.step_times_ms = []
            score.misses = 0
            score.live_label = None
            score.live_confidence = 0.0
        self._close_at = None
        self._countdown_end = now + self.countdown_seconds
        self.phase = PHASE_COUNTDOWN
        self.phase_started_at = now
        return True

    def tick(self, now: float) -> None:
        if self.phase == PHASE_COUNTDOWN and now >= self._countdown_end:
            self.phase = PHASE_PLAYING
            self.phase_started_at = now
            self._play_start = now
        if self.phase != PHASE_PLAYING:
            return
        if self._close_at is not None and now >= self._close_at:
            self._close(now)
        elif (now - self._play_start) >= self.race_timeout:
            self._close(now)

    def observe(
        self,
        player: int,
        label: str | None,
        confidence: float,
        now: float,
    ) -> bool:
        """Registra una prediccion. ``True`` si el jugador avanzo un paso."""
        if self.phase != PHASE_PLAYING:
            return False
        if not 0 <= player < self.num_players:
            return False
        score = self.players[player]
        score.live_label = label
        score.live_confidence = confidence
        if score.progress >= len(self.chain):
            return False

        expected = self.chain[score.progress]
        if label is None or label != expected or confidence < self.confidence_threshold:
            if label is not None:
                score.misses += 1
            return False

        elapsed = max(0, int(round((now - self._play_start) * 1000.0)))
        # Se guarda la duracion de este paso, no el tiempo acumulado.
        score.step_times_ms.append(max(0, elapsed - sum(score.step_times_ms)))
        score.progress += 1
        if score.progress >= len(self.chain):
            score.finished_ms = elapsed
            if self._close_at is None:
                self._close_at = now + self.grace_seconds
            if all(other.done for other in self.players):
                self._close(now)
        return True

    def reset(self) -> None:
        self.players = [ChainPlayer(index=i) for i in range(self.num_players)]
        self.chain = ()
        self.phase = PHASE_IDLE
        self.winner = None
        self.phase_started_at = 0.0
        self._last_chain = None
        self._countdown_end = 0.0
        self._play_start = 0.0
        self._close_at = None

    # -- Internos ------------------------------------------------------------
    def _close(self, now: float) -> None:
        finished = {p.index: p.finished_ms for p in self.players if p.done}
        winner: int | None = None
        if finished:
            best = min(finished.values())
            leaders = [index for index, ms in finished.items() if ms == best]
            winner = leaders[0] if len(leaders) == 1 else None
        self.winner = winner
        self.phase = PHASE_FINISHED
        self.phase_started_at = now
        self._close_at = None

    def _pick_chain(self) -> tuple[str, ...]:
        pool = list(self.gestures)
        if self.chain_length <= len(pool):
            chain = tuple(self._rng.sample(pool, self.chain_length))
        else:
            # Mas pasos que gestos disponibles: se permiten repeticiones, pero
            # nunca dos iguales seguidos.
            steps: list[str] = []
            previous: str | None = None
            for _ in range(self.chain_length):
                options = [g for g in pool if g != previous]
                previous = self._rng.choice(options)
                steps.append(previous)
            chain = tuple(steps)

        if chain == self._last_chain:
            # Evita repetir exactamente la cadena anterior.
            chain = chain[1:] + chain[:1]
        self._last_chain = chain
        return chain

