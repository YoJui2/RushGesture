"""Pruebas de la fuente de datos simulada (modo --demo).

No abren ventana ni camara: simulan el bucle del juego a mano.
"""

from __future__ import annotations

import random

from gesture_rush.game import ChainRace, Duel
from gesture_rush.live import DemoPipeline


def run_loop(game, pipeline, seconds: float = 60.0, dt: float = 1 / 60) -> float:
    """Bucle minimo equivalente al de la interfaz. Devuelve el tiempo final."""
    now = 0.0
    game.start(now)
    steps = int(seconds / dt)
    for _ in range(steps):
        now += dt
        game.tick(now)
        frame, readings = pipeline.read(now)
        assert frame is None, "el modo demo no produce imagen de camara"
        for reading in readings:
            if reading.stable is not None:
                game.observe(reading.player, reading.stable, reading.confidence, now)
        if game.phase == "finished":
            break
    return now


def test_demo_pipeline_completes_a_chain_race():
    race = ChainRace(chain_length=5, rng=random.Random(1))
    pipeline = DemoPipeline(race, rng=random.Random(2), min_delay=0.05, max_delay=0.2)

    run_loop(race, pipeline)

    assert race.phase == "finished"
    assert race.winner is not None
    assert race.players[race.winner].progress == race.num_steps
    assert all(p.done for p in race.players), "los dos terminan dentro del margen"


def test_demo_pipeline_completes_a_duel_round():
    duel = Duel(total_rounds=1, rng=random.Random(3))
    pipeline = DemoPipeline(duel, rng=random.Random(4), min_delay=0.05, max_delay=0.2)

    run_loop(duel, pipeline)

    assert duel.phase == "finished"
    assert duel.winner is not None
    assert duel.history, "la ronda quedo registrada"


def test_bind_switches_the_simulated_game():
    duel = Duel()
    chain = ChainRace()
    pipeline = DemoPipeline(duel)
    assert pipeline.game is duel

    pipeline.bind(chain)
    assert pipeline.game is chain

    pipeline.bind(chain)  # repetir no rompe nada
    assert pipeline.game is chain


def test_force_makes_a_player_advance_once():
    race = ChainRace(chain_length=3, rng=random.Random(7))
    pipeline = DemoPipeline(race, rng=random.Random(8), min_delay=30.0, max_delay=31.0)

    now = 0.0
    race.start(now)
    for _ in range(200):
        now += 1 / 60
        race.tick(now)
        if race.phase == "playing":
            break
    assert race.phase == "playing"
    assert race.players[0].progress == 0

    pipeline.force(0)
    _frame, readings = pipeline.read(now)
    for reading in readings:
        if reading.stable is not None:
            race.observe(reading.player, reading.stable, reading.confidence, now)
    assert race.players[0].progress == 1

    # El forzado se consume: no completa la cadena de golpe.
    for _ in range(3):
        now += 1 / 60
        race.tick(now)
        _frame, readings = pipeline.read(now)
        for reading in readings:
            if reading.stable is not None:
                race.observe(reading.player, reading.stable, reading.confidence, now)
    assert race.players[0].progress == 1
