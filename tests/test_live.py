"""Pruebas de la fuente de datos simulada (modo --demo).

No abren ventana ni camara: simulan el bucle del juego a mano.
"""

from __future__ import annotations

import random

from gesture_rush import config
from gesture_rush.game import ChainRace, Duel, PracticeSession
from gesture_rush.live import CameraPipeline, DemoPipeline


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


def test_demo_practice_feeds_free_gestures_with_top_k():
    practice = PracticeSession()
    pipeline = DemoPipeline(practice, rng=random.Random(9))

    now = 0.0
    seen: set[str] = set()
    for _ in range(240):  # 8 s a 30 fps: da tiempo a cambiar de gesto
        now += 1 / 30
        _frame, readings = pipeline.read(now, probe=True)
        # Una sola vista de camara: una unica lectura.
        assert len(readings) == 1
        for reading in readings:
            assert reading.present and reading.label in practice.gestures
            assert reading.top and len(reading.top) == 3
            assert reading.top[0][0] == reading.label
            assert reading.top[0][1] == reading.confidence
            assert 0 <= reading.vote_count <= reading.vote_total
            assert reading.vote_total <= config.SMOOTHING_WINDOW
        seen.add(readings[0].label)

    assert len(seen) >= 2, "el gesto simulado cambia con el tiempo"


def test_demo_practice_vote_confirms_after_the_window():
    practice = PracticeSession()
    pipeline = DemoPipeline(practice, rng=random.Random(4))

    now = 0.0
    confirmed_at = None
    for step in range(300):
        now += 1 / 30
        _frame, readings = pipeline.read(now, probe=True)
        if readings[0].stable is not None and confirmed_at is None:
            confirmed_at = step
            # El suavizado exige varios votos, no la primera prediccion.
            assert readings[0].vote_count >= config.SMOOTHING_MIN_VOTES
    assert confirmed_at is not None
    assert confirmed_at > 0


def test_bind_to_practice_resets_the_simulation():
    duel = Duel()
    pipeline = DemoPipeline(duel, rng=random.Random(11), min_delay=0.05, max_delay=0.1)
    duel.start(0.0)
    duel.tick(3.0)
    pipeline.read(3.1)

    practice = PracticeSession()
    pipeline.bind(practice)
    assert pipeline.game is practice
    assert pipeline._labels == {} and pipeline._plan == {}


def test_camera_pipeline_switches_to_one_band_in_practice():
    pipeline = CameraPipeline(predictor=object(), num_players=2)
    assert pipeline.tracker.num_players == 2

    pipeline.bind(PracticeSession())
    assert pipeline.tracker.num_players == 1

    pipeline.bind(Duel())
    assert pipeline.tracker.num_players == 2


def test_demo_shows_thumbs_up_to_approve_before_starting():
    duel = Duel(num_players=2, rng=random.Random(1))
    pipeline = DemoPipeline(duel, rng=random.Random(2))

    _frame, readings = pipeline.read(0.0)
    assert all(not reading.present for reading in readings)

    pipeline.force(0)
    _frame, readings = pipeline.read(0.1)
    assert readings[0].stable == "thumbs_up"
    assert readings[1].stable is None

    pipeline.force(1)
    _frame, readings = pipeline.read(0.2)
    assert [reading.stable for reading in readings] == ["thumbs_up", "thumbs_up"]

    # Al arrancar, el pulgar con el que se aprobo no vale como acierto.
    duel.start(0.3)
    pipeline.read(0.4)  # cuenta atras
    duel.tick(3.4)
    _frame, readings = pipeline.read(3.5)

    assert duel.phase == "playing"
    assert all(reading.stable != "thumbs_up" for reading in readings)
    assert all(value is None for value in duel.times_ms.values())


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
