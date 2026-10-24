"""Pruebas del modo cadena (sin pygame ni camara)."""

from __future__ import annotations

import random

import pytest

from gesture_rush import config
from gesture_rush.game import (
    PHASE_COUNTDOWN,
    PHASE_FINISHED,
    PHASE_IDLE,
    PHASE_PLAYING,
    ChainRace,
)


def make_race(**kwargs) -> ChainRace:
    options = dict(num_players=2, chain_length=3, rng=random.Random(5))
    options.update(kwargs)
    return ChainRace(**options)


def start_and_play(race: ChainRace, play_at: float = 3.0) -> float:
    assert race.start(0.0)
    assert race.phase == PHASE_COUNTDOWN
    race.tick(play_at)
    assert race.phase == PHASE_PLAYING
    return play_at


def finish_chain(race: ChainRace, player: int, play_at: float, step_ms: int = 400) -> float:
    """Hace que ``player`` complete la cadena; devuelve el tiempo final."""
    now = play_at
    for _ in range(len(race.chain)):
        now += step_ms / 1000.0
        assert race.observe(player, race.current_target(player), 0.95, now)
    return now


def test_race_goes_from_countdown_to_playing():
    race = make_race()
    assert race.phase == PHASE_IDLE
    assert race.start(0.0)
    assert race.phase == PHASE_COUNTDOWN
    assert len(race.chain) == 3
    assert set(race.chain) <= set(config.GESTURES)

    race.tick(1.0)
    assert race.phase == PHASE_COUNTDOWN
    race.tick(3.0)
    assert race.phase == PHASE_PLAYING


def test_chain_can_repeat_gestures_but_never_twice_in_a_row():
    race = make_race(chain_length=6)
    repeated = False
    for _ in range(60):
        race.start(0.0)
        assert len(race.chain) == 6
        assert set(race.chain) <= set(config.GESTURES)
        for previous, current in zip(race.chain, race.chain[1:]):
            assert previous != current, "nunca dos gestos iguales seguidos"
        repeated = repeated or len(set(race.chain)) < len(race.chain)
        race.reset()
    assert repeated, "los gestos pueden repetirse dentro de la cadena"


def test_players_advance_only_with_the_expected_gesture():
    race = make_race()
    play_at = start_and_play(race)
    chain = race.chain

    assert race.current_target(0) == chain[0]
    # Un gesto distinto, o el siguiente de la cadena, no avanza.
    other = next(g for g in config.GESTURES if g not in chain)
    assert not race.observe(0, other, 0.99, play_at + 0.1)
    assert not race.observe(0, chain[1], 0.99, play_at + 0.2)
    assert race.players[0].progress == 0

    assert race.observe(0, chain[0], 0.99, play_at + 0.3)
    assert race.players[0].progress == 1
    assert race.current_target(0) == chain[1]


def test_low_confidence_does_not_advance():
    race = make_race()
    play_at = start_and_play(race)

    assert not race.observe(0, race.chain[0], 0.2, play_at + 0.2)
    assert race.players[0].progress == 0


def test_each_player_has_its_own_progress():
    race = make_race()
    play_at = start_and_play(race)

    race.observe(0, race.chain[0], 0.99, play_at + 0.2)
    assert race.players[0].progress == 1
    assert race.players[1].progress == 0
    assert race.current_target(1) == race.chain[0]


def test_first_to_finish_wins_and_times_are_recorded():
    race = make_race()
    play_at = start_and_play(race)

    finish_chain(race, 0, play_at, step_ms=300)
    assert race.players[0].done
    assert race.phase == PHASE_PLAYING, "espera al rival dentro del margen"

    finish_chain(race, 1, play_at, step_ms=500)
    assert race.phase == PHASE_FINISHED
    assert race.winner == 0
    assert race.players[0].finished_ms == 900
    assert race.players[1].finished_ms == 1500

    record = race.players[0]
    assert sum(record.step_times_ms) == record.finished_ms
    assert race.average_step_ms(0) == 300


def test_race_closes_after_grace_when_rival_never_finishes():
    race = make_race(grace_seconds=2.0)
    play_at = start_and_play(race)

    last = finish_chain(race, 0, play_at, step_ms=300)
    assert race.phase == PHASE_PLAYING

    race.tick(last + 2.1)
    assert race.phase == PHASE_FINISHED
    assert race.winner == 0
    assert race.players[1].finished_ms is None
    assert race.players[1].progress < race.num_steps


def test_timeout_without_winner():
    race = make_race(race_timeout=20.0)
    play_at = start_and_play(race)

    race.tick(play_at + 20.1)
    assert race.phase == PHASE_FINISHED
    assert race.winner is None


def test_equal_times_are_a_draw():
    race = make_race()
    play_at = start_and_play(race)

    now = play_at
    for _ in range(len(race.chain)):
        now += 0.4
        assert race.observe(0, race.current_target(0), 0.95, now)
        assert race.observe(1, race.current_target(1), 0.95, now)

    assert race.phase == PHASE_FINISHED
    assert race.players[0].finished_ms == race.players[1].finished_ms
    assert race.winner is None


def test_observations_are_ignored_outside_playing():
    race = make_race()
    assert not race.observe(0, "fist", 1.0, 0.0)

    race.start(0.0)
    assert race.phase == PHASE_COUNTDOWN
    assert not race.observe(0, "fist", 1.0, 1.0)


def test_start_is_rejected_while_playing():
    race = make_race()
    start_and_play(race)
    assert not race.start(4.0)


def test_new_race_after_finish_resets_everything():
    race = make_race()
    play_at = start_and_play(race)
    finish_chain(race, 0, play_at, step_ms=300)
    finish_chain(race, 1, play_at, step_ms=500)
    assert race.phase == PHASE_FINISHED

    assert race.start(play_at + 5.0)
    assert race.phase == PHASE_COUNTDOWN
    assert all(p.progress == 0 and not p.done for p in race.players)
    assert race.winner is None


def test_standings_puts_the_winner_first():
    race = make_race()
    play_at = start_and_play(race)
    finish_chain(race, 1, play_at, step_ms=200)
    finish_chain(race, 0, play_at, step_ms=400)

    standings = race.standings()
    assert [p.index for p in standings] == [1, 0]


def test_solo_chain_closes_as_soon_as_it_is_completed():
    race = make_race(num_players=1, chain_length=3)
    play_at = start_and_play(race)

    finish_chain(race, 0, play_at, step_ms=250)

    assert race.phase == PHASE_FINISHED, "sin rival no se espera el margen"
    assert race.winner == 0
    assert race.players[0].finished_ms == 750
    assert race.average_step_ms(0) == 250


def test_rematch_never_deals_the_exact_same_chain():
    race = make_race(chain_length=4, gestures=("fist", "victory"))
    play_at = start_and_play(race)
    first = race.chain

    finish_chain(race, 0, play_at, step_ms=100)
    finish_chain(race, 1, play_at, step_ms=150)
    assert race.phase == PHASE_FINISHED

    assert race.start(20.0)
    assert len(race.chain) == 4
    assert race.chain != first


def test_chain_longer_than_vocabulary_avoids_repeats_in_a_row():
    race = make_race(chain_length=len(config.GESTURES) + 4)
    race.start(0.0)
    assert len(race.chain) == len(config.GESTURES) + 4
    for previous, current in zip(race.chain, race.chain[1:]):
        assert previous != current


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        ChainRace(num_players=0)
    with pytest.raises(ValueError):
        ChainRace(gestures=())
    with pytest.raises(ValueError):
        ChainRace(gestures=("fist",))
