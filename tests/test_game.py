"""Pruebas del motor del Duelo de gestos (sin pygame ni camara)."""

from __future__ import annotations

import random

import pytest

from gesture_rush import config
from gesture_rush.game import (
    PHASE_COUNTDOWN,
    PHASE_FINISHED,
    PHASE_IDLE,
    PHASE_PLAYING,
    PHASE_RESULT,
    Duel,
)


def make_duel(**kwargs) -> Duel:
    options = dict(num_players=2, total_rounds=3, rng=random.Random(7))
    options.update(kwargs)
    return Duel(**options)


def start_and_play(duel: Duel, play_at: float = 3.0) -> float:
    assert duel.start(0.0)
    assert duel.phase == PHASE_COUNTDOWN
    duel.tick(play_at)
    assert duel.phase == PHASE_PLAYING
    return play_at


def test_round_goes_from_countdown_to_playing():
    duel = make_duel(countdown_seconds=3.0)
    assert duel.phase == PHASE_IDLE
    assert duel.start(0.0)
    assert duel.phase == PHASE_COUNTDOWN
    assert duel.target in config.GESTURES

    duel.tick(1.5)
    assert duel.phase == PHASE_COUNTDOWN, "la cuenta atras todavia no termino"

    duel.tick(3.0)
    assert duel.phase == PHASE_PLAYING


def test_first_correct_gesture_wins_and_scores():
    duel = make_duel()
    play_at = start_and_play(duel)

    assert duel.observe(1, duel.target, 0.95, play_at + 0.25)
    assert duel.times_ms[1] == 250
    assert duel.phase == PHASE_PLAYING, "la ronda sigue abierta para el rival"

    assert duel.observe(0, duel.target, 0.90, play_at + 0.60)
    assert duel.times_ms[0] == 600
    assert duel.phase == PHASE_RESULT

    assert duel.winner == 1
    assert duel.players[1].points == 1
    assert duel.players[0].points == 0
    assert duel.history[-1].margin_ms == 350


def test_round_closes_after_grace_if_rival_never_answers():
    duel = make_duel(grace_seconds=2.0)
    play_at = start_and_play(duel)

    assert duel.observe(0, duel.target, 0.95, play_at + 0.30)
    assert duel.phase == PHASE_PLAYING

    duel.tick(play_at + 2.4)
    assert duel.phase == PHASE_RESULT
    assert duel.winner == 0
    assert duel.current_round.times_ms[1] is None
    assert duel.current_round.margin_ms is None


def test_timeout_without_winner_gives_no_points():
    duel = make_duel(round_timeout=8.0)
    play_at = start_and_play(duel)

    duel.tick(play_at + 8.1)
    assert duel.phase == PHASE_RESULT
    assert duel.winner is None
    assert all(p.points == 0 for p in duel.players)


def test_equal_times_are_a_draw():
    duel = make_duel()
    play_at = start_and_play(duel)

    assert duel.observe(0, duel.target, 0.95, play_at + 0.5)
    assert duel.observe(1, duel.target, 0.95, play_at + 0.5)
    assert duel.phase == PHASE_RESULT
    assert duel.winner is None
    assert all(p.points == 0 for p in duel.players)


def test_wrong_label_or_low_confidence_does_not_score():
    duel = make_duel()
    play_at = start_and_play(duel)

    other = next(g for g in config.GESTURES if g != duel.target)
    assert not duel.observe(0, other, 0.99, play_at + 0.2)
    assert not duel.observe(0, duel.target, 0.1, play_at + 0.3)
    assert not duel.observe(0, None, 0.0, play_at + 0.4)
    assert duel.times_ms[0] is None
    assert duel.players[0].points == 0
    assert duel.players[0].misses >= 1


def test_observations_are_ignored_outside_playing():
    duel = make_duel()
    assert not duel.observe(0, "fist", 1.0, 0.0)

    duel.start(0.0)
    assert duel.phase == PHASE_COUNTDOWN
    assert not duel.observe(0, duel.target, 1.0, 1.0)


def test_second_start_is_rejected_while_playing():
    duel = make_duel()
    start_and_play(duel)
    assert not duel.start(4.0)


def test_match_finishes_after_last_round_and_restarts():
    duel = make_duel(total_rounds=1)
    play_at = start_and_play(duel)
    duel.observe(0, duel.target, 0.9, play_at + 0.4)
    duel.observe(1, duel.target, 0.9, play_at + 0.9)

    assert duel.phase == PHASE_FINISHED
    assert duel.match_winner == 0

    assert duel.start(play_at + 2.0)
    assert duel.phase == PHASE_COUNTDOWN
    assert duel.rounds_played == 0
    assert duel.players[0].points == 0


def test_targets_do_not_repeat_back_to_back():
    duel = make_duel(total_rounds=0)
    now = 0.0
    previous = None
    for _ in range(6):
        duel.start(now)
        now += 3.0
        duel.tick(now)
        assert duel.target != previous
        previous = duel.target
        duel.observe(0, duel.target, 0.99, now + 0.2)
        duel.observe(1, duel.target, 0.99, now + 0.4)
        now += 1.0


def test_elapsed_is_frozen_after_a_hit():
    duel = make_duel()
    play_at = start_and_play(duel)
    duel.observe(0, duel.target, 0.95, play_at + 0.4)
    assert duel.elapsed_ms(0, play_at + 0.4) == 400
    assert duel.elapsed_ms(0, play_at + 1.5) == 400
    assert duel.elapsed_ms(1, play_at + 1.5) == 1500


def test_reset_clears_everything():
    duel = make_duel(total_rounds=1)
    play_at = start_and_play(duel)
    duel.observe(0, duel.target, 0.9, play_at + 0.3)
    duel.observe(1, duel.target, 0.9, play_at + 0.6)
    duel.reset()

    assert duel.phase == PHASE_IDLE
    assert duel.history == []
    assert duel.target is None
    assert all(p.points == 0 for p in duel.players)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        Duel(num_players=0)
    with pytest.raises(ValueError):
        Duel(gestures=())
