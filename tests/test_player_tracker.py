"""Pruebas de la asociacion de manos a jugadores y de la lectura por jugador."""

from __future__ import annotations

import numpy as np

from gesture_rush.player_tracker import DuelTracker, assign_hands, zone_center
from gesture_rush.predictor import NO_HAND


class FakeHand:
    """Mano minima compatible con ``assign_hands``."""

    def __init__(self, x: float, code: int = 0, index: int = 0) -> None:
        self.centroid = np.array([x, 0.5])
        self.landmarks = np.zeros((21, 3))
        self.landmarks[0, 0] = float(code)
        self.handedness = "Right"
        self.index = index


class FakePredictor:
    """Clasificador falso: la etiqueta viaja en ``landmarks[0, 0]``."""

    LABELS = {0: "fist", 1: "victory"}

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, landmarks):
        if landmarks is None:
            return NO_HAND, 0.0
        self.calls += 1
        label = self.LABELS.get(int(landmarks[0, 0]), "fist")
        return label, 0.92


def test_zone_centers_split_the_frame():
    assert zone_center(0, 2) == 0.25
    assert zone_center(1, 2) == 0.75
    assert zone_center(1, 3) == 0.5


def test_one_hand_per_half():
    left = FakeHand(0.18, index=0)
    right = FakeHand(0.81, index=1)

    mapping = assign_hands([left, right])

    assert mapping[0] is left
    assert mapping[1] is right


def test_single_hand_goes_to_its_own_half():
    right = FakeHand(0.85, index=0)
    mapping = assign_hands([right])

    assert mapping[0] is None
    assert mapping[1] is right


def test_two_hands_in_the_same_half_leave_the_other_player_empty():
    closer = FakeHand(0.10, index=0)
    farther = FakeHand(0.44, index=1)

    mapping = assign_hands([farther, closer])

    assert mapping[0] is closer
    assert mapping[1] is None


def test_three_player_bands():
    hands = [FakeHand(0.15, index=0), FakeHand(0.50, index=1), FakeHand(0.86, index=2)]
    mapping = assign_hands(hands, num_players=3)

    assert mapping[0] is hands[0]
    assert mapping[1] is hands[1]
    assert mapping[2] is hands[2]


def test_tracker_predicts_per_player_and_ignores_missing_hands():
    tracker = DuelTracker(FakePredictor(), num_players=2, window=3, min_votes=2)

    readings = tracker.update([FakeHand(0.2, code=0, index=0), FakeHand(0.8, code=1, index=1)])

    assert readings[0].present and readings[0].label == "fist"
    assert readings[1].present and readings[1].label == "victory"
    # Un solo voto todavia no alcanza para declarar la etiqueta estable.
    assert readings[0].stable is None
    assert readings[1].stable is None

    readings = tracker.update([FakeHand(0.2, code=0, index=0), FakeHand(0.8, code=1, index=1)])
    assert readings[0].stable == "fist"
    assert readings[1].stable == "victory"


def test_tracker_reports_absent_player():
    tracker = DuelTracker(FakePredictor(), num_players=2, window=3, min_votes=2)

    readings = tracker.update([])

    assert [r.present for r in readings] == [False, False]
    assert [r.stable for r in readings] == [None, None]


def test_tracker_reset_forgets_smoothing():
    tracker = DuelTracker(FakePredictor(), num_players=2, window=3, min_votes=2)
    hands = [FakeHand(0.2, code=0, index=0), FakeHand(0.8, code=1, index=1)]
    tracker.update(hands)
    tracker.update(hands)
    assert tracker.update(hands)[0].stable == "fist"

    tracker.reset()
    assert tracker.update(hands)[0].stable is None
