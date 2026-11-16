# The "right answers" of the value head (wm2/value.py): metres driven in the next steps of the same trip.
import numpy as np

from wm2.value import future_metres


def test_sums_the_next_steps_inside_one_trip():
    # Arrange: one trip of 6 moments, 1 m per step, no crash
    trip, progress, danger = np.zeros(6, int), np.ones(6, np.float32), np.zeros(6, np.uint8)

    # Act
    target, known = future_metres(trip, progress, danger, steps=3)

    # Assert: moments 0..3 have 3 full steps ahead; 4 and 5 run out of trip without a crash -> unknown
    assert target[:4].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert known.tolist() == [True, True, True, True, False, False]


def test_a_crash_ends_the_sum_and_the_answer_is_known():
    # Arrange: the trip crashes at its last moment (danger 1 there, no progress on the crash row)
    trip = np.zeros(5, int)
    progress = np.array([1, 1, 1, 1, 0], np.float32)
    danger = np.array([0, 0, 1, 1, 1], np.uint8)

    # Act
    target, known = future_metres(trip, progress, danger, steps=3)

    # Assert: near the crash the car only drives what is left
    assert target.tolist() == [3.0, 3.0, 2.0, 1.0, 0.0]
    assert known.all()


def test_the_sum_never_crosses_into_the_next_trip():
    # Arrange: trip 0 crashes after 2 moments, trip 1 drives on
    trip = np.array([0, 0, 1, 1, 1, 1])
    progress = np.array([1, 0, 5, 5, 5, 5], np.float32)
    danger = np.array([1, 1, 0, 0, 0, 0], np.uint8)

    # Act
    target, known = future_metres(trip, progress, danger, steps=3)

    # Assert
    assert target[:2].tolist() == [1.0, 0.0]
    assert target[2] == 15.0
    assert known.tolist() == [True, True, True, True, False, False]
