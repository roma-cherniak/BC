# The scenes of the corridor test (autoresearch/eval/corridor_test.py): is the drawn lidar scan the right one?
import math

import numpy as np

from autoresearch.eval.corridor_test import BEAMS, scan_of, scenes, truth, walls


def beam(angle_deg: float) -> int:
    # the beam that looks this many degrees to the left of straight ahead
    return round((angle_deg + 135) / 270 * (BEAMS - 1))


def test_a_straight_corridor_shows_its_walls_at_their_distance():
    # Arrange: 1.3 m to each side, the car 0.4 m left of the middle, heading along the corridor
    segments = walls(1.3, None)

    # Act
    scan = scan_of(0.0, 0.4, 0.0, segments)

    # Assert: the beams at -90 and +90 degrees hit the walls at right angles; ahead is beyond the lidar's range
    assert abs(scan[beam(-90)] - 1.7) < 1e-6
    assert abs(scan[beam(90)] - 0.9) < 1e-6
    assert scan[beam(0)] == 10.0


def test_a_slanted_beam_is_longer_by_the_sine_rule():
    # Arrange
    segments = walls(1.0, None)

    # Act
    scan = scan_of(0.0, 0.0, 0.0, segments)

    # Assert: the beam 30 degrees left of ahead meets the left wall after 1 / sin(30 deg) = 2 m
    assert abs(scan[beam(30)] - 1.0 / math.sin(math.radians(30))) < 1e-6


def test_a_left_bend_is_the_mirror_image_of_a_right_bend():
    # Arrange: the same place in a bend to the left and in a bend to the right
    left = scan_of(0.0, 0.2, math.radians(10), walls(0.8, 4.0))
    right = scan_of(0.0, -0.2, math.radians(-10), walls(0.8, -4.0))

    # Assert: reading one scan from the other end gives the other scan
    assert np.allclose(left, right[::-1], atol=1e-3)


def test_in_a_bend_the_inner_wall_is_nearer_than_the_outer_one_ahead():
    # Arrange: a left bend of 4 m radius, 0.8 m to each wall, the car in the middle
    scan = scan_of(0.0, 0.0, 0.0, walls(0.8, 4.0))

    # Assert: both side walls 0.8 m away; straight ahead the car looks at the OUTER wall, a few metres off
    assert abs(scan[beam(90)] - 0.8) < 1e-3 and abs(scan[beam(-90)] - 0.8) < 1e-3
    ahead = math.sqrt(4.8 ** 2 - 4.0 ** 2)              # where the tangent from the middle meets the outer wall
    assert abs(scan[beam(0)] - ahead) < 1e-2


def test_truth_gives_the_nearest_wall_of_each_of_the_13_sectors():
    # Arrange
    scan = scan_of(0.0, 0.0, 0.0, walls(0.65, None))

    # Act
    sectors = truth(scan)

    # Assert: right and left sectors see the wall at 0.65 m; the list has every width and no empty scene
    assert sectors.shape == (13,)
    assert abs(sectors[0] - 0.65) < 1e-6 and abs(sectors[12] - 0.65) < 1e-6
    assert len(scenes()) > 100 and all(np.isfinite(scan).all() for _, scan in scenes())
