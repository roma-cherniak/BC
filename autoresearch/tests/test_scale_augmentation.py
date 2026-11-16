# The "bigger / smaller world" view of a training window (wm2/data.py windows, scale_steer).
import math

import torch

from wm2.data import random_scales, scale_steer, windows


def make_data():
    # 6 moments, 5 beams: walls at 1 m and 2 m, one beam at the lidar's range (1.0 = 10 m)
    lidar = torch.tensor([[0.1, 0.2, 1.0, 0.2, 0.1]] * 6).half()
    state = torch.tensor([[0.5, 0.4]] * 6)              # speed 1.5 m/s, wheel 12 degrees
    action = torch.tensor([[0.6, 0.4]] * 6)
    return {"lidar": lidar, "state": state, "action": action}


def test_scale_one_changes_nothing():
    # Arrange
    data, starts = make_data(), torch.tensor([0, 2])

    # Act
    plain = windows(data, starts, 3)
    scaled = windows(data, starts, 3, None, torch.tensor([1.0, 1.0]))

    # Assert
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(plain, scaled))


def test_a_world_twice_as_big_doubles_distances_speed_and_throttle():
    # Arrange
    data = make_data()

    # Act
    lidar, state, action = windows(data, torch.tensor([0]), 2, None, torch.tensor([2.0]))

    # Assert: walls at 2 m and 4 m; the beam at the range stays at the range
    assert torch.allclose(lidar[0, 0], torch.tensor([0.2, 0.4, 1.0, 0.4, 0.2]), atol=1e-3)
    assert abs(state[0, 0, 0].item() - 1.0) < 1e-6 and abs(action[0, 0, 0].item() - 1.2) < 1e-6


def test_a_beam_pushed_beyond_the_range_is_cut_to_the_range():
    # Arrange: the 2 m wall times 8 would be 16 m
    data = make_data()

    # Act
    lidar, _, _ = windows(data, torch.tensor([0]), 1, None, torch.tensor([8.0]))

    # Assert
    assert lidar.max().item() <= 1.0 and abs(lidar[0, 0, 0].item() - 0.8) < 1e-3


def test_the_wheel_angle_keeps_the_same_path_in_the_bigger_world():
    # Arrange: 12 degrees of wheel = a circle of radius wheelbase / tan(12 deg)
    steer = torch.tensor(0.4)

    # Act
    bigger = scale_steer(steer, torch.tensor(2.0))

    # Assert: twice the radius needs tan(angle) / 2; the sign is kept; scale 1 keeps the angle
    assert abs(math.tan(bigger.item() * 0.5236) - math.tan(0.4 * 0.5236) / 2) < 1e-6
    assert scale_steer(-steer, torch.tensor(2.0)).item() == -bigger.item()
    assert abs(scale_steer(steer, torch.tensor(1.0)).item() - 0.4) < 1e-6


def test_random_scales_stay_in_range_and_leave_some_windows_alone():
    # Act
    torch.manual_seed(0)
    scale = random_scales(10000, (0.6, 2.2), 0.7)

    # Assert
    assert scale.min().item() >= 0.6 and scale.max().item() <= 2.2
    assert 0.25 < (scale == 1.0).float().mean().item() < 0.35
