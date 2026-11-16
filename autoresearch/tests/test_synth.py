# Tests of the synthetic road generator and its trip files.
#   venv\Scripts\python -m pytest autoresearch/tests/test_synth.py -q
import math

import numpy as np

from autoresearch.synth import collect, tracks
from core.sensors import clean_lidar


def test_valid_roads_keep_the_limits():
    for road in tracks.roads(0, 15):
        assert tracks.radius_of(road["line"]).min() >= tracks.MIN_RADIUS
        assert tracks.WIDTH_RANGE[0] * 0.75 <= road["width"].min()
        assert road["width"].max() <= tracks.WIDTH_RANGE[1] * 1.25
        assert tracks.closest_other_part(road["line"], 3 * road["width"].max() + 2) > road["width"].max()


def test_same_seed_gives_the_same_road():
    a, b = tracks.make_road(4), tracks.make_road(4)
    assert np.allclose(a["line"], b["line"]) and np.allclose(a["width"], b["width"])


def test_both_directions_appear():
    turns = []
    for road in tracks.roads(0, 30):
        line = road["line"]
        turns.append(np.sign(np.sum(line[:, 0] * np.roll(line[:, 1], -1) - np.roll(line[:, 0], -1) * line[:, 1])))
    assert (np.array(turns) > 0).sum() >= 5 and (np.array(turns) < 0).sum() >= 5


def test_lidar_sees_the_walls_of_a_straight_corridor():
    walls = np.array([[-50, 1.0, 50, 1.0], [-50, -0.6, 50, -0.6]])
    scan = collect.lidar(0.0, 0.0, 0.0, walls)
    left, right = np.argmin(np.abs(collect.ANGLES - math.pi / 2)), np.argmin(np.abs(collect.ANGLES + math.pi / 2))
    assert abs(scan[left] - 1.0) < 1e-6 and abs(scan[right] - 0.6) < 1e-6
    assert scan[np.argmin(np.abs(collect.ANGLES))] == collect.MAX_RANGE      # straight ahead: nothing within 10 m
    i10 = np.argmin(np.abs(collect.ANGLES - math.radians(10)))
    assert abs(scan[i10] - 1.0 / math.sin(collect.ANGLES[i10])) < 1e-6         # 10 deg left hits the left wall


def test_raw_beams_come_back_through_clean_lidar():
    scan = np.random.default_rng(0).uniform(0.2, 10, collect.BEAMS)
    assert np.allclose(clean_lidar(collect.to_raw(scan)) * 10, scan)


def test_driver_keeps_off_the_walls_of_a_wide_oval():
    road = tracks.make_road(1)                                    # an oval, 2.9 m wide
    walls = tracks.walls_of(road)
    seconds = [len(collect.drive_trip(road, walls, np.random.default_rng(s))["rows"]) * collect.DT for s in range(4)]
    assert np.mean(seconds) > 20


def test_speed_is_capped_by_the_top_and_slows_before_a_wall():
    open_road, wall_ahead = np.full(collect.BEAMS, 10.0), np.full(collect.BEAMS, 10.0)
    wall_ahead[collect.FRONT] = 0.8
    assert collect.target_speed(2.0, open_road) == 2.0
    assert collect.target_speed(4.0, wall_ahead) < 2.0


def test_trip_file_is_read_by_wm2_prepare(tmp_path):
    from wm2.prepare import trip_rows
    road = tracks.make_road(1)
    rows = [(collect.to_raw(np.full(collect.BEAMS, 2.0)), 0.06, 0.1, 1.0, k * collect.DT, (0.05 * k, 0.0), 0.0, 0.05,
             "gap") for k in range(40)]
    path = str(tmp_path / "drive_00001_0.npz")
    assert collect.save_trip(path, {"rows": rows, "reason": "timeout"}, road) == 40
    out = trip_rows(path, 0)
    assert out["lidar"].shape == (40, 109)
    assert abs(float(out["lidar"][0, 0]) - 0.2) < 1e-3
    assert np.allclose(out["progress"][:-1], 0.05, atol=1e-6)
