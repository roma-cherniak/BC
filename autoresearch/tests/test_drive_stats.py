# Tests of the judge's first part (autoresearch/eval/drive_stats.py) on small made-up drive logs: no sim needed.
#
# Run from the BC folder:
#     venv\Scripts\python -m pytest autoresearch/tests -q

import csv
import math
from collections.abc import Callable
from pathlib import Path

import pytest

from autoresearch.eval.drive_stats import count_crashes, drive_stats, find_logs, lap_times, read_log, wall_clearance

HEADER = ["seconds", "steering", "throttle", "speed", "real_front", "real_right", "real_left", "crashes", "plan_ms",
          "lap_count", "last_lap", "best_lap"]
RATE = 50       # packets per s in the made-up logs
INF = "Infinity"    # what the sim writes before the first lap and after a reset


def make_log(folder: Path, name: str = "drive_practice_try.csv", seconds: float = 60.0,
             crashes: Callable[[float], int] = lambda t: 0, speed: Callable[[float], float] = lambda t: 2.0,
             throttle: Callable[[float], float] = lambda t: 0.12, right: float = 0.6, left: float = 0.7,
             laps: tuple[tuple[float, int, float | str], ...] = ()) -> str:
    # laps: (from this second, the lap counter, the last lap time), e.g. (14.0, 1, 25.0)
    path = folder / name
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for i in range(int(seconds * RATE) + 1):
            t = i / RATE
            lap_count, last_lap = 0, INF
            for start, count, lap in laps:
                if t >= start:
                    lap_count, last_lap = count, lap
            writer.writerow([round(t, 2), 0.0, throttle(t), speed(t), 5.0, right, left, crashes(t), 40,
                             lap_count, last_lap, INF])
    return str(path)


def test_crash_in_the_warmup_is_not_counted(tmp_path: Path) -> None:
    # Arrange: the counter is already 1 in the first row (the sim resets the car), one real crash at 30 s
    log = read_log(make_log(tmp_path, crashes=lambda t: 1 if t < 30 else 2))

    # Act
    crashes = count_crashes(log["seconds"], log["crashes"])

    # Assert
    assert crashes == 1


def test_no_crash_after_the_warmup_gives_zero(tmp_path: Path) -> None:
    log = read_log(make_log(tmp_path, crashes=lambda t: 1))

    assert count_crashes(log["seconds"], log["crashes"]) == 0


def test_laps_drop_the_first_lap_resets_and_restarts(tmp_path: Path) -> None:
    # Arrange: first lap 25 s (began under the driver before), 15 s, a reset, 14 s, and a 45 s "lap" (a restart)
    laps = ((10.0, 1, 25.0), (25.0, 2, 15.0), (30.0, 0, INF), (44.0, 1, 14.0), (89.0, 2, 45.0))
    log = read_log(make_log(tmp_path, seconds=95.0, laps=laps))

    # Act
    times = lap_times(log["lap_count"], log["last_lap"])

    # Assert
    assert times == [15.0, 14.0]


def test_a_drive_without_a_finished_lap_has_no_laps(tmp_path: Path) -> None:
    log = read_log(make_log(tmp_path))

    assert lap_times(log["lap_count"], log["last_lap"]) == []


def test_log_shorter_than_the_warmup_is_refused(tmp_path: Path) -> None:
    path = make_log(tmp_path, seconds=3.0)

    with pytest.raises(ValueError, match="no drive"):
        read_log(path)


def test_file_that_is_not_a_drive_log_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "drive_practice_try_imagined.csv"
    path.write_text("seconds,imagined_front\n0.0,1.0\n")

    with pytest.raises(ValueError, match="not a drive log"):
        read_log(str(path))


def test_half_written_last_row_is_dropped(tmp_path: Path) -> None:
    # Arrange: the driver was killed in the middle of a row
    path = make_log(tmp_path, seconds=10.0)
    with open(path, "a") as f:
        f.write("10.02,0.0,0.1")

    # Act
    log = read_log(path)

    # Assert
    assert log["seconds"][-1] == 10.0


def test_find_logs_skips_the_imagined_files(tmp_path: Path) -> None:
    make_log(tmp_path, name="drive_practice_a.csv", seconds=6.0)
    make_log(tmp_path, name="drive_practice_a_imagined.csv", seconds=6.0)

    found = find_logs([str(tmp_path / "drive_practice_*.csv")])

    assert [Path(path).name for path in found] == ["drive_practice_a.csv"]


# ---- the tests below need wall_clearance (TODO(human) in drive_stats.py): red until it is written ----

def test_wall_clearance_is_smaller_for_a_drive_closer_to_the_wall(tmp_path: Path) -> None:
    # Arrange
    middle = read_log(make_log(tmp_path, name="middle.csv", right=0.65, left=0.65))
    close = read_log(make_log(tmp_path, name="close.csv", right=0.30, left=1.00))

    # Act
    middle_m = wall_clearance(middle["real_front"], middle["real_right"], middle["real_left"])
    close_m = wall_clearance(close["real_front"], close["real_right"], close["real_left"])

    # Assert
    assert 0 <= close_m < middle_m


def test_drive_stats_gives_the_numbers_of_one_drive(tmp_path: Path) -> None:
    # Arrange: 65 s = 60 s judged; the car stands for the 5 s of warmup, then 2 m/s; one real crash; brakes 65-end
    laps = ((10.0, 1, 25.0), (25.0, 2, 15.0), (39.0, 3, 14.0))
    path = make_log(tmp_path, seconds=65.0, crashes=lambda t: 1 if t < 30 else 2,
                    speed=lambda t: 0.0 if t < 5 else 2.0, throttle=lambda t: 0.0 if t < 5 else 0.12, laps=laps)

    # Act
    stats = drive_stats(path)

    # Assert
    assert stats.name == "drive_practice_try"
    assert stats.seconds == pytest.approx(60.0)
    assert stats.crashes == 1
    assert stats.crashes_per_min == pytest.approx(1.0)
    assert (stats.laps, stats.lap_median, stats.lap_best) == (2, 14.5, 14.0)
    assert stats.mean_speed == pytest.approx(2.0)
    assert stats.gas_top == pytest.approx(0.12)
    assert stats.gas0 == pytest.approx(5 / 65, abs=0.01)
    assert stats.packets_per_s == pytest.approx(RATE, rel=0.01)
    assert stats.plan_ms == 40
    assert math.isfinite(stats.wall_clearance)
