# The judge, part 1: ONE drive log -> the numbers of that drive. Agents never edit this folder.
#
# A drive log is runs/<run>/drive_<track>_<tag>.csv, written by wm2/drive.py: one row per sim packet (~45 per s) with
# seconds, steering, throttle, speed, real_front / real_right / real_left (m), crashes (a counter that only grows),
# plan_ms and the sim's lap counter. This file only reads the log: no sim, no model, no graphics card.
#
# The rules come from archive/research/run_trial.py (2026-11-09) and from the test logs of 2026-11-10:
#   - a crash in the first WARMUP s is not this driver's: 9 of the 10 test logs start with the counter already at 1
#     (the sim resets the car at once), so the counter at the end of the warmup is where this driver starts
#   - the sim's lap counter jumps back to 0 on every reset, so laps are counted where it CHANGES, never read from the
#     last row. The first lap is dropped (it began under the driver before, or from a standing start), and so is a
#     "lap" longer than MAX_LAP s
#   - speed and wall clearance use the packets after the warmup only
# Gates (0 crashes, enough laps, sim fast enough) and the ONE score are not here: score.py + config.py (Mon 16).
#
# Run from the BC folder; it prints a markdown table, one row per log (wildcards work, *_imagined.csv is skipped):
#     venv\Scripts\python -m autoresearch.eval.drive_stats runs/wm2_self2/drive_practice_self2_*.csv

import csv
import glob
import math
import os
import statistics
import sys
from dataclasses import dataclass

import numpy as np

WARMUP = 5.0        # s: crashes before this belong to the driver before
MAX_LAP = 40.0      # s: a longer "lap" is a restart, not driving
REQUIRED = ("seconds", "throttle", "speed", "real_front", "real_right", "real_left", "crashes", "plan_ms")
LAP_COLUMNS = ("lap_count", "last_lap")     # missing in the oldest logs: then the drive has no laps


@dataclass(frozen=True)
class DriveStats:
    name: str               # the log's file name without ".csv"
    seconds: float          # s that were judged (after the warmup)
    crashes: int
    crashes_per_min: float
    laps: int               # full laps
    lap_median: float       # s (nan without a full lap)
    lap_best: float         # s (nan without a full lap)
    mean_speed: float       # m/s
    wall_clearance: float   # m: how close the car kept to the walls (see wall_clearance)
    gas_top: float          # the highest throttle in the log = the gas ceiling of this drive
    gas0: float             # share of the packets with throttle 0 (braking)
    packets_per_s: float    # below ~30 the sim was slow: on 2026-11-09 that turned 0 crashes into 12 for the same model
    plan_ms: float          # median time of one plan


def _number(text: str | None) -> float:
    try:
        return float(text or "")
    except ValueError:      # "" when the sim sent no lap fields
        return math.nan


def read_log(path: str) -> dict[str, np.ndarray]:
    # the log as one array per column; a half-written last row (the driver was killed) is dropped
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [name for name in REQUIRED if name not in header]
        if missing:
            raise ValueError(f"{path}: not a drive log (no column {', '.join(missing)})")
        rows = [row for row in reader if all(row[name] is not None for name in header)]
    log = {name: np.array([_number(row.get(name)) for row in rows]) for name in REQUIRED + LAP_COLUMNS}
    if len(rows) == 0 or not log["seconds"][-1] > WARMUP:
        raise ValueError(f"{path}: no drive after the first {WARMUP:.0f} s")
    return log


def count_crashes(seconds: np.ndarray, crashes: np.ndarray) -> int:
    # the counter only grows: what it gained after the warmup is this driver's
    start = int(np.argmax(seconds >= WARMUP))
    return int(crashes[-1] - crashes[start])


def lap_times(lap_count: np.ndarray, last_lap: np.ndarray) -> list[float]:
    # the full laps (s), in the order they were driven
    changed = np.flatnonzero(lap_count[1:] != lap_count[:-1]) + 1
    laps = [float(last_lap[i]) for i in changed if math.isfinite(last_lap[i])]    # a reset shows "Infinity"
    return [lap for lap in laps[1:] if lap < MAX_LAP]


def wall_clearance(front: np.ndarray, right: np.ndarray, left: np.ndarray) -> float:
    # ONE number in metres for "how close did the car keep to the walls in this drive" (bigger = safer).
    # front / right / left: the nearest wall of each sector (m), one value per packet after the warmup.
    # TODO(human): choose the definition and write it (2-6 lines).
    raise NotImplementedError("wall_clearance has no definition yet: see TODO(human) in autoresearch/eval/drive_stats.py")


def drive_stats(path: str) -> DriveStats:
    log = read_log(path)
    seconds = log["seconds"]
    judged = seconds >= WARMUP
    judged_seconds = float(seconds[-1] - WARMUP)
    crashes = count_crashes(seconds, log["crashes"])
    laps = lap_times(log["lap_count"], log["last_lap"])
    return DriveStats(
        name=os.path.splitext(os.path.basename(path))[0],
        seconds=judged_seconds,
        crashes=crashes,
        crashes_per_min=crashes / (judged_seconds / 60),
        laps=len(laps),
        lap_median=statistics.median(laps) if laps else math.nan,
        lap_best=min(laps) if laps else math.nan,
        mean_speed=float(log["speed"][judged].mean()),
        wall_clearance=float(wall_clearance(log["real_front"][judged], log["real_right"][judged],
                                            log["real_left"][judged])),
        gas_top=float(log["throttle"].max()),
        gas0=float((log["throttle"] == 0).mean()),
        packets_per_s=len(seconds) / max(float(seconds[-1] - seconds[0]), 1e-9),
        plan_ms=float(np.median(log["plan_ms"])),
    )


def table(stats: list[DriveStats]) -> str:
    # a markdown table: readable in the terminal, and it can be pasted into a report.md
    lines = ["| log | min | crashes | per min | laps | lap median s | lap best s | speed m/s | wall m | gas top | gas 0 "
             "| packets/s | plan ms |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in stats:
        lines.append(f"| {s.name} | {s.seconds / 60:.1f} | {s.crashes} | {s.crashes_per_min:.2f} | {s.laps} "
                     f"| {s.lap_median:.2f} | {s.lap_best:.2f} | {s.mean_speed:.2f} | {s.wall_clearance:.2f} "
                     f"| {s.gas_top:.2f} | {s.gas0:.0%} | {s.packets_per_s:.0f} | {s.plan_ms:.0f} |")
    return "\n".join(lines)


def find_logs(patterns: list[str]) -> list[str]:
    # Windows does not expand wildcards itself; the *_imagined.csv files next to the logs are not drive logs
    paths = [path for pattern in patterns for path in (sorted(glob.glob(pattern)) or [pattern])]
    return [path for path in paths if not path.endswith("_imagined.csv")]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: venv\\Scripts\\python -m autoresearch.eval.drive_stats <drive log .csv> [more ...]")
    print(table([drive_stats(path) for path in find_logs(sys.argv[1:])]))
