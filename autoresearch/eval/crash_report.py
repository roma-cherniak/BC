# Crash report: every crash of one or more drive logs (runs/<model>/drive_<track>_<tag>.csv) with WHERE and WHY.
#   venv\Scripts\python -m autoresearch.eval.crash_report runs/wm2_c4/drive_compete_c4night_*.csv
# Where: the sim puts the car back at the start after a crash and the lap counter goes to 0, so the metres driven since
#        the last reset or lap line = the place on the track (integrated speed; the log has no position).
# Why (the WINDOW seconds before the crash): speed, steering, which wall came closest, and what the planner had
#        imagined for those moments (<log>_imagined.csv) against what really came -> was the wall a surprise?
# Crashes in the first WARMUP s belong to the driver before (autoresearch/eval/drive_stats.py).

import glob
import sys
from dataclasses import dataclass

import numpy as np

from autoresearch.eval.drive_stats import WARMUP

WINDOW = 1.5            # s before a crash that are looked at
START_FRONT = 5.3       # m: the free road ahead at the start place (5.37 on compete) -> a reset row
SURPRISE = 0.3          # m: the real wall this much closer than imagined = the model did not see it coming


@dataclass(frozen=True)
class Crash:
    log: str
    seconds: float
    place_m: float          # metres from the start / lap line
    speed: float            # m/s at the last row before the crash
    top_speed: float        # m/s, highest in the window
    steering: float         # mean steering of the last 0.5 s (+ = left)
    wall: str               # front / right / left: the closest at the last row
    wall_m: float
    standing_before: bool   # the car stood (< 0.1 m/s) in the window: it started from a stop into the wall
    imagined_m: float       # imagined distance to that wall for the last rows (nan: no imagination logged)
    surprise: bool          # real wall SURPRISE m closer than imagined


def read_csv(path: str) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def places(d: np.ndarray) -> np.ndarray:
    # metres since the last reset (crash) or lap line, per row
    dt = np.diff(d["seconds"], prepend=d["seconds"][0])
    out, metres = np.zeros(len(d)), 0.0
    crashes, laps = d["crashes"], d["lap_count"]
    for i in range(len(d)):
        if i and (crashes[i] > crashes[i - 1] or laps[i] != laps[i - 1]):
            metres = 0.0
        metres += d["speed"][i] * dt[i]
        out[i] = metres
    return out


def imagined_walls(path: str) -> np.ndarray | None:
    try:
        return read_csv(path.replace(".csv", "_imagined.csv"))
    except OSError:
        return None


def crashes_of(path: str) -> list[Crash]:
    d = read_csv(path)
    imag = imagined_walls(path)
    where = places(d)
    s = d["seconds"]
    out = []
    for i in np.flatnonzero(np.diff(d["crashes"]) > 0) + 1:
        if s[i] < WARMUP:
            continue
        last = i - 1                                            # the last row before the sim moved the car
        win = (s >= s[i] - WINDOW) & (s < s[i])
        recent = (s >= s[i] - 0.5) & (s < s[i])
        walls = {"front": d["real_front"][last], "right": d["real_right"][last], "left": d["real_left"][last]}
        wall = min(walls, key=walls.get)
        imagined = float("nan")
        if imag is not None and len(imag):
            near = (imag["target_seconds"] >= s[i] - 0.5) & (imag["target_seconds"] < s[i])
            if near.any():
                imagined = float(np.min(imag[wall][near]))
        out.append(Crash(log=path, seconds=float(s[i]), place_m=float(where[last]), speed=float(d["speed"][last]),
                         top_speed=float(d["speed"][win].max()), steering=float(d["steering"][recent].mean()),
                         wall=wall, wall_m=float(walls[wall]), standing_before=bool((d["speed"][win] < 0.1).any()),
                         imagined_m=imagined,
                         surprise=bool(np.isfinite(imagined) and imagined - walls[wall] > SURPRISE)))
    return out


def lap_length(paths: list[str]) -> float | None:
    # median metres of the whole laps (lap counter up without a crash between)
    lengths = []
    for path in paths:
        d = read_csv(path)
        where = places(d)
        for i in np.flatnonzero(np.diff(d["lap_count"]) > 0) + 1:
            lengths.append(where[i - 1])
    lengths = [x for x in lengths if x > 5]
    return float(np.median(lengths)) if lengths else None


def report(paths: list[str], bin_m: float = 2.0) -> str:
    crashes = [c for p in paths for c in crashes_of(p)]
    lines = [f"{len(crashes)} crashes in {len(paths)} logs; lap length ~{lap_length(paths) or float('nan'):.1f} m", ""]
    lines.append("| log | s | place m | speed | top | steering | wall | wall m | imagined m | surprise | from stop |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted(crashes, key=lambda c: c.place_m):
        lines.append(f"| {c.log.split('drive_')[-1][:-4]} | {c.seconds:.1f} | {c.place_m:.1f} | {c.speed:.2f} | "
                     f"{c.top_speed:.2f} | {c.steering:+.2f} | {c.wall} | {c.wall_m:.2f} | {c.imagined_m:.2f} | "
                     f"{'yes' if c.surprise else ''} | {'yes' if c.standing_before else ''} |")
    if crashes:
        bins = np.floor(np.array([c.place_m for c in crashes]) / bin_m) * bin_m
        values, counts = np.unique(bins, return_counts=True)
        order = np.argsort(-counts)
        lines += ["", f"Crash places ({bin_m:.0f} m bins), most first:"]
        lines += [f"  {values[k]:5.0f} - {values[k] + bin_m:3.0f} m: {counts[k]}" for k in order]
        walls = {w: sum(c.wall == w for c in crashes) for w in ("front", "right", "left")}
        lines += ["", f"closest wall: {walls}; surprises (real {SURPRISE} m closer than imagined): "
                      f"{sum(c.surprise for c in crashes)}; from a stop: {sum(c.standing_before for c in crashes)}; "
                      f"mean speed at the crash {np.mean([c.speed for c in crashes]):.2f} m/s"]
    return "\n".join(lines)


if __name__ == "__main__":
    files = sorted({f for pattern in sys.argv[1:] for f in glob.glob(pattern) if not f.endswith("_imagined.csv")})
    print(report(files))
