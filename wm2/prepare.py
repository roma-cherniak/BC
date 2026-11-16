# wm2 data: every recorded trip as a SEQUENCE (each moment stored once + its trip number), not as (now, next) pairs.
# The wm2 model looks at a history of frames and imagines many steps ahead, so it needs whole stretches of trips.
#
# Per split (the same trips as data/train.txt, val.txt, test.txt -> fair comparison with the old models) it writes
# data/wm2/<split>.npz (or data/wm2_fast/ ...) with one row per recorded moment:
#   lidar     (N, 109) float16  cleaned like core.sensors.clean_lidar (0..1), what the car sees
#   state     (N, 2)   float32  [speed / SPEED_MAX, wheel angle / MAX_STEER_RAD]: the AIL Cup car reports both too
#   action    (N, 2)   float32  [throttle / THROTTLE_MAX, steering -1..1]: what was commanded at this moment
#   trip      (N,)     int32    trip number: a window of frames is valid only inside one trip
#   progress  (N,)     float32  m driven forward (along the car's heading) from this moment to the next one
#   danger    (N,)     uint8    1 = within DANGER_STEPS the safety driver had to take over, or the trip crashed
#   mode, time                  who steered (random / gap / safety / warmup) and the trip clock, for the checks
# Mirroring (left <-> right) is NOT stored: wm2/data.py does it on the fly while training.
#
# Run from the BC folder:  venv\Scripts\python -m wm2.prepare [old|fast|self]   (fast = old + data/fast_practice trips)
# New trips -> split lists first:  venv\Scripts\python -m wm2.prepare split data/self_practice self

import glob
import os
import random
import sys

import numpy as np

from config import SPEED_MAX, THROTTLE_MAX
from core.sensors import clean_lidar

# data sets: name -> output folder and the split lists it joins (trip numbers stay unique across the lists)
SETS = {
    "old": ("data/wm2", {"train": ["data/train.txt"], "val": ["data/val.txt"], "test": ["data/test.txt"]}),
    "fast": ("data/wm2_fast", {s: [f"data/{s}.txt", f"data/fast_{s}.txt"] for s in ["train", "val", "test"]}),
    # + the trips the planner drove itself (wm2/drive.py ... explore -> data/self_practice, split with "split")
    "self": ("data/wm2_self", {s: [f"data/{s}.txt", f"data/fast_{s}.txt", f"data/self_{s}.txt"]
                               for s in ["train", "val", "test"]}),
    # round C2 (2026-11-10): + the trips of the faster driver with soft brakes (data/self2_practice)
    "self2": ("data/wm2_self2", {s: [f"data/{s}.txt", f"data/fast_{s}.txt", f"data/self_{s}.txt", f"data/self2_{s}.txt"]
                                 for s in ["train", "val", "test"]}),
}
# 2026-11-11, the first set with TWO tracks: self2 + the compete trips of the night 11 -> 12 Nov. "fastc" = random
# trips (data/fast_compete), "selfc" = self-driven trips (data/self_compete). The night may skip the self-driven
# part, so a list that does not exist is left out.
SETS["c1"] = ("data/wm2_c1", {s: SETS["self2"][1][s] + [p for p in (f"data/fastc_{s}.txt", f"data/selfc_{s}.txt")
                                                        if os.path.exists(p)] for s in ["train", "val", "test"]})
# 2026-11-12: + the compete trips wm2_c1 drove itself after the cost and look-ahead changes of E15 - E17:
# "selfc2" = data/self2_compete (centre term off, 30 min), "selfc3" = data/self3_compete (capped centre term,
# 1.5 s look-ahead, the first trips with whole laps)
# + "selfc4" - "selfc6" = data/self4_compete - self6_compete: the three 10 min rounds with the value head (E20)
SETS["c2"] = ("data/wm2_c2", {s: SETS["c1"][1][s] + [p for p in (f"data/selfc{n}_{s}.txt" for n in range(2, 7))
                                                     if os.path.exists(p)] for s in ["train", "val", "test"]})
# 2026-11-14: the same trips as c2, stored with track + source per moment (A1 + A2 of docs/data_solve.md, wm2_c4)
SETS["c4"] = ("data/wm2_c4", SETS["c2"][1])
MAX_STEER_RAD = 0.5236      # rad, full steering of the AutoDRIVE car (30 degrees): wheel angle -> -1 .. 1
DANGER_STEPS = 10           # 10 x 0.05 s = 0.5 s, the same window as checks/check_planner
GAP = 0.3                   # s: rows further apart than this are two pieces of the trip (the game paused sending:
                            # self trip 2 of 2026-11-09 had a 21 s hole). Old trips: at most 0.067 s, so unchanged.


def trip_rows(path, number):
    with np.load(path) as f:
        lidar = clean_lidar(f["lidar"]).astype(np.float16)
        state = np.stack([f["speed"] / SPEED_MAX, f["real_steering"] / MAX_STEER_RAD], axis=1)
        action = np.stack([f["throttle"] / THROTTLE_MAX, f["steering"]], axis=1)
        position, yaw = f["position"].astype(float), f["orientation"].astype(float)
        mode, time, crashed = f["mode"], f["time"], str(f["reason"]) == "crash"
        track = str(f["track"]) if "track" in f.files else "practice"

    n = len(lidar)
    step = np.diff(position, axis=0)
    progress = np.zeros(n, np.float32)
    progress[:-1] = step[:, 0] * np.cos(yaw[:-1]) + step[:, 1] * np.sin(yaw[:-1])
    if crashed:
        progress[-2:] = 0           # the simulator moves the car on the crash row: that jump is not driving
    hole = np.diff(time) > GAP                                       # row i -> i+1 jumps over a hole
    progress[:-1][hole] = 0
    piece = np.concatenate([[0], np.cumsum(hole)])                   # 0, 0, ..., 1, 1, ... after every hole

    safety = mode == "safety"
    danger = np.array([safety[i + 1 : i + 1 + DANGER_STEPS].any() for i in range(n)])
    if crashed:
        danger[-DANGER_STEPS:] = True

    return {"lidar": lidar, "state": state.astype(np.float32), "action": action.astype(np.float32),
            "trip": (number + piece).astype(np.int32), "progress": progress, "danger": danger.astype(np.uint8),
            "mode": mode.astype("U7"), "time": time.astype(np.float32),
            # 2026-11-14 (A1 of docs/data_solve.md): where each moment comes from, for the balanced sampling
            "track": np.full(n, track[:8], dtype="U8"),
            "source": np.full(n, os.path.basename(os.path.dirname(os.path.abspath(path)))[:16], dtype="U16")}


def prepare(out_dir, split, lists):
    files = []
    for path in lists:
        with open(path) as f:
            files += f.read().split()
    parts, number = [], 0
    for path in files:                                               # every piece of a trip gets its own number
        parts.append(trip_rows(path, number))
        number = int(parts[-1]["trip"][-1]) + 1
    data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    np.savez(f"{out_dir}/{split}.npz", **data)
    print(f"{split}: {len(files)} trips, {len(data['trip'])} moments, danger {data['danger'].mean():.1%}, "
          f"progress {data['progress'].mean() / 0.05:.2f} m/s on average")


def split_trips(folder, name, seed=42):
    # every trip file of the folder -> data/<name>_train.txt / _val.txt / _test.txt, 80 / 10 / 10 by trip (like fast_*)
    files = sorted(glob.glob(os.path.join(folder, "drive_*.npz")))
    random.Random(seed).shuffle(files)
    n_val = n_test = max(1, len(files) // 10)
    parts = {"test": files[:n_test], "val": files[n_test : n_test + n_val], "train": files[n_test + n_val :]}
    for split, part in parts.items():
        path = f"data/{name}_{split}.txt"
        if os.path.exists(path):
            exit(f"{path} already exists: nothing is overwritten")
        with open(path, "w") as f:
            f.write("\n".join(part) + "\n")
        print(f"{path}: {len(part)} trips")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "split":          # -m wm2.prepare split data/self_practice self
        split_trips(sys.argv[2], sys.argv[3])
        sys.exit()
    out_dir, splits = SETS[sys.argv[1] if len(sys.argv) > 1 else "old"]
    os.makedirs(out_dir, exist_ok=True)
    for split, lists in splits.items():
        prepare(out_dir, split, lists)
