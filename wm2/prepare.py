# wm2 data: every recorded trip as a SEQUENCE (each moment stored once + its trip number), not as (now, next) pairs.
# The wm2 model looks at a history of frames and imagines many steps ahead, so it needs whole stretches of trips.
#
# Per split (the same trips as data/train.txt, val.txt, test.txt -> fair comparison with the old models) it writes
# data/wm2/<split>.npz with one row per recorded moment:
#   lidar     (N, 109) float16  cleaned like core.sensors.clean_lidar (0..1), what the car sees
#   state     (N, 2)   float32  [speed / SPEED_MAX, wheel angle / MAX_STEER_RAD]: the AIL Cup car reports both too
#   action    (N, 2)   float32  [throttle / THROTTLE_MAX, steering -1..1]: what was commanded at this moment
#   trip      (N,)     int32    trip number: a window of frames is valid only inside one trip
#   progress  (N,)     float32  m driven forward (along the car's heading) from this moment to the next one
#   danger    (N,)     uint8    1 = within DANGER_STEPS the safety driver had to take over, or the trip crashed
#   mode, time                  who steered (random / gap / safety / warmup) and the trip clock, for the checks
# Mirroring (left <-> right) is NOT stored: wm2/data.py does it on the fly while training.
#
# Run from the BC folder:  venv\Scripts\python -m wm2.prepare

import os

import numpy as np

from config import SPEED_MAX, THROTTLE_MAX
from core.sensors import clean_lidar

OUT_DIR = "data/wm2"
MAX_STEER_RAD = 0.5236      # rad, full steering of the AutoDRIVE car (30 degrees): wheel angle -> -1 .. 1
DANGER_STEPS = 10           # 10 x 0.05 s = 0.5 s, the same window as checks/check_planner


def trip_rows(path, number):
    with np.load(path) as f:
        lidar = clean_lidar(f["lidar"]).astype(np.float16)
        state = np.stack([f["speed"] / SPEED_MAX, f["real_steering"] / MAX_STEER_RAD], axis=1)
        action = np.stack([f["throttle"] / THROTTLE_MAX, f["steering"]], axis=1)
        position, yaw = f["position"].astype(float), f["orientation"].astype(float)
        mode, time, crashed = f["mode"], f["time"], str(f["reason"]) == "crash"

    n = len(lidar)
    step = np.diff(position, axis=0)
    progress = np.zeros(n, np.float32)
    progress[:-1] = step[:, 0] * np.cos(yaw[:-1]) + step[:, 1] * np.sin(yaw[:-1])
    if crashed:
        progress[-2:] = 0           # the simulator moves the car on the crash row: that jump is not driving

    safety = mode == "safety"
    danger = np.array([safety[i + 1 : i + 1 + DANGER_STEPS].any() for i in range(n)])
    if crashed:
        danger[-DANGER_STEPS:] = True

    return {"lidar": lidar, "state": state.astype(np.float32), "action": action.astype(np.float32),
            "trip": np.full(n, number, np.int32), "progress": progress, "danger": danger.astype(np.uint8),
            "mode": mode.astype("U7"), "time": time.astype(np.float32)}


def prepare(split):
    with open(f"data/{split}.txt") as f:
        files = f.read().split()
    parts = [trip_rows(path, number) for number, path in enumerate(files)]
    data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    np.savez(f"{OUT_DIR}/{split}.npz", **data)
    print(f"{split}: {len(files)} trips, {len(data['trip'])} moments, danger {data['danger'].mean():.1%}, "
          f"progress {data['progress'].mean() / 0.05:.2f} m/s on average")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in ["train", "val", "test"]:
        prepare(split)
