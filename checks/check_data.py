# Check every episode file before the big (night) collection.
# Run from the BC folder:  venv\Scripts\python -m checks.check_data

import glob
import os
import numpy as np
from config import (DATA_DIR, THROTTLE_MIN, THROTTLE_MAX, EPISODE_TIME, BRAKE_TIME,
                    WAIT_MIN, WAIT_MAX, RECORD_STEP)     # the same numbers collection_random.py uses

MODES = ("random", "gap", "safety")   # who steered (policy.py)

BEAMS = 1081            # lidar beams in one scan
MAX_STEER_RAD = 0.5236  # full steering (30 degrees) in radians
LAG_GAP = 0.1           # time gap between two rows bigger than this = the game was slow for a moment (s)
MAX_GAP = 0.2           # bigger than this = the game froze -> problem (s)
MAX_JUMP = 0.5         # biggest allowed position jump between two rows (m)
MAX_SPEED = 5           # m/s, our car never drives faster
SHORT_ROWS = 20         # episodes shorter than this (1 s) are "very short"

ROW_KEYS = ["lidar", "throttle", "steering", "speed", "time", "collisions", "image",
            "position", "orientation", "yaw_rate", "real_steering", "mode"]  # one value per row
EPISODE_KEYS = ["track", "reason", "warmup"]                                 # one value per file


def check_shapes(d):
    problems = []
    rows = len(d["time"])
    for k in ROW_KEYS:
        if len(d[k]) != rows:
            problems.append(f"{k} has {len(d[k])} rows, time has {rows}")
    if d["lidar"].shape[1:] != (BEAMS,):
        problems.append(f"lidar shape {d['lidar'].shape}")
    if d["position"].shape[1:] != (2,):
        problems.append(f"position shape {d['position'].shape}")
    for k in ROW_KEYS:
        if k not in ("image", "mode") and np.isnan(d[k].astype(float)).any():
            problems.append(f"{k} has NaN")
    return problems


def check_time(d):
    problems = []
    t = d["time"]
    if t[0] > LAG_GAP:
        problems.append(f"first row at {t[0]:.3f} s (should be near 0)")
    if t[-1] > EPISODE_TIME + LAG_GAP:
        problems.append(f"last row at {t[-1]:.2f} s")
    if len(t) > 1:
        steps = np.diff(t)
        if (steps <= 0).any():
            problems.append("time goes backwards or stands still")
        if steps.max() > MAX_GAP:
            problems.append(f"time gap {steps.max():.3f} s (game froze)")
    return problems


def check_end(d, reason):
    problems = []
    hits = d["collisions"]
    if reason == "timeout":
        if hits.any():
            problems.append("timeout episode has collisions")
        if d["time"][-1] < EPISODE_TIME - 1:
            problems.append(f"timeout, but last row at {d['time'][-1]:.1f} s")
    elif reason == "crash":
        if hits[-1] == 0 or hits[:-1].any():
            problems.append("crash episode: the collision must be only in the last row")
    else:
        problems.append(f"unknown reason '{reason}'")
    if str(d["track"]) not in ("practice", "compete"):
        problems.append(f"unknown track '{d['track']}'")
    if not WAIT_MIN <= int(d["warmup"]) <= WAIT_MAX:
        problems.append(f"warmup {int(d['warmup'])} is outside {WAIT_MIN}..{WAIT_MAX}")
    return problems


def check_actions(d):
    problems = []
    t, throttle = d["time"], d["throttle"]
    driving = throttle[t <= BRAKE_TIME]
    braking = throttle[t > BRAKE_TIME]
    ok = (driving == 0) | ((driving >= THROTTLE_MIN) & (driving <= THROTTLE_MAX))     # 0 = coasting
    if not ok.all():
        problems.append(f"throttle outside 0 or {THROTTLE_MIN}..{THROTTLE_MAX}")
    if (braking != 0).any():
        problems.append("throttle is not 0 after the brake time")
    if np.abs(d["steering"]).max() > 1:
        problems.append("steering is outside -1..1")
    if np.abs(d["real_steering"]).max() > MAX_STEER_RAD + 0.01:
        problems.append("real_steering is bigger than 30 degrees")
    unknown = set(d["mode"].tolist()) - set(MODES)
    if unknown:
        problems.append(f"unknown mode {unknown}")
    return problems


def check_sensors(d, reason):
    problems = []
    if d["speed"].min() < 0 or d["speed"].max() > MAX_SPEED:
        problems.append(f"speed {d['speed'].min():.2f}..{d['speed'].max():.2f} m/s")
    if d["lidar"].min() < 0:
        problems.append("lidar has negative distances")
    not_jpeg = sum(img[:2] != b"\xff\xd8" for img in d["image"])     # every JPEG starts with FF D8
    if not_jpeg > 0:
        problems.append(f"{not_jpeg} images are not JPEG")
    o = d["orientation"]
    if o.min() < -0.01 or o.max() > 2 * np.pi + 0.01:   # -0.001 happens near the 0/6.28 wrap - that's fine
        problems.append(f"orientation {o.min():.2f}..{o.max():.2f} is outside 0..6.28")
    jumps = np.linalg.norm(np.diff(d["position"], axis=0), axis=1)
    if reason == "crash":
        jumps = jumps[:-1]               # the simulator moves the car on the crash row - that's normal
    if len(jumps) > 0 and jumps.max() > MAX_JUMP:
        problems.append(f"position jumps {jumps.max():.2f} m between two rows")
    return problems


def check_file(path):
    """Return (problems, info). Empty problems list = the file is good."""
    with np.load(path) as f:
        keys = set(f.files)
        expected = set(ROW_KEYS + EPISODE_KEYS)
        if keys != expected:
            return [f"keys: missing {expected - keys}, extra {keys - expected}"], None
        d = {k: f[k] for k in f.files}

    problems = check_shapes(d)
    if problems:                         # wrong shapes break the other checks
        return problems, None
    reason = str(d["reason"])
    problems = check_time(d) + check_end(d, reason) + check_actions(d) + check_sensors(d, reason)
    driving = d["time"] <= BRAKE_TIME                        # only while driving, the brake stops the car
    if reason == "crash":
        driving[-1] = False                                  # the car stands at the wall on the crash row
    speed = d["speed"][driving]
    if len(speed) == 0:
        speed = d["speed"]
    info = {"rows": len(d["time"]),
            "reason": reason,
            "throttle": d["throttle"][d["time"] <= BRAKE_TIME],
            "speed_min": speed.min(),
            "speed_max": speed.max(),
            "speed_spread": speed.max() - speed.min(),        # how much the speed changes inside one trip
            "mode": d["mode"],
            "steering": d["steering"],
            "lags": int((np.diff(d["time"]) > LAG_GAP).sum()),
            "start": d["position"][0]}
    return problems, info


def print_summary(files, infos, bad_files):
    numbers = [int(os.path.basename(p)[6:10]) for p in files]          # drive_0007.npz -> 7
    missing = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers))
    size_mb = sum(os.path.getsize(p) for p in files) / 1e6
    good = [i for i in infos if i is not None]
    if not good:
        print("No readable files")
        return

    rows = sum(i["rows"] for i in good)
    crashes = sum(i["reason"] == "crash" for i in good)
    short = sum(i["rows"] < SHORT_ROWS for i in good)
    lags = sum(i["lags"] for i in good)
    lag_files = sum(i["lags"] > 0 for i in good)
    throttles = np.concatenate([i["throttle"] for i in good])
    modes = np.concatenate([i["mode"] for i in good])
    steering = np.concatenate([i["steering"] for i in good])
    random_steer = steering[modes == "random"]
    starts = np.array([i["start"] for i in good])
    per_episode = size_mb / len(files)

    print()
    print(f"Files:       {len(files)}   missing numbers: {missing if missing else 'none'}")
    print(f"Rows:        {rows}  = {rows * RECORD_STEP / 60:.1f} min of driving")
    print(f"End:         {len(good) - crashes} timeout, {crashes} crash ({100 * crashes / len(good):.0f} %),"
          f" very short (<1 s): {short}")
    print(f"Throttle:    {throttles.min():.3f} .. {throttles.max():.3f},"
          f" coasting (0): {100 * (throttles == 0).mean():.0f} % of rows")
    print(f"Speed:       {min(i['speed_min'] for i in good):.2f} .. {max(i['speed_max'] for i in good):.2f} m/s"
          f" (before the brake), inside one trip changes by {np.mean([i['speed_spread'] for i in good]):.2f} m/s")
    print("Modes:       " + ", ".join(f"{m} {100 * (modes == m).mean():.0f} %" for m in MODES))
    if len(random_steer) > 0:
        print(f"Random steer: |s| > 0.5 in {100 * (np.abs(random_steer) > 0.5).mean():.0f} % of random rows"
              f" (left {100 * (random_steer > 0.5).mean():.0f} %, right {100 * (random_steer < -0.5).mean():.0f} %)")
    print(f"Lags:        {lags} gaps > {LAG_GAP} s in {lag_files} files (game was slow for a moment)")
    print(f"Start x:     {starts[:, 0].min():.1f} .. {starts[:, 0].max():.1f} m,"
          f"  y: {starts[:, 1].min():.1f} .. {starts[:, 1].max():.1f} m")
    print(f"Size:        {size_mb:.0f} MB = {per_episode:.1f} MB per episode"
          f" -> 700 episodes = {700 * per_episode / 1000:.1f} GB")
    print(f"Bad files:   {bad_files}" + ("   -> ALL GOOD" if bad_files == 0 else ""))


def main():
    files = sorted(glob.glob(f"{DATA_DIR}/drive_*.npz"))
    if not files:
        print(f"No files in {DATA_DIR}")
        return

    infos = []
    bad_files = 0
    for path in files:
        problems, info = check_file(path)
        infos.append(info)
        if problems:
            bad_files += 1
            print(f"{os.path.basename(path)}: PROBLEM -> {'; '.join(problems)}")
    print_summary(files, infos, bad_files)



if __name__ == "__main__": main()
