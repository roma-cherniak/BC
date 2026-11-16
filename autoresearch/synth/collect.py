# Synthetic trips on random roads (autoresearch/synth/tracks.py), saved in the trip format of data/*/drive_*.npz so
# wm2/prepare.py reads them unchanged. Settings: docs/track_generator.md. The car is the kinematic bicycle of
# ail_cup/garage_sim.py (no slip, guessed lags): use these trips for the ENCODER and the WALL HEADS only, never the
# predictor (docs/track_generator.md section 6).
# Who drives: the driver of the real random data (data_collection/policy.py) + recovery pushes; the safety driver
# takes over near a wall. Speed: the policy's random throttle, capped by a top speed per trip and lower before corners.
#   venv\Scripts\python -m autoresearch.synth.collect <n_roads> [first_seed] [out_dir]
#   -> <out_dir>/drive_<seed>_<k>.npz  +  data/<out folder name>_{train,val,test}.txt split by ROAD (test roads never in training)

import math
import os
import random
import sys
import time

import numpy as np

from autoresearch.synth.tracks import roads, walls_of
from data_collection import policy

DT = 0.05                                   # s, 20 Hz like the sim
WHEELBASE, MAX_STEER = 0.3302, 0.5236       # m; rad (30 degrees, the AutoDRIVE car: data/ trips use this scale)
SPEED_LAG, STEER_RATE = 0.29, 3.25          # s; rad/s: fitted on 500 real trips (practice_check, 14 Nov)
LIDAR_FORWARD = 0.20                        # m, the lidar sits ahead of the car's position point (practice_check)
CAR_RADIUS = 0.17                           # m, closer = crash
RAW_BEAMS, BEAMS, FOV, MAX_RANGE = 1081, 109, math.radians(270), 10.0
THROTTLE_PER_MS = 0.0428                    # throttle per m/s of steady speed: fitted on real trips (practice_check)
TRIPS_PER_ROAD, TRIP_SECONDS = 4, 60.0
PUSH_EVERY, PUSH_LENGTH = (5.0, 15.0), (0.3, 0.8)   # s: recovery pushes
SLOW_SHARE, SLOW_TOP, FAST_TOP = 0.75, (0.5, 2.5), (2.5, 4.0)    # 75 % of trips top speed 0.5-2.5 m/s, 25 % 2.5-4
LIDAR_NOISE, DROP_SHARE = 0.015, 0.03       # m; share of beams dropped to max range
BRAKE_K = 2.0                               # m/s per sqrt(m): v = 2 sqrt(free road - 0.3 m)
ANGLES = np.linspace(-FOV / 2, FOV / 2, BEAMS)
FRONT = np.abs(ANGLES) < math.radians(10)


def lidar(x: float, y: float, yaw: float, walls: np.ndarray) -> np.ndarray:
    seg = walls[:, 2:] - walls[:, :2]                               # keep only segments within the lidar's range
    s = np.clip(((x - walls[:, 0]) * seg[:, 0] + (y - walls[:, 1]) * seg[:, 1]) / np.maximum((seg ** 2).sum(1), 1e-12),
                0, 1)
    near = walls[np.hypot(walls[:, 0] + s * seg[:, 0] - x, walls[:, 1] + s * seg[:, 1] - y) < MAX_RANGE]
    if len(near) == 0:
        return np.full(BEAMS, MAX_RANGE)
    dx, dy = np.cos(yaw + ANGLES)[:, None], np.sin(yaw + ANGLES)[:, None]
    x1, y1 = near[None, :, 0] - x, near[None, :, 1] - y
    ex, ey = near[None, :, 2] - near[None, :, 0], near[None, :, 3] - near[None, :, 1]
    det = dx * ey - dy * ex
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    t, u = (x1 * ey - y1 * ex) / det, (x1 * dy - y1 * dx) / det
    hit = np.where((t > 0) & (u >= 0) & (u <= 1), t, np.inf)
    return np.minimum(hit.min(axis=1), MAX_RANGE)


def wall_distance(x: float, y: float, walls: np.ndarray) -> float:
    seg = walls[:, 2:] - walls[:, :2]
    s = np.clip(((x - walls[:, 0]) * seg[:, 0] + (y - walls[:, 1]) * seg[:, 1]) / np.maximum((seg ** 2).sum(1), 1e-12),
                0, 1)
    return float(np.hypot(walls[:, 0] + s * seg[:, 0] - x, walls[:, 1] + s * seg[:, 1] - y).min())


def sensor_scan(x: float, y: float, yaw: float, walls: np.ndarray) -> np.ndarray:
    # the scan of the lidar where it really sits: LIDAR_FORWARD ahead of the car's position point
    return lidar(x + LIDAR_FORWARD * math.cos(yaw), y + LIDAR_FORWARD * math.sin(yaw), yaw, walls)


def to_raw(scan: np.ndarray) -> np.ndarray:
    # 109 beams -> 1081 like the sim; every 10th raw beam IS the synthetic beam, so core.sensors.clean_lidar gets it back
    return np.interp(np.arange(RAW_BEAMS), np.arange(BEAMS) * 10, scan)


def target_speed(top: float, scan: np.ndarray) -> float:
    # the speed the driver wants now (m/s): braking distance ~ v^2, so v ~ sqrt(free road ahead); floor 0.4 m/s (no
    # standing car, P5 of docs/data_problems.md), cap = the trip's top speed.
    # A first version by Claude (14 Nov) to test the collection; the user may replace it (the Learn-by-Doing part).
    front = float(scan[FRONT].min())
    return min(top, max(0.4, BRAKE_K * math.sqrt(max(front - 0.3, 0.0))))


class Driver:
    # the driver of the real random data (data_collection.policy: random / gap modes, random throttle, a safety
    # driver that looks farther when faster) + recovery pushes: every 5 - 15 s steer hard to one side for
    # 0.3 - 0.8 s (P3, P6 of docs/data_problems.md); the safety driver always wins over a push.
    def __init__(self, rng: np.random.Generator, top: float):
        self.rng, self.top = rng, top
        self.next_push, self.push_until, self.push_side = rng.uniform(*PUSH_EVERY), -1.0, 0.0
        random.seed(int(rng.integers(2 ** 31)))
        policy.reset_policy()

    def act(self, scan: np.ndarray, t: float, speed: float) -> tuple[float, float, str]:
        # -> (speed wanted m/s, steering -1 .. 1, mode)
        throttle, steering, mode = policy.decide(to_raw(scan), t, False, speed)
        if t >= self.next_push:
            self.push_until = t + self.rng.uniform(*PUSH_LENGTH)
            self.push_side = float(self.rng.choice([-1.0, 1.0]))
            self.next_push = self.push_until + self.rng.uniform(*PUSH_EVERY)
        if mode != "safety" and t < self.push_until:
            steering, mode = self.push_side, "push"
        return min(throttle / THROTTLE_PER_MS, target_speed(self.top, scan)), float(steering), mode


def start_pose(road: dict, rng: np.random.Generator) -> tuple[float, float, float]:
    line, i = road["line"], int(rng.integers(len(road["line"])))
    ahead = line[(i + 1) % len(line)] - line[i - 1]
    yaw = math.atan2(ahead[1], ahead[0])
    normal = np.array([-math.sin(yaw), math.cos(yaw)])
    x, y = line[i] + rng.uniform(-0.4, 0.4) * (road["width"][i] / 2 - CAR_RADIUS) * normal
    return float(x), float(y), yaw + math.radians(rng.uniform(-30, 30))


def drive_trip(road: dict, walls: np.ndarray, rng: np.random.Generator) -> dict:
    x, y, yaw = start_pose(road, rng)
    top = rng.uniform(*(SLOW_TOP if rng.random() < SLOW_SHARE else FAST_TOP))
    speed = steer = 0.0
    driver, rows, reason = Driver(rng, top), [], "timeout"
    for k in range(int(TRIP_SECONDS / DT)):
        scan = sensor_scan(x, y, yaw, walls)
        if scan.min() < CAR_RADIUS or wall_distance(x, y, walls) < CAR_RADIUS:     # front (lidar) or middle of the car
            reason = "crash"
            break
        noisy = np.where(rng.random(BEAMS) < DROP_SHARE, MAX_RANGE, scan + rng.normal(0, LIDAR_NOISE, BEAMS))
        want_speed, command, mode = driver.act(scan, k * DT, speed)
        rows.append((to_raw(np.clip(noisy, 0.02, MAX_RANGE)), want_speed * THROTTLE_PER_MS, command, speed, k * DT,
                     (x, y), yaw, steer, mode))
        want = command * MAX_STEER
        new_steer = steer + float(np.clip(want - steer, -STEER_RATE * DT, STEER_RATE * DT))
        speed += (want_speed - speed) * DT / SPEED_LAG
        x, y = x + speed * math.cos(yaw) * DT, y + speed * math.sin(yaw) * DT
        yaw_rate = speed / WHEELBASE * math.tan(new_steer)
        yaw, steer = (yaw + yaw_rate * DT + math.pi) % (2 * math.pi) - math.pi, new_steer
    return {"rows": rows, "reason": reason}


def save_trip(path: str, trip: dict, road: dict) -> int:
    rows = trip["rows"]
    if len(rows) < 20:
        return 0
    lid, thr, cmd, spd, t, pos, yaw, real, mode = zip(*rows)
    yaw = np.array(yaw)
    np.savez_compressed(
        path, lidar=np.array(lid, np.float16), throttle=np.array(thr), steering=np.array(cmd), speed=np.array(spd),
        time=np.array(t), collisions=np.zeros(len(rows), np.int64), position=np.array(pos), orientation=yaw,
        yaw_rate=np.concatenate([np.diff(np.unwrap(yaw)) / DT, [0.0]]), real_steering=np.array(real),
        mode=np.array(["synth_" + m if m != "safety" else "safety" for m in mode]), track=np.array(f"gen_{road['seed']}"),
        reason=np.array(trip["reason"]), warmup=np.array(0), family=np.array(road["family"]),
        width=np.array(float(road["width"].mean())))
    return len(rows)


def split_by_road(paths_by_seed: dict, name: str = "synth", seed: int = 42) -> None:
    seeds = sorted(paths_by_seed)
    np.random.default_rng(seed).shuffle(seeds)
    n = max(1, len(seeds) // 10)
    parts = {"test": seeds[:n], "val": seeds[n:2 * n], "train": seeds[2 * n:]}
    for split, part in parts.items():
        path = f"data/{name}_{split}.txt"
        if os.path.exists(path):
            sys.exit(f"{path} already exists: nothing is overwritten")
        with open(path, "w") as f:
            f.write("\n".join(p for s in part for p in paths_by_seed[s]) + "\n")
        print(f"{path}: {len(part)} roads")


def main(n_roads: int, first_seed: int = 0, out_dir: str = "data/synthetic_v1") -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng, began, frames, ends = np.random.default_rng(first_seed + 12345), time.time(), 0, {"crash": 0, "timeout": 0}
    paths_by_seed = {}
    for i, road in enumerate(roads(first_seed, n_roads)):
        walls = walls_of(road)
        for k in range(TRIPS_PER_ROAD):
            trip = drive_trip(road, walls, rng)
            path = os.path.join(out_dir, f"drive_{road['seed']:05d}_{k}.npz")
            n = save_trip(path, trip, road)
            if n:
                frames += n
                ends[trip["reason"]] += 1
                paths_by_seed.setdefault(road["seed"], []).append(path)
        print(f"road {i + 1}/{n_roads} ({road['family']}, seed {road['seed']}): {frames} frames, {ends}, "
              f"{time.time() - began:.0f} s", flush=True)
    split_by_road(paths_by_seed, os.path.basename(out_dir.rstrip("/\\")))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(int(args[0]) if args else 200, int(args[1]) if len(args) > 1 else 0,
         args[2] if len(args) > 2 else "data/synthetic_v1")
