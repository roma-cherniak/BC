# Does the synthetic sim match the real one? The `practice` track is rebuilt in the format of tracks.py (centreline +
# width) from recorded trips, then two checks:
#   1. LIDAR: at real poses, the synthetic 109-beam scan vs the recorded scan (beam error in m)
#   2. CAR: from a real pose, replay the real commands with the kinematic car of collect.py for 1 s and 2 s;
#      position error vs the real path (also with the REAL speed and wheel angle, to split speed model / steering model)
#   venv\Scripts\python -m autoresearch.synth.practice_check
#   -> numbers on the screen + docs/data/synthetic_v1/practice_check.png

import glob
import math
import os

import numpy as np

from autoresearch.synth import collect, tracks
from core.sensors import clean_lidar

FOLDER = "data/clean_practice"      # the gap driver: smooth laps near the middle of the road
MAP_TRIPS, TEST_TRIPS = 120, 40     # trips used to build the map / other trips used to test it
RAW_ANGLES = np.radians(np.linspace(-135, 135, 1081))
OUT = "docs/data/synthetic_v1/practice_check.png"


def load_trips(n_skip: int, n: int) -> list[dict]:
    files = sorted(glob.glob(f"{FOLDER}/drive_*.npz"))
    out = []
    for f in files[n_skip:]:
        with np.load(f) as z:
            if len(z["time"]) > 200:
                out.append({k: z[k] for k in ("lidar", "position", "orientation", "speed", "throttle", "steering",
                                              "real_steering", "time")})
        if len(out) == n:
            break
    return out


def wall_points(trips: list[dict], every: int = 10) -> np.ndarray:
    # every lidar hit (< 9 m) of every `every`-th frame, in world coordinates
    pts = []
    for t in trips:
        lid, pos, yaw = t["lidar"][::every].astype(float), t["position"][::every], t["orientation"][::every]
        for r, p, a in zip(lid, pos, yaw):
            p = p + collect.LIDAR_FORWARD * np.array([math.cos(a), math.sin(a)])
            hit = r < 9
            ang = a + RAW_ANGLES[hit]
            pts.append(np.column_stack([p[0] + r[hit] * np.cos(ang), p[1] + r[hit] * np.sin(ang)]))
    return np.vstack(pts)


def one_lap(trips: list[dict]) -> np.ndarray:
    # the path of the first trip that closes a lap: back within 0.3 m of its start after > 10 m
    for t in trips:
        p = t["position"]
        along = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        back = np.where((along > 10) & (np.linalg.norm(p - p[0], axis=1) < 0.3))[0]
        if len(back):
            return p[: back[0]]
    raise SystemExit("no trip with a whole lap")


def smooth_closed(line: np.ndarray, k: int = 15) -> np.ndarray:
    pad = np.vstack([line[-k:], line, line[:k]])
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    return np.column_stack([np.convolve(pad[:, i], kernel, "valid") for i in (0, 1)])


def nearest_band(dist: np.ndarray, thickness: float = 0.35) -> float:
    # the middle of the nearest wall: hits form a band ~0.3 m thick; take the median of the band that starts at the
    # 5th percentile (the farther walls of other parts of the track are left out)
    start = np.percentile(dist, 5)
    return float(np.median(dist[dist < start + thickness]))


def fit_road(lap: np.ndarray, hits: np.ndarray) -> dict:
    # centreline + width: per point the nearest wall on each side, from the hit cloud; centre moved to the middle
    line = tracks.resample(smooth_closed(tracks.resample(lap, 0.05)), tracks.STEP)
    ahead = np.roll(line, -1, axis=0) - np.roll(line, 1, axis=0)
    ahead /= np.linalg.norm(ahead, axis=1, keepdims=True)
    normal = np.column_stack([-ahead[:, 1], ahead[:, 0]])
    left, right = np.zeros(len(line)), np.zeros(len(line))
    for i, (c, a, n) in enumerate(zip(line, ahead, normal)):
        d = hits - c
        near = np.abs(d @ a) < 0.1
        side = d[near] @ n
        left[i], right[i] = nearest_band(side[(side > 0.05) & (side < 3)]), nearest_band(-side[(side < -0.05) & (side > -3)])
    left, right = (smooth_closed(np.column_stack([v, v]), 5)[:, 0] for v in (left, right))
    centre = line + ((left - right) / 2)[:, None] * normal
    return {"line": centre, "width": left + right, "family": "practice", "seed": -1}


def lidar_check(road: dict, trips: list[dict], every: int = 7) -> np.ndarray:
    walls = tracks.walls_of(road)
    errors = []
    for t in trips:
        for r, p, a in zip(t["lidar"][::every], t["position"][::every], t["orientation"][::every]):
            real = clean_lidar(r.astype(float)) * collect.MAX_RANGE
            errors.append(collect.sensor_scan(p[0], p[1], a, walls) - real)
    return np.array(errors)


def replay(trip: dict, start: int, steps: int, real_inputs: bool) -> np.ndarray:
    x, y = trip["position"][start]
    yaw, speed, steer = trip["orientation"][start], trip["speed"][start], trip["real_steering"][start]
    path = []
    for k in range(start, start + steps):
        if real_inputs:
            speed, steer = trip["speed"][k], trip["real_steering"][k]
        else:
            want = trip["steering"][k] * collect.MAX_STEER
            steer += float(np.clip(want - steer, -collect.STEER_RATE * collect.DT, collect.STEER_RATE * collect.DT))
            speed += (trip["throttle"][k] / collect.THROTTLE_PER_MS - speed) * collect.DT / collect.SPEED_LAG
        x, y = x + speed * math.cos(yaw) * collect.DT, y + speed * math.sin(yaw) * collect.DT
        yaw += speed / collect.WHEELBASE * math.tan(steer) * collect.DT
        path.append((x, y))
    return np.array(path)


def car_check(trips: list[dict]) -> dict:
    out = {}
    for real_inputs in (True, False):
        for secs in (1.0, 2.0):
            steps, err = int(secs / collect.DT), []
            for t in trips:
                for s in range(0, len(t["time"]) - steps - 1, 40):
                    if np.any(np.diff(t["time"][s:s + steps + 1]) > 0.1):
                        continue
                    err.append(np.linalg.norm(replay(t, s, steps, real_inputs)[-1] - t["position"][s + steps]))
            out[("real speed+wheel" if real_inputs else "commands", secs)] = np.array(err)
    return out


def draw(road: dict, hits: np.ndarray, errors: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 5.5))
    sub = hits[np.random.default_rng(0).choice(len(hits), min(60000, len(hits)), replace=False)]
    a.scatter(sub[:, 0], sub[:, 1], s=0.2, c="tab:blue", alpha=0.3, label="real lidar hits")
    w = tracks.walls_of(road)
    a.plot(w[:, [0, 2]].T, w[:, [1, 3]].T, "r-", lw=0.8)
    a.plot([], [], "r-", label="synthetic walls")
    a.set_aspect("equal"), a.legend(loc="upper right", fontsize=8), a.set_title("practice: real hits vs rebuilt road")
    b.hist(np.clip(errors.ravel(), -1, 1), bins=100, color="tab:gray")
    b.set_title("synthetic - real, per beam (m), clipped to +-1"), b.set_yscale("log")
    deg = np.linspace(-135, 135, collect.BEAMS)
    b2 = np.median(np.abs(errors), axis=0)
    c.plot(deg, b2, "k-")
    c.set_title("median |error| per beam angle (m)"), c.set_xlabel("beam angle (deg, + = left)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    map_trips, test_trips = load_trips(0, MAP_TRIPS), load_trips(MAP_TRIPS + 200, TEST_TRIPS)
    hits = wall_points(map_trips)
    road = fit_road(one_lap(map_trips), hits)
    print(f"practice rebuilt: {tracks.length_of(road['line']):.1f} m long, width {road['width'].min():.2f} - "
          f"{road['width'].max():.2f} m (mean {road['width'].mean():.2f}), tightest corner "
          f"{tracks.radius_of(road['line']).min():.2f} m")
    err = lidar_check(road, test_trips)
    near = err[np.abs(err) < 9.5]
    print(f"LIDAR ({len(err)} frames of {len(test_trips)} other trips): median |error| {np.median(np.abs(err)):.3f} m, "
          f"within 5 cm {100 * np.mean(np.abs(err) < 0.05):.0f} %, within 10 cm {100 * np.mean(np.abs(err) < 0.1):.0f} %, "
          f"mean (bias) {near.mean():+.3f} m")
    closest = [np.argmin(np.abs(e)) for e in err]
    for name, e in car_check(test_trips).items():
        print(f"CAR {name[0]:>17}, {name[1]:.0f} s: position error median {np.median(e):.3f} m, "
              f"p90 {np.percentile(e, 90):.3f} m ({len(e)} windows)")
    draw(road, hits, err)
    print(OUT)
