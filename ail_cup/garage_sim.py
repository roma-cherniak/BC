# A small stand-in for the AIL Cup Garage, to try a controller on this PC before the user uploads it.
#   venv\Scripts\python -m ail_cup.garage_sim ail_cup.wm2_controller oval [model.npz] [--cw] [--seconds 90]
# It calls Controller(init) and Controller.step(obs) like the Garage (the keys seen in run #1259), every dt = 0.05 s of
# SIM time, however long the call takes. The car is a kinematic bicycle (wheelbase 0.3302 m, +-24 degrees), the lidar
# has 109 beams over 270 degrees up to 10 m, the walls are the centreline moved half a track width to each side.
# NOT the Garage: no tyre slip, guessed speed and steering lags, made-up tracks (the real maps are not on this PC):
#   oval     a stadium, 79.5 m long like the Garage's first map, 2.57 m wide
#   wavy     a closed curve with three bends each way and tighter corners
#   hairpin  a long loop with two 180 degree turns of 2.2 m radius
# So a controller that fails here will fail there; one that works here is only worth an upload.

import importlib
import math
import sys
import time

import numpy as np

DT = 0.05
WHEELBASE, MAX_STEER = 0.3302, 0.4189
BEAMS, FOV, MAX_RANGE = 109, math.radians(270), 10.0
CAR_RADIUS = 0.17               # m: closer than this to a wall = a crash
SPEED_LAG, STEER_RATE = 0.25, 3.5      # s to follow a speed command; rad/s the wheels can turn (guesses)


def centreline(kind: str) -> tuple[np.ndarray, float]:
    # -> closed centreline (N, 2), points ~0.2 m apart, driven counter-clockwise; the track width (m)
    if kind == "oval":
        r, width = 6.0, 2.57
        straight = (79.5 - 2 * math.pi * r) / 2
        pts = [(x, -r) for x in np.arange(-straight / 2, straight / 2, 0.2)]
        pts += [(straight / 2 + r * math.sin(a), -r * math.cos(a)) for a in np.arange(0, math.pi, 0.2 / r)]
        pts += [(x, r) for x in np.arange(straight / 2, -straight / 2, -0.2)]
        pts += [(-straight / 2 - r * math.sin(a), r * math.cos(a)) for a in np.arange(0, math.pi, 0.2 / r)]
        return np.array(pts), width
    if kind == "wavy":
        a = np.linspace(0, 2 * math.pi, 2000, endpoint=False)
        r = 9.0 * (1 + 0.28 * np.cos(3 * a))
        return resample(np.column_stack([r * np.cos(a), r * np.sin(a)])), 2.4
    if kind == "hairpin":
        r, length = 2.2, 16.0
        pts = [(x, -r) for x in np.arange(-length / 2, length / 2, 0.2)]
        pts += [(length / 2 + r * math.sin(a), -r * math.cos(a)) for a in np.arange(0, math.pi, 0.2 / r)]
        pts += [(x, r) for x in np.arange(length / 2, -length / 2, -0.2)]
        pts += [(-length / 2 - r * math.sin(a), r * math.cos(a)) for a in np.arange(0, math.pi, 0.2 / r)]
        return np.array(pts), 2.2
    sys.exit(f"unknown track {kind!r} (oval, wavy or hairpin)")


def resample(pts: np.ndarray, step: float = 0.2) -> np.ndarray:
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    along = np.concatenate([[0], np.cumsum(seg)])
    want = np.arange(0, along[-1], step)
    closed = np.vstack([pts, pts[:1]])
    return np.column_stack([np.interp(want, along, closed[:, 0]), np.interp(want, along, closed[:, 1])])


def walls_of(line: np.ndarray, width: float) -> np.ndarray:
    # -> wall segments (M, 4): x1, y1, x2, y2 of the inner and the outer wall
    ahead = np.roll(line, -1, axis=0) - np.roll(line, 1, axis=0)
    normal = np.column_stack([-ahead[:, 1], ahead[:, 0]]) / np.linalg.norm(ahead, axis=1, keepdims=True)
    segments = []
    for side in (+1, -1):
        wall = line + side * width / 2 * normal
        segments.append(np.column_stack([wall, np.roll(wall, -1, axis=0)]))
    return np.vstack(segments)


def lidar(x: float, y: float, yaw: float, walls: np.ndarray) -> np.ndarray:
    angles = yaw + np.linspace(-FOV / 2, FOV / 2, BEAMS)
    dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]                 # (B, 1)
    x1, y1 = walls[None, :, 0] - x, walls[None, :, 1] - y
    ex, ey = walls[None, :, 2] - walls[None, :, 0], walls[None, :, 3] - walls[None, :, 1]
    det = dx * ey - dy * ex
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    t = (x1 * ey - y1 * ex) / det                                              # distance along the ray
    u = (x1 * dy - y1 * dx) / det                                              # place on the segment, 0 .. 1
    hit = np.where((t > 0) & (u >= 0) & (u <= 1), t, np.inf)
    return np.minimum(hit.min(axis=1), MAX_RANGE)


def run(module: str, kind: str, model_path: str | None, clockwise: bool, seconds: float, quiet: bool = True) -> dict:
    line, width = centreline(kind)
    if clockwise:
        line = line[::-1].copy()
    walls = walls_of(line, width)
    x, y = line[0]
    yaw = math.atan2(line[1][1] - y, line[1][0] - x)
    model = dict(np.load(model_path)) if model_path else None
    init = {"type": "init", "mode": "time_trial", "laps": 1, "dt": DT, "latency_cap_ms": 250.0,
            "car": {"wheelbase": WHEELBASE, "max_steer": MAX_STEER, "v_max": 30.0, "mass": 3.47},
            "lidar": {"n_beams": BEAMS, "fov": FOV, "max_range": MAX_RANGE, "angle_min": -FOV / 2,
                      "angle_increment": FOV / (BEAMS - 1)},
            "track": {"centerline": line.tolist()}, "model": model}
    out = sys.stdout
    if quiet:
        sys.stdout = open("ail_cup/garage_sim_last.log", "w", encoding="utf-8")
    try:
        car = importlib.import_module(module).Controller(init)
        speed = steer = progress = 0.0
        nearest, calls, turn, swing, closest = 0, [], [], [], []
        result = {"end": "time up"}
        for k in range(int(seconds / DT)):
            scan = lidar(x, y, yaw, walls)
            obs = {"type": "obs", "t": k * DT, "pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
                   "speed": speed, "steer": steer, "lap": 0, "progress": progress, "lidar": scan.tolist()}
            began = time.perf_counter()
            act = car.step(obs)
            calls.append(1000 * (time.perf_counter() - began))
            want = max(-MAX_STEER, min(MAX_STEER, float(act["steer"])))
            new_steer = steer + max(-STEER_RATE * DT, min(STEER_RATE * DT, want - steer))
            swing.append(abs(new_steer - steer))
            steer = new_steer
            speed += (float(act["speed"]) - speed) * DT / SPEED_LAG
            x, y = x + speed * math.cos(yaw) * DT, y + speed * math.sin(yaw) * DT
            yaw += speed / WHEELBASE * math.tan(steer) * DT
            turn.append(abs(steer))
            closest.append(float(scan.min()))
            d = np.linalg.norm(line - [x, y], axis=1)
            near = int(np.argmin(d[(nearest + np.arange(-10, 40)) % len(line)])) - 10
            nearest = (nearest + near) % len(line)
            progress += max(0, near) * 0.2
            if scan.min() < CAR_RADIUS or d.min() > width / 2 - CAR_RADIUS + 0.05:
                result = {"end": f"CRASH at {k * DT:.1f} s"}
                break
            if progress >= len(line) * 0.2:
                result = {"end": f"lap in {k * DT:.1f} s"}
                break
    finally:
        if quiet:
            sys.stdout.close()
            sys.stdout = out
    result.update(progress=round(progress, 1), track_m=round(len(line) * 0.2, 1), mean_speed=round(progress / max(k * DT, DT), 2),
                  closest_wall=round(min(closest), 2), wall_p5=round(float(np.percentile(closest, 5)), 2),
                  wheel_deg_per_s=round(math.degrees(float(np.mean(swing))) / DT, 1),
                  mean_wheel_deg=round(math.degrees(float(np.mean(turn))), 1),
                  call_ms=round(float(np.median(calls))), slowest_ms=round(max(calls)))
    return result


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    secs = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 120.0
    if "--seconds" in sys.argv:
        args.remove(sys.argv[sys.argv.index("--seconds") + 1])
    print(run(args[0], args[1], args[2] if len(args) > 2 else None, "--cw" in sys.argv, secs))
    print("the controller's own lines: ail_cup/garage_sim_last.log")
