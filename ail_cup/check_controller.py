# Does the pure-Python AIL Cup controller compute the same plan as the torch model, and how long does a call take?
#   venv\Scripts\python -m ail_cup.check_controller wm2_c2 [trip.npz]
# 1. the same 4 real frames go through Controller.think() and through wm2 (encoder, predictor, heads, value head)
#    with the controller's cost formula -> the option costs must agree (float16 weights: to ~0.05)
# 2. 60 calls of Controller.step() on the frames of a recorded trip -> the slowest call (the Garage cap is 250 ms)
# No simulator and no graphics card are needed.

import sys
import time

import numpy as np
import torch

from ail_cup import claude_controller as cc
from config import RUNS_DIR, SPEED_MAX, THROTTLE_MAX
from core.sensors import clean_lidar
from wm2.heads import load_heads
from wm2.model import HISTORY, LATENT, Encoder, Predictor, rollout
from wm2.value import load_value

MAX_STEER_RAD = 0.5236


def torch_costs(run, lidar, state, past, steer_now, with_value, options):
    enc, pred = Encoder(), Predictor()
    enc.load_state_dict(torch.load(f"{RUNS_DIR}/{run}/encoder.pt", map_location="cpu"))
    pred.load_state_dict(torch.load(f"{RUNS_DIR}/{run}/predictor.pt", map_location="cpu"))
    enc.eval(), pred.eval()
    heads = load_heads(run, "cpu", "sectors")
    n = len(options)
    plans = torch.zeros(n, cc.STEPS, 2)
    plans[..., 0] = cc.SPEED * cc.GAS_PER_MS / cc.THROTTLE_MAX
    plans[..., 1] = torch.tensor(options)[:, None]
    with torch.no_grad():
        z = enc(torch.tensor(lidar), torch.tensor(state)).view(1, HISTORY, LATENT).expand(n, -1, -1)
        z_im, speed = rollout(pred, z, torch.cat([torch.tensor(past)[None].expand(n, -1, -1), plans], 1), 0)
        walls, _, progress = heads(z_im)
        v = (speed * SPEED_MAX).clamp(min=0)
        front = walls[..., 5:8].min(-1).values
        side = torch.cat([walls[..., :5], walls[..., 8:]], -1).min(-1).values
        center = (walls[..., 12] - walls[..., 0]).clamp(-cc.CENTER_CLAMP, cc.CENTER_CLAMP) ** 2
        cost = (-progress + cc.W_WALL * ((cc.FRONT_BASE + v ** 2 / (2 * cc.DECEL) - front).clamp(min=0) ** 2
                                         + (cc.SIDE_MARGIN - side).clamp(min=0) ** 2) + cc.W_CENTER * center).sum(1)
        if with_value:
            ahead = load_value(run, "cpu")(z_im[:, -cc.VALUE_STEPS:]).mean(1).clamp(0, cc.VALUE_CAP)
            cost = cost - cc.W_VALUE * ahead
        cost = cost + cc.W_SWITCH * (torch.tensor(options) - steer_now).abs()
    return cost.tolist()


def main(run, trip_path):
    model = dict(np.load(f"ail_cup/{run}_value_garage.npz"))
    print(f"model file: {len(model)} arrays")
    d = np.load(trip_path, allow_pickle=True)
    i = len(d["time"]) // 2
    lidar = clean_lidar(d["lidar"][i - 3 : i + 1].astype(np.float32)).astype(np.float32)
    state = np.stack([d["speed"][i - 3 : i + 1] / SPEED_MAX, d["real_steering"][i - 3 : i + 1] / MAX_STEER_RAD], 1).astype(np.float32)
    past = np.stack([d["throttle"][i - 3 : i] / THROTTLE_MAX, d["steering"][i - 3 : i]], 1).astype(np.float32)
    steer_now = float(state[-1, 1])

    car = cc.Controller({"car": {"max_steer": 0.4189}, "lidar": None, "model": model})
    frames = [(lidar[k].tolist(), state[k].tolist()) for k in range(HISTORY)]
    plan, pieces = car.think(frames, past.tolist(), steer_now), []
    while True:
        t0 = time.perf_counter()
        try:
            next(plan)
        except StopIteration:
            break
        pieces.append(time.perf_counter() - t0)
    ours = car.last_costs
    theirs = torch_costs(run, lidar, state, past, steer_now, car.has_value, car.last_options)
    print("steering options:", car.last_options)
    print("option costs, controller:", [round(c, 3) for c in ours])
    print("option costs, torch     :", [round(c, 3) for c in theirs])
    print(f"largest difference {max(abs(a - b) for a, b in zip(ours, theirs)):.3f}; same choice: "
          f"{ours.index(min(ours)) == theirs.index(min(theirs))}")
    print(f"one plan: {sum(pieces):.2f} s of work in {len(pieces)} pieces, the longest piece {1000 * max(pieces):.1f} ms")

    car = cc.Controller({"car": {"max_steer": 0.4189}, "lidar": None, "model": model})
    times = []
    for k in range(min(60, len(d["time"]))):
        obs = {"lidar": d["lidar"][k].astype(float).tolist(), "speed": float(d["speed"][k]), "steer": float(d["real_steering"][k])}
        t0 = time.perf_counter()
        car.step(obs)
        times.append(1000 * (time.perf_counter() - t0))
    print(f"60 calls of step(): median {np.median(times):.0f} ms, slowest {max(times):.0f} ms, plans finished {car.plans}")


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "wm2_c2"
    main(run, sys.argv[2] if len(sys.argv) > 2 else "data/self6_compete/drive_0005.npz")
