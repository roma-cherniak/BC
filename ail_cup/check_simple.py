# Does the numpy model of simple_controller.py compute what the torch model computes, and how long does one call take?
#   venv\Scripts\python -m ail_cup.check_simple wm2_c3 [trip.npz]
# 1. the same 4 real frames through Wm2 (numpy, float16 weights of the export) and through wm2 (torch, float32):
#    z, then the controller's N plans imagined for HORIZON steps -> imagined z, speed, walls, progress, value
# 2. the time of the model part of one call (encode + imagine + heads + value), 30 times; the Garage cap is 250 ms
# No simulator and no graphics card are needed. choose() is not called (only its inputs are checked).

import sys
import time

import numpy as np
import torch

from ail_cup import simple_controller as sc
from config import RUNS_DIR
from core.sensors import clean_lidar
from wm2.heads import load_heads
from wm2.model import HISTORY, Encoder, Predictor, rollout
from wm2.value import load_value


def torch_model(run):
    enc, pred = Encoder(), Predictor()
    enc.load_state_dict(torch.load(f"{RUNS_DIR}/{run}/encoder.pt", map_location="cpu"))
    pred.load_state_dict(torch.load(f"{RUNS_DIR}/{run}/predictor.pt", map_location="cpu"))
    return enc.eval(), pred.eval(), load_heads(run, "cpu", "sectors"), load_value(run, "cpu")


def frames(trip_path):
    d = np.load(trip_path, allow_pickle=True)
    i = len(d["time"]) // 2
    lidar = clean_lidar(d["lidar"][i - 3 : i + 1].astype(np.float32)).astype(np.float32)
    state = np.stack([d["speed"][i - 3 : i + 1] / sc.SPEED_MAX,
                      d["real_steering"][i - 3 : i + 1] / sc.MODEL_MAX_STEER], 1).astype(np.float32)
    past = np.stack([d["throttle"][i - 3 : i] / sc.THROTTLE_MAX, d["steering"][i - 3 : i]], 1).astype(np.float32)
    return lidar, state, past


def gap(name, ours, theirs):
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    print(f"  {name:9s} largest difference {np.abs(ours - theirs).max():.4f}   (values up to {np.abs(theirs).max():.2f})")


def main(run, trip_path):
    model = sc.Wm2(dict(np.load(f"ail_cup/{run}_value_garage.npz")))
    car = sc.Controller({"model": None})                               # only for its plan grid
    lidar, state, past = frames(trip_path)
    print(f"{model.name}; {len(car.plans)} plans x {sc.HORIZON} steps")

    z_np = np.stack([model.encode(lidar[k], state[k]) for k in range(HISTORY)])
    z_im, v_im = model.imagine(z_np, past, car.plans)
    walls, progress = model.heads(z_im)
    value = model.value(z_im[:, -sc.VALUE_STEPS:]).mean(1)

    enc, pred, heads, value_head = torch_model(run)
    n = len(car.plans)
    with torch.no_grad():
        z_t = enc(torch.tensor(lidar), torch.tensor(state))
        actions = torch.cat([torch.tensor(past)[None].expand(n, -1, -1), torch.tensor(car.plans)], 1)
        z_im_t, v_im_t = rollout(pred, z_t[None].expand(n, -1, -1), actions, 0)
        walls_t, _, progress_t = heads(z_im_t)
        value_t = value_head(z_im_t[:, -sc.VALUE_STEPS:]).mean(1)
    print("numpy against torch:")
    gap("z", z_np, z_t)
    gap("imagined", z_im, z_im_t)
    gap("speed", v_im * sc.SPEED_MAX, v_im_t * sc.SPEED_MAX)
    gap("walls", walls, walls_t)
    gap("progress", progress, progress_t)
    gap("value", value, value_t)

    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        z = model.encode(lidar[-1], state[-1])
        z_im, _ = model.imagine(np.concatenate([z_np[1:], z[None]]), past, car.plans)
        model.heads(z_im)
        model.value(z_im[:, -sc.VALUE_STEPS:])
        times.append(1000 * (time.perf_counter() - t0))
    print(f"model part of one call: median {np.median(times):.0f} ms, slowest {max(times):.0f} ms (Garage cap 250 ms)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "wm2_c3",
         sys.argv[2] if len(sys.argv) > 2 else "data/self6_compete/drive_0005.npz")
