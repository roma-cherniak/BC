# wm2 data on the graphics card: windows of consecutive moments inside one trip, mirrored on the fly.

import json

import numpy as np
import torch

DATA_DIR = "data/wm2"


def run_data_dir(run):
    # the data a wm2 model was trained on (settings.json "WM2_DATA"; the first models: data/wm2)
    with open(f"runs/{run}/settings.json") as f:
        return json.load(f).get("WM2_DATA", DATA_DIR)


def load_split(split, device, data_dir=DATA_DIR):
    # <data_dir>/<split>.npz (made by wm2/prepare.py) -> dict of tensors on the device (+ numpy mode/time for checks)
    with np.load(f"{data_dir}/{split}.npz") as f:
        d = {key: f[key] for key in f.files}
    out = {key: torch.from_numpy(d[key]).to(device) for key in ["lidar", "state", "action", "trip", "progress", "danger"]}
    out["mode"], out["time"] = d["mode"], d["time"]
    for key in ("track", "source"):                     # data sets made since 14 Nov (wm2/prepare.py)
        if key in d:
            out[key] = d[key]
    return out


def window_starts(trip, length):
    # every start i whose `length` moments i .. i+length-1 all belong to one trip (trips lie one after another)
    trip = trip.cpu().numpy()
    ok = trip[: len(trip) - length + 1] == trip[length - 1 :]
    return torch.from_numpy(np.flatnonzero(ok))


MAX_STEER_RAD = 0.5236      # rad of steering 1.0 (wm2/prepare.py)


def scale_steer(steer, scale):
    # the wheel angle (model units) that drives the same path in a world `scale` times bigger: a circle of radius r
    # becomes one of radius scale x r, and the radius is wheelbase / tan(angle)
    return torch.atan(torch.tan(steer * MAX_STEER_RAD) / scale) / MAX_STEER_RAD


def windows(data, starts, length, mirror=None, scale=None):
    # moments starts .. starts+length-1 -> lidar (B, L, beams) float, state (B, L, 2), action (B, L, 2)
    # mirror: bool tensor (B,): flip those windows left <-> right (beams reversed, wheel angle and steering negated),
    # so the model meets every left turn also as a right turn
    # scale: float tensor (B,) or None: show those windows as the same drive in a world that many times bigger
    # (2026-11-13, E25 / E26: the models read walls right only at the widths of the two training tracks). Every
    # distance and the speed are multiplied, the throttle too (steady speed is ~ throttle), the wheel angles follow
    # scale_steer; a beam at the lidar's range stays there, and a beam pushed beyond it is cut to the range.
    idx = starts.to(data["lidar"].device)[:, None] + torch.arange(length, device=data["lidar"].device)
    lidar, state, action = data["lidar"][idx].float(), data["state"][idx].clone(), data["action"][idx].clone()
    if mirror is not None:
        m = mirror.to(lidar.device)
        lidar[m] = lidar[m].flip(-1)
        state[m, :, 1] = -state[m, :, 1]
        action[m, :, 1] = -action[m, :, 1]
    if scale is not None:
        k = scale.to(lidar.device).float()[:, None]
        lidar = torch.where(lidar >= 0.999, lidar, (lidar * k[:, :, None]).clamp(max=1.0))
        state[:, :, 0] = state[:, :, 0] * k
        state[:, :, 1] = scale_steer(state[:, :, 1], k)
        action[:, :, 0] = action[:, :, 0] * k
        action[:, :, 1] = scale_steer(action[:, :, 1], k)
    return lidar, state, action


def random_scales(n, low_high, share):
    # n scales: 1.0 for (1 - share) of them, the others log-uniform between low and high
    low, high = low_high
    scale = torch.exp(torch.empty(n).uniform_(float(np.log(low)), float(np.log(high))))
    scale[torch.rand(n) >= share] = 1.0
    return scale


def scale_setting():
    # WM2_SCALE=0.6,2.2 switches the scale augmentation on for wm2.train, wm2.heads and wm2.value (default: off)
    import os
    text = os.environ.get("WM2_SCALE")
    return tuple(float(v) for v in text.split(",")) if text else None


# --- balanced sampling (2026-11-14, A1 + A2 of docs/data_solve.md; switched on by WM2_BALANCE=1, default off) ---
SPEED_MAX = 3.0             # m/s (config.py): state[:, 0] * SPEED_MAX = speed
STANDING_SPEED, STANDING_WEIGHT = 0.2, 0.2   # A2: a window starting slower than 0.2 m/s is drawn 1 in 5 as often
FAST_SPEED, FAST_WEIGHT = 1.5, 2.0           # A1: windows starting faster than 1.5 m/s twice as often
NEW_WEIGHT = 2.0                             # A1: trips the planner drove after the fixes of 12 Nov, twice as often
NEW_SOURCES = ("self2_compete", "self3_compete", "self4_compete", "self5_compete", "self6_compete")


def balance_setting():
    import os
    return os.environ.get("WM2_BALANCE") == "1"


def sample_weights(data, starts):
    # weight of every window start: each track gets the same total (A1), x NEW_WEIGHT for the newest compete trips,
    # x FAST_WEIGHT for fast starts, x STANDING_WEIGHT for a standing car (A2). Needs a data set with "track"/"source".
    if "track" not in data:
        raise SystemExit("WM2_BALANCE=1 needs a data set with track + source per moment (wm2.prepare c4)")
    i = starts.cpu().numpy()
    speed = data["state"][starts.to(data["state"].device), 0].float().cpu().numpy() * SPEED_MAX
    w = np.ones(len(i))
    w[np.isin(data["source"][i], NEW_SOURCES)] *= NEW_WEIGHT
    w[speed > FAST_SPEED] *= FAST_WEIGHT
    w[speed < STANDING_SPEED] *= STANDING_WEIGHT
    track = data["track"][i]
    for name in np.unique(track):                   # every track gets the same share of the draws
        m = track == name
        w[m] /= w[m].sum()
    return torch.from_numpy(w / w.sum())


def draw_order(starts, weights):
    # one epoch = as many windows as there are starts, drawn by weight (with replacement)
    return starts[torch.multinomial(weights, len(starts), replacement=True)]
