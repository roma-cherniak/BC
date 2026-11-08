# wm2 data on the graphics card: windows of consecutive moments inside one trip, mirrored on the fly.

import numpy as np
import torch

DATA_DIR = "data/wm2"


def load_split(split, device):
    # data/wm2/<split>.npz (made by wm2/prepare.py) -> dict of tensors on the device (+ numpy mode/time for checks)
    with np.load(f"{DATA_DIR}/{split}.npz") as f:
        d = {key: f[key] for key in f.files}
    out = {key: torch.from_numpy(d[key]).to(device) for key in ["lidar", "state", "action", "trip", "progress", "danger"]}
    out["mode"], out["time"] = d["mode"], d["time"]
    return out


def window_starts(trip, length):
    # every start i whose `length` moments i .. i+length-1 all belong to one trip (trips lie one after another)
    trip = trip.cpu().numpy()
    ok = trip[: len(trip) - length + 1] == trip[length - 1 :]
    return torch.from_numpy(np.flatnonzero(ok))


def windows(data, starts, length, mirror=None):
    # moments starts .. starts+length-1 -> lidar (B, L, beams) float, state (B, L, 2), action (B, L, 2)
    # mirror: bool tensor (B,): flip those windows left <-> right (beams reversed, wheel angle and steering negated),
    # so the model meets every left turn also as a right turn
    idx = starts.to(data["lidar"].device)[:, None] + torch.arange(length, device=data["lidar"].device)
    lidar, state, action = data["lidar"][idx].float(), data["state"][idx].clone(), data["action"][idx].clone()
    if mirror is not None:
        m = mirror.to(lidar.device)
        lidar[m] = lidar[m].flip(-1)
        state[m, :, 1] = -state[m, :, 1]
        action[m, :, 1] = -action[m, :, 1]
    return lidar, state, action
