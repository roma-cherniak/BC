# Fork replay (E32): which way does the planner want to go at the fork of `compete` (E31), and does a heavier value
# head (cem.W_VALUE) change it? No sim: real moments from recorded self-driven trips, the driver's own grid of plans
# (wm2/drive.py: 9 steering values held for the plan x the gas levels of grid_gas(0.12)), the driver's cost settings.
#   venv\Scripts\python -m autoresearch.eval.fork_replay [model]          (default wm2_c4)
# At the fork, steering RIGHT (< 0) goes up into the dead-end wedge, LEFT (> 0) follows the road (docs/data/compete_corner.png).

import glob
import sys

import numpy as np
import torch

from config import RUNS_DIR, SPEED_MAX, THROTTLE_MAX
from core.sensors import clean_lidar
from wm2 import cem
from wm2.heads import load_heads
from wm2.model import HISTORY, LATENT, Encoder, Predictor
from wm2.value import load_value

FORK = np.array([-2.4, 4.8])            # where the middle channel opens (E31)
START = np.array([0.74, 2.84])          # start place of compete
WINDOW = (-10, 10)                      # frames (0.05 s) around the closest point to the fork: the moments
DRIVER_COST = dict(FRONT_BASE=0.25, W_CENTER=1.0, CENTER_CLAMP=0.3, SIDE_MARGIN=0.45)   # E20 / E21 best, without W_VALUE
W_VALUES = [0.0, 0.3, 1.0, 2.0, 3.0]
GRID_STEER = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
GAS = [0.04, 0.08, 0.12]                # grid_gas(0.12) without 0 (the car is moving)
HORIZON, W_SWITCH = 20, 0.3
MAX_STEER_RAD = 0.5236


def trip_label(position: np.ndarray, reason: str) -> str | None:
    # "passed": went on down-left after the fork; "wedge": crashed up in the wedge; None: neither / not from the start
    if np.linalg.norm(position[0] - START) > 0.5:
        return None
    after = position[np.argmin(np.linalg.norm(position - FORK, axis=1)):]
    if (after[:, 1] < 3.5).any() and (after[:, 0] < -3).any():
        return "passed"
    end = position[-3]                                  # the last rows are the reset to the start
    if reason == "crash" and end[1] > 5.5 and end[0] < -2.2:
        return "wedge"
    return None


def moments(files: list[str]) -> list[dict]:
    out = []
    for f in files:
        with np.load(f) as z:
            pos = z["position"]
            label = trip_label(pos, str(z["reason"]))
            if label is None:
                continue
            d = np.linalg.norm(pos[:-3] - FORK, axis=1)
            c = int(np.argmin(d))
            if d[c] > 1.0:
                continue
            idx = [i for i in range(c + WINDOW[0], c + WINDOW[1] + 1, 2) if HISTORY <= i < len(d)]
            lidar = clean_lidar(z["lidar"]).astype(np.float32)
            state = np.stack([z["speed"] / SPEED_MAX, z["real_steering"] / MAX_STEER_RAD], 1).astype(np.float32)
            action = np.stack([z["throttle"] / THROTTLE_MAX, z["steering"]], 1).astype(np.float32)
            for i in idx:
                out.append({"label": label, "lidar": lidar[i - HISTORY + 1:i + 1], "state": state[i - HISTORY + 1:i + 1],
                            "past": action[i - HISTORY + 1:i], "steer_now": float(z["steering"][i]),
                            "driven": float(np.mean(z["steering"][i:i + 10])), "offset": i - c})
    return out


def grid_plans(device: str) -> tuple[torch.Tensor, np.ndarray]:
    plans, steers = [], []
    for s in GRID_STEER:
        for g in GAS:
            p = torch.zeros(HORIZON, 2)
            p[:, 0], p[:, 1] = g / THROTTLE_MAX, s
            plans.append(p)
            steers.append(s)
    return torch.stack(plans).to(device), np.array(steers)


@torch.no_grad()
def choices(model: str, ms: list[dict], device: str = "cuda") -> dict[float, np.ndarray]:
    encoder, predictor = Encoder().to(device), Predictor().to(device)
    encoder.load_state_dict(torch.load(f"{RUNS_DIR}/{model}/encoder.pt", map_location=device))
    predictor.load_state_dict(torch.load(f"{RUNS_DIR}/{model}/predictor.pt", map_location=device))
    heads, value = load_heads(model, device, "walls5"), load_value(model, device)
    for k, v in DRIVER_COST.items():
        setattr(cem, k, v)
    plans, steers = grid_plans(device)
    n = len(plans)
    out = {w: [] for w in W_VALUES}
    for m in ms:
        lidar, state = torch.from_numpy(m["lidar"]).to(device), torch.from_numpy(m["state"]).to(device)
        z = encoder(lidar, state).view(1, HISTORY, LATENT).expand(n, -1, -1)
        past = torch.from_numpy(m["past"]).to(device)[None].expand(n, -1, -1)
        switch = W_SWITCH * np.abs(steers - m["steer_now"])
        for w in W_VALUES:
            cem.W_VALUE, cem.VALUE = w, value
            cost = cem.sequence_cost(predictor, heads, z, past, plans).cpu().numpy() + switch
            out[w].append(steers[int(np.argmin(cost))])
    return {w: np.array(v) for w, v in out.items()}


def main(model: str = "wm2_c4") -> None:
    files = sorted(glob.glob("data/self[3456]_compete/*.npz"))
    ms = moments(files)
    labels = np.array([m["label"] for m in ms])
    driven = np.array([m["driven"] for m in ms])
    print(f"{model}: {len(ms)} moments {WINDOW[0] * 0.05:+.1f} .. {WINDOW[1] * 0.05:+.1f} s around the fork "
          f"(passed trips {np.sum(labels == 'passed')}, wedge trips {np.sum(labels == 'wedge')})")
    for lab in ("passed", "wedge"):
        m = labels == lab
        print(f"  what the car really steered next 0.5 s ({lab}): mean {driven[m].mean():+.2f}, "
              f"right (< -0.1) {100 * np.mean(driven[m] < -0.1):.0f} %")
    print("\nshare of moments whose cheapest plan steers RIGHT (wedge side) / LEFT (road side):")
    print("| W_VALUE | passed trips: right / left | wedge trips: right / left | all: right / left |")
    print("|---|---|---|---|")
    for w, s in choices(model, ms).items():
        cells = [f"{100 * np.mean(s[m] < 0):.0f} % / {100 * np.mean(s[m] > 0):.0f} %"
                 for m in (labels == "passed", labels == "wedge", np.ones(len(s), bool))]
        print(f"| {w} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:])
