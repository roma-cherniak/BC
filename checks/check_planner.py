# Planner check = planner_test.py as one function.
# In "random" moments a coin picked the steering. The brain imagines all 9 options and picks the safest.
# If the brain is right, moments where the coin happened to steer like the brain end in danger less often
# than the same brain choices shuffled between moments (control).
# Run from the BC folder:  venv\Scripts\python -m checks.check_planner k10_50ep

import numpy as np
import torch
from core.data import load_pairs, pair_obs
from core.planner import STEERING, score_options
import sys
from core.models import load_run, run_input
from config import DEVICE, CURRENT_RUN


DANGER_WINDOW = 10      # 10 x 0.05 s = 0.5 s: if safety had to save the car or it crashed -> danger
BRAKE_TIME = 27.5       # s, after this the trip brakes to the end, skip those moments
AGREE = 0.5             # |coin - brain| below this = the coin steered like the brain
CHUNK = 2048            # moments per planner call: all ~20k photo moments at once do not fit into the 4 GB graphics card


def trip_info(name):

    modes, times, danger, judge = [], [], [], []
    for file in open(f"data/{name}.txt").read().split("\n"):
        with np.load(file) as f:
            mode = f["mode"][:-1]                               # the same [:-1] cut as prepare_data.py
            time = f["time"][:-1]
            crash = str(f["reason"]) == "crash"
        n = len(mode)
        j = np.arange(n)
        safety_soon = np.array([(mode[k + 1 : k + 1 + DANGER_WINDOW] == "safety").any() for k in j])
        crash_soon = crash & (j >= n - DANGER_WINDOW)           # the crash row comes right after the last pair
        modes.append(mode)
        times.append(time)
        danger.append(safety_soon | crash_soon)
        judge.append((j + DANGER_WINDOW < n) | crash_soon)      # a timeout trip must still have the whole window
    return [np.concatenate(x) for x in (modes, times, danger, judge)]


def brain_choice(encoder, predictor, zond, obs, action, speed, starts):
    best = []
    for i in range(0, len(starts), CHUNK):
        s = starts[i : i + CHUNK]
        scores = score_options(encoder, predictor, zond, obs[s], action[s, 0], speed[s])    # (chunk, 9)
        best.append(scores.argmax(dim=1))
    return torch.tensor(STEERING, device=obs.device)[torch.cat(best)]


def check_planner(encoder, predictor, zond, obs, action, speed):
    # obs, action, speed: one row per pair of data/test_pairs.npz (pair_obs gives obs for lidar or photo models)
    device = obs.device
    modes, times, danger, judge = trip_info("test")
    assert len(modes) == len(action)

    starts = np.flatnonzero((modes == "random") & (times < BRAKE_TIME) & judge)
    danger = torch.from_numpy(danger[starts]).to(device)
    starts = torch.from_numpy(starts).to(device)

    best = brain_choice(encoder, predictor, zond, obs, action, speed, starts)
    real = action[starts, 1]                                         
    dice = torch.Generator(device=device).manual_seed(0)            
    control = best[torch.randperm(len(best), generator=dice, device=device)]

    agree = (real - best).abs() < AGREE                             
    agree_control = (real - control).abs() < AGREE                  


    brain = danger[agree].float().mean()
    control = danger[agree_control].float().mean()

    return {"planner_agree_vs_control": ((brain - control) * 100).item()}


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else CURRENT_RUN
    _, action, speed, _ = load_pairs("test")
    obs = pair_obs("test", run_input(run))
    encoder, predictor, zond = load_run(run, DEVICE)
    print(check_planner(encoder, predictor, zond, obs, action, speed))
