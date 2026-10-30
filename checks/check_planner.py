# Planner check = planner_test.py as one function.
# In "random" moments a coin picked the steering. The brain imagines all 9 options and picks the safest.
# If the brain is right, moments where the coin happened to steer like the brain end in danger less often
# than the same brain choices shuffled between moments (control).
# Run from the BC folder:  venv\Scripts\python -m checks.check_planner

import numpy as np
import torch
from jepa.jepa_model import LATENT
from utils import load_pairs, load_jepa, load_zond


STEERING = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
STEPS = 20              # 20 x 0.05 s = 1 s of imagination
DANGER_WINDOW = 10      # 10 x 0.05 s = 0.5 s: if safety had to save the car or it crashed -> danger
BRAKE_TIME = 27.5       # s, after this the trip brakes to the end, skip those moments
WALLS = [0, 1, 2, 5, 6] # which facts are wall distances
AGREE = 0.5             # |coin - brain| below this = the coin steered like the brain


def score(facts):
    return facts[:, WALLS].min(dim=1).values


def trip_info(name):
    """for every pair: mode, time, danger soon (safety or crash within DANGER_WINDOW), enough future to judge"""
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


def brain_choice(encoder, predictor, zond, lidar_now, action, speed, starts):
    """for every start: imagine all 9 steering options for STEPS steps, return the safest steering"""
    device = lidar_now.device
    M, K = len(starts), len(STEERING)
    options = torch.tensor(STEERING, device=device)
    with torch.no_grad():
        z = encoder(lidar_now[starts].float()).repeat_interleave(K, dim=0)
        act = torch.stack([action[starts, 0].repeat_interleave(K), options.repeat(M)], dim=1)
        spd = speed[starts].repeat_interleave(K, dim=0)
        worst = torch.full((M * K,), float("inf"), device=device)    # the closest wall on the WHOLE imagined path
        for i in range(STEPS):
            z = predictor(z, act, spd)
            z = z / z.norm(dim=1, keepdim=True) * LATENT ** 0.5        # guard B: no drifting away
            worst = torch.minimum(worst, score(zond(z).clamp(0, 1)))
    return options[worst.view(M, K).argmax(dim=1)]


def check_planner(encoder, predictor, zond, test):
    lidar_now, action, speed, _ = test
    device = lidar_now.device
    modes, times, danger, judge = trip_info("test")
    assert len(modes) == len(action)

    starts = np.flatnonzero((modes == "random") & (times < BRAKE_TIME) & judge)
    danger = torch.from_numpy(danger[starts]).to(device)
    starts = torch.from_numpy(starts).to(device)

    best = brain_choice(encoder, predictor, zond, lidar_now, action, speed, starts)
    real = action[starts, 1]                                         # what the coin really did
    dice = torch.Generator(device=device).manual_seed(0)             # own dice: the global seed stays untouched
    control = best[torch.randperm(len(best), generator=dice, device=device)]

    agree = (real - best).abs() < AGREE                              # the coin steered like the brain
    agree_control = (real - control).abs() < AGREE                   # ... like the shuffled brain

    # TODO(human): the number from the table: danger in the agree moments minus danger in the
    # agree_control moments, in percentage points (−10.5 for the current model)



    return {"planner_agree_vs_control": ...}


if __name__ == "__main__":
    test = load_pairs("test")
    encoder, predictor = load_jepa()
    zond = load_zond("zond.pt")
    print(check_planner(encoder, predictor, zond, test))
