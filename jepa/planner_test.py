# Planner test on the whole test set, judged by what REALLY happened next.
# In "random" moments the steering was picked by a coin, not by the situation = a natural experiment:
# if the brain's choice is good, moments where the coin happened to steer like the brain must end safer
# than moments where the coin steered the other way.
# Control: the same brain choices shuffled between moments (same amount of each option, wrong moment).
# Run from the BC folder:  venv\Scripts\python jepa\planner_test.py

import numpy as np
import torch
from jepa.jepa_model import LATENT
from utils import load_pairs
from utils import load_jepa, load_zond
from config import DEVICE, WALLS


STEERING = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
STEPS = 20              # 20 x 0.05 s = 1 s of imagination
DANGER_WINDOW = 10      # 10 x 0.05 s = 0.5 s: if safety had to save the car or it crashed -> danger
BRAKE_TIME = 27.5       # s, after this the trip brakes to the end, skip those moments
ZOND_FILE = "zond.pt"                   # probe on real states (evaluate_jepa.py); "zond_imagined.pt" was worse
BINS = [0, 0.5, 1.0, 1.5, 2.01]     # |real steering - brain choice|




def score(facts):
   
    return facts[:, WALLS].min(dim=1).values


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


lidar_now, action, speed, no = load_pairs("test")
modes, times, danger, judge = trip_info("test")
assert len(modes) == len(action)

starts = np.flatnonzero((modes == "random") & (times < BRAKE_TIME) & judge)
danger = torch.from_numpy(danger[starts]).to(DEVICE)
starts = torch.from_numpy(starts).to(DEVICE)

encoder, predictor = load_jepa()
zond = load_zond(ZOND_FILE)

M, K = len(starts), len(STEERING)
with torch.no_grad():                                           # the same imagination as driver_batch.py
    z = encoder(lidar_now[starts].float()).repeat_interleave(K, dim=0)
    steer = torch.tensor(STEERING, device=DEVICE).repeat(M)
    throttle = action[starts, 0].repeat_interleave(K)
    act = torch.stack([throttle, steer], dim=1)
    spd = speed[starts].repeat_interleave(K, dim=0)
    worst = torch.full((M * K,), float("inf"), device=DEVICE)    # the closest wall on the WHOLE imagined path
    for i in range(STEPS):
        z = predictor(z, act, spd)
        z = z / z.norm(dim=1, keepdim=True) * LATENT ** 0.5        # guard B from rollout_jepa.py: no drifting away
        worst = torch.minimum(worst, score(zond(z).clamp(0, 1)))
    scores = worst.view(M, K)
    best = torch.tensor(STEERING, device=DEVICE)[scores.argmax(dim=1)]

real = action[starts, 1]                                        # what the coin really did
torch.manual_seed(0)
control = best[torch.randperm(M, device=DEVICE)]


def danger_by_distance(choice):
    diff = (real - choice).abs()
    rows = []
    for low, high in zip(BINS[:-1], BINS[1:]):
        inside = (diff >= low) & (diff < high)
        rows.append((inside.sum().item(), danger[inside].float().mean().item()))
    return rows


print(f"random moments: {M}, danger (safety or crash within {DANGER_WINDOW * 0.05:.1f} s) overall: "
      f"{danger.float().mean().item():.1%}")
print(f"\n{'|coin - brain|':<16}{'moments':>9}{'danger':>9}   {'control':>9}")
for (low, high), (n, d), (_, c) in zip(zip(BINS[:-1], BINS[1:]), danger_by_distance(best), danger_by_distance(control)):
    print(f"{low:.1f} .. {min(high, 2):.1f}{'':6}{n:>9}{d:>9.1%}   {c:>9.1%}")
print("\nbrain choices:", {s: int((best == s).sum()) for s in STEERING})
