# Planner test on many moments: for every start moment the JEPA brain imagines every steering option
# for STEPS steps, the probe reads the walls, and the option with the best score wins.
# Then we compare the brain's choice with what the real driver (find_gap + noise) did at that moment.
# Run from the BC folder:  venv\Scripts\python jepa\driver_batch.py

import random
import numpy as np
import torch
import torch.nn as nn
from jepa.jepa_model import LATENT
from utils import load_pairs, DEVICE
from utils import load_jepa


STEERING = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
STEPS = 5              # 10 x 0.05 s = 0.5 s of imagination
MOMENTS = 500           # how many random test moments
FACTS = ["front", "right", "left", "steering", "speed", "front_right", "front_left"]
CLEAR_TURN = 0.1        # |real steering| above this = the driver clearly turned



def score(facts):
    """facts: (rows, 5) probe readings, distances in 0..1 (x10 = meters) -> one score per row, bigger = better."""
    front, right, left = facts[:, 0], facts[:, 1], facts[:, 2]
    score = front + right + left
    danger = (front <= 0.05) | (left <= 0.03) | (right <= 0.03) 
    return torch.where(danger, score - 1, score) 


lidar_now, action, speed, lidar_next = load_pairs("test")

encoder, predictor = load_jepa()

zond = nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, len(FACTS))).to(DEVICE)
zond.load_state_dict(torch.load("zond.pt", map_location=DEVICE))

# safe starts: all STEPS steps stay inside one trip (the domino trick from rollout_curve.py)
same = (lidar_next[:-1] == lidar_now[1:]).all(dim=1).cpu()      # on the CPU the loop below is much faster
safe = [s for s in range(len(same) - STEPS) if same[s : s + STEPS].all()]
random.seed(0)
starts = torch.tensor(random.sample(safe, MOMENTS), device=DEVICE)

K = len(STEERING)
with torch.no_grad():
    z = encoder(lidar_now[starts].float())                                 # (M, 32)
    z = z.repeat_interleave(K, dim=0)                                      # (M*K, 32): m0 x9, m1 x9, ...
    steer = torch.tensor(STEERING, device=DEVICE).repeat(MOMENTS)          # (M*K,):    s0..s8, s0..s8, ...
    throttle = action[starts, 0].repeat_interleave(K)                      # start throttle, kept for all steps
    act = torch.stack([throttle, steer], dim=1)                            # (M*K, 2)
    spd = speed[starts].repeat_interleave(K, dim=0)                        # (M*K, 1): start speed, no peeking

    for i in range(STEPS):
        z = predictor(z, act, spd)

    facts = zond(z).clamp(0, 1)                                            # (M*K, 5), impossible distances cut
    scores = score(facts).view(MOMENTS, K)                                 # one row per moment, one column per option
    winner = scores.argmax(dim=1)                                          # (M,): number of the best option
    best = torch.tensor(STEERING, device=DEVICE)[winner]                   # (M,): its steering

real = action[starts, 1]                                                   # what the driver really did
clear = real.abs() > CLEAR_TURN

print(f"moments: {MOMENTS}, imagination: {STEPS * 0.05:.2f} s")
print(f"mean |brain - driver|    : {(best - real).abs().mean().item():.3f}")
print(f"mean |always 0 - driver| : {real.abs().mean().item():.3f}   <- lazy baseline")
same_side = (torch.sign(best[clear]) == torch.sign(real[clear])).float().mean().item()
print(f"same side on clear turns : {same_side:.0%}  ({clear.sum().item()} moments)")
print("how often each option won:")
for s, n in zip(STEERING, torch.bincount(winner, minlength=K).tolist()):
    print(f"  {s:+.2f}: {n}")
