# "Eyes for imagination": zond.pt learned to read REAL encoder states, but the planner shows it IMAGINED states.
# Here the probe learns on imagined states: the brain imagines 1..STEPS steps with the real actions,
# and the probe must read the REAL walls at the same moment.
# Saves zond_imagined.pt (the same shape as zond.pt, so planner_test.py can load it).
# Run from the BC folder:  venv\Scripts\python jepa\imagined_probe.py

import random
import numpy as np
import torch
import torch.nn as nn
from jepa.jepa_model import LATENT
from utils import load_pairs, load_jepa, load_zond
from config import DEVICE, FACTS, FRONT, RIGHT, LEFT, FRONT_RIGHT, FRONT_LEFT




BATCH = 256
LR = 0.001
PROBE_EPOCHS = 3
STEPS = 20              # the same 1 s as planner_test.py
TRAIN_STARTS = 50000    # random train starts -> TRAIN_STARTS * STEPS imagined states
TEST_STARTS = 5000

torch.manual_seed(0)
random.seed(0)


def make_targets(lidar_now, action, speed):
    """the same facts as evaluate_jepa.py"""
    front = torch.min(lidar_now[:, FRONT], dim = 1).values
    right = torch.min(lidar_now[:, RIGHT], dim = 1).values
    left = torch.min(lidar_now[:, LEFT], dim = 1).values
    front_right = torch.min(lidar_now[:, FRONT_RIGHT], dim = 1).values
    front_left = torch.min(lidar_now[:, FRONT_LEFT], dim = 1).values
    return torch.stack([front, right, left, action[:, 1], speed[:, 0], front_right, front_left], dim=1)


def pair_modes(name):
    """mode of every pair: the same file order and [:-1] cut as prepare_data.py"""
    modes = []
    for file in open(f"data/{name}.txt").read().split("\n"):
        with np.load(file) as f:
            modes.append(f["mode"][:-1])
    return np.concatenate(modes)


def safe_starts(data, name, n):
    """n random starts whose next STEPS pairs are all in one trip (the domino trick) and all "random" mode:
    in safety mode hard steering happens exactly when a wall is close, the probe would learn "hard turn = wall\""""
    lidar_now, action, speed, lidar_next = data
    same = (lidar_next[:-1] == lidar_now[1:]).all(dim=1).cpu().numpy()
    is_random = pair_modes(name) == "random"
    safe = [i for i in range(len(same) - STEPS) if same[i : i + STEPS - 1].all() and is_random[i : i + STEPS].all()]
    print(f"{name}: {len(safe)} safe random starts")
    return torch.tensor(random.sample(safe, min(n, len(safe))), device=DEVICE)


def imagine(data, starts):
    """imagined states after 1..STEPS steps (like the planner: start speed + guard B) and the real facts then"""
    lidar_now, action, speed, lidar_next = data
    zs, facts = [], []
    with torch.no_grad():
        z = encoder(lidar_now[starts].float())
        for k in range(STEPS):
            z = predictor(z, action[starts + k], speed[starts])
            z = z / z.norm(dim=1, keepdim=True) * LATENT ** 0.5
            zs.append(z)
            facts.append(make_targets(lidar_next[starts + k].float(), action[starts + k], speed[starts + k]))
    return torch.stack(zs), torch.stack(facts)          # (STEPS, starts, 32), (STEPS, starts, 7)


def r2(zond, z, facts, lazy):
    with torch.no_grad():
        return 1 - ((zond(z) - facts) ** 2).mean(dim=0) / ((lazy - facts) ** 2).mean(dim=0)


encoder, predictor = load_jepa()

train = load_pairs("train")
z_train, facts_train = imagine(train, safe_starts(train, "train", TRAIN_STARTS))
test = load_pairs("test")
z_test, facts_test = imagine(test, safe_starts(test, "test", TEST_STARTS))

z_flat, facts_flat = z_train.reshape(-1, LATENT), facts_train.reshape(-1, len(FACTS))
lazy = facts_flat.mean(dim=0)

zond = nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, len(FACTS))).to(DEVICE)
optimizer = torch.optim.Adam(zond.parameters(), lr=LR)
for epoch in range(PROBE_EPOCHS):
    order = torch.randperm(len(z_flat), device=DEVICE)
    for start in range(0, len(order), BATCH):
        idx = order[start : start + BATCH]
        loss = ((zond(z_flat[idx]) - facts_flat[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch}  loss {loss.item():.5f}")
torch.save(zond.state_dict(), "zond_imagined.pt")

old = load_zond("zond.pt")

print("\nR² on TEST imagined states (1.0 = perfect, 0 = no better than the average)")
print(f"{'fact':<12}" + "".join(f"{f'old {k * 0.05:.2f}s':>12}{f'new {k * 0.05:.2f}s':>12}" for k in (5, 20)))
for i, name in enumerate(FACTS):
    row = f"{name:<12}"
    for k in (5, 20):
        row += f"{r2(old, z_test[k - 1], facts_test[k - 1], lazy)[i].item():>12.3f}"
        row += f"{r2(zond, z_test[k - 1], facts_test[k - 1], lazy)[i].item():>12.3f}"
    print(row)
