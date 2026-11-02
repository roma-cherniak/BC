# Train the probe (zond) of one model: it learns to read FACTS from the encoder's z. Saves runs/<name>/zond.pt,
# so the probe always lies next to its own encoder. (The probe training from jepa/evaluate_jepa.py.)
# Works for lidar and photo models (settings.json INPUT): z from what the encoder eats, the right answers from the lidar.
# Run from the BC folder:  venv\Scripts\python -m training.train_zond k10_50ep

import os
import sys
import torch
from core.data import load_pairs, load_photos
from core.models import load_run, make_zond, make_targets, run_input
from config import DEVICE, RUNS_DIR, FACTS

BATCH = 256
LR = 0.001
PROBE_EPOCHS = 20
TRAIN_MOMENTS = 100000
CHUNK = 4096            # moments per encoder call: 100k photos at once would not fit into the 4 GB graphics card

RUN = sys.argv[1]
KIND = run_input(RUN)                                                  # "lidar" or "photo"
ZOND_FILE = f"{RUNS_DIR}/{RUN}/zond.pt"
if os.path.exists(ZOND_FILE):
    exit(f"{ZOND_FILE} already exists: delete it first if you really want a new probe")


def load_split(split):
    # what the encoder eats + the lidar of the same moments (the probe's right answers)
    if KIND == "photo":
        images, action, speed, _, lidar = load_photos(split)
        return images, lidar, action, speed
    lidar_now, action, speed, _ = load_pairs(split)
    return lidar_now, lidar_now, action, speed


def encode(obs):
    # z of many moments, CHUNK at a time
    with torch.no_grad():                                              # the encoder is frozen
        if KIND == "photo":
            return torch.cat([encoder(obs[i : i + CHUNK][:, None].float() / 255) for i in range(0, len(obs), CHUNK)])
        return torch.cat([encoder(obs[i : i + CHUNK].float()) for i in range(0, len(obs), CHUNK)])


torch.manual_seed(0)
encoder, predictor, _ = load_run(RUN, DEVICE, with_zond=False)

obs, lidar, action, speed = load_split("train")
idx = torch.randperm(len(obs), device=DEVICE)[:TRAIN_MOMENTS]         # 100k random moments from all trips
targets = make_targets(lidar[idx].float(), action[idx], speed[idx])
z = encode(obs[idx])

zond = make_zond().to(DEVICE)
optimizer = torch.optim.Adam(zond.parameters(), lr=LR)
for epoch in range(PROBE_EPOCHS):
    order = torch.randperm(len(z), device=DEVICE)
    for start in range(0, len(order), BATCH):
        batch = order[start : start + BATCH]
        loss = ((zond(z[batch]) - targets[batch]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch}  loss {loss.item():.5f}")
torch.save(zond.state_dict(), ZOND_FILE)
print("saved", ZOND_FILE)

obs_t, lidar_t, action_t, speed_t = load_split("test")                # how well it reads on test: R²
targets_t = make_targets(lidar_t.float(), action_t, speed_t)
with torch.no_grad():
    guess = zond(encode(obs_t))
r2 = 1 - ((guess - targets_t) ** 2).mean(dim=0) / ((targets.mean(dim=0) - targets_t) ** 2).mean(dim=0)
for name, value in zip(FACTS, r2):
    print(f"{name:<12} R² {value.item():.3f}")
