# Train the probe (zond) of one model: it learns to read FACTS from the encoder's z. Saves runs/<name>/zond.pt,
# so the probe always lies next to its own encoder. (The probe training from jepa/evaluate_jepa.py.)
# Run from the BC folder:  venv\Scripts\python -m training.train_zond k10_50ep

import os
import sys
import torch
from core.data import load_pairs
from core.models import load_run, make_zond, make_targets
from config import DEVICE, RUNS_DIR, FACTS

BATCH = 256
LR = 0.001
PROBE_EPOCHS = 20
TRAIN_MOMENTS = 100000

RUN = sys.argv[1]
ZOND_FILE = f"{RUNS_DIR}/{RUN}/zond.pt"
if os.path.exists(ZOND_FILE):
    exit(f"{ZOND_FILE} already exists: delete it first if you really want a new probe")

torch.manual_seed(0)
encoder, predictor, _ = load_run(RUN, DEVICE, with_zond=False)

lidar_now, action, speed, _ = load_pairs("train")
idx = torch.randperm(len(lidar_now), device=DEVICE)[:TRAIN_MOMENTS]   # 100k random moments from all trips
lidar_now = lidar_now[idx].float()
targets = make_targets(lidar_now, action[idx], speed[idx])
with torch.no_grad():                                                  # the encoder is frozen
    z = encoder(lidar_now)

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

lidar_t, action_t, speed_t, _ = load_pairs("test")                    # how well it reads on test: R²
targets_t = make_targets(lidar_t.float(), action_t, speed_t)
with torch.no_grad():
    guess = zond(encoder(lidar_t.float()))
r2 = 1 - ((guess - targets_t) ** 2).mean(dim=0) / ((targets.mean(dim=0) - targets_t) ** 2).mean(dim=0)
for name, value in zip(FACTS, r2):
    print(f"{name:<12} R² {value.item():.3f}")
