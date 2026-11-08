# Cost heads of wm2: what the planner asks about every imagined z (like TD-MPC's reward head, but reward-free data:
# the "right answers" come from the lidar and the trip records of the same moment).
#   walls     5 distances in metres: front, right, left, front_right, front_left (config.FRONT ...)
#   danger    logit of "within 0.5 s the safety driver takes over or the trip crashes" (data/wm2 danger)
#   progress  metres driven forward along the car's heading in the next 0.05 s (data/wm2 progress)
# The heads learn on REAL z of the frozen encoder; the planner then reads them on IMAGINED z.
# Saves runs/<RUN>/heads.pt.  Run from the BC folder:  venv\Scripts\python -m wm2.heads wm2_k10

import os
import sys

import torch
import torch.nn as nn

from config import DEVICE, FRONT, FRONT_LEFT, FRONT_RIGHT, LEFT, MAX_RANGE, RIGHT, RUNS_DIR
from wm2.data import load_split
from wm2.model import LATENT, Encoder

WALLS = [FRONT, RIGHT, LEFT, FRONT_RIGHT, FRONT_LEFT]
WALL_NAMES = ["front", "right", "left", "front_right", "front_left"]
EPOCHS = 10
BATCH = 1024
CHUNK = 8192


class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(LATENT, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.out = nn.Linear(256, 5 + 1 + 1)

    def forward(self, z):
        # z (..., LATENT) -> walls (..., 5) metres, danger logit (...), progress (...) metres per step
        h = self.out(self.body(z))
        return h[..., :5], h[..., 5], h[..., 6]


def targets(data):
    walls = torch.stack([data["lidar"][:, s].float().min(dim=1).values for s in WALLS], dim=1) * MAX_RANGE
    return walls, data["danger"].float(), data["progress"]


def encode_all(encoder, data):
    with torch.no_grad():
        return torch.cat([encoder(data["lidar"][i : i + CHUNK].float(), data["state"][i : i + CHUNK])
                          for i in range(0, len(data["lidar"]), CHUNK)])


def load_heads(run, device):
    heads = Heads().to(device)
    heads.load_state_dict(torch.load(f"{RUNS_DIR}/{run}/heads.pt", map_location=device))
    return heads.eval()


def main(run):
    path = f"{RUNS_DIR}/{run}/heads.pt"
    if os.path.exists(path):
        exit(f"{path} already exists: delete it first if you really want new heads")
    torch.manual_seed(0)
    encoder = Encoder().to(DEVICE)
    encoder.load_state_dict(torch.load(f"{RUNS_DIR}/{run}/encoder.pt", map_location=DEVICE))
    encoder.eval()
    train, test = load_split("train", DEVICE), load_split("test", DEVICE)
    z, z_t = encode_all(encoder, train), encode_all(encoder, test)
    (w, d, p), (w_t, d_t, p_t) = targets(train), targets(test)

    heads = Heads().to(DEVICE)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(EPOCHS):
        order = torch.randperm(len(z), device=DEVICE)
        for i in range(0, len(z), BATCH):
            idx = order[i : i + BATCH]
            walls, danger, progress = heads(z[idx])
            loss = ((walls - w[idx]) ** 2).mean() + bce(danger, d[idx]) + 100 * ((progress - p[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(heads.state_dict(), path)

    heads.eval()
    with torch.no_grad():
        walls, danger, progress = heads(z_t)
    err = (walls - w_t).abs().mean(dim=0)
    print("test walls, mean error m: " + ", ".join(f"{n} {e:.2f}" for n, e in zip(WALL_NAMES, err.tolist())))
    hit = ((danger > 0) == (d_t > 0.5)).float().mean().item()
    print(f"test danger: right in {hit:.1%} of moments (always 'danger' would be {d_t.mean().item():.1%})")
    print(f"test progress: mean error {(progress - p_t).abs().mean().item() * 100:.1f} cm per step "
          f"(mean progress {p_t.mean().item() * 100:.1f} cm)")
    print("saved", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "wm2_k10")
