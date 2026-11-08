# Cost heads of wm2: what the planner asks about every imagined z (like TD-MPC's reward head, but reward-free data:
# the "right answers" come from the lidar and the trip records of the same moment).
#   walls     5 distances in metres: front, right, left, front_right, front_left (config.FRONT ...)
#   danger    logit of "within 0.5 s the safety driver takes over or the trip crashes" (data/wm2 danger)
#   progress  metres driven forward along the car's heading in the next 0.05 s (data/wm2 progress)
# The heads learn on REAL z of the frozen encoder; the planner then reads them on IMAGINED z.
# Saves runs/<RUN>/heads.pt (or heads_sectors.pt).  Run from the BC folder:  venv\Scripts\python -m wm2.heads wm2_k10 sectors

import os
import sys

import torch
import torch.nn as nn

from config import DEVICE, FRONT, FRONT_LEFT, FRONT_RIGHT, LEFT, MAX_RANGE, RIGHT, RUNS_DIR, beams_between
from wm2.data import load_split
from wm2.model import LATENT, Encoder

# two kinds of heads:
#   "walls5"  (heads.pt): front, right, left, front_right, front_left - the 5 probe walls of the old models
#   "sectors" (heads_sectors.pt): the closest wall in 13 sectors of 15 degrees from -90 (right) to +90 (left).
#             The 5 walls miss the corners between them, where the car hit the walls in the sim (drive try 4).
SECTOR_ANGLES = list(range(-90, 91, 15))
KINDS = {
    "walls5": ([FRONT, RIGHT, LEFT, FRONT_RIGHT, FRONT_LEFT], ["front", "right", "left", "front_right", "front_left"],
               "heads.pt"),
    "sectors": ([beams_between(a - 7.5, a + 7.5) for a in SECTOR_ANGLES], [f"{a:+d}" for a in SECTOR_ANGLES],
                "heads_sectors.pt"),
}
KIND = "walls5"
WALLS, WALL_NAMES, FILE = KINDS[KIND]
EPOCHS = 10
BATCH = 1024
CHUNK = 8192


class Heads(nn.Module):
    def __init__(self, walls=5):
        super().__init__()
        self.walls = walls
        self.body = nn.Sequential(nn.Linear(LATENT, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.out = nn.Linear(256, walls + 1 + 1)

    def forward(self, z):
        # z (..., LATENT) -> walls (..., n) metres, danger logit (...), progress (...) metres per step
        h = self.out(self.body(z))
        return h[..., : self.walls], h[..., self.walls], h[..., self.walls + 1]


def targets(data):
    walls = torch.stack([data["lidar"][:, s].float().min(dim=1).values for s in WALLS], dim=1) * MAX_RANGE
    return walls, data["danger"].float(), data["progress"]


def encode_all(encoder, data):
    with torch.no_grad():
        return torch.cat([encoder(data["lidar"][i : i + CHUNK].float(), data["state"][i : i + CHUNK])
                          for i in range(0, len(data["lidar"]), CHUNK)])


def load_heads(run, device, kind="walls5"):
    walls, _, file = KINDS[kind]
    heads = Heads(len(walls)).to(device)
    heads.load_state_dict(torch.load(f"{RUNS_DIR}/{run}/{file}", map_location=device))
    return heads.eval()


def main(run):
    path = f"{RUNS_DIR}/{run}/{FILE}"
    if os.path.exists(path):
        exit(f"{path} already exists: delete it first if you really want new heads")
    torch.manual_seed(0)
    encoder = Encoder().to(DEVICE)
    encoder.load_state_dict(torch.load(f"{RUNS_DIR}/{run}/encoder.pt", map_location=DEVICE))
    encoder.eval()
    train, test = load_split("train", DEVICE), load_split("test", DEVICE)
    z, z_t = encode_all(encoder, train), encode_all(encoder, test)
    (w, d, p), (w_t, d_t, p_t) = targets(train), targets(test)

    heads = Heads(len(WALLS)).to(DEVICE)
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
    # venv\Scripts\python -m wm2.heads wm2_k10 [walls5|sectors]
    if len(sys.argv) > 2:
        KIND = sys.argv[2]
        WALLS, WALL_NAMES, FILE = KINDS[KIND]
    main(sys.argv[1] if len(sys.argv) > 1 else "wm2_k10")
