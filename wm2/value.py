# Value head of wm2: "from this moment, how many metres does the car still drive in the next SECONDS?"
#
# Why (docs/research_log.md E15 - E18): the plan covers 1 - 1.5 s and its cost only knows walls and "metres forward".
# On `compete` an open pocket beside the track then looks as good as the road, and the car drives into it. Wall
# distances cannot tell the two apart, but the recorded trips can: from the pocket the trips went ~1 m and crashed,
# from the road they kept driving. The head learns that number from the real trips; the planner can then ask it about
# the END of every imagined plan (a terminal value, like TD-MPC's value function, but learned from recorded returns).
#
# The "right answer" of a moment = the sum of `progress` (metres forward per step, wm2/prepare.py) over the next STEPS
# moments of the same trip. A crash ends the sum (nothing is driven after it). A trip that ends WITHOUT a crash before
# the STEPS are over has no known answer and is left out.
# Learns on REAL z of the frozen encoder, like wm2/heads.py. Saves runs/<RUN>/value.pt.
# Run from the BC folder (sim closed):  venv\Scripts\python -m wm2.value wm2_c1

import os
import sys

import numpy as np
import torch
import torch.nn as nn

from config import DEVICE, RUNS_DIR
from wm2.data import load_split, run_data_dir, scale_setting
from wm2.heads import augmented, encode_all
from wm2.model import LATENT, Encoder

SECONDS = 3.0
STEP = 0.05                     # s between two moments
STEPS = int(SECONDS / STEP)
FILE = "value.pt"
EPOCHS = 10
BATCH = 1024


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(LATENT, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, z):
        # z (..., LATENT) -> metres still driven in the next SECONDS (...)
        return self.body(z)[..., 0]


def future_metres(trip: np.ndarray, progress: np.ndarray, danger: np.ndarray, steps: int = STEPS):
    # -> target (N,) metres driven in the next `steps` moments of the same trip, known (N,) bool
    # danger of the LAST moment of a trip is 1 only when the trip crashed (wm2/prepare.py), so it marks the crash
    n = len(trip)
    last = np.flatnonzero(np.append(trip[1:] != trip[:-1], True))          # index of the last moment of every trip
    end_of = last[np.searchsorted(last, np.arange(n))]                     # for every moment: its trip's last index
    stop = np.minimum(np.arange(n) + steps, end_of + 1)                    # the sum covers moments i .. stop - 1
    total = np.concatenate([[0.0], np.cumsum(progress, dtype=np.float64)])
    target = (total[stop] - total[np.arange(n)]).astype(np.float32)
    crashed = danger[end_of] > 0
    known = (np.arange(n) + steps <= end_of + 1) | crashed
    return target, known


def targets(data):
    target, known = future_metres(data["trip"].cpu().numpy(), data["progress"].cpu().numpy(),
                                  data["danger"].cpu().numpy())
    return torch.from_numpy(target).to(DEVICE), torch.from_numpy(known).to(DEVICE)


def load_value(run, device, file=FILE):
    value = Value().to(device)
    value.load_state_dict(torch.load(f"{RUNS_DIR}/{run}/{file}", map_location=device))
    return value.eval()


def main(run, data_dir=None, file=FILE):
    # data_dir: other data than the model was trained on (the same encoder, more trips: night/value_round.py)
    path = f"{RUNS_DIR}/{run}/{file}"
    if os.path.exists(path):
        sys.exit(f"{path} already exists: delete it first if you really want a new value head")
    torch.manual_seed(0)
    encoder = Encoder().to(DEVICE)
    encoder.load_state_dict(torch.load(f"{RUNS_DIR}/{run}/encoder.pt", map_location=DEVICE))
    encoder.eval()
    data_dir = data_dir or run_data_dir(run)
    train, test = load_split("train", DEVICE, data_dir), load_split("test", DEVICE, data_dir)
    (t, known), (t_t, known_t) = targets(train), targets(test)
    z, z_t = encode_all(encoder, train)[known], encode_all(encoder, test)[known_t]
    t, t_t = t[known], t_t[known_t]
    print(f"train: {len(t):,} moments with a known answer ({known.float().mean().item():.1%}), "
          f"mean {t.mean().item():.2f} m in {SECONDS:.0f} s; test: {len(t_t):,}")

    # WM2_VALUE_BALANCE=1: every 0.5 m band of answers weighs the same in the loss. Moments shortly before a crash are
    # a few percent of the data, and they are the ones the planner needs the head for (E20).
    weight = torch.ones_like(t)
    if os.environ.get("WM2_VALUE_BALANCE") == "1":
        band = (t.clamp(0, 7.99) / 0.5).long()
        count = torch.bincount(band, minlength=16).float().clamp(min=1)
        weight = (len(t) / 16) / count[band]
        weight = weight.clamp(max=20.0)
        print("balanced: weight of a moment by its answer band (0 - 0.5 m, 0.5 - 1 m, ...): "
              + " ".join(f"{w:.1f}" for w in ((len(t) / 16) / count).clamp(max=20.0).tolist()[:10]))

    value = Value().to(DEVICE)
    optimizer = torch.optim.AdamW(value.parameters(), lr=1e-3, weight_decay=1e-4)
    # WM2_SCALE=0.6,2.2: every epoch the moments are shown mirrored and in bigger / smaller worlds, like wm2.train
    # (wm2/heads.py augmented); in a world k times bigger the car drives k times as many metres
    scale_range, t_real = scale_setting(), t
    for epoch in range(EPOCHS):
        if scale_range is not None:
            z_all, _, _, _, scale = augmented(encoder, train, scale_range)
            z, t = z_all[known], t_real * scale[known]
        order = torch.randperm(len(z), device=DEVICE)
        for i in range(0, len(z), BATCH):
            idx = order[i : i + BATCH]
            loss = (weight[idx] * nn.functional.smooth_l1_loss(value(z[idx]), t[idx], reduction="none")).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(value.state_dict(), path)

    value.eval()
    with torch.no_grad():
        v = value(z_t)
    err, naive = (v - t_t).abs().mean().item(), (t_t - t_real.mean()).abs().mean().item()
    corr = torch.corrcoef(torch.stack([v, t_t]))[0, 1].item()
    low, high = t_t < 1.0, t_t > 4.0
    print(f"test: mean error {err:.2f} m (always saying the mean: {naive:.2f} m), correlation {corr:.2f}")
    print(f"test: where the car really drove < 1 m the head says {v[low].mean().item():.2f} m ({int(low.sum()):,} "
          f"moments), where it drove > 4 m it says {v[high].mean().item():.2f} m ({int(high.sum()):,} moments)")
    print("saved", path)


if __name__ == "__main__":
    # venv\Scripts\python -m wm2.value wm2_c1 [data folder] [file name in runs/wm2_c1]
    main(sys.argv[1] if len(sys.argv) > 1 else "wm2_c1", *sys.argv[2:4])
