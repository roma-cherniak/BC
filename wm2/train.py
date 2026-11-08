# Train wm2 (wm2/model.py) on data/wm2 (wm2/prepare.py). Saves runs/<RUN>/ like training/train_jepa.py:
# encoder.pt, predictor.pt, settings.json ("MODEL": "wm2"), loss.csv. Never overwrites an existing model.
#
# One training example = a window of HISTORY + K moments of one trip (mirrored with probability 1/2):
#   the encoder turns every moment into z; the predictor gets the first HISTORY real z's and imagines K steps
#   with the real actions; loss = mean error of the imagined z vs the encoder's real z over the K steps
#   + SPEED_WEIGHT x error of the imagined speed + SIGREG_WEIGHT x SIGReg (per moment of the window, like LeWM).
#
# Run from the BC folder (no sim at the same time: 4 GB graphics card):  venv\Scripts\python -m wm2.train
# A short try under another name:  venv\Scripts\python -m wm2.train wm2_try 2      (arguments: RUN, EPOCHS)

import csv
import datetime
import json
import os
import sys
import time

import torch

from config import DEVICE, RUNS_DIR
from core.sigreg import sigreg
from wm2.data import load_split, window_starts, windows
from wm2.model import BEAMS, HISTORY, LATENT, LAYERS, WIDTH, Encoder, Predictor, count, rollout

RUN = "wm2_k10"         # the name of the NEW model: change it before every new training
K = 10                  # steps imagined during training: 10 x 0.05 s = 0.5 s
EPOCHS = 20
BATCH = 256
LR = 3e-4
WEIGHT_DECAY = 1e-3
SIGREG_WEIGHT = 0.1     # LeWM's lambda
SPEED_WEIGHT = 1.0
MIRROR = 0.5            # share of windows flipped left <-> right
LENGTH = HISTORY + K


def losses(encoder, predictor, data, starts, mirror):
    lidar, state, action = windows(data, starts, LENGTH, mirror)
    B = len(starts)
    z = encoder(lidar.flatten(0, 1), state.flatten(0, 1)).view(B, LENGTH, LATENT)
    z_imag, speed_imag = rollout(predictor, z[:, :HISTORY], action[:, : LENGTH - 1])
    pred = ((z_imag - z[:, HISTORY:]) ** 2).mean()
    speed = ((speed_imag - state[:, HISTORY:, 0]) ** 2).mean()
    reg = torch.stack([sigreg(z[:, t]) for t in range(LENGTH)]).mean()
    copy = ((z[:, HISTORY - 1 : HISTORY] - z[:, HISTORY:]) ** 2).mean()      # "nothing changes" baseline
    return pred, speed, reg, copy, z


def main():
    global RUN, EPOCHS
    if len(sys.argv) > 1:
        RUN = sys.argv[1]
    if len(sys.argv) > 2:
        EPOCHS = int(sys.argv[2])
    folder = f"{RUNS_DIR}/{RUN}"
    os.makedirs(folder)     # no exist_ok: if this model already exists, the training stops here and overwrites nothing
    settings = {"MODEL": "wm2", "INPUT": "lidar", "BEAMS": BEAMS, "LATENT": LATENT, "HISTORY": HISTORY, "K": K,
                "WIDTH": WIDTH, "LAYERS": LAYERS, "EPOCHS": EPOCHS, "BATCH": BATCH, "LR": LR,
                "SIGREG_WEIGHT": SIGREG_WEIGHT, "SPEED_WEIGHT": SPEED_WEIGHT, "MIRROR": MIRROR,
                "DATA_DIR": "data/wm2 (from data/random_practice, split data/train.txt, val.txt, test.txt)",
                "date": datetime.datetime.now().isoformat(timespec="minutes")}
    with open(f"{folder}/settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    train, val = load_split("train", DEVICE), load_split("val", DEVICE)
    train_starts, val_starts = window_starts(train["trip"], LENGTH), window_starts(val["trip"], LENGTH)
    encoder, predictor = Encoder().to(DEVICE), Predictor().to(DEVICE)
    print(f"encoder {count(encoder):,} + predictor {count(predictor):,} numbers; "
          f"{len(train_starts):,} train windows, {len(val_starts):,} val windows")
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    steps = EPOCHS * ((len(train_starts) + BATCH - 1) // BATCH)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=steps, pct_start=0.03)

    log = open(f"{folder}/loss.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "train_pred", "train_speed", "sigreg", "val_pred", "val_copy", "val_pred_over_copy",
                     "val_speed", "spread"])
    for epoch in range(EPOCHS):
        t0 = time.time()
        encoder.train(), predictor.train()
        order = train_starts[torch.randperm(len(train_starts))]
        sums = torch.zeros(3)
        for start in range(0, len(order), BATCH):
            starts = order[start : start + BATCH]
            pred, speed, reg, _, _ = losses(encoder, predictor, train, starts, torch.rand(len(starts)) < MIRROR)
            loss = pred + SPEED_WEIGHT * speed + SIGREG_WEIGHT * reg
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            sums += torch.tensor([pred.item(), speed.item(), reg.item()]) * len(starts)
        train_pred, train_speed, train_reg = (sums / len(order)).tolist()

        encoder.eval(), predictor.eval()
        with torch.no_grad():
            vsum, zs = torch.zeros(3), []
            for start in range(0, len(val_starts), 4 * BATCH):
                starts = val_starts[start : start + 4 * BATCH]
                pred, speed, _, copy, z = losses(encoder, predictor, val, starts, None)
                vsum += torch.tensor([pred.item(), copy.item(), speed.item()]) * len(starts)
                zs.append(z[:, 0])
            val_pred, val_copy, val_speed = (vsum / len(val_starts)).tolist()
            spread = torch.cat(zs).std(dim=0).mean().item()

        torch.save(encoder.state_dict(), f"{folder}/encoder.pt")
        torch.save(predictor.state_dict(), f"{folder}/predictor.pt")
        writer.writerow([epoch, train_pred, train_speed, train_reg, val_pred, val_copy, val_pred / val_copy,
                         val_speed, spread])
        log.flush()
        print(f"epoch {epoch}  pred {train_pred:.4f}  speed {train_speed:.5f}  sigreg {train_reg:.2f}  "
              f"val pred/copy {val_pred / val_copy:.3f}  val speed {val_speed:.5f}  spread {spread:.3f}  "
              f"time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
