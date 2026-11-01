# Eyes check: can a small NEW probe read the FACTS from z? The trained encoder vs a random one (the same blueprint).
# If trained is not better than random, the encoder learned nothing extra that a probe can use. Nothing is saved.
# (probe_r2 from jepa/evaluate_jepa.py.)
# Run from the BC folder:  venv\Scripts\python -m checks.check_eyes k10_50ep

import sys
import torch
from core.data import load_pairs
from core.models import Encoder, load_run, make_zond, make_targets
from config import DEVICE, CURRENT_RUN, FACTS

BATCH = 256
LR = 0.001
PROBE_EPOCHS = 20
TRAIN_MOMENTS = 100000


def probe_r2(encoder, lidar, targets, lidar_t, targets_t):
    # a NEW probe learns facts from z of the train moments -> R² of every fact on test (1 = perfect, 0 = only the average)
    with torch.no_grad():                             # the encoder is frozen
        z, z_t = encoder(lidar), encoder(lidar_t)
    zond = make_zond().to(lidar.device)
    optimizer = torch.optim.Adam(zond.parameters(), lr=LR)
    for epoch in range(PROBE_EPOCHS):
        order = torch.randperm(len(z), device=lidar.device)
        for start in range(0, len(order), BATCH):
            idx = order[start : start + BATCH]
            loss = ((zond(z[idx]) - targets[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        zond_error = ((zond(z_t) - targets_t) ** 2).mean(dim=0)
        lazy_error = ((targets.mean(dim=0) - targets_t) ** 2).mean(dim=0)
    return 1 - zond_error / lazy_error


def check_eyes(encoder, predictor, zond, test):
    device = test[0].device
    gpus = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=gpus):         # own dice: the global seed stays as it was
        torch.manual_seed(0)
        lidar, action, speed, _ = load_pairs("train", device)
        idx = torch.randperm(len(lidar), device=device)[:TRAIN_MOMENTS]      # 100k random moments from all trips
        lidar = lidar[idx].float()
        targets = make_targets(lidar, action[idx], speed[idx])
        lidar_t = test[0].float()
        targets_t = make_targets(lidar_t, test[1], test[2])

        trained = probe_r2(encoder, lidar, targets, lidar_t, targets_t)
        random_encoder = probe_r2(Encoder().to(device), lidar, targets, lidar_t, targets_t)   # control: random knobs

    results = {}
    for name, a, b in zip(FACTS, trained, random_encoder):
        results[f"eyes R² {name}"] = a.item()
        results[f"eyes R² {name} (random)"] = b.item()
    return results


if __name__ == "__main__":
    test = load_pairs("test")
    encoder, predictor, zond = load_run(sys.argv[1] if len(sys.argv) > 1 else CURRENT_RUN, DEVICE)
    for name, value in check_eyes(encoder, predictor, zond, test).items():
        print(f"{name:<28}{value:.3f}")
