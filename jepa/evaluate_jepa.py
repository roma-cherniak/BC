import torch
from jepa.jepa_model import Encoder, LATENT
import torch.nn as nn
from utils import load_pairs, DEVICE, FACTS



BATCH = 256
LR = 0.001
PROBE_EPOCHS = 20
TRAIN_MOMENTS = 100000


torch.manual_seed(0)                                  # the same random moments and random encoder every run


def make_targets(lidar_now, action, speed):
    front = torch.min(lidar_now[:, 130:140], dim = 1).values
    right = torch.min(lidar_now[: , 40:50], dim = 1).values
    left = torch.min(lidar_now[: , 220:230], dim = 1).values
    steering = action[:, 1]
    speed = speed[:, 0]
    front_right = torch.min(lidar_now[:, 85:95], dim = 1).values      # -45° (index = angle + 135)
    front_left = torch.min(lidar_now[:, 175:185], dim = 1).values     # +45°
    return torch.stack([front, right, left, steering, speed, front_right, front_left], dim=1)


lidar_now, action, speed, lidar_next = load_pairs("train")
idx = torch.randperm(len(lidar_now), device=DEVICE)[:TRAIN_MOMENTS]   # 100k random moments from all trips
lidar_now = lidar_now[idx].float()
targets = make_targets(lidar_now, action[idx], speed[idx])

lidar_now_t, action_t, speed_t, lidar_next_t = load_pairs("test")
lidar_now_t = lidar_now_t.float()
targets_t = make_targets(lidar_now_t, action_t, speed_t)


def probe_r2(encoder):
    with torch.no_grad():                             # the encoder is frozen
        z = encoder(lidar_now)
        z_t = encoder(lidar_now_t)

    zond = nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, len(FACTS))).to(DEVICE)   # a NEW MLP probe for every encoder
    optimizer = torch.optim.Adam(zond.parameters(), lr=LR)

    for epoch in range(PROBE_EPOCHS):
        order = torch.randperm(len(z), device=DEVICE)
        for start in range(0, len(order), BATCH):
            idx = order[start : start + BATCH]
            loss = ((zond(z[idx]) - targets[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():                             # your R² block, now for any encoder
        zond_guess = zond(z_t)
        lazy_guess = targets.mean(dim=0)
        lazy_error = ((lazy_guess - targets_t) ** 2).mean(dim=0)
        zond_error = ((zond_guess - targets_t) ** 2).mean(dim=0)
        return 1 - zond_error / lazy_error , zond


model = Encoder().to(DEVICE)
model.load_state_dict(torch.load("jepa_encoder.pt", map_location = DEVICE))

r2_trained, train_zond = probe_r2(model)
torch.save(train_zond.state_dict(), "zond.pt")
r2_random, random_zond = probe_r2(Encoder().to(DEVICE))            # control: the same blueprint, random knobs

print(f"{'fact':<10}{'trained':>10}{'random':>10}")
for name, a, b in zip(FACTS, r2_trained, r2_random):
    print(f"{name:<10}{a.item():>10.3f}{b.item():>10.3f}")
