
import numpy as np
import torch
from config import DEVICE


def load_pairs(name, device=DEVICE):
    d = np.load(f"data/{name}_pairs.npz")
    lidar_now = torch.from_numpy(d["lidar_now"]).to(device)
    action = torch.from_numpy(d["act"]).float().to(device)
    speed = torch.from_numpy(d["spd"]).float().to(device)[:, None]
    lidar_next = torch.from_numpy(d["lidar_next"]).to(device)
    return lidar_now, action, speed, lidar_next


def safe_starts(lidar_now, lidar_next, steps):
    same = (lidar_next[:-1] == lidar_now[1:]).all(dim=1).cpu()      # on the CPU the loop below is much faster
    safe = [i for i in range(len(same) - steps) if same[i : i + steps].all()]
    return torch.tensor(safe, device=lidar_now.device)
