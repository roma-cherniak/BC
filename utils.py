import numpy as np
import torch

DEVICE = 'cuda'





def load_pairs(name):
    d = np.load(f"data/{name}_pairs.npz")
    lidar_now = torch.from_numpy(d["lidar_now"]).to(DEVICE)
    action = torch.from_numpy(d["act"]).float().to(DEVICE)
    speed = torch.from_numpy(d["spd"]).float().to(DEVICE)[:, None]
    lidar_next = torch.from_numpy(d["lidar_next"]).to(DEVICE)
    return lidar_now, action, speed, lidar_next
