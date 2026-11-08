
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

def load_photos(name, device=DEVICE):
    d = np.load(f"data/{name}_photos.npz")
    images = torch.from_numpy(d["image"]).to(device)
    action = torch.from_numpy(d["act"]).float().to(device)
    speed = torch.from_numpy(d["spd"]).float().to(device)[:, None]
    trip = torch.from_numpy(d["trip"]).to(device)
    lidar = torch.from_numpy(d["lidar"]).to(device)          # the lidar of the same frame: the probe's right answers
    return images, action, speed, trip, lidar


def pair_obs(name, kind, device=DEVICE):
    # what the encoder eats for every pair of data/{name}_pairs.npz, in the same order: lidar_now or that moment's photo
    if kind == "photo":
        images, _, _, trip, _ = load_photos(name, device)
        has_next = trip[:-1] == trip[1:]                       # a frame with a next frame in its trip = one pair
        return images[:-1][has_next][:, None].float() / 255
    return load_pairs(name, device)[0].float()


def safe_starts(lidar_now, lidar_next, steps):
    same = (lidar_next[:-1] == lidar_now[1:]).all(dim=1).cpu()      # on the CPU the loop below is much faster
    safe = [i for i in range(len(same) - steps) if same[i : i + steps].all()]
    return torch.tensor(safe, device=lidar_now.device)



def load_images(name, device = DEVICE):
    d = np.load(f"data/random_practice/drive_0001.npz")
    image = (d['image'])
    return image
