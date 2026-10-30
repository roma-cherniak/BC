import numpy as np
import torch
from jepa.jepa_model import Encoder, Predictor, LATENT
import torch.nn as nn
from config import DEVICE, FACTS




def load_pairs(name):
    d = np.load(f"data/{name}_pairs.npz")
    lidar_now = torch.from_numpy(d["lidar_now"]).to(DEVICE)
    action = torch.from_numpy(d["act"]).float().to(DEVICE)
    speed = torch.from_numpy(d["spd"]).float().to(DEVICE)[:, None]
    lidar_next = torch.from_numpy(d["lidar_next"]).to(DEVICE)
    return lidar_now, action, speed, lidar_next




def load_jepa():
    encoder = Encoder().to(DEVICE)
    encoder.load_state_dict(torch.load("jepa_encoder.pt", map_location=DEVICE))

    predictor = Predictor().to(DEVICE)
    predictor.load_state_dict(torch.load("jepa_predoctor.pt", map_location=DEVICE))

    return encoder, predictor




def load_zond(file):
    zond = nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, len(FACTS))).to(DEVICE)
    zond.load_state_dict(torch.load(file, map_location=DEVICE))

    return zond
