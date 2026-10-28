import eventlet
import eventlet.wsgi
import socketio
import base64
import gzip
from jepa.jepa_model import Encoder, Predictor, LATENT
import torch
import torch.nn as nn
import numpy as np



DEVICE = 'cuda'
STEERING = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
FACTS = ["front", "right", "left", "steering", "speed", "front_right", "front_left"]
STEPS = 20
FIRST_STEPS = 3
SECOND_STEPS = 7


predictor = Predictor().to(DEVICE)
predictor.load_state_dict(torch.load("jepa_predoctor.pt", map_location=DEVICE))


zond = nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, len(FACTS))).to(DEVICE)
zond.load_state_dict(torch.load("zond.pt", map_location=DEVICE))


def score(facts):
    front, right, left = facts[:, 0], facts[:, 1], facts[:, 2]
    score = front + right + left
    danger = (front <= 0.05) | (left <= 0.03) | (right <= 0.03) 
    return torch.where(danger, score - 1, score) 


def make_decesion(latent, speed, throttle):
    K = len(STEERING)

    steer = torch.tensor(STEERING, device=DEVICE)
    first  = steer.repeat_interleave(K) 
    second = steer.repeat(K)

    thr = torch.full((K * K,), throttle / 0.09, device=DEVICE)
    spd = torch.full((K * K, 1), speed / 3, device=DEVICE)

    with torch.no_grad():
        z = latent.repeat(81,1)

        for _ in range(FIRST_STEPS):
            z = predictor(z, torch.stack([thr, first], dim = 1), spd)


        for _ in range(SECOND_STEPS):
            z = predictor(z, torch.stack([thr, second], dim = 1), spd)


        scores = score(zond(z).clamp(0, 1)).view(K, K)
        best_first = scores.max(dim=1).values.argmax()   


        return STEERING[best_first.item()]
        

        





