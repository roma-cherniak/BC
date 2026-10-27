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


predictor = Predictor().to(DEVICE)
predictor.load_state_dict(torch.load("jepa_predoctor.pt", map_location=DEVICE))

zond = nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, len(FACTS))).to(DEVICE)
zond.load_state_dict(torch.load("zond.pt", map_location=DEVICE))


def score(facts):
    """facts: (rows, 5) probe readings, distances in 0..1 (x10 = meters) -> one score per row, bigger = better."""
    front, right, left = facts[:, 0], facts[:, 1], facts[:, 2]
    score = front + right + left
    danger = (front <= 0.05) | (left <= 0.03) | (right <= 0.03) 
    return torch.where(danger, score - 1, score) 



def make_decesion(latent, speed, throttle):

    
    facts = []

    with torch.no_grad():
        speed = torch.tensor([[speed / 3]], device=DEVICE) 
        for angle in STEERING:
            z = latent
            act = torch.tensor([[throttle / 0.09, angle]], device=DEVICE) 
            for _ in range(STEPS):  
                z = predictor(z, act, speed)

            fact = zond(z).clamp(0, 1)
            facts.append(fact)



        all_facts = torch.cat(facts)
        scores = score(all_facts)                             # one row per moment, one column per option
        winner = scores.argmax()                                          # (M,): number of the best option
        best = torch.tensor(STEERING, device=DEVICE)[winner] 

    return best.item()









