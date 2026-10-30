import torch
from jepa.jepa_model import Encoder, Predictor, LATENT
import torch.nn as nn
from config import DEVICE, FACTS
from core.data import load_pairs        # moved to core/data.py; still here so old `from utils import load_pairs` works




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
