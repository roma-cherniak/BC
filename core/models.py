# Models: the JEPA (Encoder + Predictor) and the probe (zond). One trained model = one folder runs/<name>/.

import json

import torch
import torch.nn as nn
from config import RUNS_DIR, FACTS, FRONT, RIGHT, LEFT, FRONT_RIGHT, FRONT_LEFT, PHOTO_HEIGHT,PHOTO_WIDTH


LIDAR = 271
ACTION = 2
SPEED = 1
LATENT = 32
HIDDEN = 256


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LIDAR, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, LATENT)
        )

    def forward(self, lidar):
        return self.net(lidar)



class ImageEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(                                   # sizes for a 36 x 64 photo:
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),    # -> 16 x 18 x 32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),   # -> 32 x 9 x 16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # -> 64 x 5 x 8
            nn.ReLU(),
            nn.Flatten()                                             # -> 64 * 5 * 8 = 2560 numbers in one row
        )
        with torch.no_grad():                                        # how many numbers come out of conv
            flat = self.conv(torch.zeros(1, 1, PHOTO_HEIGHT, PHOTO_WIDTH)).shape[1]
        self.out = nn.Linear(flat, LATENT)

    def forward(self, image):
        return self.out(self.conv(image))


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT + ACTION + SPEED, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, LATENT)
        )

    def forward(self, latent, action, speed):
        x = torch.cat([latent , action , speed], dim = 1)
        return self.net(x)


def make_zond():
    # a plain Sequential, not a class: zond.pt stores the keys "0.weight", ... (a class would make "net.0.weight")
    return nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, len(FACTS)))


def make_targets(lidar_now, action, speed):
    # what the probe must read, in the order of FACTS
    facts = {
        "front": torch.min(lidar_now[:, FRONT], dim=1).values,
        "right": torch.min(lidar_now[:, RIGHT], dim=1).values,
        "left": torch.min(lidar_now[:, LEFT], dim=1).values,
        "steering": action[:, 1],
        "speed": speed[:, 0],
        "front_right": torch.min(lidar_now[:, FRONT_RIGHT], dim=1).values,
        "front_left": torch.min(lidar_now[:, FRONT_LEFT], dim=1).values,
    }
    return torch.stack([facts[name] for name in FACTS], dim=1)


def run_input(name):
    # what the model eats, from runs/<name>/settings.json: "photo", or "lidar" (old models have no INPUT)
    settings = json.load(open(f"{RUNS_DIR}/{name}/settings.json"))
    return settings.get("INPUT", "lidar")


def load_run(name, device, with_zond=True):
    # runs/<name>/encoder.pt, predictor.pt, zond.pt -> encoder, predictor, zond (zond = None if with_zond=False)
    folder = f"{RUNS_DIR}/{name}"

    if run_input(name) == "photo":
        encoder = ImageEncoder().to(device)
    else :
        encoder = Encoder().to(device)
    
    encoder.load_state_dict(torch.load(f"{folder}/encoder.pt", map_location=device))
    predictor = Predictor().to(device)
    predictor.load_state_dict(torch.load(f"{folder}/predictor.pt", map_location=device))
    zond = None
    if with_zond:
        zond = make_zond().to(device)
        zond.load_state_dict(torch.load(f"{folder}/zond.pt", map_location=device))
    return encoder, predictor, zond


