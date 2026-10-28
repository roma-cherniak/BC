# JEPA driver. Every message from the game: the encoder describes the lidar, the predictor imagines 0.5 s ahead
# for 9 steering options, the probe reads the walls on the WHOLE imagined path, and the option whose closest wall
# stays farthest away is sent to the game.
# The settings (score, STEPS, no norm guard) won an offline test on the "random" test moments
# (the natural experiment of jepa/planner_test.py): moments where the coin steered like this planner
# were in danger 35.2 % of the time vs 50.1 % for the same choices shuffled between moments
# (your old sum front + right + left at 20 steps: 45.7 % vs 49.6 %).
#
# Run from the BC folder, then click "Disconnected" in the game menu:
#     venv\Scripts\python -m driver_lidar.drive_jepa

import base64
import csv
import gzip
import time

import eventlet
import eventlet.wsgi
import numpy as np
import socketio
import torch
import torch.nn as nn

from jepa.jepa_model import Encoder, Predictor, LATENT


DEVICE = "cpu"              # the game needs the graphics card; 9 options x 10 steps take a few ms on the CPU
STEERING = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
STEPS = 10                  # 10 x 0.05 s = 0.5 s of imagination (best in the offline test)
THROTTLE = 0.05             # constant; the training data had 0.03 .. 0.09
SMOOTH = 0.01               # penalty per unit of steering change (0.01 = 10 cm of wall distance) -> less zig-zag
RESET_PAUSE = 30            # messages to ignore the collision counter after a reset (the game needs a moment)
PRINT_EVERY = 60            # ~1 s
FACTS = ["front", "right", "left", "steering", "speed", "front_right", "front_left"]


def read_laser(data):
    packed = data["V1 LIDAR Range Array"]
    try:
        text = gzip.decompress(base64.b64decode(packed)).decode("utf-8")   # practice / compete builds
    except Exception:
        text = packed                                                      # some builds send plain text
    return np.array([float(v) for v in text.split()])


def clean_lidar(ranges):
    
    return np.minimum(ranges[::4], 10) / 10


def score(facts):
    """the closest wall, where a diagonal counts /1.4 (in a corridor it is naturally ~1.4x farther) and the front /2:
    bigger = safer"""
    walls = torch.stack([facts[:, 1], facts[:, 2], facts[:, 5] / 1.4, facts[:, 6] / 1.4, facts[:, 0] / 2], dim=1)
    return walls.min(dim=1).values


encoder = Encoder().to(DEVICE)
encoder.load_state_dict(torch.load("jepa_encoder.pt", map_location=DEVICE))
predictor = Predictor().to(DEVICE)
predictor.load_state_dict(torch.load("jepa_predoctor.pt", map_location=DEVICE))
zond = nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, len(FACTS))).to(DEVICE)
zond.load_state_dict(torch.load("zond.pt", map_location=DEVICE))

K = len(STEERING)
OPTIONS = torch.tensor(STEERING, device=DEVICE)


def plan(lidar, speed, previous):
    """lidar: 271 cleaned beams, speed in m/s, previous: the last steering -> the best steering (a plain number)"""
    with torch.no_grad():
        z = encoder(torch.tensor(lidar, dtype=torch.float32, device=DEVICE)[None]).repeat(K, 1)   # (9, 32)
        act = torch.stack([torch.full((K,), THROTTLE / 0.09, device=DEVICE), OPTIONS], dim=1)       # (9, 2)
        spd = torch.full((K, 1), speed / 3, device=DEVICE)                                           # (9, 1)
        worst = torch.full((K,), float("inf"), device=DEVICE)   # the closest wall on the whole imagined path
        for _ in range(STEPS):
            z = predictor(z, act, spd)
            worst = torch.minimum(worst, score(zond(z).clamp(0, 1)))
        worst = worst - SMOOTH * (OPTIONS - previous).abs()
    return STEERING[worst.argmax().item()]


sio = socketio.Server()
state = {"packets": 0, "steering": 0.0, "pause": 0, "crashes": 0, "start": None, "plan_ms": 0.0}
log_file = open("drive_log.csv", "w", newline="")
writer = csv.writer(log_file)
writer.writerow(["seconds", "steering", "speed", "real_front", "real_right", "real_left", "crashes"])


@sio.on("connect")
def connect(sid, environ):
    print("Connected! Caller ID:", sid)


@sio.on("Bridge")
def bridge(sid, data):
    state["packets"] += 1
    if state["start"] is None:
        state["start"] = time.time()
    reset = "False"

    lidar = clean_lidar(read_laser(data))
    speed = float(data["V1 Speed"])

    t0 = time.perf_counter()
    state["steering"] = plan(lidar, speed, state["steering"])
    state["plan_ms"] = (time.perf_counter() - t0) * 1000

    if state["pause"] > 0:                                  # just reset: the counter may still show the old crash
        state["pause"] -= 1
    elif int(data["V1 Collisions"]) > 0:
        state["crashes"] += 1
        state["pause"] = RESET_PAUSE
        state["steering"] = 0.0
        reset = "True"

    seconds = time.time() - state["start"]
    front, right, left = lidar[130:141].min() * 10, lidar[40:51].min() * 10, lidar[220:231].min() * 10
    writer.writerow([round(seconds, 2), state["steering"], speed, round(front, 2), round(right, 2),
                     round(left, 2), state["crashes"]])
    if state["packets"] % PRINT_EVERY == 0:
        log_file.flush()
        per_min = state["crashes"] / max(seconds / 60, 1e-9)
        print(f"{seconds:6.0f} s  steer {state['steering']:+.2f}  speed {speed:.2f} m/s  "
              f"walls F {front:.2f} R {right:.2f} L {left:.2f} m  crashes {state['crashes']} "
              f"({per_min:.1f}/min)  plan {state['plan_ms']:.1f} ms")

    sio.emit("Bridge", data={"V1 Throttle": str(THROTTLE), "V1 Steering": str(state["steering"]), "Reset": reset})


print("Door is open, waiting for the game...")
eventlet.wsgi.server(eventlet.listen(("127.0.0.1", 4567)), socketio.WSGIApp(sio), log_output=False)
