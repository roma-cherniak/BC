# JEPA driver. Every message from the game: the encoder describes the lidar, the predictor imagines 0.5 s ahead
# for 9 steering options, the probe reads the walls on the WHOLE imagined path, and the option whose closest wall
# stays farthest away is sent to the game.
# The settings (score, STEPS, no norm guard) won an offline test on the "random" test moments
# (the natural experiment, now checks/check_planner.py): moments where the coin steered like this planner
# were in danger 35.2 % of the time vs 50.1 % for the same choices shuffled between moments
# (your old sum front + right + left at 20 steps: 45.6 % vs 49.6 %).
#
# Run from the BC folder, then click "Disconnected" in the game menu:
#     venv\Scripts\python -m driver_lidar.drive_jepa

import csv
import time

import eventlet
import eventlet.wsgi
import socketio
import torch

from config import THROTTLE_MAX, SPEED_MAX, MAX_RANGE, RUNS_DIR
from core.models import load_run
from core.sensors import read_laser, clean_lidar
from core.planner import STEERING, score_options


MODEL = "k10_50ep"          # which model drives: runs/<MODEL>/
TRACK = "practice"          # which sim build is open: the log goes to runs/<MODEL>/drive_<TRACK>.csv
DEVICE = "cpu"              # the game needs the graphics card; 9 options x 10 steps take a few ms on the CPU
THROTTLE = 0.08             # constant; the training data had 0.03 .. 0.09
SMOOTH = 0.01               # penalty per unit of steering change (0.01 = 10 cm of wall distance) -> less zig-zag
RESET_PAUSE = 30            # messages to ignore the collision counter after a reset (the game needs a moment)
PRINT_EVERY = 60            # ~1 s


encoder, predictor, zond = load_run(MODEL, DEVICE)

OPTIONS = torch.tensor(STEERING, device=DEVICE)


sio = socketio.Server()
state = {"packets": 0, "steering": 0.0, "pause": 0, "crashes": 0, "start": None, "plan_ms": 0.0}
log_file = open(f"{RUNS_DIR}/{MODEL}/drive_{TRACK}.csv", "w", newline="")    # a new drive replaces the old log
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
    lidar_t = torch.tensor(lidar, dtype=torch.float32, device=DEVICE)[None]                 # (1, 271): M = 1 moment
    throttle_t = torch.tensor([THROTTLE / THROTTLE_MAX], device=DEVICE)                     # (1,)
    speed_t = torch.tensor([[speed / SPEED_MAX]], device=DEVICE)                            # (1, 1)
    scores = score_options(encoder, predictor, zond, lidar_t, throttle_t, speed_t)[0]       # (9,)
    scores = scores - SMOOTH * (OPTIONS - state["steering"]).abs()                           # less zig-zag
    state["steering"] = STEERING[scores.argmax().item()]
    state["plan_ms"] = (time.perf_counter() - t0) * 1000

    if state["pause"] > 0:                                  # just reset: the counter may still show the old crash
        state["pause"] -= 1
    elif int(data["V1 Collisions"]) > 0:
        state["crashes"] += 1
        state["pause"] = RESET_PAUSE
        state["steering"] = 0.0
        reset = "True"

    seconds = time.time() - state["start"]
    front, right, left = (lidar[130:141].min() * MAX_RANGE, lidar[40:51].min() * MAX_RANGE,
                          lidar[220:231].min() * MAX_RANGE)                          # back to meters for the log
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
