# wm2 driver for the AutoDRIVE simulator: the CEM planner (wm2/cem.py) chooses steering AND throttle.
#
# One CEM decision takes ~0.1-0.2 s on the CPU (the graphics card belongs to the game), but the game sends ~60
# packets per second. So, like every MPC controller, the planner runs in its own thread and makes a 1 s PLAN; the
# game handler answers every packet at once with the action of the current 0.05 s step of the newest plan, and the
# next plan starts from the old one (warm start). The model sees what it learned on: every 0.05 s one frame
# (lidar 109 beams + speed + wheel angle) and the last HISTORY frames.
#
# Run from the BC folder, with the practice sim open, then click "Connect" in the game:
#     venv\Scripts\python -m wm2.drive               (model wm2_k10; the log: runs/wm2_k10/drive_practice.csv)

import csv
import math
import sys
import threading
import time
from collections import deque

import eventlet
import eventlet.wsgi
import socketio
import torch

from config import FRONT, LEFT, MAX_RANGE, RIGHT, RUNS_DIR, SPEED_MAX, THROTTLE_MAX
from core.sensors import clean_lidar, read_laser
from wm2 import cem
from wm2.heads import load_heads
from wm2.model import HISTORY, LATENT, Encoder, Predictor, rollout

MODEL = sys.argv[1] if len(sys.argv) > 1 else "wm2_k10"
TRACK = "practice"
DEVICE = "cpu"
THREADS = 6                     # CPU threads for the planner
SIZE = dict(samples=32, iters=1, horizon=12, elites=6)   # try 3 (24/2/16) took 0.34 s in the sim -> 8 of 16 steps
                                                        # were fixed prefix; fewer predictor calls = faster plans
STEP = 0.05                     # s between two frames, like the training data
MAX_STEER_RAD = 0.5236          # wheel angle in the packet (rad) -> -1 .. 1, like wm2/prepare.py
START_ACTION = (0.03, 0.0)      # throttle, steering until the first plan is ready
RESET_PAUSE = 30                # packets to ignore the collision counter after a reset
PRINT_EVERY = 60                # packets (~1 s)
DUPLICATE = 0.008               # s: a packet closer than this to the last one is a copy from an old connection

# first real drive (2026-11-08): with gas up to 0.09 and 0.3 m to the side it drove 1.7-2.1 m/s and hit the INNER
# wall of right turns 6 times in 30 s -> a bit slower and a wider side margin for now
cem.THROTTLE_RANGE = (0.0, 0.06)
cem.SIDE_MARGIN = 0.4
# try 3: the imagined side walls were right within 3-4 cm, yet it hit walls -> the cost, not the model. The danger
# head knows "a wall closer than 0.3 m anywhere in +-90 degrees" (what the 5 wall beams miss): weigh it 10x more
cem.W_DANGER = 0.0
# try 4 showed why: the danger head says "danger" for HARD STEERING itself (in the data hard steering = the safety
# driver = danger), so it punished turning away from a wall (steer away 0.75-0.95 vs straight 0.45-0.61).
# Instead: walls in 13 sectors of 15 degrees (heads_sectors.pt) cover the corners between the 5 old walls.
HEADS_KIND = "sectors"
# try 5: the imagined right wall was right within ~5 cm, yet the car drifted to the inner (right) wall of the
# clockwise track until every plan was bad: the margin 0.4 m only punished at the last moment. Now it also prefers
# the middle of the corridor (a 0.3 m off-centre costs as much as ~0.1 m of progress per step).
cem.W_CENTER = 1.0
cem.SIDE_MARGIN = 0.45

torch.set_num_threads(THREADS)
encoder, predictor = Encoder().to(DEVICE), Predictor().to(DEVICE)
encoder.load_state_dict(torch.load(f"{RUNS_DIR}/{MODEL}/encoder.pt", map_location=DEVICE))
predictor.load_state_dict(torch.load(f"{RUNS_DIR}/{MODEL}/predictor.pt", map_location=DEVICE))
encoder.eval(), predictor.eval()
heads = load_heads(MODEL, DEVICE, HEADS_KIND)

lock = threading.Lock()
frames = deque(maxlen=HISTORY)  # (lidar tensor (109,), state tensor (2,), action sent then (2,) in model units)
shared = {"plan": None, "plan_time": 0.0, "plan_ms": 0.0, "plans": 0, "reset": False}


# try 6: random-shooting CEM with 32 samples gives a different answer every 0.15 s: the steering jumped by 0.27 per
# 0.1 s (median; the old driver: 0.00) and the slow wheels never followed -> crashes. The grid planner is
# deterministic: every combination of a constant steering and a constant throttle after the prefix, all in one batch,
# plus a price for changing the steering -> calm, repeatable decisions (and faster: one rollout instead of two).
PLANNER = "grid"
GRID_STEER = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
GRID_GAS = [0.0, 0.03, 0.06]
W_SWITCH = 0.3                  # cost per unit of steering change from the steering driven now
CREEP_SPEED = 0.3               # m/s: slower than this, braking (gas 0) is not an option


def grid_plan(z, past, prefix, steer_now, speed_now):
    # z (1, HISTORY, D), past (1, HISTORY - 1, 2), prefix (1, d, 2) -> the best plan (1, horizon, 2) in model units
    # try 7 drove 90 s with 1 crash at 1.5 m/s, then stopped in front of a wall for good: standing still never comes
    # closer to a wall, so "gas 0" won every time. When nearly standing, it must at least creep (gas >= 0.03).
    gas = [g for g in GRID_GAS if g > 0 or speed_now >= CREEP_SPEED]
    options = torch.tensor([[g / THROTTLE_MAX, s] for s in GRID_STEER for g in gas])
    n, horizon = len(options), SIZE["horizon"]
    plans = options[:, None].expand(n, horizon, 2).clone()
    plans[:, : prefix.shape[1]] = prefix
    cost = cem.sequence_cost(predictor, heads, z.expand(n, -1, -1), past.expand(n, -1, -1), plans)
    cost = cost + W_SWITCH * (options[:, 1] - steer_now).abs()
    return plans[cost.argmin()][None]


def action_at(t, plan, plan_time):
    # the action the game handler sends at time t (monotonic s) while this plan is the newest one
    if plan is None:
        return torch.tensor([START_ACTION[0] / THROTTLE_MAX, START_ACTION[1]])
    return plan[min(max(int((t - plan_time) / STEP), 0), len(plan) - 1)]


def planner_loop():
    # every plan starts at the newest frame. While it is being computed (~0.2 s) the car keeps driving the OLD plan,
    # so the first d actions of the new plan are not free: they are the old plan's actions for those moments
    # (a fixed prefix). Without this the imagination assumed actions the car never drove (drive try 1 and 2).
    mean, mean_time, delay = None, 0.0, 0.2
    while True:
        with lock:
            history = list(frames)
            frame_time = shared.get("frame_time", 0.0)
            old_plan, old_time, start = shared["plan"], shared["plan_time"], shared.get("start", 0.0)
            if shared["reset"]:
                mean, shared["reset"] = None, False
        if len(history) < HISTORY:
            time.sleep(0.01)
            continue
        t0 = time.perf_counter()
        d = min(int(math.ceil(delay / STEP)) + 1, SIZE["horizon"] - 4)
        prefix = torch.stack([history[-1][2]] + [action_at(frame_time + j * STEP + 1e-6, old_plan, old_time)
                                                  for j in range(1, d)])[None]
        lidar = torch.stack([f[0] for f in history])[None]                  # (1, HISTORY, 109)
        state = torch.stack([f[1] for f in history])[None]
        past = torch.stack([f[2] for f in history[:-1]])[None]              # actions between the frames
        with torch.no_grad():
            z = encoder(lidar.flatten(0, 1), state.flatten(0, 1)).view(1, HISTORY, LATENT)
            if mean is not None:                                             # warm start, moved to this frame
                for _ in range(min(int(round((frame_time - mean_time) / STEP)), SIZE["horizon"])):
                    mean = cem.shift(mean)
            if PLANNER == "grid":
                best = grid_plan(z, past, prefix, history[-1][2][1].item(), history[-1][1][0].item() * SPEED_MAX)
            else:
                best, _ = cem.plan(predictor, heads, z, past, mean, prefix=prefix, **SIZE)
            log_now = imagined_writer is not None and shared["plans"] % 3 == 0
            if log_now:                                                      # what the model expects to see
                walls, _, _ = heads(rollout(predictor, z, torch.cat([past, best], dim=1))[0])
        mean, mean_time = best, frame_time
        took = time.perf_counter() - t0
        delay = 0.8 * delay + 0.2 * took
        with lock:
            shared["plan"], shared["plan_time"] = best[0].clone(), frame_time
            shared["plan_ms"], shared["plans"] = took * 1000, shared["plans"] + 1
        if log_now:                                                          # imagined walls, to compare later
            for k in (d + 4, d + 8):
                if k < SIZE["horizon"]:
                    w = walls[0, k]                                          # sectors: -90 .. +90
                    f, r, l = w[5:8].min().item(), w[0].item(), w[12].item()
                    imagined_writer.writerow([round(frame_time + (k + 1) * STEP - start, 3), k - d + 1,
                                              round(f, 2), round(r, 2), round(l, 2), d])


sio = socketio.Server()
drive = {"packets": 0, "pause": 0, "crashes": 0, "start": None, "next_frame": 0.0,
         "reply": {"V1 Throttle": str(START_ACTION[0]), "V1 Steering": "0", "Reset": "False"},
         "action": torch.tensor([START_ACTION[0] / THROTTLE_MAX, START_ACTION[1]])}
log_file, writer, imagined_writer = None, None, None


def start_log(path):
    global log_file, writer, imagined_writer
    imagined = open(path.replace(".csv", "_imagined.csv"), "w", newline="", buffering=1)
    imagined_writer = csv.writer(imagined)
    imagined_writer.writerow(["target_seconds", "steps_after_prefix", "front", "right", "left", "prefix"])
    log_file = open(path, "w", newline="")                                  # a new drive replaces the old log
    writer = csv.writer(log_file)
    writer.writerow(["seconds", "steering", "throttle", "speed", "real_front", "real_right", "real_left",
                     "crashes", "plan_ms"])


@sio.on("connect")
def connect(sid, environ):
    print("Connected! Caller ID:", sid)


@sio.on("Bridge")
def bridge(sid, data):
    # the game sometimes keeps old connections open after a reconnect and then sends every packet 2-4 times (drive
    # try 8: 232 packets/s instead of 58). Closing them made it reconnect again and again, so instead: a packet that
    # comes less than DUPLICATE s after the last one is only answered with the current action, not processed.
    now = time.monotonic()
    if now - drive.get("last", -1.0) < DUPLICATE:
        sio.emit("Bridge", data=drive["reply"], to=sid)
        return
    drive["last"] = now
    drive["packets"] += 1
    if drive["start"] is None:
        drive["start"] = now
        with lock:
            shared["start"] = now
    reset = "False"
    lidar = torch.tensor(clean_lidar(read_laser(data)), dtype=torch.float32)
    speed = float(data["V1 Speed"])
    state = torch.tensor([speed / SPEED_MAX, float(data["V1 Steering"]) / MAX_STEER_RAD], dtype=torch.float32)

    with lock:                                                               # the action of this 0.05 s step
        plan, plan_time, plan_ms = shared["plan"], shared["plan_time"], shared["plan_ms"]
    if plan is not None:
        k = min(max(int((now - plan_time) / STEP), 0), len(plan) - 1)
        drive["action"] = plan[k]
    if now >= drive["next_frame"]:                                           # a new frame for the model, 20 per s
        with lock:
            frames.append((lidar, state, drive["action"].clone()))
            shared["frame_time"] = now
        drive["next_frame"] = max(drive["next_frame"] + STEP, now)

    if drive["pause"] > 0:
        drive["pause"] -= 1
    elif int(data["V1 Collisions"]) > 0:
        drive["crashes"] += 1
        drive["pause"] = RESET_PAUSE
        reset = "True"
        with lock:
            frames.clear()
            shared["plan"], shared["reset"] = None, True
        drive["action"] = torch.tensor([START_ACTION[0] / THROTTLE_MAX, START_ACTION[1]])

    throttle = float(drive["action"][0]) * THROTTLE_MAX
    steering = float(drive["action"][1])
    seconds = now - drive["start"]
    front, right, left = (float(lidar[s].min()) * MAX_RANGE for s in (FRONT, RIGHT, LEFT))
    writer.writerow([round(seconds, 2), round(steering, 3), round(throttle, 4), round(speed, 3), round(front, 2),
                     round(right, 2), round(left, 2), drive["crashes"], round(plan_ms)])
    if drive["packets"] % PRINT_EVERY == 0:
        log_file.flush()
        per_min = drive["crashes"] / max(seconds / 60, 1e-9)
        print(f"{seconds:6.0f} s  steer {steering:+.2f}  gas {throttle:.3f}  speed {speed:.2f} m/s  "
              f"F {front:.2f} R {right:.2f} L {left:.2f} m  crashes {drive['crashes']} ({per_min:.1f}/min)  "
              f"plan {plan_ms:.0f} ms, {shared['plans']} plans", flush=True)
    drive["reply"] = {"V1 Throttle": str(throttle), "V1 Steering": str(steering), "Reset": "False"}
    sio.emit("Bridge", data={"V1 Throttle": str(throttle), "V1 Steering": str(steering), "Reset": reset}, to=sid)


if __name__ == "__main__":
    start_log(f"{RUNS_DIR}/{MODEL}/drive_{TRACK}.csv")
    threading.Thread(target=planner_loop, daemon=True).start()
    print(f"Door is open, waiting for the game... model {MODEL}, planner {SIZE}, log runs/{MODEL}/drive_{TRACK}.csv")
    eventlet.wsgi.server(eventlet.listen(("127.0.0.1", 4567)), socketio.WSGIApp(sio), log_output=False)
