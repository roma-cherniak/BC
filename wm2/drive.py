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
#     venv\Scripts\python -m wm2.drive wm2_fast_k10 0.09 gas09   (model, top gas of the grid, log name tag)

import csv
import glob
import math
import os
import random
import sys
import threading
import time
from collections import deque

import eventlet
import eventlet.wsgi
import numpy as np
import socketio
import torch

from config import FRONT, LEFT, MAX_RANGE, RIGHT, RUNS_DIR, SPEED_MAX, THROTTLE_MAX
from core.sensors import clean_lidar, read_laser
from data_collection import recorder
from data_collection.packet import read_packet
from wm2 import cem
from wm2.heads import load_heads
from wm2.model import HISTORY, LATENT, Encoder, GraphRollout, Predictor, rollout

MODEL = sys.argv[1] if len(sys.argv) > 1 else "wm2_k10"
GAS_TOP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06     # the highest throttle the grid may choose
TAG = f"_{sys.argv[3]}" if len(sys.argv) > 3 else ""             # drive_practice<TAG>.csv: one log per try
# "explore" (4th argument): the planner drives AND records its trips for training (stage C, the car learns by itself).
# Every trip gets a random gas ceiling, and ~10% of the time a short random steering offset or brake pulse, so the
# data also show what happens when the car does NOT do what the planner wanted (mode "explore", else "self").
EXPLORE = len(sys.argv) > 4 and sys.argv[4] == "explore"
EXPLORE_GAS = [0.09, 0.12, 0.15, 0.18]
NOISE_START = 0.015             # chance per 0.05 s frame that a noise event begins (~10% of the time noisy)
NOISE_STEER = (0.3, 0.7)        # size of a steering offset (either sign), steering units -1 .. 1
NOISE_STEER_TIME = (0.2, 0.6)   # s
NOISE_BRAKE_TIME = (0.15, 0.4)  # s of gas 0
TRIP_MAX = 60.0                 # s: a longer trip is saved and a new one begins (no reset, the car keeps driving)
SELF_DIR = "data/self_practice"
TRACK = "practice"
DEVICE = "cpu"
THREADS = 6                     # CPU threads for the planner
SIZE = dict(samples=32, iters=1, horizon=20, elites=6)   # try 3 (24/2/16) took 0.34 s in the sim -> 8 of 16 steps
# horizon 12 (0.6 s) until 2026-11-09; braking from 2.6 m/s takes ~0.6 s, so the plan now looks 1 s ahead
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
# 2026-11-09, the fast GPU planner (41 ms) showed this cost is too careful: the old ~300 ms planner could not choose
# "brake NOW" (7 steps were fixed by the old plan), the fast one braked 31-36% of the time, laps 18-27 s. Split on
# real moments: the centre term was ~4x all the progress. Softer (sim, gas 0.12, 5.5 min): 0 crashes, gas 0 22%,
# lap median 15.63 s (yesterday's CPU best 15.64). Gas 0.15 with it: 0 crashes but slower (17.1 s: v^2 margin).
cem.W_CENTER = 0.3
cem.SIDE_MARGIN = 0.35
cem.FRONT_BASE = 0.4
# for tuning in the sim without editing this file:  set WM2_COST=W_CENTER=0.3,SIDE_MARGIN=0.35,FRONT_BASE=0.4
for item in filter(None, os.environ.get("WM2_COST", "").split(",")):
    name, value = item.split("=")
    if not hasattr(cem, name.strip()):
        sys.exit(f"WM2_COST: wm2/cem.py has no {name}")
    setattr(cem, name.strip(), float(value))

# 2026-11-09: the plan took 250-450 ms on the CPU, the car drove 1-2 m on an old plan. The imagination (predictor +
# heads) now runs on the graphics card as one CUDA graph (model.GraphRollout, ~55 ms, 60 MB next to the sim); the
# encoder (one frame) stays on the CPU. set WM2_PLAN_DEVICE=cpu for the old way.
PLAN_DEVICE = os.environ.get("WM2_PLAN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(THREADS)
encoder, predictor = Encoder().to(DEVICE), Predictor().to(PLAN_DEVICE)
encoder.load_state_dict(torch.load(f"{RUNS_DIR}/{MODEL}/encoder.pt", map_location=DEVICE))
predictor.load_state_dict(torch.load(f"{RUNS_DIR}/{MODEL}/predictor.pt", map_location=PLAN_DEVICE))
encoder.eval(), predictor.eval()
heads = load_heads(MODEL, PLAN_DEVICE, HEADS_KIND)
ROLL = GraphRollout(predictor) if PLAN_DEVICE == "cuda" else None      # None: model.rollout on the CPU
# first GPU drive (plan 41 ms): 0 crashes but gas 0 for 36% of the time, laps 26 s. Re-planning every 40 ms, the car
# only ever drives the first step of "brake 3 steps, then gas". WM2_PLAN_EVERY = min. seconds between plan starts
PLAN_EVERY = float(os.environ.get("WM2_PLAN_EVERY", "0"))

lock = threading.Lock()
frames = deque(maxlen=HISTORY)  # (lidar tensor (109,), state tensor (2,), action sent then (2,) in model units)
shared = {"plan": None, "plan_time": 0.0, "plan_ms": 0.0, "plans": 0, "reset": False, "gas_top": GAS_TOP,
          "epoch": 0}           # +1 at every reset: a plan begun before the reset is thrown away (see planner_loop)


# try 6: random-shooting CEM with 32 samples gives a different answer every 0.15 s: the steering jumped by 0.27 per
# 0.1 s (median; the old driver: 0.00) and the slow wheels never followed -> crashes. The grid planner is
# deterministic: every combination of a constant steering and a constant throttle after the prefix, all in one batch,
# plus a price for changing the steering -> calm, repeatable decisions (and faster: one rollout instead of two).
PLANNER = "grid"
GRID_STEER = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
# 2026-11-09: wm2_fast_k10 learned trips up to 4.7 m/s (throttle up to 0.2), so the grid may go faster:
# gas 0, 0.03, 0.06, ... up to GAS_TOP. The front margin grows with the imagined speed (cem: 0.5 + v^2 / 8).
def grid_gas(top):
    # always 4 levels: 0, 1/3, 2/3 and all of the ceiling (0.09 -> 0, .03, .06, .09 as before). The self night of
    # 2026-11-09 first used steps of 0.03: at ceiling 0.18 that is 99 plans, ~450 ms per plan in the sim (0.12: 256 ms)
    # and ~11 crashes/min, because the car drove 1 m "blind" on an old plan
    return [round(top * i / 3, 3) for i in range(4)]


GRID_GAS = grid_gas(GAS_TOP)
BRAKE_STEPS = [3, 6]            # "brake 0.15 s / 0.3 s, then gas" plans (see gas_profiles)
W_SWITCH = 0.3                  # cost per unit of steering change from the steering driven now
CREEP_SPEED = 0.3               # m/s: slower than this, braking (gas 0) is not an option


def gas_profiles(gas_top, speed_now):
    # -> list of throttle sequences for the steps after the prefix (each one value per step, len = horizon)
    # 2026-11-09, gas 0.12: 11 crashes at 2.3-2.9 m/s with the wheels already at full lock, gas 0 chosen 2% of the
    # time. A constant gas for the whole plan only offers "brake all 0.6 s" (little progress) or "go": nobody brakes
    # into a corner. So also "brake BRAKE_STEPS steps, then gas" (like a racing driver), with a 1 s horizon.
    horizon = SIZE["horizon"]
    gas = [g for g in grid_gas(gas_top) if g > 0 or speed_now >= CREEP_SPEED]
    profiles = [[g] * horizon for g in gas]
    if speed_now >= CREEP_SPEED:
        for k in BRAKE_STEPS:
            for g in {gas_top, grid_gas(gas_top)[len(grid_gas(gas_top)) // 2]}:
                profiles.append([0.0] * k + [g] * (horizon - k))
    return profiles


def grid_plan(z, past, prefix, steer_now, speed_now, gas_top=GAS_TOP, old=None):
    # z (1, HISTORY, D), past (1, HISTORY - 1, 2) on PLAN_DEVICE, prefix (1, d, 2) on the CPU
    # old (horizon, 2): the last plan moved to this frame, or None -> the best plan (1, horizon, 2), on the CPU
    # The old plan is one more candidate (a warm start, like CEM's): GPU drive 1 re-planned every 40 ms from a fresh
    # grid that has "brake 3 steps, then gas" but not "brake 2 more, then gas" -> it braked again and again and
    # never reached the gas (36% gas 0). With the old plan in the grid the car can keep a plan it already began.
    # try 7 drove 90 s with 1 crash at 1.5 m/s, then stopped in front of a wall for good: standing still never comes
    # closer to a wall, so "gas 0" won every time. When nearly standing, it must at least creep (gas >= 0.03).
    horizon, d = SIZE["horizon"], prefix.shape[1]
    profiles = torch.tensor(gas_profiles(gas_top, speed_now)) / THROTTLE_MAX           # (P, horizon)
    steer = torch.tensor(GRID_STEER)
    n_p, n_s = len(profiles), len(steer)
    plans = torch.empty(n_s, n_p, horizon, 2)
    plans[..., 0] = 0
    plans[:, :, d:, 0] = profiles[None, :, : horizon - d]                             # the profile starts after the prefix
    plans[..., 1] = steer[:, None, None]
    plans = plans.flatten(0, 1)
    keep = plans[-1] if old is None else old                                         # always n_s x n_p + 1 plans:
    plans = torch.cat([plans, keep[None]])                                           # fixed shapes for the graphs
    plans[:, :d] = prefix
    n = len(plans)
    # every plan shares z, past and the prefix: on the CPU the first d imagined steps are computed once (same=d)
    cost = cem.sequence_cost(predictor, heads, z.expand(n, -1, -1), past.expand(n, -1, -1), plans.to(PLAN_DEVICE),
                             same=d, roll=ROLL).cpu()
    cost = cost + W_SWITCH * (plans[:, min(d, horizon - 1), 1] - steer_now).abs()   # steering of the free part
    return plans[cost.argmin()][None]


def warm_up():
    # build the CUDA graphs (one per number of plans: slow = no braking plans, fast = with them) before the game
    # connects, so the first plans of the drive do not wait ~1 s for it; also times one plan of each
    z = torch.zeros(1, HISTORY, LATENT, device=PLAN_DEVICE)
    past = torch.zeros(1, HISTORY - 1, 2, device=PLAN_DEVICE)
    prefix = torch.zeros(1, 5, 2)
    tops = {GAS_TOP, *(EXPLORE_GAS if EXPLORE else [])}
    with torch.no_grad():
        for top in sorted(tops):
            for speed in (0.0, 2.5):
                grid_plan(z, past, prefix, 0.0, speed, top)
                t = time.perf_counter()
                grid_plan(z, past, prefix, 0.0, speed, top)
                print(f"warm-up: gas top {top}, speed {speed}: {(time.perf_counter() - t) * 1000:.0f} ms per plan",
                      flush=True)


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
            gas_top, epoch = shared["gas_top"], shared["epoch"]
            if shared["reset"]:
                mean, shared["reset"] = None, False
        if len(history) < HISTORY:
            time.sleep(0.01)
            continue
        if PLAN_EVERY > 0 and shared["plan"] is not None and time.monotonic() - shared["plan_time"] < PLAN_EVERY:
            time.sleep(0.005)
            continue
        t0 = time.perf_counter()
        d = min(int(math.ceil(delay / STEP)) + 1, SIZE["horizon"] - 4)
        prefix = torch.stack([history[-1][2]] + [action_at(frame_time + j * STEP + 1e-6, old_plan, old_time)
                                                  for j in range(1, d)])[None]
        lidar = torch.stack([f[0] for f in history])[None]                  # (1, HISTORY, 109)
        state = torch.stack([f[1] for f in history])[None]
        past = torch.stack([f[2] for f in history[:-1]])[None]              # actions between the frames
        with torch.no_grad():
            z = encoder(lidar.flatten(0, 1), state.flatten(0, 1)).view(1, HISTORY, LATENT).to(PLAN_DEVICE)
            past = past.to(PLAN_DEVICE)
            if mean is not None:                                             # warm start, moved to this frame
                for _ in range(min(int(round((frame_time - mean_time) / STEP)), SIZE["horizon"])):
                    mean = cem.shift(mean)
            if PLANNER == "grid":
                old = None if old_plan is None else torch.stack(
                    [action_at(frame_time + j * STEP + 1e-6, old_plan, old_time) for j in range(SIZE["horizon"])])
                best = grid_plan(z, past, prefix, history[-1][2][1].item(), history[-1][1][0].item() * SPEED_MAX,
                                 gas_top, old)
            else:
                best, _ = cem.plan(predictor, heads, z, past, None if mean is None else mean.to(PLAN_DEVICE),
                                   prefix=prefix.to(PLAN_DEVICE), **SIZE)
                best = best.cpu()
            log_now = imagined_writer is not None and shared["plans"] % 3 == 0
            if log_now:                                                      # what the model expects to see
                walls = heads(rollout(predictor, z, torch.cat([past, best.to(PLAN_DEVICE)], dim=1))[0])[0].cpu()
        mean, mean_time = best, frame_time
        took = time.perf_counter() - t0
        delay = 0.8 * delay + 0.2 * took
        with lock:
            # 2026-11-09 self1 drive: crashes came in pairs < 1 s apart. The plan computed from the frames BEFORE a
            # crash was written AFTER the reset, and the car drove it from the new place (often at full lock).
            if shared["epoch"] != epoch:
                mean = None
                continue
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
                     "crashes", "plan_ms", "lap_count", "last_lap", "best_lap"])


trip = {"number": 0, "start": None, "noise": None}       # explore mode: the trip being recorded


def next_trip_number():
    os.makedirs(SELF_DIR, exist_ok=True)
    old = sorted(glob.glob(f"{SELF_DIR}/drive_*.npz"))
    return int(os.path.basename(old[-1])[6:10]) if old else 0          # continue after the last saved trip


def new_trip():
    # a new trip: a new gas ceiling for the planner, an empty recorder
    recorder.new_episod()
    trip["start"], trip["noise"] = None, None
    with lock:
        shared["gas_top"] = random.choice(EXPLORE_GAS)


def save_trip(path, rows, reason):
    # like data_collection.recorder.save, but for a copy of the rows: it runs in its own thread, because compressing
    # a trip with its camera images takes long enough to make the game wait for its answers
    columns = {key: [row[key] for row in rows] for key in recorder.KEYS}
    columns["lidar"] = np.array(columns["lidar"], dtype=np.float16)
    np.savez_compressed(path, **columns, track=TRACK, reason=reason, warmup=0)


def end_trip(reason):
    if recorder.rows:
        trip["number"] += 1
        path, rows = f"{SELF_DIR}/drive_{trip['number']:04d}.npz", list(recorder.rows)
        threading.Thread(target=save_trip, args=(path, rows, reason)).start()
        print(f"trip {trip['number']}: {len(rows)} rows, {reason}, gas top {shared['gas_top']}", flush=True)
    new_trip()


def explore_action(action, now):
    # the planner's action (model units) -> the action really sent, and the mode ("self" or "explore")
    noise = trip["noise"]
    if noise is not None and now >= noise[2]:
        noise = trip["noise"] = None
    if noise is None and now >= drive["next_frame"] and random.random() < NOISE_START:
        if random.random() < 0.5:
            offset = random.uniform(*NOISE_STEER) * random.choice([-1, 1])
            noise = trip["noise"] = ("steer", offset, now + random.uniform(*NOISE_STEER_TIME))
        else:
            noise = trip["noise"] = ("brake", 0.0, now + random.uniform(*NOISE_BRAKE_TIME))
    if noise is None:
        return action, "self"
    sent = action.clone()
    if noise[0] == "steer":
        sent[1] = max(-1.0, min(1.0, float(sent[1]) + noise[1]))
    else:
        sent[0] = 0.0
    return sent, "explore"


def record_row(data, sent, mode, now):
    if trip["start"] is None:
        trip["start"] = now
    obs = read_packet(data)
    recorder.add_row(obs, float(sent[1]), float(sent[0]) * THROTTLE_MAX, now - trip["start"], mode)


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
    sent, mode = explore_action(drive["action"], now) if EXPLORE else (drive["action"], "self")
    crashed = drive["pause"] == 0 and int(data["V1 Collisions"]) > 0
    if now >= drive["next_frame"]:                                           # a new frame for the model, 20 per s
        with lock:
            frames.append((lidar, state, sent.clone()))                      # what was really sent (noise included)
            shared["frame_time"] = now
        drive["next_frame"] = max(drive["next_frame"] + STEP, now)
        if EXPLORE and drive["pause"] == 0 and not crashed:
            record_row(data, sent, mode, now)
            if now - trip["start"] >= TRIP_MAX:
                end_trip("timeout")

    if drive["pause"] > 0:
        drive["pause"] -= 1
    elif crashed:
        if EXPLORE:
            record_row(data, sent, mode, now)                                # the crash row, like the collector
            end_trip("crash")
        drive["crashes"] += 1
        drive["pause"] = RESET_PAUSE
        reset = "True"
        with lock:
            frames.clear()
            shared["plan"], shared["reset"] = None, True
            shared["epoch"] += 1
        drive["action"] = torch.tensor([START_ACTION[0] / THROTTLE_MAX, START_ACTION[1]])
        sent = drive["action"]

    throttle = float(sent[0]) * THROTTLE_MAX
    steering = float(sent[1])
    seconds = now - drive["start"]
    front, right, left = (float(lidar[s].min()) * MAX_RANGE for s in (FRONT, RIGHT, LEFT))
    laps = (data.get("V1 Lap Count", ""), data.get("V1 Last Lap Time", ""), data.get("V1 Best Lap Time", ""))
    writer.writerow([round(seconds, 2), round(steering, 3), round(throttle, 4), round(speed, 3), round(front, 2),
                     round(right, 2), round(left, 2), drive["crashes"], round(plan_ms), *laps])
    if drive["packets"] % PRINT_EVERY == 0:
        log_file.flush()
        per_min = drive["crashes"] / max(seconds / 60, 1e-9)
        print(f"{seconds:6.0f} s  steer {steering:+.2f}  gas {throttle:.3f}  speed {speed:.2f} m/s  "
              f"F {front:.2f} R {right:.2f} L {left:.2f} m  crashes {drive['crashes']} ({per_min:.1f}/min)  "
              f"plan {plan_ms:.0f} ms, {shared['plans']} plans  laps {laps[0]} last {laps[1]} best {laps[2]}", flush=True)
    drive["reply"] = {"V1 Throttle": str(throttle), "V1 Steering": str(steering), "Reset": "False"}
    sio.emit("Bridge", data={"V1 Throttle": str(throttle), "V1 Steering": str(steering), "Reset": reset}, to=sid)


if __name__ == "__main__":
    log_path = f"{RUNS_DIR}/{MODEL}/drive_{TRACK}{TAG}.csv"
    start_log(log_path)
    if EXPLORE:
        trip["number"] = next_trip_number()
        new_trip()
        print(f"explore: trips -> {SELF_DIR}, next trip {trip['number'] + 1}, gas ceilings {EXPLORE_GAS}", flush=True)
    warm_up()
    threading.Thread(target=planner_loop, daemon=True).start()
    print(f"Door is open, waiting for the game... model {MODEL}, planner {SIZE} on {PLAN_DEVICE}, gas {GRID_GAS}, "
          f"log {log_path}")
    eventlet.wsgi.server(eventlet.listen(("127.0.0.1", 4567)), socketio.WSGIApp(sio), log_output=False)
