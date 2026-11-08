# One status line about the night plan (night/supervisor.py): what phase, trips collected, training progress.
# Run from the BC folder:  venv\Scripts\python night\status.py

import glob
import os
import subprocess
import time

import numpy as np

BC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(BC, "night")


def last_state():
    path = os.path.join(HERE, "state.txt")
    if not os.path.exists(path):
        return "no state yet"
    lines = open(path, encoding="utf-8").read().strip().splitlines()
    return lines[-1] if lines else "no state yet"


def trips():
    files = sorted(glob.glob(os.path.join(BC, "data", "fast_practice", "drive_*.npz")))
    if not files:
        return "0 trips"
    newest = time.time() - os.path.getmtime(files[-1])
    rows, crashes, fast = 0, 0, 0
    for path in files[-30:]:                                                  # the newest 30 trips: crash share, speed
        try:
            with np.load(path) as f:
                crashes += str(f["reason"]) == "crash"
                fast += int((f["speed"] > 2.2).sum())
                rows += len(f["time"])
        except Exception:
            pass
    total_rows = 0
    for path in files:
        total_rows += os.path.getsize(path)
    sample = min(30, len(files))
    return (f"{len(files)} trips (~{len(files) * 0.4:.0f} min of driving), newest {newest / 60:.0f} min ago, "
            f"last {sample}: crash {crashes / sample:.0%}, >2.2 m/s {fast / max(rows, 1):.0%}")


def training():
    path = os.path.join(HERE, "train.log")
    if not os.path.exists(path):
        return ""
    epochs = [l for l in open(path, encoding="utf-8", errors="ignore") if l.startswith("epoch")]
    errors = [l for l in open(path, encoding="utf-8", errors="ignore") if "Error" in l or "Traceback" in l]
    text = f" | training: {len(epochs)}/20 epochs"
    if epochs:
        text += " | " + epochs[-1].strip()
    if errors:
        text += " | ERROR: " + errors[-1].strip()
    return text


def sim():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq AutoDRIVE Simulator.exe"], capture_output=True, text=True)
    return "sim on" if "autodrive" in out.stdout.lower() else "sim off"


print(f"{last_state()} | {sim()} | {trips()}{training()}")
