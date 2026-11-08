# The report of one model. Run from the BC folder:  venv\Scripts\python -m checks.run_all k10_50ep
# Writes runs/<name>/report.md and runs/<name>/report.json.

import csv
import datetime
import glob
import json
import os
import sys
from core.data import load_pairs, pair_obs, safe_starts
from core.models import load_run, run_input
from checks.check_code import check_code
from checks.check_imagination import check_imagination, STEPS as IMAGINE_STEPS
from config import DEVICE, RUNS_DIR, CURRENT_RUN
from checks.check_planner import check_planner
from checks.check_training import check_training

def drive_results(folder):
    # crashes per minute from every runs/<name>/drive_<track>.csv the driver wrote
    results = {}
    for path in sorted(glob.glob(f"{folder}/drive_*.csv")):
        track = os.path.basename(path)[len("drive_"):-len(".csv")]
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            minutes = float(rows[-1]["seconds"]) / 60
            results[f"drive {track}: crashes per minute"] = int(rows[-1]["crashes"]) / max(minutes, 1e-9)
            results[f"drive {track}: minutes"] = minutes
    return results



run = sys.argv[1] if len(sys.argv) > 1 else CURRENT_RUN
folder = f"{RUNS_DIR}/{run}"
kind = run_input(run)                       # "lidar" or "photo": the only place where the two kinds differ
test = load_pairs("test")                   # actions, speed and the trips (domino) are the same for both
obs = pair_obs("test", kind)                # what the encoder eats, one row per test pair
_, action, speed, _ = test

encoder, predictor, zond = load_run(run, DEVICE)


if not check_code(encoder, predictor, zond, test, obs)["code_ok"]:
    exit("code brocken")

results = {}
safe = safe_starts(test[0], test[3], IMAGINE_STEPS)
results.update(check_imagination(encoder, predictor, obs, action, speed, safe))
results.update(drive_results(folder))
results.update(check_planner(encoder, predictor, zond, obs, action, speed))
results.update(check_training(folder))

with open(f"{folder}/report.json", "w") as f:
    json.dump({"run": run, "input": kind, "date": datetime.datetime.now().isoformat(timespec="minutes"), **results},
              f, indent=2)
with open(f"{folder}/report.md", "w", encoding="utf-8") as f:
    f.write(f"# Report: {run}\n\n| check | value |\n|---|---|\n")
    for name, value in results.items():
        f.write(f"| {name} | {value:.4g} |\n")

print(results)
print("saved", f"{folder}/report.md", "and report.json")


