# Loop 1 of the autoresearch: ONE trial = one fixed 5.5-minute drive of wm2/drive.py in the practice sim, judged the
# same way every time. The search (research/loop1.py) only chooses the settings; it must never change this file
# (like prepare.py in Karpathy's autoresearch: the judge is fixed, otherwise a "better" score may only mean a softer
# judge).
#
# Judge, in this order:
#   1. valid?  the sim sends >= MIN_RATE packets/s (2026-11-09: a slow sim turned 0 crashes into 12 for the SAME
#              model), the driver connected and drove >= SECONDS
#   2. gate:   real crashes <= MAX_CRASHES (a crash in the first WARMUP s is the car still rolling from the last
#              driver, not this one) and >= MIN_LAPS laps
#   3. score:  median lap time (s, smaller = better) of the full laps (the first, incomplete lap and laps > 40 s out)
# Every trial is one row in research/results.tsv (also invalid and failed ones).
#
# Run from the BC folder with the practice sim open (and nothing else driving):
#   venv\Scripts\python -m research.run_trial <tag> <model> <gas> [NAME=value,...]
#   e.g. venv\Scripts\python -m research.run_trial base_a wm2_self2 0.12 FRONT_BASE=0.2

import csv
import datetime
import os
import statistics
import subprocess
import sys
import time

BC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(BC, "venv", "Scripts", "python.exe")
HERE = os.path.join(BC, "research")
RESULTS = os.path.join(HERE, "results.tsv")
SIM = "AutoDRIVE Simulator.exe"

SECONDS = 330           # s of driving per trial: ~20 laps of ~15 s; one trial's lap median is good to ~+-0.3 s
WARMUP = 5              # s: crashes before this belong to the previous driver
MAX_CRASHES = 0         # the user's main task: "fast and WITHOUT crashes"
MIN_LAPS = 10
MIN_RATE = 30           # packets/s
MAX_LAP = 40            # s: longer "laps" are restarts, not driving
CONNECT_TIMEOUT = 180   # s until the game must have connected
STALE = 60              # s without a new log row = the sim stopped sending
COLUMNS = ["time", "tag", "model", "gas", "cost", "valid", "why", "crashes", "laps", "median", "best", "gas0",
           "mean_speed", "pkt_s", "plan_ms", "score"]


def sim_running():
    out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {SIM}"], capture_output=True, text=True).stdout
    return SIM.lower() in out.lower()


def other_driver():
    # another wm2.drive (or collector) would hold port 4567: then this trial cannot connect
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          "Where-Object { $_.CommandLine -like '*wm2.drive*' -or $_.CommandLine -like '*collection*' } | "
                          "Measure-Object | Select-Object -ExpandProperty Count"], capture_output=True, text=True).stdout
    return int(out.strip() or 0) > 0


def judge(path):
    # the drive log (one row per packet) -> the numbers of one trial
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 100:
        return {"valid": False, "why": "no drive"}
    t = [float(r["seconds"]) for r in rows]
    crashes_at_warmup = next((int(r["crashes"]) for r in rows if float(r["seconds"]) >= WARMUP), 0)
    laps, seen = [], None
    for r in rows:
        if r["lap_count"] != seen and seen is not None and r["last_lap"] not in ("", "Infinity", "inf"):
            laps.append(float(r["last_lap"]))
        seen = r["lap_count"]
    full = [lap for lap in laps[1:] if lap < MAX_LAP]                       # the first lap started mid-track
    out = {"crashes": int(rows[-1]["crashes"]) - crashes_at_warmup, "laps": len(full),
           "median": statistics.median(full) if full else float("nan"), "best": min(full) if full else float("nan"),
           "gas0": sum(float(r["throttle"]) == 0 for r in rows) / len(rows),
           "mean_speed": statistics.mean(float(r["speed"]) for r in rows),
           "pkt_s": len(rows) / max(t[-1] - t[0], 1e-9),
           "plan_ms": statistics.median(float(r["plan_ms"]) for r in rows)}
    if t[-1] < SECONDS - 10:
        out.update(valid=False, why=f"drove only {t[-1]:.0f} s")
    elif out["pkt_s"] < MIN_RATE:
        out.update(valid=False, why=f"sim slow: {out['pkt_s']:.0f} packets/s")
    elif out["crashes"] > MAX_CRASHES:
        out.update(valid=True, why=f"gate: {out['crashes']} crashes")
    elif out["laps"] < MIN_LAPS:
        out.update(valid=True, why=f"gate: {out['laps']} laps")
    else:
        out.update(valid=True, why="ok")
    out["score"] = out["median"] if out["why"] == "ok" else float("inf")
    return out


def kill_tree(process):
    if process.poll() is None:                    # the venv python.exe is a launcher with a child: kill both
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
        process.wait(timeout=30)


def drive(tag, model, gas, cost, env=None):
    # one drive -> path of its log; raises if the sim is closed or someone else drives
    if not sim_running():
        raise RuntimeError("the practice sim is not running")
    if other_driver():
        raise RuntimeError("another driver is running (port 4567)")
    path = os.path.join(BC, "runs", model, f"drive_practice_{tag}.csv")
    if os.path.exists(path):
        raise RuntimeError(f"{path} exists: every trial needs a new tag")
    environment = dict(os.environ, PYTHONUNBUFFERED="1", WM2_PLAN_EVERY="",
                       WM2_COST=",".join(f"{k}={v}" for k, v in cost.items()), **(env or {}))
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log = open(os.path.join(HERE, "logs", f"{tag}.log"), "w", encoding="utf-8")
    process = subprocess.Popen([PY, "-u", "-m", "wm2.drive", model, str(gas), tag], cwd=BC, env=environment,
                               stdout=log, stderr=subprocess.STDOUT)
    began, last_size, last_growth = time.time(), -1, time.time()
    try:
        while True:
            time.sleep(5)
            if process.poll() is not None:
                break
            if not os.path.exists(path):
                if time.time() - began > CONNECT_TIMEOUT:
                    break
                continue
            size = os.path.getsize(path)
            if size != last_size:
                last_size, last_growth = size, time.time()
            elif time.time() - last_growth > STALE:
                break
            with open(path, "rb") as f:                                 # the last row: seconds driven so far
                f.seek(max(0, size - 300))
                tail = f.read().decode(errors="ignore").strip().splitlines()
            if tail and tail[-1].split(",")[0].replace(".", "", 1).isdigit() and float(tail[-1].split(",")[0]) >= SECONDS:
                break
    finally:
        kill_tree(process)
        log.close()
    time.sleep(3)
    return path


def run_trial(tag, model, gas, cost, env=None):
    path = drive(tag, model, gas, cost, env)
    result = judge(path) if os.path.exists(path) else {"valid": False, "why": "never connected", "score": float("inf")}
    row = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "tag": tag, "model": model, "gas": gas,
           "cost": ",".join(f"{k}={v}" for k, v in cost.items()), **result}
    record(row)
    return row


def record(row):
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, COLUMNS, delimiter="\t", extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})


def describe(row):
    return (f"{row['tag']:28s} {row['why']:22s} crashes {row.get('crashes', '-')}  laps {row.get('laps', '-')}  "
            f"median {row.get('median', float('nan')):.2f}  best {row.get('best', float('nan')):.2f}  "
            f"gas0 {row.get('gas0', 0) * 100:.0f}%  {row.get('pkt_s', 0):.0f} pkt/s  plan {row.get('plan_ms', 0):.0f} ms")


if __name__ == "__main__":
    cost = dict(item.split("=") for item in sys.argv[4].split(",")) if len(sys.argv) > 4 and sys.argv[4] else {}
    print(describe(run_trial(sys.argv[1], sys.argv[2], float(sys.argv[3]), cost)))
