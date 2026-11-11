# Night plan 2026-11-10 (user: "так, все дозволяю, запуск"): stage C, round 2 - faster trips with the new driver.
# The same steps as night/self_night.py (round 1), with:
#   - driver: wm2_self1 + soft brakes + the front margin picked in the evening tuning (COST below),
#     gas ceilings 0.12 / 0.15 / 0.18 (round 1: 0.09 .. 0.18 with a ~400 ms planner -> ~9 crashes/min),
#     trips -> data/self2_practice (round 1 stays untouched in data/self_practice)
#   - training: wm2_self1 -> wm2_self2, EPOCHS epochs at lr 1e-4 on data/wm2_self2 (old + fast + self + self2)
# Runs detached; writes what it is doing to night/state2.txt, logs to night/self2_*.log.
# Start (from the BC folder):  venv\Scripts\python night\self2_night.py

import datetime
import os
import subprocess
import sys
import time

BC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(BC, "venv", "Scripts", "python.exe")
HERE = os.path.join(BC, "night")
SIM = "AutoDRIVE Simulator.exe"
MAX_COLLECT = float(os.environ.get("NIGHT2_HOURS", "2.5")) * 3600     # s
CHECK_EVERY = 30                # s
STALL = 120                     # s without a new log line -> restart the driver
TEACHER, STUDENT = "wm2_self1", "wm2_self2"
EPOCHS = os.environ.get("NIGHT2_EPOCHS", "3")
DRIVER_ENV = {"WM2_COST": os.environ.get("NIGHT2_COST", "FRONT_BASE=0.25"),
              "WM2_SOFT": os.environ.get("NIGHT2_SOFT", "1"),
              "WM2_EXPLORE_GAS": "0.12,0.15,0.18",
              "WM2_SELF_DIR": "data/self2_practice"}
STEPS = [
    ("split", [PY, "-u", "-m", "wm2.prepare", "split", "data/self2_practice", "self2"]),
    ("prepare", [PY, "-u", "-m", "wm2.prepare", "self2"]),
    ("train", [PY, "-u", "-m", "wm2.train", STUDENT, EPOCHS, "data/wm2_self2", TEACHER, "1e-4"]),
    ("heads sectors", [PY, "-u", "-m", "wm2.heads", STUDENT, "sectors"]),
    ("heads walls5", [PY, "-u", "-m", "wm2.heads", STUDENT, "walls5"]),
]


def say(text):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {text}"
    with open(os.path.join(HERE, "state2.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sim_running():
    out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {SIM}"], capture_output=True, text=True).stdout
    return SIM.lower() in out.lower()


def kill_tree(process):
    # the venv python.exe is a launcher with a child process: kill both
    if process.poll() is None:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
        process.wait(timeout=30)


def start_driver(n):
    log = open(os.path.join(HERE, f"self2_drive_{n}.log"), "a", encoding="utf-8")
    env = {**os.environ, **DRIVER_ENV, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen([PY, "-u", "-m", "wm2.drive", TEACHER, "0.18", f"self2_{n}", "explore"], cwd=BC,
                               stdout=log, stderr=subprocess.STDOUT, env=env)
    return process, os.path.join(BC, "runs", TEACHER, f"drive_practice_self2_{n}.csv")


def collect():
    began = time.time()
    say(f"collect: start, {MAX_COLLECT / 3600:.1f} h, driver {DRIVER_ENV}")
    n = 1
    driver, csv_path = start_driver(n)
    last_size, last_growth = -1, time.time()
    while True:
        time.sleep(CHECK_EVERY)
        if not sim_running():
            say("collect: the simulator was closed -> stop collecting")
            break
        if time.time() - began > MAX_COLLECT:
            say(f"collect: {MAX_COLLECT / 3600:.1f} h reached -> stop collecting")
            break
        size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
        if size != last_size:
            last_size, last_growth = size, time.time()
        stalled = time.time() - last_growth > STALL
        if driver.poll() is not None or stalled:
            why = f"exit {driver.returncode}" if driver.poll() is not None else f"no new log line for {STALL} s"
            n += 1
            say(f"collect: driver problem ({why}) -> restart {n - 1}")
            if n > 20:
                say("collect: too many restarts -> stop collecting")
                break
            kill_tree(driver)
            time.sleep(10)
            driver, csv_path = start_driver(n)
            last_size, last_growth = -1, time.time()
    kill_tree(driver)
    time.sleep(5)                                       # the last trip is saved in a thread
    say("collect: driver stopped")


def close_sim():
    if sim_running():
        subprocess.run(["taskkill", "/IM", SIM], capture_output=True)          # ask it to close
        for _ in range(12):
            time.sleep(5)
            if not sim_running():
                break
        if sim_running():
            subprocess.run(["taskkill", "/F", "/IM", SIM], capture_output=True)
        say("sim: closed")
    time.sleep(20)                                      # let the graphics card free its memory


def train():
    if os.path.exists(os.path.join(HERE, "NO_TRAIN")):
        say("train: night/NO_TRAIN exists -> no training (the user said so)")
        return
    if os.path.exists(os.path.join(BC, "runs", STUDENT)):
        say(f"train: runs/{STUDENT} already exists -> no training (nothing is overwritten)")
        return
    for name, command in STEPS:
        say(f"train: {name} - start")
        with open(os.path.join(HERE, f"self2_{name.split()[0]}.log"), "a", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=BC, stdout=log, stderr=subprocess.STDOUT).returncode
        say(f"train: {name} - finished, exit code {code}")
        if code != 0:
            say("train: stopped here because of the error")
            return
    say("night: ALL DONE")


def main():
    collect()
    close_sim()
    train()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:                          # never die silently
        say(f"supervisor crashed: {error!r}")
        sys.exit(1)
