# Night plan (user, 2026-11-08): collect fast practice data until the user closes the simulator or for 12 h at most,
# then stop the collector, close the simulator and train wm2 (wm2_k10, 20 epochs), unless the user says otherwise.
# Runs detached from the chat; writes what it is doing to night/state.txt and its logs to night/*.log.
# Start (from the BC folder):  venv\Scripts\python night\supervisor.py

import datetime
import os
import subprocess
import sys
import time

BC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(BC, "venv", "Scripts", "python.exe")
HERE = os.path.join(BC, "night")
SIM = "AutoDRIVE Simulator.exe"
MAX_COLLECT = 12 * 3600          # s
CHECK_EVERY = 30                 # s
TRAIN = [PY, "-u", "-m", "wm2.train", "wm2_k10", "20"]


def say(text):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {text}"
    with open(os.path.join(HERE, "state.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sim_running():
    out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {SIM}"], capture_output=True, text=True).stdout
    return SIM.lower() in out.lower()


def start_collector():
    log = open(os.path.join(HERE, "collect.log"), "a", encoding="utf-8")
    return subprocess.Popen([PY, "-u", "-m", "data_collection.collection_random", "practice"], cwd=BC,
                            stdout=log, stderr=subprocess.STDOUT)


def main():
    began = time.time()
    say("collect: start")
    collector, restarts = start_collector(), 0
    while True:
        time.sleep(CHECK_EVERY)
        if not sim_running():
            say("collect: the simulator was closed -> stop collecting")
            break
        if time.time() - began > MAX_COLLECT:
            say("collect: 12 h reached -> stop collecting")
            break
        if collector.poll() is not None:
            restarts += 1
            say(f"collect: the collector stopped (exit {collector.returncode}), restart {restarts}")
            if restarts > 5:
                say("collect: too many restarts -> stop collecting")
                break
            time.sleep(10)
            collector = start_collector()

    if collector.poll() is None:
        collector.terminate()
        collector.wait(timeout=30)
    say("collect: collector stopped")

    if sim_running():
        subprocess.run(["taskkill", "/IM", SIM], capture_output=True)          # ask it to close
        for _ in range(12):
            time.sleep(5)
            if not sim_running():
                break
        if sim_running():
            subprocess.run(["taskkill", "/F", "/IM", SIM], capture_output=True)
        say("sim: closed")
    time.sleep(20)                                                            # let the graphics card free its memory

    if os.path.exists(os.path.join(HERE, "NO_TRAIN")):
        say("train: night/NO_TRAIN exists -> no training (the user said so)")
        return
    if os.path.exists(os.path.join(BC, "runs", "wm2_k10")):
        say("train: runs/wm2_k10 already exists -> no training (nothing is overwritten)")
        return
    say("train: start wm2_k10, 20 epochs")
    with open(os.path.join(HERE, "train.log"), "a", encoding="utf-8") as log:
        code = subprocess.run(TRAIN, cwd=BC, stdout=log, stderr=subprocess.STDOUT).returncode
    say(f"train: finished, exit code {code}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:                                                 # never die silently
        say(f"supervisor crashed: {error!r}")
        sys.exit(1)
