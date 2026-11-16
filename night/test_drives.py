# Test drives of one model on the track of the open sim, like the tests of E13 / E14:
# gas 0.12, default cost (FRONT_BASE 0.25), soft brakes, no explore, SECONDS of driving each.
#   venv\Scripts\python night\test_drives.py wm2_c1 compete c1_test_a c1_test_b
# -> runs/<model>/drive_<track>_<tag>.csv; progress in night/state_tests.txt; driver output in night/test_<tag>.log
# Other cost or length:  set TEST_COST=FRONT_BASE=0.25,W_CENTER=0.1  and  set TEST_SECONDS=120  before starting.
# It refuses to start if the open sim build is not the build of this track (the log would get the wrong track name).

import csv
import datetime
import os
import subprocess
import sys
import time

BC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(BC, "venv", "Scripts", "python.exe")
HERE = os.path.join(BC, "night")
STATE = os.path.join(HERE, "state_tests.txt")
GAS_TOP = "0.12"
SECONDS = int(os.environ.get("TEST_SECONDS", "330"))            # s of driving per test
COST = os.environ.get("TEST_COST", "FRONT_BASE=0.25")           # WM2_COST of the driver; every tag gets the same
CONNECT_TIMEOUT = 600           # s to wait for the first packet (the user may have to click "Connect")
STALL = 120                     # s without a new log row after the drive began
CHECK_EVERY = 5


def say(text: str) -> None:
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {text}"
    print(line, flush=True)
    with open(STATE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sim_folder() -> str | None:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-Process 'AutoDRIVE Simulator' -ErrorAction SilentlyContinue | Select-Object -First 1).Path"],
                         capture_output=True, text=True).stdout.strip()
    return out or None


def driven_seconds(path: str) -> float:
    # the "seconds" of the last complete row of the drive log (0 if there is no row yet)
    if not os.path.exists(path):
        return 0.0
    with open(path, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.reader(f) if row]
    for row in reversed(rows[1:]):
        try:
            return float(row[0])
        except ValueError:
            continue
    return 0.0


def kill_tree(process: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)


def one_drive(model: str, track: str, tag: str) -> bool:
    path = os.path.join(BC, "runs", model, f"drive_{track}_{tag}.csv")
    if os.path.exists(path):
        say(f"{tag}: {path} already exists -> not driven again")
        return False
    # WM2_HORIZON and other driver settings pass through from the environment of this script
    env = {**os.environ, "WM2_TRACK": track, "WM2_COST": COST, "WM2_SOFT": "1"}
    say(f"{tag}: start ({model} on {track}, gas {GAS_TOP}, {SECONDS} s, cost {COST})")
    with open(os.path.join(HERE, f"test_{tag}.log"), "a", encoding="utf-8") as log:
        driver = subprocess.Popen([PY, "-u", "-m", "wm2.drive", model, GAS_TOP, tag], cwd=BC, env=env,
                                  stdout=log, stderr=subprocess.STDOUT)
    began, last_seen, last_change = time.time(), 0.0, time.time()
    try:
        while True:
            time.sleep(CHECK_EVERY)
            seen = driven_seconds(path)
            if seen >= SECONDS:
                say(f"{tag}: finished, {seen:.0f} s")
                return True
            if seen != last_seen:
                last_seen, last_change = seen, time.time()
            if driver.poll() is not None:
                say(f"{tag}: STOPPED, the driver exited with code {driver.returncode} after {seen:.0f} s")
                return False
            if seen == 0.0 and time.time() - began > CONNECT_TIMEOUT:
                say(f"{tag}: STOPPED, no packet from the sim in {CONNECT_TIMEOUT} s (click Connect?)")
                return False
            if seen > 0.0 and time.time() - last_change > STALL:
                say(f"{tag}: STOPPED, no new log row for {STALL} s at {seen:.0f} s")
                return False
    finally:
        kill_tree(driver)
        time.sleep(5)                                   # the port is free before the next driver starts


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[2] not in ("practice", "compete"):
        sys.exit("usage: night\\test_drives.py <model> <practice|compete> <tag> [<tag> ...]")
    model, track, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
    folder = sim_folder()
    if folder is None:
        sys.exit(f"the simulator is not running: open the {track.upper()} sim first")
    if f"simulator_{track}_" not in folder.lower():
        sys.exit(f"the open sim is not the {track} build: {folder}")
    if not os.path.exists(os.path.join(BC, "runs", model, "heads.pt")):
        sys.exit(f"runs/{model}/heads.pt is missing")
    for tag in tags:
        if not one_drive(model, track, tag):
            say("tests: stopped early")
            return
    say("tests: ALL DONE")


if __name__ == "__main__":
    main()
