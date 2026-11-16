# Night plan 2026-11-11 (week 1, day 2): the first data from the COMPETE track. No training tonight.
#   1. test    wm2_self2 drives compete 2 x 5.5 min exactly like the practice tests of 2026-11-10 (gas 0.12, default
#              cost, no explore) -> runs/wm2_self2/drive_compete_self2_test_a/b.csv
#              = the answer to "how does it drive a track it never saw" (an older model hit the first wall there)
#   2. random  data_collection.collection_random compete, RANDOM_HOURS -> data/fast_compete
#   3. self    wm2_self2 drives and records itself (explore), SELF_HOURS -> data/self_compete
#              ONLY if both tests had at most MAX_TEST_CRASHES crashes per minute and the car really drove. A car that
#              only drives from the start line into the first wall records nothing useful, so then phase 2 gets
#              these hours too.
#   then the sim is closed.
# The sim reconnects by itself to every new driver, so "Connect" is clicked once.
#
# The user: open the COMPETE sim (BP\autodrive_simulator_compete_windows), start this script from the BC folder,
# click "Connect" in the game within 10 minutes:
#     venv\Scripts\python night\compete_night.py
# What it is doing: night/state_compete.txt. Logs: night/compete_*.log. To stop early: close the sim.
# Other hours:  set NIGHTC_RANDOM_HOURS=4  and  set NIGHTC_SELF_HOURS=2  before starting.

import datetime
import glob
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

BC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BC)                                   # started as a file: BC is not on the path by itself
from autoresearch.eval.drive_stats import WARMUP, count_crashes, read_log   # noqa: E402  (the judge's own rules)

PY = os.path.join(BC, "venv", "Scripts", "python.exe")
HERE = os.path.join(BC, "night")
SIM = "AutoDRIVE Simulator.exe"
TRACK = "compete"
MODEL = os.environ.get("NIGHTC_MODEL", "wm2_self2")                 # 12 Nov: wm2_c1 with NIGHTC_COST=...,W_CENTER=0
TEST_TAG = os.environ.get("NIGHTC_TEST_TAG", "self2_test")          # the tests: drive_<track>_<tag>_{a,b}.csv
SELF_NAME = os.environ.get("NIGHTC_SELF_DIR", f"data/self_{TRACK}")  # another driver or cost -> another folder
SELF_TAG = os.environ.get("NIGHTC_SELF_TAG", "self")                 # in the names of the self-driving logs
RANDOM_DIR = os.path.join(BC, "data", f"fast_{TRACK}")
SELF_DIR = os.path.join(BC, *SELF_NAME.split("/"))

RANDOM_HOURS = float(os.environ.get("NIGHTC_RANDOM_HOURS", "3"))
SELF_HOURS = float(os.environ.get("NIGHTC_SELF_HOURS", "3"))
MAX_TEST_CRASHES = float(os.environ.get("NIGHTC_MAX_TEST_CRASHES", "10"))    # per minute; round 1 on practice: ~8
MIN_TEST_SPEED = 0.5            # m/s: slower on average = the car stands in front of a wall instead of driving
NIGHT = f"{datetime.datetime.now():%m%d}"       # in the names of this night's self-driving logs
TEST_SECONDS = 330              # s of driving per test, like archive/research/run_trial.py and the practice tests
TEST_GAS = "0.12"
EXPLORE_GAS = os.environ.get("NIGHTC_EXPLORE_GAS", "0.09,0.12,0.15")       # slower than on practice: a new track
DRIVER_ENV = {"WM2_TRACK": TRACK, "WM2_COST": os.environ.get("NIGHTC_COST", "FRONT_BASE=0.25"), "WM2_SOFT": "1"}
CHECK_EVERY = 10                # s
CONNECT_TIMEOUT = 600           # s a new driver may wait for the first packet (the user clicks "Connect")
STALL_DRIVE = 120               # s without a new log row -> restart the driver
STALL_RANDOM = 300              # s without a new trip file (a random trip is 30 s + a wait of up to ~30 s)
MAX_RESTARTS = 20               # per phase
PORT_PAUSE = 10                 # s after a driver is killed, until port 4567 is free


@dataclass(frozen=True)
class Phase:
    name: str                               # also the log: night/compete_<name>.log
    command: Callable[[int], list[str]]     # start number (1, 2, ...) -> what comes after "python -u -m"
    env: dict[str, str]
    seconds: float                          # the longest this phase may take
    progress: Callable[[], float]           # a number that changes while the phase is alive
    stall: float                            # s without a change of `progress` -> restart
    done: Callable[[], bool] | None = None  # the phase is finished before its time is up


def say(text: str) -> None:
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {text}"
    with open(os.path.join(HERE, "state_compete.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sim_running() -> bool:
    out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {SIM}"], capture_output=True, text=True).stdout
    return SIM.lower() in out.lower()


def sim_folder() -> str:
    # where the running sim was started from: the practice and the compete build have the same .exe name
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-CimInstance Win32_Process -Filter \"Name='{SIM}'\" | Select-Object -First 1)"
                          ".ExecutablePath"], capture_output=True, text=True).stdout
    return out.strip()


def kill_tree(process: subprocess.Popen) -> None:
    # the venv python.exe is a launcher with a child process: kill both
    if process.poll() is None:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
        process.wait(timeout=30)


def start(phase: Phase, n: int) -> subprocess.Popen:
    with open(os.path.join(HERE, f"compete_{phase.name}.log"), "a", encoding="utf-8") as log:
        return subprocess.Popen([PY, "-u", "-m", *phase.command(n)], cwd=BC, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, **phase.env, "PYTHONUNBUFFERED": "1"})


def problem_of(process: subprocess.Popen, phase: Phase, moved: bool, quiet: float) -> str | None:
    # why the driver must be restarted, or None when it is fine
    if process.poll() is not None:
        return f"exit {process.returncode}"
    if moved and quiet > phase.stall:
        return f"nothing new for {phase.stall:.0f} s"
    if not moved and quiet > CONNECT_TIMEOUT:
        return f"the game did not connect in {CONNECT_TIMEOUT} s"
    return None


def run_phase(phase: Phase, after_start: Callable[[], None] | None = None) -> bool:
    # one phase with restarts -> False when the sim was closed (then the night ends)
    # after_start (14 Nov): called once when the first driver listens, e.g. open the sim and click "Connect"
    say(f"{phase.name}: start, at most {phase.seconds / 60:.0f} min")
    began, n, sim_open = time.time(), 1, True
    process = start(phase, n)
    if after_start is not None:
        after_start()
    last, last_change, moved = phase.progress(), time.time(), False
    while True:
        time.sleep(CHECK_EVERY)
        if not sim_running():
            say(f"{phase.name}: the simulator was closed -> the night ends here")
            sim_open = False
            break
        if phase.done is not None and phase.done():
            say(f"{phase.name}: finished")
            break
        if time.time() - began > phase.seconds:
            say(f"{phase.name}: its time is up")
            break
        now = phase.progress()
        if now != last:
            last, last_change, moved = now, time.time(), True
        problem = problem_of(process, phase, moved, time.time() - last_change)
        if problem is None:
            continue
        n += 1
        if n > MAX_RESTARTS:
            say(f"{phase.name}: {problem}, too many restarts -> this phase is given up")
            break
        say(f"{phase.name}: {problem} -> restart {n - 1}")
        kill_tree(process)
        time.sleep(PORT_PAUSE)
        process = start(phase, n)
        last, last_change, moved = phase.progress(), time.time(), False
    kill_tree(process)
    time.sleep(PORT_PAUSE)
    return sim_open


def size_of(pattern: str) -> float:
    # bytes of the drive logs that match (they grow while the car drives); the *_imagined.csv next to them are left out
    return float(sum(os.path.getsize(p) for p in glob.glob(pattern) if not p.endswith("_imagined.csv")))


def trips_in(folder: str) -> int:
    return len(glob.glob(os.path.join(folder, "drive_*.npz")))


def driven_seconds(path: str) -> float:
    # the "seconds" of the last row of a drive log; a half-written last number is only ever too small
    if not os.path.exists(path):
        return 0.0
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - 400))
        lines = f.read().decode(errors="ignore").splitlines()
    for line in reversed(lines):
        try:
            return float(line.split(",")[0])
        except ValueError:
            continue
    return 0.0


def test_log(letter: str) -> str:
    return os.path.join(BC, "runs", MODEL, f"drive_{TRACK}_{TEST_TAG}_{letter}.csv")


def test_phase(letter: str) -> Phase:
    path = test_log(letter)
    return Phase(name=f"test_{letter}", command=lambda n: ["wm2.drive", MODEL, TEST_GAS, f"{TEST_TAG}_{letter}"],
                 env=DRIVER_ENV, seconds=TEST_SECONDS + CONNECT_TIMEOUT + 300,
                 progress=lambda: size_of(path), stall=STALL_DRIVE, done=lambda: driven_seconds(path) >= TEST_SECONDS)


def random_phase(hours: float) -> Phase:
    return Phase(name="random", command=lambda n: ["data_collection.collection_random", TRACK], env={},
                 seconds=hours * 3600, progress=lambda: float(trips_in(RANDOM_DIR)), stall=STALL_RANDOM)


def self_phase(hours: float) -> Phase:
    logs = os.path.join(BC, "runs", MODEL, f"drive_{TRACK}_{SELF_TAG}_*.csv")
    env = {**DRIVER_ENV, "WM2_EXPLORE_GAS": EXPLORE_GAS, "WM2_SELF_DIR": SELF_NAME}
    top = EXPLORE_GAS.split(",")[-1]
    return Phase(name="self", command=lambda n: ["wm2.drive", MODEL, top, f"{SELF_TAG}_{NIGHT}_{n}", "explore"], env=env,
                 seconds=hours * 3600, progress=lambda: size_of(logs), stall=STALL_DRIVE)


def why_no_self_driving() -> str | None:
    # the two tests by the judge's rules -> why the car should NOT record itself tonight, or None when it may
    reasons = []
    for letter in "ab":
        try:
            log = read_log(test_log(letter))
        except (OSError, ValueError) as error:
            say(f"test_{letter}: no result ({error})")
            reasons.append(f"test_{letter} gave no result")
            continue
        judged = log["seconds"] >= WARMUP
        minutes = (log["seconds"][-1] - WARMUP) / 60
        per_min = count_crashes(log["seconds"], log["crashes"]) / minutes
        speed = float(log["speed"][judged].mean())
        say(f"test_{letter}: {minutes:.1f} min, {per_min:.1f} crashes per min, mean speed {speed:.2f} m/s")
        if per_min > MAX_TEST_CRASHES:
            reasons.append(f"test_{letter}: {per_min:.1f} crashes per min (the limit is {MAX_TEST_CRASHES:.0f})")
        elif speed < MIN_TEST_SPEED:
            reasons.append(f"test_{letter}: mean speed {speed:.2f} m/s, the car mostly stands")
    return "; ".join(reasons) or None


def close_sim() -> None:
    if sim_running():
        subprocess.run(["taskkill", "/IM", SIM], capture_output=True)          # ask it to close
        for _ in range(12):
            time.sleep(5)
            if not sim_running():
                break
        if sim_running():
            subprocess.run(["taskkill", "/F", "/IM", SIM], capture_output=True)
        say("sim: closed")


def wrong_sim() -> str | None:
    # why the night must not start, or None
    if not sim_running():
        return "the simulator is not running: open the COMPETE sim first"
    folder = sim_folder()
    say(f"sim: {folder or 'running (its folder is unknown)'}")
    if "practice" in folder.lower() and TRACK not in folder.lower() and not os.environ.get("NIGHTC_ANY_SIM"):
        return "this is the PRACTICE sim: its trips would be saved as compete (set NIGHTC_ANY_SIM=1 if I am wrong)"
    return None


def main() -> None:
    say(f"night: {TRACK}, tests 2 x {TEST_SECONDS} s, random {RANDOM_HOURS} h, self {SELF_HOURS} h, driver {DRIVER_ENV}"
        + ("" if MODEL == "wm2_self2" else f", model {MODEL}, self trips -> {SELF_NAME}"))
    reason = wrong_sim()
    if reason is not None:
        say(f"night: NOT STARTED, {reason}")
        sys.exit(reason)
    random_before, self_before = trips_in(RANDOM_DIR), trips_in(SELF_DIR)
    sim_open = True
    for letter in "ab":
        if driven_seconds(test_log(letter)) >= TEST_SECONDS:
            say(f"test_{letter}: its log is already complete -> not driven again (nothing is overwritten)")
        elif sim_open:
            sim_open = run_phase(test_phase(letter))
    if sim_open:
        random_hours, self_hours = RANDOM_HOURS, SELF_HOURS
        reason = why_no_self_driving()
        if reason is not None:
            say(f"self: skipped ({reason}) -> its {self_hours} h go to random")
            random_hours, self_hours = random_hours + self_hours, 0.0
        if random_hours > 0:
            sim_open = run_phase(random_phase(random_hours))
        if sim_open and self_hours > 0:
            sim_open = run_phase(self_phase(self_hours))
    close_sim()
    say(f"night: new trips: random {trips_in(RANDOM_DIR) - random_before}, self {trips_in(SELF_DIR) - self_before}")
    say("night: ALL DONE" if sim_open else "night: ended early (the sim was closed)")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:                          # never die silently
        say(f"supervisor crashed: {error!r}")
        sys.exit(1)
