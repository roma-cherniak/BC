# Night 13 -> 14 Nov 2026 (the user's go, in the chat): test drives of wm2_c4 + wm2_c3, then B1, the speed ladder.
# Per track (compete first, then practice; the sim build is switched once by night/open_sim.ps1):
#   1. tests   wm2_c4 and wm2_c3, 2 x 5.5 min each, the best setting of E20 / E21 (gas 0.12, own value head)
#              -> runs/<model>/drive_<track>_c4night_{a,b}.csv
#   2. ladder  data_collection.collection_random (the random / gap / safety driver of the fast data) with the random
#              throttle ceiling 0.15 -> 0.20 -> 0.25 -> 0.30, STEP_MINUTES each -> data/speed_<track>_t<ceiling>/
#              A step with more than MAX_CRASHES_PER_MIN crashes per minute ends the ladder on that track
#              (the safety driver is too late at that speed: higher steps would only crash).
# No training, no change of driver settings. The sim is closed at the end.
#   venv\Scripts\python night\ladder_night.py        (from BC; the sim may be closed: this script opens it)
# Progress: night/state_compete.txt. Logs: night/compete_<phase>.log. To stop early: close the sim.

import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compete_night as cn                                                          # noqa: E402
from autoresearch.eval.drive_stats import WARMUP, count_crashes, read_log           # noqa: E402

MODELS = ["wm2_c4", "wm2_c3"]
TEST_TAG = "c4night"
TEST_SECONDS = 330
DRIVER_COST = "FRONT_BASE=0.25,W_CENTER=1.0,CENTER_CLAMP=0.3,SIDE_MARGIN=0.45,W_VALUE=0.3"     # E20 / E21 best
CEILINGS = [0.15, 0.20, 0.25, 0.30]
STEP_MINUTES = float(os.environ.get("LADDER_STEP_MINUTES", "30"))
MAX_CRASHES_PER_MIN = 4.0
TRACKS = ["compete", "practice"]


def open_sim(track: str):
    def action() -> None:
        cn.say(f"sim: opening the {track} build and clicking Connect")
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        os.path.join(HERE, "open_sim.ps1"), track], capture_output=True, timeout=300)
    return action


def test_log(model: str, track: str, letter: str) -> str:
    return os.path.join(cn.BC, "runs", model, f"drive_{track}_{TEST_TAG}_{letter}.csv")


def test_phase(model: str, track: str, letter: str) -> cn.Phase:
    path = test_log(model, track, letter)
    env = {"WM2_TRACK": track, "WM2_COST": DRIVER_COST, "WM2_SOFT": "1"}
    return cn.Phase(name=f"test_{model}_{track}_{letter}",
                    command=lambda n: ["wm2.drive", model, "0.12", f"{TEST_TAG}_{letter}"], env=env,
                    seconds=TEST_SECONDS + cn.CONNECT_TIMEOUT + 300, progress=lambda: cn.size_of(path),
                    stall=cn.STALL_DRIVE, done=lambda: cn.driven_seconds(path) >= TEST_SECONDS)


def ladder_dir(track: str, ceiling: float) -> str:
    return f"data/speed_{track}_t{round(ceiling * 100):03d}"


def ladder_phase(track: str, ceiling: float) -> cn.Phase:
    folder = os.path.join(cn.BC, *ladder_dir(track, ceiling).split("/"))
    env = {"COLLECT_THROTTLE_TOP": f"{ceiling}", "COLLECT_OUT_DIR": ladder_dir(track, ceiling)}
    return cn.Phase(name=f"ladder_{track}_t{round(ceiling * 100):03d}",
                    command=lambda n: ["data_collection.collection_random", track], env=env,
                    seconds=STEP_MINUTES * 60, progress=lambda: float(cn.trips_in(folder)), stall=cn.STALL_RANDOM)


def test_summary(model: str, track: str, letter: str) -> None:
    try:
        log = read_log(test_log(model, track, letter))
    except (OSError, ValueError) as error:
        cn.say(f"test {model} {track} {letter}: no result ({error})")
        return
    judged = log["seconds"] >= WARMUP
    minutes = max((log["seconds"][-1] - WARMUP) / 60, 1e-9)
    cn.say(f"test {model} {track} {letter}: {minutes:.1f} min, "
           f"{count_crashes(log['seconds'], log['crashes']) / minutes:.2f} crashes/min, "
           f"mean speed {float(log['speed'][judged].mean()):.2f} m/s")


def ladder_summary(track: str, ceiling: float) -> float:
    # -> crashes per minute of this step; also writes trips, minutes and speed shares
    files = glob.glob(os.path.join(cn.BC, ladder_dir(track, ceiling), "drive_*.npz"))
    seconds, crashes, speeds = 0.0, 0, []
    for f in files:
        with np.load(f) as z:
            seconds += float(z["time"][-1] - z["time"][0])
            crashes += str(z["reason"]) == "crash"
            speeds.append(z["speed"])
    if not files:
        cn.say(f"ladder {track} {ceiling}: no trips")
        return float("inf")
    v = np.concatenate(speeds)
    per_min = crashes / max(seconds / 60, 1e-9)
    cn.say(f"ladder {track} {ceiling}: {len(files)} trips, {seconds / 60:.0f} min, {per_min:.2f} crashes/min, "
           f"speed mean {v.mean():.2f} max {v.max():.2f} m/s, > 2 m/s {100 * (v > 2).mean():.0f} %, "
           f"> 3 m/s {100 * (v > 3).mean():.0f} %")
    return per_min


def night_on(track: str, first: bool) -> bool:
    # -> False when the sim was closed by someone (the night ends)
    if not first:
        cn.close_sim()
    opener = open_sim(track)
    for model in MODELS:
        for letter in "ab":
            if cn.driven_seconds(test_log(model, track, letter)) >= TEST_SECONDS:
                cn.say(f"test {model} {track} {letter}: already complete -> not driven again")
                continue
            if not cn.run_phase(test_phase(model, track, letter), after_start=opener):
                return False
            opener = None
            test_summary(model, track, letter)
    for ceiling in CEILINGS:
        if not cn.run_phase(ladder_phase(track, ceiling), after_start=opener):
            return False
        opener = None
        if ladder_summary(track, ceiling) > MAX_CRASHES_PER_MIN:
            cn.say(f"ladder {track}: more than {MAX_CRASHES_PER_MIN} crashes/min at {ceiling} -> no higher step")
            break
    return True


def main() -> None:
    cn.say(f"ladder night: tests {MODELS} 2 x {TEST_SECONDS} s per track, ladder {CEILINGS} x {STEP_MINUTES:.0f} min, "
           f"tracks {TRACKS}")
    if cn.sim_running():
        cn.close_sim()
    for i, track in enumerate(TRACKS):
        if not night_on(track, first=(i == 0)):
            cn.say("ladder night: ended early (the sim was closed)")
            return
    cn.close_sim()
    cn.say("ladder night: ALL DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:                          # never die silently
        cn.say(f"ladder night: supervisor crashed: {error!r}")
        cn.close_sim()
        sys.exit(1)
