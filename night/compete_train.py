# After the compete night (night/compete_night.py): train wm2_c1, the first model that saw BOTH tracks.
# The user allowed it on 2026-11-11 ("1 - yes you can", leaving for 16 hours).
#   waits until night/state_compete.txt says the night is over and the sim is closed (4 GB card: never both), then
#   split    data/fast_compete -> data/fastc_*.txt, data/self_compete -> data/selfc_*.txt (80 / 10 / 10 by trip)
#   prepare  wm2.prepare c1 -> data/wm2_c1 (everything wm2_self2 learned on + the compete trips)
#   train    wm2_self2 -> wm2_c1, EPOCHS epochs at lr 1e-4 (like self1 -> self2), then both kinds of heads
# Nothing is overwritten: it stops if runs/wm2_c1 exists. It does not train on fewer than MIN_TRIPS compete trips.
# Writes to night/state_compete.txt too; logs: night/c1_*.log.
# Start (from the BC folder, any time during or after the night):  venv\Scripts\python night\compete_train.py

import datetime
import glob
import os
import subprocess
import sys
import time

BC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(BC, "venv", "Scripts", "python.exe")
HERE = os.path.join(BC, "night")
STATE = os.path.join(HERE, "state_compete.txt")
SIM = "AutoDRIVE Simulator.exe"
TEACHER = os.environ.get("NIGHTC_TEACHER", "wm2_self2")         # 12 Nov, second round: wm2_c1 -> wm2_c2
STUDENT = os.environ.get("NIGHTC_STUDENT", "wm2_c1")            # its data set: wm2.prepare <the name without "wm2_">
MIN_NEW = int(os.environ.get("NIGHTC_MIN_NEW", "0"))            # trips the newest folder must hold, or no training
EPOCHS = os.environ.get("NIGHTC_EPOCHS", "3")
MIN_TRIPS = 50                  # random compete trips (~1 minute each): fewer is not worth a training
MAX_WAIT = 12 * 3600            # s to wait for the night to end
OVER = ("night: ALL DONE", "night: ended early", "supervisor crashed", "night: NOT STARTED")
FOLDERS = {"fastc": "data/fast_compete", "selfc": "data/self_compete"}
for item in filter(None, os.environ.get("NIGHTC_EXTRA", "").split(",")):       # selfc2=data/self2_compete,...
    FOLDERS[item.split("=")[0]] = item.split("=")[1]


def say(text: str) -> None:
    with open(STATE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {text}\n")


def running(image: str, words: tuple[str, ...] = ()) -> bool:
    # is a program with this image name running (and, for python, with one of these words in its command line)?
    if not words:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}"], capture_output=True, text=True).stdout
        return image.lower() in out.lower()
    test = " -or ".join(f"$_.CommandLine -like '*{w}*'" for w in words)
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"@(Get-CimInstance Win32_Process -Filter \"Name='{image}'\" | Where-Object {{ {test} }}).Count"],
                         capture_output=True, text=True).stdout
    return int(out.strip() or 0) > 0


def night_is_over() -> bool:
    # the last "night:" start line and what came after it
    with open(STATE, encoding="utf-8") as f:
        lines = f.read().splitlines()
    starts = [i for i, line in enumerate(lines) if "night: compete, tests" in line]
    return bool(starts) and any(word in line for line in lines[starts[-1]:] for word in OVER)


def trips(name: str) -> int:
    return len(glob.glob(os.path.join(BC, FOLDERS[name], "drive_*.npz")))


def steps() -> list[tuple[str, list[str]]]:
    todo = []
    for name, folder in FOLDERS.items():
        if os.path.exists(os.path.join(BC, "data", f"{name}_train.txt")):
            say(f"train: data/{name}_*.txt exist -> not split again")
        elif trips(name) >= 10:
            todo.append((f"split {name}", [PY, "-u", "-m", "wm2.prepare", "split", folder, name]))
        else:
            say(f"train: {folder} has {trips(name)} trips -> left out")
    return todo + [
        ("prepare", [PY, "-u", "-m", "wm2.prepare", STUDENT.removeprefix("wm2_")]),
        ("train", [PY, "-u", "-m", "wm2.train", STUDENT, EPOCHS, f"data/{STUDENT}", TEACHER, "1e-4"]),
        ("heads sectors", [PY, "-u", "-m", "wm2.heads", STUDENT, "sectors"]),
        ("heads walls5", [PY, "-u", "-m", "wm2.heads", STUDENT, "walls5"]),
    ]


def why_not() -> str | None:
    if os.path.exists(os.path.join(HERE, "NO_TRAIN")):
        return "night/NO_TRAIN exists (the user said so)"
    if os.path.exists(os.path.join(BC, "runs", STUDENT)):
        return f"runs/{STUDENT} already exists (nothing is overwritten)"
    if running(SIM):
        return "the simulator is still open (4 GB card: never the sim and a training together)"
    if running("python.exe", ("wm2.drive", "collection_random", "wm2.train")):
        return "a driver, a collector or another training is still running"
    newest = list(FOLDERS)[-1]
    if trips(newest) < MIN_NEW:
        return f"{FOLDERS[newest]} has only {trips(newest)} trips (too little new driving for a training)"
    if trips("fastc") < MIN_TRIPS:
        return f"only {trips('fastc')} random compete trips (at least {MIN_TRIPS} are needed)"
    return None


def main() -> None:
    say(f"train: waiting for the night to end (then {TEACHER} -> {STUDENT}, {EPOCHS} epochs)")
    began = time.time()
    while not night_is_over():
        if time.time() - began > MAX_WAIT:
            say("train: the night did not end in 12 h -> no training")
            return
        time.sleep(60)
    time.sleep(60)                                      # the night closes the sim; the card frees its memory
    reason = why_not()
    if reason is not None:
        say(f"train: NOT STARTED, {reason}")
        return
    for name, command in steps():
        say(f"train: {name} - start")
        with open(os.path.join(HERE, f"{STUDENT.removeprefix('wm2_')}_{name.split()[0]}.log"), "a",
                  encoding="utf-8") as log:
            code = subprocess.run(command, cwd=BC, stdout=log, stderr=subprocess.STDOUT).returncode
        say(f"train: {name} - finished, exit code {code}")
        if code != 0:
            say("train: stopped here because of the error")
            return
    say("train: ALL DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:                          # never die silently
        say(f"train supervisor crashed: {error!r}")
        sys.exit(1)
