# Loop 1 of the autoresearch: the car's DRIVER tunes itself in the sim; the model (world model) stays the same.
#
# Search = coordinate search with repeats (simple, explainable; ~9 trials per hour):
#   - the LEADER is the best settings so far; its reference time = median of its last 3 scores
#   - one trial changes ONE number of the leader by one step up or down (numbers in turn)
#   - a trial better than the reference by > MARGIN s is driven AGAIN (one 5.5-min drive is good to ~+-0.3 s);
#     only if the two together are still better it becomes the new leader
#   - when both directions of a number failed, its step is halved; below its smallest step it is done
#   - every CONTROL_EVERY-th trial drives the leader again: the sim drifts (2026-11-09: the same model went from 0 to
#     12 crashes when the sim got slower), so the leader is always compared with ITS newest times, not old ones
# The judge (research/run_trial.py: gate "0 crashes, >= 10 laps", then the median lap) is fixed.
# State in research/loop1_state.json (stop any time, start again = it goes on). Every trial: research/results.tsv.
# Stop: after --hours, when every step is at its minimum, when the sim is closed, or when research/STOP exists.
#
# Run from the BC folder with the practice sim open (and no training: the 4 GB graphics card):
#   venv\Scripts\python -m research.loop1 --hours 4
#   venv\Scripts\python -m research.loop1 --dry          (no sim: fake trials, to test the search itself)

import argparse
import datetime
import json
import math
import os
import random
import statistics
import time

from research import run_trial as judge

HERE = os.path.dirname(os.path.abspath(__file__))
START = {"model": "wm2_self2", "gas": 0.12,                     # the best of the night of 2026-11-10 (CHAT_CONTEXT)
         "FRONT_BASE": 0.2, "SIDE_MARGIN": 0.35, "W_CENTER": 0.3, "DECEL": 4.0}
SPACE = {                                                       # number: (first step, smallest step, low, high)
    "FRONT_BASE": (0.05, 0.0125, 0.05, 0.6),                    # m kept in front even when slow
    "DECEL": (1.0, 0.25, 2.0, 10.0),                            # m/s2 in the front margin v^2 / (2 DECEL)
    "SIDE_MARGIN": (0.05, 0.0125, 0.15, 0.6),                   # m to the side walls
    "W_CENTER": (0.15, 0.04, 0.0, 1.5),                         # keep to the middle of the corridor
    "gas": (0.02, 0.005, 0.08, 0.2),                            # the highest throttle the grid may choose
}
MARGIN = 0.3            # s: a trial must beat the leader by more than this to be driven again
CONTROL_EVERY = 4


def cost_of(config):
    return {k: v for k, v in config.items() if k not in ("model", "gas")}


def key(config):
    return json.dumps({k: config[k] for k in sorted(config)})


def new_state():
    return {"leader": dict(START), "leader_scores": [], "steps": {k: v[0] for k, v in SPACE.items()},
            "failed": {k: [] for k in SPACE}, "tried": [], "turn": 0, "trials": 0, "history": []}


def reference(state):
    good = [s for s in state["leader_scores"][-3:] if math.isfinite(s)]
    return statistics.median(good) if good else math.inf


def propose(state):
    # the next number to try (in turn) and its direction; None when every step is at its minimum
    names = list(SPACE)
    for _ in range(len(names)):
        name = names[state["turn"] % len(names)]
        step, smallest, low, high = state["steps"][name], SPACE[name][1], SPACE[name][2], SPACE[name][3]
        if step >= smallest:
            for direction in (+1, -1):
                if direction in state["failed"][name]:
                    continue
                candidate = dict(state["leader"])
                candidate[name] = round(min(high, max(low, candidate[name] + direction * step)), 4)
                if candidate[name] == state["leader"][name] or key(candidate) in state["tried"]:
                    state["failed"][name].append(direction)             # at a bound or already driven
                    continue
                return candidate, name, direction
            state["steps"][name] = step / 2                             # both directions failed: smaller step
            state["failed"][name] = []                                  # (tried again, finer, on its next turn)
        state["turn"] += 1                                              # the next number
    return None


def trial(config, label, dry):
    tag = f"l1_{datetime.datetime.now():%m%d_%H%M%S}_{label}"
    if dry:
        return fake_trial(tag, config)
    return judge.run_trial(tag, config["model"], config["gas"], cost_of(config))


def fake_trial(tag, config):
    # a made-up sim for --dry: best near FRONT_BASE 0.17, DECEL 5, crashes when the margins get too small
    t = (14.2 + 40 * (config["FRONT_BASE"] - 0.17) ** 2 + 0.05 * (config["DECEL"] - 5) ** 2
         + 8 * (config["SIDE_MARGIN"] - 0.33) ** 2 + 0.5 * (config["W_CENTER"] - 0.25) ** 2
         - 6 * (config["gas"] - 0.12) + 200 * max(0, config["gas"] - 0.14) ** 2 + random.gauss(0, 0.25))
    crashes = int(config["FRONT_BASE"] < 0.12 or config["SIDE_MARGIN"] < 0.25 or random.random() < 0.05)
    why = "ok" if crashes == 0 else f"gate: {crashes} crashes"
    row = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "tag": tag, "model": config["model"],
           "gas": config["gas"], "cost": json.dumps(cost_of(config)), "valid": True, "why": why, "crashes": crashes,
           "laps": 20, "median": t, "best": t - 1, "gas0": 0.2, "mean_speed": 2.0, "pkt_s": 45, "plan_ms": 40,
           "score": t if crashes == 0 else math.inf}
    judge.record(row)
    return row


def log(state, text):
    line = f"{datetime.datetime.now():%H:%M:%S}  {text}"
    print(line, flush=True)
    state["history"].append(line)


def run(hours, dry):
    path = os.path.join(HERE, "loop1_state_dry.json" if dry else "loop1_state.json")
    if dry:
        judge.RESULTS = os.path.join(HERE, "results_dry.tsv")
        random.seed(0)
    state = json.load(open(path)) if os.path.exists(path) else new_state()
    deadline = time.time() + hours * 3600

    def save():
        with open(path, "w") as f:
            json.dump(state, f, indent=1)

    def drive(config, label):
        if not dry and not judge.sim_running():
            raise RuntimeError("the sim was closed")
        row = trial(config, label, dry)
        state["trials"] += 1
        state["tried"].append(key(config))
        log(state, judge.describe(row) if not dry else f"{row['tag']:34s} {row['why']:18s} median {row['median']:.2f}")
        return row

    try:
        while len([s for s in state["leader_scores"] if math.isfinite(s)]) < 2 and len(state["leader_scores"]) < 4:
            state["leader_scores"].append(drive(state["leader"], "base")["score"])     # the leader's own time first
            save()
        log(state, f"leader {json.dumps(state['leader'])}  reference {reference(state):.2f} s")
        while time.time() < deadline and not os.path.exists(os.path.join(HERE, "STOP")):
            if state["trials"] % CONTROL_EVERY == CONTROL_EVERY - 1:
                state["leader_scores"].append(drive(state["leader"], "control")["score"])
                log(state, f"control: reference now {reference(state):.2f} s")
                save()
                continue
            proposal = propose(state)
            if proposal is None:
                log(state, "every step is at its minimum: done")
                break
            candidate, name, direction = proposal
            label = f"{name}{'+' if direction > 0 else '-'}{candidate[name]}"
            ref = reference(state)
            first = drive(candidate, label)["score"]
            better = False
            if first < ref - MARGIN:
                second = drive(candidate, label + "_again")["score"]
                better = math.isfinite(second) and (first + second) / 2 < ref - MARGIN / 2
                if better:
                    state["leader"], state["leader_scores"] = candidate, [first, second]
                    state["failed"] = {k: [] for k in SPACE}
                    log(state, f"NEW LEADER {label}: {first:.2f}, {second:.2f} vs reference {ref:.2f}")
            if not better:
                state["failed"][name].append(direction)
            save()
    except RuntimeError as error:
        log(state, f"stopped: {error}")
    save()
    log(state, f"END after {state['trials']} trials. Leader {json.dumps(state['leader'])}, "
               f"reference {reference(state):.2f} s, steps {json.dumps(state['steps'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=4)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    run(0.02 if args.dry and args.hours == 4 else args.hours, args.dry)
