# Recovery pushes for self-driving trips (B4 of docs/data_solve.md, 14 Nov 2026): every PUSH_EVERY seconds the
# steering is held at full lock to one side for PUSH_TIME seconds, then the planner drives again. The data then show
# the car off its usual line and how it gets back (E28 P3: self-driven trips saw a place from only ~2 of 8 headings;
# P6: few moments close to a wall). Used by wm2/drive.py explore mode when WM2_PUSH=1 (default off).
# NOT run yet: its first drive must be watched (the user's rule; a push at full lock can crash the car).

import random

PUSH_EVERY = (5.0, 15.0)        # s between the end of one push and the start of the next
PUSH_TIME = (0.3, 0.8)          # s of full lock


def new_push_state(now: float, rng: random.Random) -> dict:
    return {"next": now + rng.uniform(*PUSH_EVERY), "until": -1.0, "side": 0.0}


def push_steer(state: dict, now: float, rng: random.Random) -> tuple[dict, float | None]:
    # -> (new state, the steering to send (-1 or +1) while a push is on, else None). The state is never changed.
    if now < state["until"]:
        return state, state["side"]
    if now >= state["next"]:
        until = now + rng.uniform(*PUSH_TIME)
        side = rng.choice([-1.0, 1.0])
        return {"next": until + rng.uniform(*PUSH_EVERY), "until": until, "side": side}, side
    return state, None
