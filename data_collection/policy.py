import math
import numpy as np
import random
from config import THROTTLE_MIN, THROTTLE_TOP

# working range from 180 to 900
# 900 - left , 180 - rinht

STEERING_NOISE = 0.3
r_steering = 0
random_steering = 0

# --- variety for the camera dataset ---
THROTTLE_HOLD = (1, 3)          # s, one throttle value is kept this long
# --- fast data (2026-11-08): throttle up to THROTTLE_TOP, short brake pulses, safety that looks farther when fast
BRAKE_SHARE = 0.1               # share of throttle picks that are a brake pulse (throttle 0)
BRAKE_PULSE = (0.1, 0.4)        # s: throttle 0 brakes hard (1.4 m/s -> 0 in 0.3 s), so only short pulses
DECEL = 4.0                     # m/s2, measured braking of throttle 0 (4.3 in the first 0.15 s)
RANDOM_THROTTLE = 0.12          # random steering: not faster than ~2.9 m/s
WARMUP_THROTTLE = 0.07         # warm-up drives to a random place on the track: not too fast
SAFETY_BRAKE_SPEED = 2.0        # m/s: faster than this, safety brakes (throttle 0) instead of only slowing down
MODE_HOLD = (2, 5)              # s, one driving mode is kept this long
RANDOM_SHARE = 0.6              # how often the "random" mode is chosen (else "gap")
RANDOM_HOLD = (0.2, 1.2)        # s, one random steering value is kept this long
SAFE_FRONT = 0.75                # m, wall closer than this in front (±20°) -> safety
SAFE_AROUND = 0.3               # m, wall closer than this anywhere in ±90° -> safety
SAFETY_TIME = 0.3             # s, safety keeps driving this long after the danger
SAFETY_THROTTLE = 0.04          # slow down while saving the car

last_time = 0
throttle_now, next_throttle = 0, 0
mode_now, next_mode = "gap", 0
random_now, next_random = 0, 0
safety_until = -1


def dangarous_angle(distance, half_width=0.25):
    distance = max(distance, 0.1)
    angle = math.degrees(math.atan(half_width / distance))
    return int(angle * 4)


def reset_policy():             # new trip: every timer starts again from 0
    global r_steering, last_time, next_throttle, next_mode, next_random, safety_until
    r_steering = 0
    last_time = 0
    next_throttle = next_mode = next_random = 0
    safety_until = -1

def add_steering_noise(steering, time):
    global r_steering, random_steering
    if time > r_steering :
        random_steering = random.uniform(-STEERING_NOISE , STEERING_NOISE)
        r_steering += random.uniform(0.3 , 2)


    steering = steering + random_steering
    return max(-1, min(1, steering))



def find_gap(ranges):   # ranges has type list
    front = ranges[180:901]  # range that i will work with from right to left
    max_count = 0
    index = 0
    front = [min(r,3) for r in front]
    original = front.copy()
    jumps = np.diff(front)
    corners = np.where(np.abs(jumps) > 0.5)[0]

    for corner in corners:

        near = min(original[corner],original[corner + 1])
        dan_angle = dangarous_angle(near)
        for k in range((corner - dan_angle), (corner + dan_angle)):      # Safty buble
            if 0 <= k < len(front):
                front[k] = 0



    winners = []
    for i in range(0, len(front) - 80 + 1, 20):     # Choose the best 20 degres angle to ride
        count = sum(front[i: i + 80])
        if count > max_count:
            max_count = count
            winners = [i]
        elif count == max_count:
            winners.append(i)

    groups = [[winners[0]]]
    for w in winners[1:]:
        if w - groups[-1][-1] == 20:
            groups[-1].append(w)
        else:
            groups.append([w])
    best = groups[0]
    for g in groups:
        if len(g) > len(best):
            best = g


    index = (best[0] + best[-1]) / 2
    index = index + 180
    center = index + 40
    angle = (center / 4) - 135
    steering_angle = max(-1, min(1, angle / 30))





    SIDE_LIMIT = 0.35                            # if wall is near from side just move forward
    right_side = min(ranges[90:300])
    left_side = min(ranges[780:990])
    if steering_angle < 0 and right_side < SIDE_LIMIT:
        steering_angle = 0
    if steering_angle > 0 and left_side < SIDE_LIMIT:
        steering_angle = 0


    return steering_angle


def pick_throttle(time):
    global throttle_now, next_throttle
    # independent of the situation (like the random steering): the model must learn what the throttle DOES
    if time >= next_throttle:
        if random.random() < BRAKE_SHARE:
            throttle_now = 0
            next_throttle = time + random.uniform(*BRAKE_PULSE)
        else:
            throttle_now = random.uniform(THROTTLE_MIN, THROTTLE_TOP)
            next_throttle = time + random.uniform(*THROTTLE_HOLD)
    return throttle_now


def pick_mode(time):
    global mode_now, next_mode
    if time >= next_mode:
        mode_now = "random" if random.random() < RANDOM_SHARE else "gap"
        next_mode = time + random.uniform(*MODE_HOLD)
    return mode_now


def random_steering_at(time):
    # does NOT look at the lidar: the brain must learn what steering DOES, not what find_gap would do
    global random_now, next_random
    if time >= next_random:
        random_now = random.uniform(-1, 1)
        next_random = time + random.uniform(*RANDOM_HOLD)
    return random_now


def wall_is_close(ranges, speed=0.0):
    # the faster the car, the farther ahead it must look: + braking distance speed^2 / (2 DECEL)
    front = min(ranges[460:621])                 # ±20°  (index = (angle + 135) * 4)
    around = min(ranges[180:901])                # ±90°
    return front < SAFE_FRONT + speed ** 2 / (2 * DECEL) or around < SAFE_AROUND + 0.05 * speed


def decide(ranges, time, warming_up, speed=0.0):
    """Return (throttle, steering, mode) for this moment. speed: m/s now (for the safety distance)."""
    global last_time, safety_until
    if time < last_time:                         # the clock went back to 0 = a new trip
        reset_policy()
    last_time = time

    throttle = pick_throttle(time)
    if warming_up:                               # drive normally to a random place on the track
        return min(throttle, WARMUP_THROTTLE), add_steering_noise(find_gap(ranges), time), "warmup"

    if wall_is_close(ranges, speed):
        safety_until = time + SAFETY_TIME
    if time < safety_until:
        save_throttle = 0 if speed > SAFETY_BRAKE_SPEED else min(throttle, SAFETY_THROTTLE)
        return save_throttle, find_gap(ranges), "safety"

    if pick_mode(time) == "random":
        # random steering at 4 m/s on the tight practice track only crashes (try run: 60 % crash trips): the coin
        # steers at moderate speeds, the fast driving comes from "gap" (find_gap + noise) with the full throttle
        return min(throttle, RANDOM_THROTTLE), random_steering_at(time), "random"
    return throttle, add_steering_noise(find_gap(ranges), time), "gap"
