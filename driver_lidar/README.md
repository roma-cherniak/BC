# Lidar JEPA driver (AI generated)

`drive_jepa.py` drives the car in the AutoDRIVE simulator with the JEPA world model trained on lidar.
It does not copy a driver. Each time the game sends a message, it **imagines** what would happen with 9 different
steering options and picks the one that keeps the car farthest from the walls.

## The loop (about 60 times per second)

```
game ──► lidar 1081 beams ──► clean (271 beams, 0..1) ──► ENCODER ──► z (32 numbers)
                                                                  │
            ┌─────────────────────────────────────────────────────┘
            ▼   9 copies of z, one per steering option -1 ... +1
        PREDICTOR  x 10 steps (0.5 s)  ──►  PROBE after every step  ──►  walls on the whole imagined path
                                                                                │
game ◄── throttle 0.05 + the steering whose closest wall stayed farthest ◄─────┘
```

## Files it needs

| File | What it is | Made by |
|---|---|---|
| `driver_lidar/drive_jepa.py` | the driver (this folder) | |
| `jepa/jepa_model.py` | the `Encoder` and `Predictor` blueprints | |
| `jepa_encoder.pt` | encoder weights ("the eye") | `jepa/train_jepa.py` |
| `jepa_predoctor.pt` | predictor weights ("the imagination") | `jepa/train_jepa.py` |
| `zond.pt` | probe weights ("reads walls from a description") | `jepa/evaluate_jepa.py` |

The three `.pt` files are a set: if the encoder is retrained, the probe must be retrained too,
because the 32 numbers of a description mean something else after retraining.

## Step by step

**1. Read and clean the lidar** (`read_laser`, `clean_lidar`)
The game sends 1081 distances (gzip + base64 text, beam 0 = -135° on the right, beam 1080 = +135° on the left).
They are cleaned **exactly like in `data_prep/prepare_data.py`**: every 4th beam (1081 → 271, one beam per degree,
beam `b` looks at `b - 135` degrees), `inf` and everything beyond 10 m → 10 m, then divided by 10 → numbers 0..1.
Any difference here would give the brain data it never saw during training.

**2. Describe** (`encoder`)
271 numbers → a description `z` of 32 numbers. Encoder: 271 → 256 → ReLU → 32.

**3. Imagine 9 options at once** (`plan`)
`z` is copied 9 times, one row per steering option `[-1, -0.75, ..., +0.75, +1]` (+ = left, full = 30°).
Every row gets the same throttle (`0.05 / 0.09`, the same scaling as `clean_action`) and the same current speed
(`m/s / 3`). Then the predictor (35 → 256 → 256 → 32) steps all 9 rows together 10 times = 0.5 s.
The steering stays the same during the whole imagined 0.5 s; only the first step of the winner is really driven,
and 0.017 s later everything is planned again with a new lidar scan.

**4. Read the walls** (`zond`)
After every imagined step the probe (32 → 256 → ReLU → 7) turns each description into 7 facts:
front, right, left, steering, speed, front_right (-45°), front_left (+45°). Distances are cut to 0..1
(`clamp`), because the imagination can drift to impossible values like -0.4 m.

**5. Score and choose** (`score`)
For each option and each step: the closest wall, where a diagonal counts `/1.4` (in a corridor it is naturally
about 1.4x farther than the side) and the front counts `/2`. For each option the **worst** step of its whole path is
kept (a car can hit a wall in the middle of a plan, not only at the end). The option with the best worst step wins.
A small penalty `SMOOTH x |option - previous steering|` breaks near-ties in favour of not changing the steering,
so the car zig-zags less.

**6. Send, and reset after a crash** (`bridge`)
Throttle `THROTTLE` and the chosen steering go back to the game. If `V1 Collisions > 0`, the driver sends
`Reset: True` and ignores the counter for `RESET_PAUSE` messages, because the game needs a moment to clear it.

## Where the models come from

1. `data/random_practice`: 1,646 trips, 12 h. Steering came in segments of 0.3–1 s: **random**
   (a coin picked a steering from -1..+1, nothing to do with the lidar), **gap** (`find_gap` + noise) or **safety**
   (`find_gap` took over near a wall). Throttle changed every few seconds.
2. `data_prep/prepare_data.py` → `data/*_pairs.npz` (now, action, speed, next), split by trip.
3. `jepa/train_jepa.py`: 50 epochs, loss = prediction MSE + 0.09 × SIGReg. Last epoch: spread 1.00
   (no collapse), val prediction 0.026 vs copy 0.046.
4. `jepa/evaluate_jepa.py`: trains the probe on 100,000 train moments and saves `zond.pt`.

### Why the data had to be random

The first dataset was driven only by `find_gap` + small noise. There the steering is chosen *by the lidar*
(near the right wall → steer left), so the model learned "steering left means the right wall is close" instead of
"steering left moves the right wall away". With that model, imagined right distance **fell** when steering left,
which is physically wrong, and a planner chose like a coin (50 % right side).
With the random segments the same test is physical: steering from -1 to +1 moves the imagined right wall from
0.81 m to 1.46 m and the left wall from 1.31 m to 0.91 m (0.5 s ahead, average of 500 test moments), and the
difference between options (signal) is larger than the probe error (noise).

## How the settings were chosen (offline, without the game)

`jepa/planner_test.py` uses the **random** test moments as a natural experiment: a coin steered there, so we can
check what really happened next. "Danger" = the safety driver had to take over, or the car crashed, within 0.5 s.
If the planner's choice is good, moments where the coin happened to steer like the planner (`|coin - planner| < 0.5`)
must be less dangerous than the same planner choices shuffled between moments (control).

| Score | Imagination | Danger, coin ≈ planner | Control |
|---|---|---|---|
| `front + right + left` (first version) | 1 s | 45.6 % | 49.6 % |
| closest of 5 walls | 0.5 s | 41.2 % | 51.4 % |
| closest of right, left, diagonals /1.4, front /2 (**used**) | 0.5 s | **35.2 %** | 50.1 % |

Sanity check on test moments: with the right wall closer than 0.35 m the planner steers left 38 %, right 17 %;
with the left wall that close it steers right 42 %, left 10 %; on an open road it goes straight 69 %.
One decision takes about 3 ms on the CPU.

## Run it

Start the game first:

```bash
venv\Scripts\python -m driver_lidar.drive_jepa
```

and click "Disconnected" in the game menu. 

## What it prints and saves

Once per second (`PRINT_EVERY` messages):
```
  42 s  steer +0.25  speed 1.21 m/s  walls F 3.10 R 0.62 L 0.95 m  crashes 1 (1.4/min)  plan 2.8 ms
```
`walls` are the **real** distances from the current lidar, not the imagined ones.

`drive_log.csv` (in `BC`, one row per message): `seconds, steering, speed, real_front, real_right, real_left,
crashes`. Rows just before a jump of `crashes` show where and how the car hit a wall.

## Settings

| Constant | Value | Meaning |
|---|---|---|
| `DEVICE` | `cpu` | the game uses the graphics card |
| `STEERING` | 9 values -1 .. +1 | the options that are imagined |
| `STEPS` | 10 | imagined steps of 0.05 s (0.5 s) |
| `THROTTLE` | 0.05 | constant; the data had 0.03 .. 0.09, try 0.04 if it crashes often |
| `SMOOTH` | 0.01 | penalty per unit of steering change (0.01 = 10 cm of wall distance), 0 = off |
| `RESET_PAUSE` | 30 | messages to ignore the collision counter after a reset |

## Known limits

- Even the best setting is far from perfect offline: 35 % danger, and near the right wall 17 % of the choices
  still steer towards it.
- Near walls the training data is weakest: 38 % of the rows are "safety" rows, where `find_gap` drove,
  so close to walls the steering again partly depends on the lidar.
- Only the practice track, only one constant throttle while driving, and the planner keeps the steering constant
  for the whole imagined 0.5 s.
- The probe is an MLP here: good as the planner's eyes, but for judging the quality of the JEPA descriptions
  a linear probe is the fair test.
