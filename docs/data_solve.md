# Data solutions: how each data problem is solved

The problems are in [data_problems.md](data_problems.md) (P1 – P9).
Every row here is a **proposal** until the user decides; the decision is then written in the "Decision" column, in
[research_log.md](research_log.md) as a D entry, and in [goals/communication.md](../goals/communication.md) → Decisions.
Rule: **no sim collection without the user's go** (decision of 12 Nov).

## Overview
| # | Solution | Solves | Needs | Cost | Status | Decision |
|---|---|---|---|---|---|---|
| A1 | Balance the training mix: ~50 / 50 practice / compete, more weight on new compete trips and on frames > 1.5 m/s | P4, P1 (a little) | training only | small code change in `wm2/data.py` | **training wm2_c4 (14 Nov)** | **yes (user, 14 Nov), D18** |
| A2 | Keep only 1 of 5 frames of a standing car (< 0.2 m/s) | P5 | training only | small | **training wm2_c4 (14 Nov)** | **yes (user, 14 Nov), D18** |
| A3 | Sensor noise: beam dropout, small distance noise, a few wrong beams | AIL Cup lidar differences | training only | small | proposed | |
| A4 | `clean_practice` at a low weight, or not at all | P8 | training only | none | proposed: leave it out | |
| B1 | Speed ladder: random steering with the safety driver at throttle 0.15 → 0.20 → 0.25 → 0.30, both tracks | **P1** | 1 night of sim | `data_collection` settings | **running night 13 → 14 Nov** (`night/ladder_night.py`, 0.15 – 0.30, compete + practice, 30 min per step) | **yes (user, 14 Nov)** |
| B2 | Reverse direction on both tracks (start facing the other way, or a U-turn at the start) | P2 | 1 night of sim | depends on the sim: question 13 | open question | |
| B3 | Start near the crash places, with steering noise | P4, P6 | ½ night of sim | the crash list of today's plan | planned 13 Nov (day file) | |
| B4 | Recovery pushes: while self-driving, steer to a wall for 0.3 – 0.8 s, then the planner takes back | **P3**, P6 | 1 night of sim | change of `explore` in `wm2/drive.py` | **code done 14 Nov** (`wm2/push.py`, `WM2_PUSH=1`, 4 tests); first drive must be watched | **yes (user, 14 Nov): code only tonight** |
| C1 | Synthetic tracks from `ail_cup/garage_sim.py` (random shapes and widths, random steering) for the encoder and the wall heads only | P9, P7 | no sim, CPU | new generator + mixing | proposed as a loop experiment (week 2) | |
| — | Scale + mirror augmentation (`WM2_SCALE`) | P7 | done | — | **done (E26)**, sim drive pending | |

## Recommended order
1. **Now (training only):** A1 + A2 in the next training.
2. **Next night (after the user's go):** B1 + B4 in one night (speed ladder, then self-driving with recovery pushes).
3. **When the sim answer is known:** B2.
4. **Week 2, as an experiment of the loop:** C1.

## Details

### A1. Balance the training mix (P4)
- **How:** a weight per trip when windows are drawn: per track to ~50 / 50, ×2 for trips of the last 3 days,
  ×2 for windows with speed > 1.5 m/s.
- **Check:** `wm2.plan_test` and the corridor test do not get worse; compete crashes/min in a sim drive.
- **Risk:** the practice result (crash-free) could get worse; check it in the same drive test.

### A2. Fewer frames of a standing car (P5)
- **How:** in `wm2/data.py`, drop 4 of 5 windows whose first frame is slower than 0.2 m/s.
- **Risk:** the start from standstill must stay in the data (it is kept: 1 of 5).

### A3. Sensor noise (AIL Cup)
- **How:** per window, 0 – 5 % of beams set to the max range, ±1 – 2 cm noise, rarely one beam much shorter.
- **Check:** the corridor test with noisy scans.

### A4. `clean_practice` (P8)
- **Proposal:** leave it out; it adds nothing that `random_practice` does not have, and makes P2 and P4 worse.

### B1. Speed ladder (P1)
- **How:** the random collector (`data_collection.collection_random`, mode random + gap + safety) with the throttle
  ceiling stepped up: 0.15, 0.20, 0.25, 0.30, ~1 h each, on `practice` and on `compete`.
- **Check:** the report shows ≥ 20 % of frames above 2 m/s; crashes are allowed (the safety driver takes over).
- **Risk:** at 0.30 the safety driver may be too late; stop the ladder at the step where crashes/min explode.

### B2. Reverse direction (P2)
- **Open:** can the sim start the car facing the other way? (question 13 in `goals/communication.md`)
- **If not:** a scripted U-turn at the start of each trip, in a wide part of the track.

### B3. Starts near the crash places (P4, P6)
- **How:** from the crash list of 13 Nov; lower gas at those places, or starts there, with steering noise.

### B4. Recovery pushes (P3, P6)
- **How:** in `explore` mode, every 5 – 15 s steer to one side for 0.3 – 0.8 s (random side and length), then the
  planner drives again. Record everything; these moments are marked so the value head learns from them.
- **Check:** headings per 0.5 m cell rise from ~2 toward 4+ in the report; closest wall < 0.2 m in more than 2 %.

### C1. Synthetic tracks (P9)
- **How:** `garage_sim.py` makes random closed tracks (shape, width 1.2 – 3 m, corner radius), a car with random steering
  records 109-beam scans; mixed into the training of the encoder and the wall heads.
- **Risk:** its physics are guesses (no slip, guessed lags), so it must not teach the speed dynamics.
