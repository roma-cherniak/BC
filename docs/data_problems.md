# Data problems: what is missing or one-sided in the training data

Found on 13 Nov 2026 by `autoresearch/data_report.py` (E28 in [research_log.md](research_log.md)).
Full numbers per folder: [data/data_report_2026-11-13.txt](data/data_report_2026-11-13.txt).
How each problem will be solved: [data_solve.md](data_solve.md).

## The data in one table
About **44 h** of trips in 12 folders. The training set `data/wm2_c2` uses all of them except `clean_practice`.

| Folder | Track | Min | Who drove | Crashes/min | Mean speed m/s | Used in wm2_c2 |
|---|---|---|---|---|---|---|
| clean_practice | practice | 791 | gap driver | 0.27 | 1.35 | **no** |
| random_practice | practice | 720 | random + gap + safety | 0.52 | 1.18 | yes |
| fast_practice | practice | 403 | random + gap + safety | 1.12 | 1.41 | yes |
| fast_compete | compete | 110 | random + gap + safety | 1.19 | 1.42 | yes |
| self_practice | practice | 243 | the planner (self) | 9.22 | 1.85 | yes |
| self_compete | compete | 166 | the planner | 7.51 | 1.30 | yes |
| self2_practice | practice | 145 | the planner | 1.15 | 1.92 | yes |
| self2_compete | compete | 27 | the planner | 7.34 | 1.65 | yes |
| self3_compete | compete | 39 | the planner | 3.40 | 1.38 | yes |
| self4_compete | compete | 10 | the planner + value head | 4.09 | 1.32 | yes |
| self5_compete | compete | 9 | the planner + value head | 3.45 | 1.57 | yes |
| self6_compete | compete | 10 | the planner + value head | 2.43 | 1.25 | yes |

## The problems, biggest first

### P1. No fast driving
- **What:** in every folder only **0 – 4 % of frames are faster than 2 m/s**; the throttle never goes above 0.12 – 0.20.
- **Why it matters:** week 2 (the autoresearch loop) is about making the car **faster**. At a higher speed the planner
  would trust the model's imagination in states it has never seen.
- **Evidence:** speed rows of every folder in the report (bin 2 – 3 m/s: 0 – 4 %, bin 3 – 5 m/s: 0 %).

### P2. Each track is driven in one direction only
- **What:** `practice` almost only **clockwise** (random 1373 cw / 206 ccw, self 781 / 22, self2 210 / 4);
  `compete` almost only **counter-clockwise** (self 1061 / 0, self3 121 / 1).
- **Why it matters:** every corner is learned from one side only. The mean steering is one-sided too
  (practice −0.10 … −0.18, compete +0.05 … +0.14). The AIL Cup tracks may go either way.
- **Note:** the mirror augmentation in `wm2/data.py` makes a mirror-image track, not the same corner from the other side.

### P3. Self-driven trips repeat the same line
- **What:** in the car's own trips each 0.5 m spot is seen from only **1.5 – 2.5 of 8 headings**, and up to
  **54 %** of spots from a single heading (self4). The random-steering data has 4 – 6 headings per spot.
- **Why it matters:** more self-driving rounds make the data **bigger, not wider**. The model never sees how to get back
  to the line after a mistake.

### P4. Too little `compete`
- **What:** in the training set ~1,510 min of `practice` against ~370 min of `compete` (about 4 : 1).
  The newest and best compete trips (self4 – self6) are only **29 min**.
- **Why it matters:** the car is crash-free on `practice` but not on `compete` (0.6 – 1.1 crashes/min, E21).

### P5. Many frames of a standing car
- **What:** **17 – 23 %** of the self-driven compete frames are slower than 0.5 m/s (the car stops at corners, E15).
- **Why it matters:** these frames teach almost nothing and take training time.

### P6. Few moments close to a wall
- **What:** the closest wall is under 0.2 m in only **0 – 2 %** of frames.
- **Why it matters:** these are the moments just before a crash, where the wall heads and the value head must be right.

### P7. Narrow corridors on `practice`, few wide ones
- **What:** on `practice` 53 – 62 % of frames have a corridor under 1.6 m wide; only ~15 % of all frames are 2.6 m or
  wider, and the AIL Cup oval is ~2.6 m wide.
- **Status:** mostly solved by the scale augmentation of E26 (`wm2_c3`); its drives in the sim are still not done.

### P8. `clean_practice` is not used
- **What:** 791 min (the largest folder) are not in any training set.
- **Why it may not matter:** slow (mean 1.35 m/s), only clockwise, mean steering −0.18, low throttle (≤ 0.09).

### P9. Only two track shapes
- **What:** every trip is on `practice` or `compete` (`explore` is not a track, decision of 11 Nov).
- **Why it matters:** the model learns these two shapes; the AIL Cup has three other ones (E26: "the shapes stay those
  of the two tracks").

## How the numbers were measured
- Corridor width = lidar beam at +90° + beam at −90° (each capped at 10 m). In corners this is larger than the real width.
- Direction = the sign of the car's turning around the trip's centre, only for trips longer than 100 frames.
- Coverage = 0.5 m cells the car visited and how many of 8 heading bins (45° each) were seen in each cell.
- Lidar numbers use every 5th frame.
