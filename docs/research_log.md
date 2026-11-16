# Research log: JEPA world models for self-driving (AutoDRIVE + AIL Cup)

The starting point of the project documentation. **Every decision and every experiment is written here**, with its date,
its reason and its evidence (numbers + the file they come from), so the final write-up can be built from this file.

## How to add an entry
- **Decision** (`D<n>`): what was chosen, why, what else was considered, the evidence. A decision is never deleted:
  if it is changed, add a new one and write "replaced by D<m>" in the old one.
- **Experiment** (`E<n>`): the question, the setup, the result with numbers, the conclusion, where the files are.
  Failed experiments are written too: they explain the decisions.
- Newest entries at the **bottom** of each list; numbers keep growing (D15, D16, ... / E15, E16, ...).
- Where a number comes from: a `runs/<run>/` folder, a `night/*.log`, or a report. Never a number from memory.

---

## The project in one paragraph
A small car in the AutoDRIVE simulator learns a **world model** from its lidar: a JEPA encoder turns each scan into a short
description `z`, a predictor imagines how `z` changes for a given action, and a planner tries many action sequences in
that imagination and drives the safest / fastest one. The same model drives in the **AIL Cup** (pure-Python controller).
Goal of November 2026: an **autoresearch loop** where AI agents (Claude, Gemini, ...) improve the model on their own,
compared on one graph ([goals/month.md](../goals/month.md)).

## Timeline
| Dates | Stage | What happened |
|---|---|---|
| 06 – 13 Oct 2026 | data | sim connection, `find_gap` driver, noise, `.npz` recording, train/val/test split by trip |
| 14 – 17 Oct | A: MLP | MLP that predicts the next lidar scan (E1) |
| 18 – 28 Oct | B: lidar JEPA | JEPA + SIGReg (E2–E4), random-steering data (E5), planner score (E6), first JEPA driver |
| 30 Oct – 03 Nov | structure | `core/`, `checks/`, one planner for all, image JEPA try (E9) |
| 07 Nov | B | 109 beams like the AIL Cup lidar (E8) |
| 08 – 10 Nov | C: wm2 | transformer world model + CEM, self-driving data rounds, planner on the GPU (E10–E13) |
| 11 Nov | plan | new goal: autoresearch loop (D14), architecture kept (D13) |

---

## Decisions

### D1 — Predict in a learned description (JEPA), not the raw lidar (Oct 2026)
- **Why:** the MLP that predicts the raw next scan is good for one step but its imagination falls apart after ~1.1 s (E1).
- **Considered:** keep the MLP and make it bigger.
- **Evidence:** E1, E3, E4.

### D2 — SIGReg is required (19 Oct 2026)
- **Why:** without it the JEPA collapses in 10 epochs (every `z` the same) (E2).
- **Setting:** weight 0.09 in stage B, 0.1 in wm2 (LeWM's λ).

### D3 — Training data with random steering segments, not only `find_gap` (Oct 2026)
- **Why:** when only `find_gap` drives, steering is *caused by* the lidar, so the model learns "steer left means the
  right wall is close" instead of "steering left moves the right wall away". Random segments make the data causal.
- **Evidence:** E5. Data: `data/random_practice` (1,646 trips, 12 h).

### D4 — Planner score: the closest wall over the whole imagined path (Oct 2026)
- **What:** for each option, the worst moment of its path counts; diagonals count `/1.4`, the front `/2`.
- **Evidence:** E6 (35.2 % danger vs 50.1 % control). Replaced in wm2 by the CEM cost (D10).

### D5 — Multi-step training (K = 10) instead of one step (Oct 2026)
- **Why:** the planner imagines many steps, so the model must be trained on many steps.
- **Evidence:** E7 (1 s imagination error vs copy: 0.68 one-step vs 0.23 with K = 10).

### D6 — Lidar is the main input, not the camera (Nov 2026)
- **Why:** the AIL Cup car has a lidar, and the image JEPA plans much worse (E9).

### D7 — 109 beams, like the AIL Cup lidar (07 Nov 2026)
- **Why:** one model for the sim and the AIL Cup without re-sampling.
- **Evidence:** E8 (planner score -12.9, about the same as 271 beams).

### D8 — Probe for the planner: MLP; probe for judging `z`: linear (Oct 2026)
- **Why:** the MLP probe reads the walls well enough for planning; a linear probe is the fair test of how good a description is.

### D9 — Planning is offline-tested with a natural experiment (Oct 2026)
- **What:** in random-steering test moments, are moments where the coin steered like the planner less dangerous than a
  shuffled control? Reported as "agree vs control" points (more negative = better). `find_gap` ≈ -23.
- **Why:** a cheap planner check without the sim. Used for every model since.

### D10 — wm2: a LeWM-style world model with history, CEM over steering + throttle, cost heads (08 Nov 2026)
- **What:** conv encoder (109 beams + speed + wheel angle → 64), transformer predictor with 4 frames of history and
  AdaLN-zero action input, multi-step loss + speed loss + SIGReg; CEM planner; heads read walls / danger / progress from `z`.
- **Why:** stage B could only steer at a fixed throttle; the AIL Cup needs speed control too.
- **Evidence:** E10. Code: `wm2/`.

### D11 — The car collects its own data (self-driving rounds) (09 Nov 2026)
- **Why:** random data are slow; at higher speed the model only sees what it needs if it drives itself.
- **Evidence:** E11, E13.

### D12 — The sim planner runs on the GPU as a CUDA graph (09 Nov 2026)
- **Why:** 320 ms per plan is too slow at 60 packets/s; on the GPU it is 41 ms.
- **Evidence:** E12.

### D13 — Keep the wm2 architecture; do not rebuild it like LeWM (11 Nov 2026)
- **Why:** wm2 already has what makes LeWM work (SIGReg, end-to-end training, AdaLN action input, CEM). The differences
  are adaptations to the lidar, the 4 GB GPU, the AIL Cup time limit and racing without a goal image.
- **Compared with LeWM** (from memory, not yet checked against the LeWM code in `BP/autodrive-lewm/`):

| Part | LeWM | wm2 | Verdict |
|---|---|---|---|
| Encoder | ViT on camera **pixels** | 1D conv on **109 lidar beams** + speed / wheel angle | ✅ the AIL Cup gives lidar; a ViT would be far too slow for the pure-Python controller |
| Projector after the encoder | Linear + BatchNorm | Linear + BatchNorm (`wm2/model.py`) | ✅ same |
| Predictor | transformer, actions via **AdaLN-zero** | transformer, AdaLN-zero | ✅ same idea |
| Anti-collapse | SIGReg, λ = 0.1 | SIGReg, 0.1 (`wm2/train.py`) | ✅ same |
| History | several past frames | 4 frames (0.2 s) | ✅ |
| Training target | the next embedding | **K = 10 imagined steps** + speed loss | ✅ trains the long imagination the planner uses |
| Planning | CEM towards a **goal image** embedding | CEM with **cost heads** (walls, danger, progress) | ✅ racing has no goal image |
| Size | ~15M parameters | much smaller | ✅ must fit 4 GB and the AIL Cup time per call |

- **Rule:** architecture changes are tried by the autoresearch agents in week 2, measured by the frozen judge,
  not by hand in week 1. If the AIL Cup is too slow (Thu 12), the model must get **smaller**.

### D14 — New goal: an autoresearch loop for JEPA driving models (11 Nov 2026)
- **What:** step 1 (Tue 10 → Tue 17): a solid baseline car on `practice` + `compete` and in the AIL Cup, plus a frozen
  judge and the baseline score ± noise. Step 2 (Tue 17 → Tue 24): the loop with several AI agents, the final graph.
- **Metric:** the sim drive inside the loop (automatic), the AIL Cup for each agent's best (manual upload).
- **Code:** new code in `autoresearch/` ([README](../autoresearch/README.md)). Plans in `goals/`.
- **Why "solid" and not "fastest" in step 1:** the loop has to have room to improve, and speed by hand would take weeks.

### D15 — The judge reads a drive log with the rules of `research/run_trial.py` (11 Nov 2026)
- **What:** `autoresearch/eval/drive_stats.py` turns one drive log into numbers (crashes, crashes/min, laps, lap median / best,
  mean speed, gas ceiling, packets/s, plan time). Three rules: a crash in the first 5 s is not counted; laps are counted
  where the sim's lap counter changes; the first lap and "laps" over 40 s are dropped.
- **Why:** the last row of a log is misleading. In the 10 test logs of 2026-11-10
  (`runs/wm2_self{1,2}/drive_practice_self{1,2}_{test,front02,front015}_{a,b}.csv`) 9 start with the crash counter already
  at 1, and the lap counter goes back to 0 on every reset (`..._self2_front015_a.csv`: 21 → 0 at 294.7 s).
- **Considered:** calling `research.run_trial.judge` directly. Not done: that file also starts the driver and holds the
  gates (0 crashes, ≥ 10 laps), which belong to `score.py` / `config.py` (Mon 16).
- **Open:** the definition of the "closest wall" number (`wall_clearance`, the user's choice), and what happens to the
  `research/` folder, which is in no plan file (question 7 in [goals/communication.md](../goals/communication.md)).
- **Tests:** `autoresearch/tests/test_drive_stats.py`.

### D16 — The user's rules for the loop (11 Nov 2026)
- **What** (the user's answers 3–6 in [goals/communication.md](../goals/communication.md), Archive):
  an agent may change **everything** except the judge; the agents are Claude (Pro subscription) and other models through
  free APIs; the dates do not move (week 2 starts Tue 17 whatever the baseline is); the AIL Cup has 3 tracks, unlimited
  uploads, a model up to 50 MB and up to 64 arrays (`ail_cup/export_wm2.py` already packs wm2 into 49 arrays).
- **What follows (Claude's reading, not yet agreed):** `wm2/drive.py` counts its own crashes and writes the log the judge
  reads, so with "change everything" the frozen judge must own the sim connection and the log (Mon 16, `drive_eval.py`).
  Usage limits make hours an unfair budget: count experiments per agent instead (to check Wed 18).
- **Still open:** the AIL Cup time limit per call and which 3 tracks (question 9). → answered in D17.

### D17 — The AIL Cup tracks; `research/` archived (11 Nov 2026)
- **AIL Cup maps** (the user's screenshots, `docs/ail_cup_maps/`): (1) a plain oval; (2) a winding track with a hairpin
  and an S-bend; (3) an oval with two chicanes. All are driven clockwise from a start line on the bottom straight, and the
  road looks wide compared with the car. Time limit: a control step over 250 ms ends the run (`BP/CHAT_CONTEXT.md`).
- **What follows (Claude's reading):** the baseline has to drive tracks it was not trained on, so "drives an unseen track"
  (E14) is a main result, not a side check. Whether the sim builds `practice` / `compete` / `explore` are these same
  three tracks is not known yet (question 10).
- **Answer to question 10 (the user, 11 Nov):** the sim tracks are **different** from the AIL Cup maps. So the AIL Cup
  always tests the car on tracks it never trained on, and the sim score inside the loop measures something else than
  the AIL Cup result unless the judge also drives a track that was held out of training. Proposed, not yet agreed:
  collect on `explore` too (question 11), and decide on Mon 16 which sim track the judge keeps unseen.
- **Answer to question 11 (the user, 11 Nov):** `explore` is **not a track**, so it is not used for data collection.
  The sim has two tracks, `practice` and `compete`. Holding one out for the judge would leave one training track, so the
  unseen-track check is the AIL Cup itself (unlimited uploads), not a sim track.
- **`research/`** (the driver-tuning loop of 10 Nov) was moved to `archive/research/` on the user's decision; the loop is
  written new in `autoresearch/`. `drive_stats.py` keeps its rules (D15).

---

### D18 — Balanced sampling for wm2 training (A1 + A2), first in `wm2_c4` (14 Nov 2026)
- **Decision (the user, 14 Nov):** "Now (training only): A1 + A2 in the next training", then "start".
- **What:** `WM2_BALANCE=1` (`wm2/data.py` `sample_weights`, `wm2/train.py`): windows are drawn by weight instead of
  all alike: each track gets the same share of the draws (A1, `practice` : `compete` was ~4 : 1, E28 P4), the trips
  `self2` – `self6_compete` x2, starts faster than 1.5 m/s x2, a standing car (< 0.2 m/s) x0.2 (A2, E28 P5).
  One epoch draws as many windows as before (with replacement). Needs `track` + `source` per moment, which
  `wm2/prepare.py` stores since 14 Nov (set `c4` = the trips of `c2`). Default off. Tests: `autoresearch/tests/test_balance.py`.
- **Why only for the world model:** heads and value head keep their own sampling (unchanged).
- **Other options:** fewer `practice` trips (loses data); weights per trip only (cannot drop standing moments).
- **Evidence it helps:** E30 (`wm2_c3` -> `wm2_c4`, `night/c4_train.sh`).

### D19 — `compete` is dropped; the focus is the AIL Cup tracks with a numpy controller (14 Nov 2026)
- **Decision (the user, 14 Nov):** "lets forget about this compete track because its too hard for me to make it right;
  lets focus on normal tracks ... ride on AIL Cup tracks with numpy".
- **Why:** every `compete` crash is one fork with a dead-end branch (E31), a route problem the value head gets wrong
  (E32); the AIL Cup maps are closed loops without forks (D17, `docs/ail_cup_maps/`).
- **What changes:** the week-1 goal "0 crashes on `compete`" is replaced by "full laps on the AIL Cup maps"; `practice`
  stays the sim check; the Garage controller is `ail_cup/simple_controller.py` (numpy, E27). The user confirms that the
  Garage allows numpy (question 12).
- **Kept:** `compete` data stays in training (it is real driving), E31 / E32 stay as the record of the fork.

## Experiments

### E1 — MLP predicts the next lidar scan (14 – 16 Oct 2026)
- **Question:** can a plain network learn how the lidar changes?
- **Setup:** 1,722 trips (13.2 h, practice); MLP 274 → 512 → 512 → 271 (542k parameters), 40 epochs.
- **Result:** one step: brain / copy = 0.24. Imagination: 0.25 @0.05 s, 0.35 @0.5 s, 0.67 @1 s, worse than copy after ~1.1 s;
  some imaginations explode (values < 0 or > 1).
- **Conclusion:** motion is learned, long imagination is not → D1. Files: `archive/results_MLP.md`.

### E2 — JEPA without SIGReg (Oct 2026)
- **Result:** full collapse in 10 epochs (spread 1.7e-6, val loss ~4e-10: the model predicts a constant). → D2. Source: `notes.md`.

### E3 — JEPA with SIGReg + probes (19 Oct 2026)
- **Setup:** encoder 271 → 256 → 32, predictor 35 → 256 → 256 → 32 (161k parameters), SIGReg 0.09, 50 epochs.
- **Result:** spread 1.01, one step val / copy = 0.24 (latent). Probe R² on test (trained / random encoder):
  linear front 0.16 / 0.90, steering 0.19 / 0.60; MLP front 0.99 / 0.98, steering 0.83 / 0.69, track position x / y 0.99 / 0.99.
- **Conclusion:** the information is kept but folded (nonlinear); the encoder learned where the car is. → D8. Source: `notes.md`.

### E4 — Long imagination in latent space, with and without a guard (24 Oct 2026)
- **Result** (500 test starts, MSE vs copy): no guard: better than copy until ~0.75 s, then explodes (5.8 @1 s), |z| grows 5.5 → 7.5.
  Guard B (rescale |z| to √32 every step): better than copy for the whole 2 s (0.95 vs 1.9 @1 s). Source: `notes.md`.

### E5 — Causal test: `find_gap` data vs random-steering data (Oct 2026)
- **Result:** with `find_gap`-only data the imagined right wall got *closer* when steering left (physically wrong; the planner chose
  like a coin). With random segments: steering -1 → +1 moves the imagined right wall 0.81 → 1.46 m and the left 1.31 → 0.91 m (0.5 s ahead).
- **Conclusion:** → D3. Source: `driver_lidar/README.md`.

### E6 — Which planner score (Oct 2026)
| Score | Imagination | Danger, coin ≈ planner | Control |
|---|---|---|---|
| front + right + left | 1 s | 45.6 % | 49.6 % |
| closest of 5 walls | 0.5 s | 41.2 % | 51.4 % |
| closest of R, L, diagonals /1.4, front /2 | 0.5 s | **35.2 %** | 50.1 % |
- → D4. Source: `driver_lidar/README.md`.

### E7 — One-step vs K = 10 training (25 – 26 Oct 2026)
| Run | 1 s imagined / copy | Planner (agree vs control) |
|---|---|---|
| `onestep_k1` | 0.678 | – |
| `k10_50ep` | 0.231 | -14.9 |
- → D5. Source: `runs/*/report.md`.

### E8 — 109 beams (07 Nov 2026)
- `k10_50ep_b109`: 1 s imagined / copy 0.293, planner -12.9 (vs -14.9 with 271 beams): a little worse, accepted for the AIL Cup. → D7.

### E9 — Image JEPA (camera) (02 – 03 Nov 2026)
- `Image_jepa_K10_50ep`: 1 s imagined / copy 0.131 (good), but planner only **-4.4** (lidar: -14.9). → D6.

### E10 — wm2 first models and driver (08 Nov 2026)
- `wm2_k10` (20 epochs, random data), `wm2_fast_k10` (20 epochs, + fast data).
- Offline planner (5,000 random test moments, `night/plan_test_fast.log`): wm2_fast_k10 **-14.1**, wm2_k10 **-13.7** (find_gap ≈ -23).
- In the sim: the first wm2 driver drove a clean 5 minutes (commit 2026-11-08 "wm2 driver ... drives a clean 5 minutes").

### E11 — Self-driving round 1 → `wm2_self1` (night 08 → 09 Nov 2026)
- **Setup:** the car drives itself (explore gas 0.09 … 0.18), then wm2_fast_k10 is fine-tuned 6 epochs → `wm2_self1`.
- **Collection:** ~8.1 crashes/min with the ~340 ms CPU planner (`night/self_drive_3.log`).
- **Imagination error on fast self-driven trips, 1 s, brain / copy** (`night/self_compare.log`):
  wm2_k10 0.209, wm2_fast_k10 0.189, **wm2_self1 0.058** (about 3× better). On slow trips all ≈ 0.015–0.017 (no loss).
- **Conclusion:** the model improves exactly where the car drives itself → D11.

### E12 — Planner on the GPU (09 Nov 2026)
- 320 ms → **41 ms** per plan (CUDA graph, warm start, softer cost). A single rollout: 253 ms CPU, 119 ms normal GPU, ~54 ms graph.
- **Sim result, wm2_self1, gas 0.15** (`night/drive_self1_gas15_gpu_keep_soft.log`): **0 crashes in ~5 min, 18 laps, best lap 15.04 s**. → D12.

### E13 — Self-driving round 2 → `wm2_self2` (night 09 → 10 Nov 2026)
- **Setup:** wm2_self1 with the GPU planner, `FRONT_BASE=0.25`, soft brakes, explore gas 0.12 / 0.15 / 0.18, 2.1 h → `data/self2_practice`;
  fine-tune 3 epochs (lr 1e-4) → `wm2_self2` (75 min of training).
- **Collection:** **1.2 crashes/min** (round 1: 8.1), laps ~15.3 – 15.7 s (`night/self2_drive_1.log`).
- **Test drives:** `runs/wm2_self2/drive_practice_self2_*.csv` (test, front015, front02; a/b). **Analysis: open (Wed 11 Nov).**
- **Result (11 Nov):** 2 drives of 5.5 min for each row, gas 0.12, soft brakes, one sim session (42–48 packets/s), self1 and
  self2 driven in turn. Numbers by the rules of D15 from `runs/wm2_self{1,2}/drive_practice_self{1,2}_{test,front02,front015}_{a,b}.csv`;
  `test` = the default `FRONT_BASE` 0.25 (`BP/claude_tools/batch_*.txt`, `BP/CHAT_CONTEXT.md`, which has the same table).

| Model | FRONT_BASE | Crashes after 5 s (a, b) | Lap median s (a, b) | Best lap s |
|---|---|---|---|---|
| self1 | 0.25 | 0, 0 | 14.30, 14.23 | 12.74 |
| self2 | 0.25 | 0, 0 | 14.69, 14.73 | 13.21 |
| self1 | 0.2 | 0, 1 | 13.97, 13.97 | 12.16 |
| self2 | 0.2 | 0, 0 | 14.18, 14.23 | 12.17 |
| self2 | 0.15 | 1, 0 | 14.08, 13.75 | 12.32 |

- **Conclusion:** self2 is **not faster** than self1: at the same settings its lap median is 0.2–0.45 s slower. Crashes do
  not separate them (0 or 1 per 5.5 min, 2 drives each). The fastest setting without a crash in both drives is self2 with
  `FRONT_BASE` 0.2. Round 2 collected almost no faster driving (> 3 m/s in ~1 % of the rows, `BP/CHAT_CONTEXT.md`), which
  fits a model that did not get faster. Still to write: `runs/wm2_self2/report.md`.

### E14 — The first night on `compete` (night 11 → 12 Nov 2026) — done, `wm2_c1` trained, not driven yet
- **Question:** how does wm2_self2 drive a track it never saw, and what data does `compete` give?
  Expected: badly. An older lidar JEPA trained on practice only hit the first wall on `compete`, and its 0.5 s imagination
  on old compete recordings was no better than copy (0.99) (`BP/CHAT_CONTEXT.md`, section 5).
- **Setup** (`night/compete_night.py`, state in `night/state_compete.txt`):
  1. two test drives of wm2_self2, 5.5 min each, gas 0.12, default cost (`FRONT_BASE` 0.25), no explore: the same as the
     practice `test` drives of E13 → `runs/wm2_self2/drive_compete_self2_test_{a,b}.csv`;
  2. random collection (`data_collection.collection_random compete`), 3 h → `data/fast_compete`;
  3. self-driving with noise (explore gas 0.09 / 0.12 / 0.15), 3 h → `data/self_compete`, **only if** both tests had at most
     10 crashes/min and a mean speed of at least 0.5 m/s; otherwise step 2 gets these hours. No training.
- **Why the condition:** a car that only drives from the start line into the first wall records the same 3 seconds again and
  again. Round 1 on practice was still useful at ~8 crashes/min (E11), so the limit is a little above that.
- **After the night** (allowed by the user on 11 Nov): `night/compete_train.py` fine-tunes wm2_self2 → `wm2_c1` on
  `data/wm2_c1` (everything self2 learned on + the compete trips; `wm2.prepare c1`), 3 epochs, lr 1e-4, then the heads.
- **Started:** 18:01 on 11 Nov. The first 3.5 min of test a: about 9 crashes/min (`night/compete_test_a.log`).
- **Result (12 Nov, 01:36; everything from `night/state_compete.txt`, no restart in the whole night):**
  - **wm2_self2 on a track it never saw** (rules of D15, `runs/wm2_self2/drive_compete_self2_test_{a,b}.csv`):
    43 and 45 crashes after the warmup = **7.8 and 8.3 crashes/min**, mean speed 1.40 / 1.41 m/s, **no finished lap**
    (the raw lap counter reached 1 once in a, 0 in b). On practice the same settings give 0 crashes and ~2.05 m/s (E13).
    Nearer side wall: 5th percentile 0.39 m, median 0.72 / 0.71 m. Better than expected (it moves and is under the limit),
    but it does not drive this track.
  - **Random collection** 18:13 – 21:13 → `data/fast_compete`: **300 trips, 110 min**, 133 crash / 167 timeout (44 % crash, the
    same share as on practice; the 68 % of the first 25 trips was not typical).
  - **Self-driving collection** 21:13 – 00:13 → `data/self_compete`: **1245 trips, 166 min**, every trip ended in a crash,
    median trip 7.9 s (90 % under 12.1 s, longest 50.9 s); 90 trips have fewer than 50 frames. So this is "8 seconds, then a
    wall" data, not laps. (`data/self_practice` for comparison: 2292 trips, 243 min, median 1.9 s.)
  - **Training** `wm2_self2 → wm2_c1` (`night/c1_*.log`, `runs/wm2_c1/loss.csv`): split 80/10/10 by trip (fastc 240/30/30,
    selfc 997/124/124); `data/wm2_c1` = 5486 / 684 / 685 trips, 1.74 M train moments; 3 epochs, 78 min.
    val pred/copy 0.034 → 0.031 → **0.030**, spread 0.990 (no collapse). wm2_self2 ended at 0.0296 on its own val set, which
    had no compete trips, so the two numbers are **not comparable**; an honest comparison needs both models on one val set.
    Heads on the test set: walls 0.04 – 0.11 m by beam (walls5: front 0.10, sides 0.05, diagonals 0.07), danger right in
    88.8 % of moments (always "danger" = 44.3 %), progress error 0.8 cm per step of 6.5 cm.
  - **Not known yet:** whether wm2_c1 drives `compete` better, and whether it still drives `practice` crash-free.
    Both need the sim (Thu 12).
- **Test drives of wm2_c1 on `compete` (12 Nov 01:52 – 02:04, `night/test_drives.py`, the same settings as the self2 tests;
  `runs/wm2_c1/drive_compete_c1_test_{a,b}.csv`):** 43 and 42 crashes = **7.9 and 7.7 crashes/min** (self2: 7.8, 8.3),
  mean speed 1.19 / 1.31 m/s (self2: 1.40 / 1.41), no finished lap. **Not better.**
  - **Where:** metres driven between two crashes (speed × time; the car is put back at the start after a crash): c1
    85 of 85 crashes after 6 – 12 m; self2 87 of 88 after 8 – 15 m (one 79 m run). So both models crash at **one place,
    about 10 m after the start**, and almost never pass it. The 166 min of `data/self_compete` are these 10 m again and again.
  - **How** (c1 test b, mean over 42 crashes): 3 s before the crash front wall 1.67 m, speed 1.2 m/s; then the front wall
    comes closer (0.89 → 0.48 → 0.21 m) while the side walls stay far (right 1.0 – 1.4 m, left 0.8 – 1.7 m), speed falls
    to 0.3 – 0.6 m/s, and the steering has no direction (mean −0.1 … +0.03, mean |steer| 0.3 – 0.6: left in some crashes,
    right in others, straight in others). The car creeps head-on into a wall with room on both sides.
  - **Reading (Claude's, not proven):** this is a choice problem of the planner at one corner, not a lack of data about
    walls: it sees the wall and brakes, but does not pick a side. The random trips do get past this place
    (path median 39.8 m, 225 of 300 longer than 15 m), so the model has data from behind it.
  - **Consequence:** more self-driving collection with this driver adds nothing. Next: find what the planner sees there
    (the cost of left / right / straight at that place) before any more training.
- **wm2_c1 on `practice` (12 Nov 02:11 – 02:23, `runs/wm2_c1/drive_practice_c1_test_{a,b}.csv`):** 0 and 0 crashes, 19 laps
  each, lap median **15.31 / 15.27 s** (self2: 14.69 / 14.73), best 14.15 / 13.89, mean speed 1.97 / 2.01 m/s (self2 2.04 /
  2.07). Still crash-free, about 0.6 s per lap slower.

### E15 — Why the car stops at one corner of `compete` (12 Nov 2026, no sim: replay on recorded trips)
- **Question:** is the crash place of E14 a fault of the model's imagination, of missing data, or of the planner's cost?
- **Data:** `data/self_compete` (driven by wm2_self2; trips that end within 1.5 m of the median crash point x −0.87, y 9.33)
  and `data/fast_compete` (random trips). Model in the replay: wm2_c1 with the sector heads, on the CPU. Scratch scripts,
  not kept in the repo.
- **1. The place is a left turn that the random driver passes:** 230 random trips came within 2.5 m of it, 214 passed,
  16 crashed there; the wheel was turned left in 94 % of the passes (mean yaw rate +0.61 rad/s, speed 1.28 m/s).
- **2. The imagination is right there:** 1 s ahead with the real actions, sector walls against the real lidar (val + test
  trips only): random trips near the corner 0.12 m mean error (copy 0.79), elsewhere on compete 0.16 m (copy 0.81), self
  trips 2 / 1.5 s before the crash 0.06 m (copy 0.33). So the model is not the weak part.
- **3. The wrong choice is 3 – 4 s before the crash:** until 4 s before it the car is on the line of the passing random
  trips (heading within 4°, speed 1.7 m/s against their 1.6). From 3 s on the passing trips keep the wheel left
  (+0.27 … +0.43) while the car holds −0.06 … −0.14 and brakes: heading 27° behind theirs at −3 s, 56° at −2 s, speed
  0.5 m/s against 0.97. 2 s before the crash every sector from −90° to +90° is 0.8 – 1.2 m away: it has parked itself
  facing the outer wall.
- **4. What the cost says at that moment** (constant steering × constant gas for 1 s, the driver's settings `W_CENTER` 0.3,
  `SIDE_MARGIN` 0.35, `FRONT_BASE` 0.25; mean over 119 moments, 3.5 s before the crash, speed 1.45 m/s): progress is worth
  0.24 – 1.8 for every option, but with gas 0.08 the front-wall term is 12 – 51 and with gas 0.12 it is 83 – 180 **for every
  steering**, because the imagined front wall after 1 s is 0.1 – 0.5 m whatever the wheel does. The centre term adds
  5 – 10 for a full left turn (1 for straight). The cheapest plans are "gas 0" (total 0.7 – 1.3) and "gas 0.04, straight"
  (0.2). So the planner brakes and goes straight.
- **Reading (Claude's):** the cost was tuned on `practice`, a corridor of one width. In this tighter turn the front wall
  stays close while the car turns past it, and the rule "keep the braking distance to the wall in the ±15° sectors" plus
  "stay in the middle" make every moving plan look worse than stopping. A fault of the cost, not of the model or the data.
- **Not proven:** the replay uses constant gas, not the driver's brake-then-gas plans, and no prefix. The test is a
  drive: `compete` with a softer front / centre term (`WM2_COST`), then the same settings on `practice`.
- **Result:** open (Thu 12 Nov).

---

## Open questions (move to a decision or experiment when answered)
- Is wm2_self2 better than wm2_self1 in the sim? → answered in E13 (11 Nov): no, a little slower
- How does the car drive on `compete`, a track it never saw? (Thu 12)
- Is the model fast enough for the AIL Cup time per call? (Thu 12)
- The sim score formula for the loop ([goals/month.md](../goals/month.md))

### E16 — Cost settings for `compete` and `practice` with wm2_c1 (12 Nov 2026, 02:32 – 03:19, sim)
- **Question:** does a softer cost let the car pass the corner of E15, and does `practice` survive the same setting?
- **Setup:** `night/test_drives.py` (gas 0.12, soft brakes, `FRONT_BASE` 0.25 unless said; cost through `WM2_COST`), model
  wm2_c1. Logs `runs/wm2_c1/drive_<track>_c1_<tag>.csv`. "Past corner 1" = more than 15 m driven before a crash
  (speed × time between two crashes). The sim builds were switched by `night/open_sim.ps1` (allowed by the user 12 Nov).
- **`compete`:**

| Tag | Cost | s | Crashes/min | Past corner 1 | Mean speed | Lap times seen |
|---|---|---|---|---|---|---|
| `test_a` (E14) | default: `W_CENTER` 0.3, `SIDE_MARGIN` 0.35 | 333 | 7.9 | 0 / 44 | 1.19 | none |
| `c01` | `W_CENTER` 0.1 | 125 | 10.5 | 0 / 22 | 1.44 | none |
| `center0` | `W_CENTER` 0 | 125 | 5.0 | 9 / 11 | 1.72 | 19.5 |
| `c0_f015` | `W_CENTER` 0, `FRONT_BASE` 0.15 | 125 | 6.0 | 7 / 13 | 1.88 | 18.6 |
| `c0_d8` | `W_CENTER` 0, `DECEL` 8 | 124 | 8.1 | 9 / 17 | 1.99 | none |
| `c0_s025` | `W_CENTER` 0, `SIDE_MARGIN` 0.25 | 124 | 5.0 | 8 / 11 | 1.90 | 18.8, 19.9 |
| `c0_s045` | `W_CENTER` 0, `SIDE_MARGIN` 0.45 | 202 | 5.5 | 10 / 19 | 1.68 | 20.6, 21.1 |
| `c0_s05` | `W_CENTER` 0, `SIDE_MARGIN` 0.5 | 205 | 5.7 | 10 / 20 | 1.59 | 22.0 |
| `c0_s055` | `W_CENTER` 0, `SIDE_MARGIN` 0.55 | 200 | 5.8 | 11 / 20 | 1.48 | 22.0 |

- **`practice`:**

| Tag | Cost | s | Crashes | Lap median s |
|---|---|---|---|---|
| `test_a`, `test_b` (E14) | default | 332, 332 | 0, 0 | 15.31, 15.27 |
| `c0_test_a`, `c0_test_b` | `W_CENTER` 0 | 333, 332 | 8, 12 | 13.1, 12.7 |
| `c0_s045` | `W_CENTER` 0, `SIDE_MARGIN` 0.45 | 201 | 1 | 13.9 |
| `c0_s055` | `W_CENTER` 0, `SIDE_MARGIN` 0.55 | 201 | 0 | 16.5 |

- **Findings:**
  1. **The centre term blocks corner 1 of `compete`** (E15 confirmed by driving): with any `W_CENTER` > 0 tried the car never
     passes it, with 0 it passes in about half to four fifths of the tries. The first finished laps of wm2 on `compete`.
  2. **On `practice` the centre term is what kept the car off the walls:** without it 8 and 12 crashes in 5.5 min, laps
     2.5 s faster. A wider side margin does that job without asking for symmetry: 0.45 m → 1 crash in 200 s and laps
     1.4 s faster than the default; 0.55 m → 0 crashes, laps 1.2 s slower.
  3. **`compete` has a second hard place about 15 – 16 m after the start** (a right turn after the left one): with
     `W_CENTER` 0 most crashes are there, whatever the side margin; 3 s before them the wheel is at −0.6, in the last
     0.3 s the speed rises from ~1.1 to 1.4 – 2.0 m/s with the left wall at 0.4 – 0.5 m. Corner 1 still takes some tries.
  4. Still about 5 – 6 crashes/min on `compete`: no setting here is a baseline for both tracks.
- **Caveat:** single drives of 2 – 3.5 min; differences of a few crashes between the `W_CENTER` 0 rows are noise.
- **Next:** 2 × 5.5 min + 30 min of self-driving with `W_CENTER` 0, `SIDE_MARGIN` 0.45 on `compete` → `data/self2_compete`
  (trips with positions from corner 2), then the replay of E15 at corner 2.

### E17 — A capped centre term and a longer look-ahead (12 Nov 2026, 03:19 – 05:00, sim)
- **Question:** E16 left two crash places on `compete`. What is the second one, and is there one setting for both tracks?
- **Code:** `cem.CENTER_CLAMP` (new, default 2.0 = as before): |left − right| in the centre term counts up to this many
  metres. `WM2_HORIZON` (new, default 20): steps of 0.05 s the driver's plan looks ahead. `night/test_drives.py` and
  `night/compete_night.py` take the model, cost and folders from the environment. Defaults drive as before.
- **1. Full drives with `W_CENTER` 0, `SIDE_MARGIN` 0.45 on `compete`** (`drive_compete_c1_c0s045_test_{a,b}.csv`, 5.5 min):
  6.5 and 5.2 crashes/min, mean speed 1.64 / 1.72 m/s, lap times seen 19.3 – 21.3 s (3 laps).
- **2. Corner 2** (30 min of self-driving with that setting → `data/self2_compete`, 199 trips, 27 min; 77 trips end within
  1.2 m of x −2.76, y 6.69; the last row of a crash trip is already the reset position, so the row before it is used):
  after the left hairpin the track is an S-bend, right then left. Up to 3 s before the crash the car is on the line of
  the passing random trips; from 2 s on they steer left (+0.09 … +0.50) and the car keeps right (−0.29 … −0.01): heading
  30° off theirs at −2 s, 86° at −1 s. It makes a U-turn into the open space on the right, speeds up (0.8 → 1.6 m/s) and
  hits the wall. Reading: without any centre term, "metres driven forward" is all the planner wants, and in a wide
  place the open space is as good as the track.
- **3. A capped centre term on `compete`** (200 s each, 1 s look-ahead; lap ≈ 38 m):

| Tag | `W_CENTER`, `CENTER_CLAMP` | Crashes/min | Crashes after 12 – 20 m (corner 2) | Runs > 15 m | Laps seen |
|---|---|---|---|---|---|
| `cc10` | 0.3, 1.0 | 10.1 | 0 | 0 / 35 | 0 |
| `cc05` | 0.3, 0.5 | 5.8 | 0 | 4 / 20 | 3 |
| `w1cc03` | 1.0, 0.3 | 3.0 | 2 | 4 / 11 | 5 |
| `h20_w1cc03_b` (repeat) | 1.0, 0.3 | 5.7 | 2 | 3 / 20 | 2 |
| `w1cc02` | 1.0, 0.2 | 5.2 | 2 | 7 / 18 | 5 |
| `w03cc03` | 0.3, 0.3 | 6.1 | 11 | 10 / 21 | 2 |
| `w2cc015` | 2.0, 0.15 | 6.1 | 6 | 5 / 22 | 3 |

  A centre pull of about 0.075 – 0.09 per step at the cap removes corner 2 but again stops the car at corner 1 in two
  thirds of the tries; a weaker one (0.027) does the opposite. **The repeat of the same setting gave 3.0 and 5.7
  crashes/min: a 200 s drive cannot rank settings that differ by less than a factor of two.**
- **4. The look-ahead** (`W_CENTER` 1.0, `CENTER_CLAMP` 0.3; 200 s on `compete`):

| Tag | Look-ahead | Crashes/min | Runs > 15 m | Lap times s | Plan ms (median) |
|---|---|---|---|---|---|
| `h20_w1cc03_b` | 1.0 s | 5.7 | 3 / 20 | 21.8, 22.2 | 40 |
| `h30_w1cc03` | 1.5 s | 1.5 | 6 / 6 | 26.5 – 28.1 | 59 |
| `h40_w1cc03` | 2.0 s | 2.7 | 2 / 10 | 31.5, 36.0 | 75 |

- **5. Full drives with the 1.5 s look-ahead** (5.5 min each, `drive_<track>_c1_h30_test_{a,b}.csv`):

| Track | Crashes | Crashes/min | Laps counted | Lap median s | Mean speed | Longest run |
|---|---|---|---|---|---|---|
| `compete` | 7, 7 | 1.3, 1.3 | 7, 7 | 27.6, 29.0 | 1.44, 1.52 | 204 m, 197 m |
| `practice` | 1, 1 | 0.2, 0.2 | 13, 13 | 21.5, 20.7 | 1.34, 1.37 | 425 m, 437 m |

  For comparison (E14): default on `compete` 7.9 / 7.7 crashes/min and no lap; default on `practice` 0 crashes, lap 15.3 s.
- **Findings:**
  1. **The 1 s look-ahead was the reason for corner 1**, as E15 suggested: with 1.5 s the car sees that "straight a little
     longer" ends at a wall. 2 s is worse (slower plans, and the model was trained on 0.5 s rollouts).
  2. **This is the first setting that drives laps on both tracks**: `compete` from ~8 to 1.3 crashes/min. The price:
     `practice` is 40 % slower (21 s laps instead of 15.3 s) and has one crash per drive, both about 25 m after the start.
  3. Not a baseline yet: the target is crash-free on both tracks.
- **Open:** where the remaining `compete` crashes are (5 of 14 runs end 13 – 15 m in, corner 2; the others after 24 – 204 m);
  why `practice` crashes once at 25 m; whether gas or the front margin can win back the speed.
- **Next (running since 05:00):** 2 h of self-driving on `compete` with this setting → `data/self3_compete`, the first
  self-driven trips with whole laps; then `wm2_c1 → wm2_c2` on everything (`wm2.prepare c2`), allowed by the user on 12 Nov.

### E18 — A reward for free road in front at the end of the plan (12 Nov 2026, morning, sim)
- **Before it:** Windows restarted the PC at ~05:45 (boot 05:49). The collection of E17 ran 05:03 – 05:44:
  `data/self3_compete` = 136 trips, 39 min, 24 trips longer than a lap (38 m), 3.3 crashes/min with the explore noise.
  `wm2_c2` was not trained.
- **Idea (the user's question: "does the driver see 10 m or only 3?"):** the lidar and the model see 10 m, but the plan
  covers 1 – 1.5 s (1.5 – 3 m) and the cost only punishes near walls. So reward what is seen far away:
  `− W_FREE × free road`, free road = the imagined front wall (the ±15° sectors), mean of the last 5 steps of the plan,
  up to 4 m (`W_FREE`, `FREE_CAP`, `FREE_STEPS` in `wm2/cem.py`; `W_FREE` 0 = off, the default).
- **Test:** wm2_c1 on `compete`, 200 s each, `W_CENTER` 1.0, `CENTER_CLAMP` 0.3, `FRONT_BASE` 0.25
  (`runs/wm2_c1/drive_compete_c1_{fr03,fr06,h30_fr03}.csv`):

| Tag | Look-ahead | `W_FREE` | Crashes/min | Runs > 15 m | Crashes after 12 – 20 m | Without `W_FREE` (E17) |
|---|---|---|---|---|---|---|
| `fr03` | 1.0 s | 0.3 | 5.5 | 3 / 19 | 0 | 3.0 and 5.7 |
| `fr06` | 1.0 s | 0.6 | 6.0 | 3 / 21 | 3 | 3.0 and 5.7 |
| `h30_fr03` | 1.5 s | 0.3 | 3.4 | 5 / 12 | 8 | 1.5 (200 s), 1.3 and 1.3 (5.5 min) |

- **Result: it does not help.** With the 1 s look-ahead corner 1 stops the car as before (16 of 19 and 17 of 21 runs end at
  8 m). With 1.5 s it is worse than without: 8 of 11 crashes are back at corner 2.
- **Reading (Claude's):** "a far front wall" is true for the road but also for the open pocket beside it at corner 2, so
  the reward pulls the car into the pocket it was meant to avoid. Free space is not the same as the track. The signal
  that tells them apart must know the track's direction (the second idea: progress along the track, learned from the
  trips that pass). One 200 s drive per setting: by E17 that only shows differences of a factor of two, which this is for
  `h30_fr03` against 1.3.
- **Left in the code**, off by default.

### E19 — A value head: "how far does the car still get from here?" (12 Nov 2026) — replay done, no drive yet
- **Decision (the user, 12 Nov: "yes"):** after E18, learn the difference between road and pocket from the recorded
  trips instead of from wall distances.
- **What:** `wm2/value.py`. The answer for a moment = metres driven (`progress`) in the next 3 s of the same trip; a crash
  ends the sum; a trip that ends without a crash inside the 3 s is left out. A head on the frozen z (64 → 256 → 256 → 1,
  smooth L1, 10 epochs) → `runs/wm2_c1/value.pt`. Tests of the answers: `autoresearch/tests/test_value_targets.py`.
- **Data:** `data/wm2_c1` as it is (no new trips): 1,635,250 train moments with a known answer (94.0 %), mean 3.75 m.
- **Test split:** mean error 0.69 m (always saying the mean: 1.05 m), correlation 0.74. Where the car really drove
  < 1 m the head says 1.01 m (10,587 moments); where it drove > 4 m it says 4.44 m (82,437 moments).
- **Replay at the two corners** (no sim; model wm2_c1, constant steering × constant gas, value read on the last 5 imagined
  steps; cost as the best driver setting of E17; "wins" = which steering has the lowest cost − `W_VALUE` × value).
  Corner 1: 160 moments from the val + test self trips of `data/self_compete`. Corner 2: 81 moments from
  `data/self2_compete`, trips the head never saw. The passing trips turn left (+) at both.

| Place, moment, plan | Value at the plan's end: full right / straight / full left (m) | Full left wins: `W_VALUE` 0 → 1 → 2 |
|---|---|---|
| Corner 1, 3.0 s before, 1.0 s plan, gas 0.04 | 1.62 / 0.96 / 2.09 | 13 → 85 → 90 of 160 |
| Corner 1, 3.5 s before, 1.5 s plan, gas 0.04 | 1.33 / 1.15 / 2.74 | 8 → 74 → 79 |
| Corner 1, 3.0 s before, 1.5 s plan, gas 0.04 | 1.76 / 0.60 / 2.19 | 14 → 67 → 74 |
| Corner 1, 3.0 s before, 1.0 s plan, gas 0.08 | 0.34 / 0.59 / 0.70 | 21 → 33 → 34 |
| Corner 2, 2.0 s before, 1.0 s plan, gas 0.04 | 1.37 / 2.54 / 2.79 | 5 → 39 → 42 of 81 |
| Corner 2, 2.0 s before, 1.5 s plan, gas 0.04 | 0.44 / 1.33 / 2.73 | 9 → 35 → 35 |
| Corner 2, 2.5 s before, 1.0 s plan, gas 0.04 | 2.26 / 2.75 / 2.14 | 8 → 18 → 18 (the passing trips still steer right here) |

- **Reading:** at the moments where the planner went wrong, the head ranks the turn the passing trips took highest for
  the slow plans, and with `W_VALUE` ≥ 1 that turn wins most often. Two warnings: (1) at corner 1 "full right" also gets
  a high value (1.3 – 2.1 m), and with gas 0.08 some answers are below 0: on imagined z far from the data the head is
  unreliable; (2) for the faster plans the wall terms (12 +) still decide, the value hardly moves the choice.
- **In the cost:** `− W_VALUE × value` (kept between 0 and `VALUE_CAP` 6 m), `cem.W_VALUE` 0 = off (default). The driver
  loads the head when `WM2_COST` has `W_VALUE` > 0. Checked: the default cost is unchanged.
- **Not done:** no drive. Proposed test: `compete`, `W_VALUE` 1.0 with the 1.0 s and the 1.5 s look-ahead, then `practice`.
- **Drive test of E19 (12 Nov, the user: "test"; `compete`, 200 s each, `W_CENTER` 1.0, `CENTER_CLAMP` 0.3, `W_VALUE` 1.0;
  `runs/wm2_c1/drive_compete_c1_{val1,h30_val1}.csv`):**

| Tag | Look-ahead | Crashes/min | Runs that end at corner 1 (~8 m) | Runs that end after 10 – 13 m | Mean speed | Without `W_VALUE` (E17) |
|---|---|---|---|---|---|---|
| `val1` | 1.0 s | 4.9 | 0 of 17 | 16 | 1.21 | 3.0 and 5.7, 16 of 20 runs end at corner 1 |
| `h30_val1` | 1.5 s | 4.8 | 0 of 17 | 15 | 0.93 | 1.5, 1.3, 1.3 |

  - **Corner 1 is gone:** not one run ends there (before: most of them with the 1 s look-ahead). The replay was right.
  - **A new crash place 10 – 13 m after the start takes its place**, in 31 of 32 crashes, so the total is worse than
    the 1.5 s look-ahead alone (1.3). In the last second before these crashes the narrow front / right / left beams all
    show 1.7 – 2.3 m of room and the speed is ~1.1 m/s: the car hits something outside these three beams. Guess, not
    proven: the end of a free-standing wall (the user's map picture shows such wall ends after corner 1), on a line
    the earlier drivers never took. The test drives record neither the full lidar nor the position, so this needs a
    short recording with this setting.
  - **So:** the value head does what it was built for, but one head with one weight is not a finished driver.

### E20 — The value head learns from the driver's own crashes, in rounds (12 Nov 2026, 13:20 – 15:17, sim)
- **Decision (the user, 12 Nov 13:23):** "now you have 2 hours for improving my driver so it can ride compete and
  practice, you can do everything you need". Sim tests, recordings and head trainings without asking, until ~15:25.
- **What the new crash place of E19 is** (10 min of self-driving with `W_VALUE` 1.0, head `value.pt` →
  `data/self4_compete`, 39 trips, 37 crashes): 22 of 37 crashes are at x −3 … −2, y 4 (the S-bend after corner 1), the
  nearest lidar beam 0.17 – 0.25 m away on the LEFT (36° … 135°), speed 1.36 m/s. The value term pulls the car into the
  left turn early and outweighs the side-wall term.
- **Method:** a round = record 10 min on `compete` with the current driver (explore noise, gas ceiling 0.12) → add the
  trips to the data → train a new value head on the frozen wm2_c1 encoder (`night/value_round.py`, 3 min, sim closed)
  → test drives. The world model is not retrained. Heads: `runs/wm2_c1/value_<round>.pt`, data: `data/wm2_<round>`.
  "b" = balanced loss (`WM2_VALUE_BALANCE=1`): every 0.5 m band of answers weighs the same (bands under 2 m get
  2.0 – 2.7 ×, bands over 3 m 0.3 – 0.6 ×).
- **Rounds** (all: wm2_c1, gas 0.12, `W_CENTER` 1.0, `CENTER_CLAMP` 0.3, `FRONT_BASE` 0.25, 1.0 s look-ahead unless said;
  `runs/wm2_c1/drive_compete_c1_<tag>.csv`, 160 s each):

| Head (trained on) | Tag | `W_VALUE` | Crashes/min | Longest run m | Lap times seen | Mean speed |
|---|---|---|---|---|---|---|
| `value.pt` (c1 data) | `val1` (200 s, E19) | 1.0 | 4.9 | 51 | 1 | 1.21 |
| v1 (+ self2, 3, 4) | `v1_w10` | 1.0 | 5.0 | 13 | 0 | 1.08 |
| v1 | `v1_w05` | 0.5 | 2.7 | 106 | 4 | 1.60 |
| v1 | `v1_h30_w05` (1.5 s) | 0.5 | 1.9 | 100 | 3 | 1.16 |
| v2 (+ self5) | `v2_w05` | 0.5 | 4.6 | 60 | 1 | 1.33 |
| v2b (same data, balanced) | `v2b_w05` | 0.5 | 1.5 | 148 | 4 | 1.53 |
| v2b | `v2b_h30_w05` (1.5 s) | 0.5 | 2.6 | 74 | 2 | 1.13 |
| v2b | `v2b_w10` | 1.0 | 0.8 | 148 | 4 | 1.33 |

  Recordings (with explore noise): self5 with v1, `W_VALUE` 0.5: 32 crash trips in 10 min; self6 with v2b, `W_VALUE` 1.0:
  27 trips, 2.3 crashes/min, 6 laps, longest run 152 m.
- **`practice` with v3b (+ self6), `W_VALUE` 1.0** (`drive_practice_c1_v3b_w10.csv`, 200 s): 0 crashes, but lap median
  26.7 s (default driver 15.3 s), mean speed 1.11 m/s. Gas 0 is chosen 38 % of the time (default driver 16 %,
  1.5 s look-ahead without value 27 %): with the value term the planner brakes far more.
- **Round 3 at lower strengths** (v3b, 200 s each; `practice`: `drive_practice_c1_v3b_w{10,05,03}.csv`):

| `W_VALUE` | `practice` crashes | `practice` lap median s | `compete` crashes/min | `compete` longest run m |
|---|---|---|---|---|
| 1.0 | 0 | 26.7 | 2.5 | 58 |
| 0.5 | 0 | 20.7 | 3.9 | 58 |
| 0.3 | 0 | 19.8 | 2.7 | 58 |

  `compete` with v3b is not better than with v2b: the rounds did not converge in three steps, or 160 – 200 s cannot show
  it. 13 of 30 crashes are 12 – 13 m into the lap (the left wall of the S-bend), most of the rest 57 – 58 m (the same
  place on the next lap).
- **A wider side margin with the value head** (v3b, `SIDE_MARGIN` 0.45, `compete`, 190 s): `W_VALUE` 0.3 → 3 crashes
  (1.0/min), longest run 207 m, lap median 24.3 s; `W_VALUE` 1.0 → 1 crash (0.3/min), runs of 106 and 179 m, 28.4 s.
- **Final drives, 5.5 min each, one per setting and track** (v3b, `W_CENTER` 1.0, `CENTER_CLAMP` 0.3, `SIDE_MARGIN` 0.45,
  1.0 s look-ahead, gas 0.12; `runs/wm2_c1/drive_<track>_c1_final_w{10,03}.csv`):

| Track | `W_VALUE` | Crashes | Crashes/min | Laps counted | Lap median s | Mean speed | Longest run m |
|---|---|---|---|---|---|---|---|
| `compete` | 1.0 | 10 | 1.8 | 5 | 28.5 | 1.38 | 105 |
| `compete` | 0.3 | 9 | 1.7 | 9 | 23.3 | 1.76 | 152 |
| `practice` | 1.0 | 0 | 0.0 | 12 | 26.2 | 1.18 | 398 (the whole drive) |
| `practice` | 0.3 | 0 | 0.0 | 15 | 20.7 | 1.49 | 500 (the whole drive) |

  For comparison, 5.5 min each: the old driver (E14) `compete` 7.9 / 7.7 crashes/min and no lap, `practice` 0 crashes and
  15.3 s; the 1.5 s look-ahead without a value head (E17) `compete` 1.3 / 1.3 with laps of 27.6 / 29.0 s, `practice`
  0.2 / 0.2 with 21.5 / 20.7 s.
- **Result:**
  1. **Best driver after the two hours: `W_VALUE` 0.3 with head v3b, `SIDE_MARGIN` 0.45, capped centre term, 1.0 s
     look-ahead.** `practice` crash-free at 20.7 s per lap; `compete` 1.7 crashes/min with 9 laps of 23.3 s in 5.5 min.
     Against E17's best it is crash-free on `practice` (was 0.2/min) and about 5 s per lap faster on `compete`; the
     `compete` crash rate is the same within the noise (1.7 against 1.3).
  2. **The 190 s results of 0.3 and 1.0 crashes/min did not hold over 5.5 min (1.8 and 1.7).** Again: short drives flatter.
  3. **`compete` is not crash-free.** 7 of the 19 crashes in the two final drives are 12 – 13 m into the lap, the others
     48 – 152 m (multiples of the lap + the same places). `practice` is 5 s per lap slower than the old driver.
  4. Both strengths crash equally often on `compete`; 1.0 only costs speed. So the value term is needed for corner 1
     but more of it does not buy safety.
- **Not done / open:** the world model was not retrained on the new laps (`data/self3 … self6_compete`, about 70 min);
  more rounds; why the 12 – 13 m place survives; speed on `practice`. The best setting as a command:
  `set WM2_VALUE_FILE=value_v3b.pt` and `set WM2_COST=FRONT_BASE=0.25,W_CENTER=1.0,CENTER_CLAMP=0.3,SIDE_MARGIN=0.45,W_VALUE=0.3`,
  then `venv\Scripts\python -m wm2.drive wm2_c1 0.12 <tag>` (`set WM2_TRACK=compete` for that track).

### E21 — `wm2_c2`: the model retrained on its own laps on `compete` (12 Nov 2026, 15:24 – 17:11)
- **Decision (the user, 12 Nov):** "you can start training and tests after training".
- **Setup** (`night/c2_train_and_test.sh`, logs `night/c2_*.log`): `wm2.prepare c2` → `data/wm2_c2` = the data of wm2_c1
  + `data/self2 … self6_compete` (the self-driven trips of E17 – E20, about 100 min, the first with whole laps):
  5839 / 724 / 725 trips, 1.83 M train moments. Fine-tune `wm2_c1 → wm2_c2`, 3 epochs, lr 1e-4 (81 min); sector and
  walls5 heads; a balanced value head (`runs/wm2_c2/value.pt`).
- **Training:** val pred/copy 0.035 → 0.032 → 0.031, spread 0.991 (wm2_c1 ended at 0.030 on a val set without the new
  trips: not comparable). Heads on the test set: walls 0.04 – 0.11 m, danger right in 88.2 %, progress error 1.0 cm per step.
- **Test drives** (the best setting of E20: `W_VALUE` 0.3, `SIDE_MARGIN` 0.45, `W_CENTER` 1.0, `CENTER_CLAMP` 0.3,
  `FRONT_BASE` 0.25, 1.0 s look-ahead, gas 0.12; 5.5 min each; `runs/wm2_c2/drive_<track>_c2_test_{a,b}.csv`):

| Track | Model | Crashes | Crashes/min | Laps counted | Lap median s | Mean speed | Longest run m |
|---|---|---|---|---|---|---|---|
| `compete` | wm2_c1 (E20, 1 drive) | 9 | 1.7 | 9 | 23.3 | 1.76 | 152 |
| `compete` | **wm2_c2** | 3, 6 | **0.6, 1.1** | 10, 9 | 26.1, 25.9 | 1.63, 1.61 | 233, 197 |
| `practice` | wm2_c1 (E20, 1 drive) | 0 | 0.0 | 15 | 20.7 | 1.49 | the whole drive |
| `practice` | **wm2_c2** | 0, 0 | **0.0, 0.0** | 15, 17 | 20.2, 18.5 | 1.51, 1.62 | the whole drive |

- **Result:** the best driver so far on both tracks. `compete`: 9 crashes in 11 min (wm2_c1: 9 in 5.5 min), runs of up
  to 233 m (about 6 laps); laps about 2.7 s slower than wm2_c1's. `practice`: crash-free in both drives and 0.5 – 2 s
  per lap faster than wm2_c1 with the same setting (the old driver: 15.3 s).
- **How sure:** `compete` 0.6 and 1.1 against one drive at 1.7. Two drives each would be needed for wm2_c1 too; by the
  lesson of E17 / E20 this is "probably better", not proven. 2 of the 9 crashes are still 12 – 13 m into the lap.
- **Still open:** `compete` is not crash-free; `practice` is 3 – 5 s per lap slower than the old driver; nothing was
  tried on a track the model never saw (the AIL Cup export).

### E22 — AIL Cup: export of `wm2_c2` and controller v18 (12 Nov 2026, evening) — files ready, no Garage run yet
- **Decision (the user):** "Do A now and B after it". A = the new model with the old controller (a "before" number),
  B = today's fixes carried into the controller.
- **What the AIL Cup controller is** (`ail_cup/claude_controller.py`): pure Python without numpy, ~90 ms per call, so
  one plan = 3 steering options × 10 steps (0.5 s) at a fixed 1.2 m/s, spread over ~15 calls (0.75 s); between two
  decisions real-lidar reflexes and the track map's "forward" guard steer. Much smaller than the sim driver
  (~40 plans × 20 steps on the graphics card), with the cost rules of before E15.
- **A:** `python -m ail_cup.export_wm2 wm2_c2` → `ail_cup/wm2_c2_garage.npz` (3.3 MB, 50 arrays; the same names and
  shapes as `wm2_self1_garage.npz`, no NaN, largest number 2.67). Controller: `ail_cup/claude_controller_v17.py`,
  an untouched copy of the controller of 10 Nov.
- **B:** controller **v18**: `CENTER_CLAMP` 0.3 with `W_CENTER` 1.0 (was 0.3 uncapped), `SIDE_MARGIN` 0.45 (was 0.35),
  and the value head at the end of every option (mean of the last 3 imagined moments, 0 … 6 m, `W_VALUE` 0.3 × 10 / 20
  = 0.15 because the plan has 10 steps, not 20). The value head is optional: a model file without `val_*` arrays plans
  as before. `export_wm2` adds the head of `runs/<run>/value.pt` when it exists → `ail_cup/wm2_c2_value_garage.npz`
  (3.4 MB, 56 arrays of the 64 allowed). Not ported: the controller keeps its own `FRONT_BASE` 0.4, speed and reflexes.
- **Check without the Garage** (`python -m ail_cup.check_controller wm2_c2`, new): the same 4 real frames through the
  controller and through the torch model with the controller's cost → option costs 0.286 / 0.032 / −0.090 against
  0.287 / 0.032 / −0.090 (largest difference 0.001, same choice). One plan = 1.18 s of work in 501 pieces, the longest
  7.1 ms. 60 calls of `step()` on a recorded trip: median 88 ms, slowest 91 ms (the Garage cap: 250 ms), 4 plans.
- **Expectation (Claude's):** modest. The world model's decision is 0.75 s old when it is driven and covers only 0.5 s;
  wm2_self1 with v11 – v17 finished no lap. The value head is the part most likely to help such a short plan.
- **Result:** open, waits for the user's Garage runs (laps, crashes, time per call, errors) of A and B on the 3 maps.
- **First Garage runs (the user, 12 Nov evening):**
  - **A** (`wm2_c2_garage.npz` + controller v17): crashed at the start (the user's words; no log pasted).
  - **B** (`wm2_c2_value_garage.npz` + controller v18): **run #1258 "finished in 68.81 s"**, 0.4259 Wh, `step()` p95
    185 ms (cap 250 ms). The pasted end of the log (t 49.0 – 68.5 s): progress 53.7 → 79.1 m at 1.12 – 1.20 m/s, 58 plans,
    calls 88 – 93 ms. **Every one of these 40 status lines is in mode REFLEX**: the left wall is 0.35 – 0.60 m away
    (`REFLEX_M` 0.7), the right one 2.0 – 2.2 m, the track's forward direction 18 – 30° to the right. So in this part the
    simple rule steered, not the world model; the model's option costs mostly prefer right / straight
    (e.g. −0.46 −0.48 −0.31), in line with the map.
  - **Reading:** the first finished Garage run of wm2 (v11 – v17 with wm2_self1: none). What it does NOT show: how much
    of it is the value head / the new cost, since the pasted part is driven by the reflex. A and B differ only in the
    cost of the plan, so the difference at the start is the only evidence so far. Which map, and the start of both
    logs: asked from the user.
  - **The user's answers:** run #1258 was the first map, the oval, one lap of ~80 m. Seen from the start: the car goes
    to the left wall, turns right at the last moment, and repeats. The user's reading, shared by Claude: the model did
    not carry this ride. Why: the simple rule steers into the deepest opening (on a left-turning oval that is along
    the inner wall) and pushes away only under 0.5 m (`SIDE_MIN`); with `REFLEX_M` 0.7 it keeps the car inside its own
    take-over distance, so the model never steers again.
- **Controller v19 (12 Nov, the user: "write new controller"):** one change, the reflex only as an emergency:
  `REFLEX_M` 0.7 → 0.4 m, `FRONT_REFLEX_M` 1.0 → 0.6 m. Everything else as v18 (kept as
  `ail_cup/claude_controller_v18.py`). `check_controller`: costs equal to the torch model (0.001), calls 88 – 91 ms.
  Risk: one decision per ~0.75 s, 0.9 m driven between two decisions at 1.2 m/s. **Result: open** (upload:
  `wm2_c2_value_garage.npz` + `claude_controller.py`).
- **Run #1259 (v19, the oval): DNF, hit the wall at ~7 s** (0.0437 Wh, `step()` p95 185 ms). The model steered
  ("brain" in the log): 0 – 3.5 s straight while the track bends left (forward direction +2° → +19°, right wall 1.28 →
  1.00 m; costs −0.40 / −1.23 / −0.39); at 4.0 s it chose left (−0.11 / −0.61 / −0.93), which is right; the wheel then
  stayed at +0.8 until the next plan and the car swung to 77° LEFT of the track by 5.0 s; the track guard steered −1.0,
  speed fell to 0.5 m/s; at 6.5 s the model chose +0.8 again with the left wall at 0.48 m → crash. The user saw the
  same in the replay (4.1 s straight, 4.6 s pointing across the track): "the model makes the right decision but the
  steering angle is too big".
- **Cause:** three options (−0.8 / 0 / +0.8) held until the next plan, ~0.75 s later: full lock at 1.2 m/s turns the
  car about 90° per second, and the plan behind the choice imagined only 0.5 s of it. The reflex of v18 had hidden this.
- **Controller v20:** the options are the real wheel angle (rounded to 0.05) − 0.25 / + 0 / + 0.25 (7.5° steps, within
  ±0.8) instead of full lock. v19 kept as `ail_cup/claude_controller_v19.py`. `check_controller`: costs equal to the
  torch model, calls 88 – 91 ms. Weak point: full lock now needs 3 plans (~2 s); in a tight corner the emergency
  reflex and the track guard have to cover that. **Result: open.**

### E23 — Why wm2 swerves in the Garage: it misreads wide corridors (12 Nov 2026, evening, no Garage needed; its "front" numbers are corrected in E25)
- **Trigger:** the user, after v20: "let's make a controller from scratch, because this problem is still chasing me".
- **Tool (new):** `ail_cup/garage_sim.py`, a stand-in for the Garage on this PC: `Controller(init)` / `step(obs)` with the
  keys of run #1259, dt 0.05 s of sim time per call, a kinematic bicycle, 109 lidar beams, made-up tracks (an oval of
  79.5 m × 2.57 m like the Garage's first map, a wavy loop, a hairpin loop). Not the Garage (no slip, guessed lags).
- **1. The stand-in reproduces the fault.** v20 on the oval's first STRAIGHT (walls 1.29 m on both sides, wheel straight):
  the first plan prefers "a step right" (costs −0.64 / −0.41 / −0.32) and the car zigzags 40° off the track's direction
  within 2 s; 55.6 m in 60 s, wheel moving 74°/s on average.
- **2. The cost parts in that symmetric scene** (controller v20's own code): without the value term −0.076 / 0.180 /
  0.214; the value head says 4.76 / 4.19 / 3.51 m for right / straight / left; the imagined walls after 0.5 s of
  "straight" are right 1.20 m, left 0.83 m. So the pull to the right is in the heads' reading, not in the steering size.
- **3. The wall heads on the REAL frame, no imagination** (wm2_c2, straight corridor, car in the middle, 1.2 m/s):

| True distance to each side wall (m) | Head: right | Head: left | Front sector: true → head |
|---|---|---|---|
| 0.50 | 1.19 | 0.89 | 3.83 → 2.34 |
| 0.65 | 0.68 | 0.77 | 4.98 → 1.85 |
| 0.80 | 0.75 | 0.80 | 6.13 → 1.66 |
| 1.00 | 0.87 | 0.94 | 7.66 → 2.73 |
| 1.29 (the Garage oval) | 1.39 | 0.79 | 9.88 → 3.25 |
| 1.60 | 1.48 | 1.03 | 10.00 → 3.52 |

  A real sim frame for comparison (compete): true right 1.99, left 1.11 → head 1.91, 0.98. wm2_c1 and wm2_self2 show
  the same kind of error at 1.29 m (c1: 1.37 / 0.97; self2: 0.79 / 1.05).
- **Finding:** the model reads the side walls right only at 0.65 – 1.0 m (the width of `practice` and `compete`) and never
  reads a free road longer than ~3.5 m. The Garage oval (1.29 m to each side, straights of 7 – 10 m) is outside what it
  learned, and there it is wrong by up to 0.5 m, not even symmetrically. v18 – v20 steered away from a wall that was not
  there. No controller around the model can repair this; the heads were also never trained on mirrored frames
  (`wm2/train.py` mirrors windows for the predictor, `wm2/heads.py` and `wm2/value.py` do not).
- **Consequences:** (1) the Garage controller must steer from the real lidar every call and use the model only in scenes
  like its training; (2) the unseen-track question of the project has a first concrete answer: wm2 does not transfer to
  a track of another width; candidates to fix it: mirrored training of the heads, scale changes of the lidar in
  training, or synthetic tracks of many widths. Not done.

### E24 — A new Garage controller: the lidar steers, wm2 advises (12 Nov 2026, evening) — no Garage run yet
- **Decision (the user):** "let's make a controller from scratch". With E23, the design follows from the finding: the
  steering must come from the real lidar every call; the model may only advise where a scene is like its training.
- **`ail_cup/wm2_controller.py` v1** (pure Python, one file): every call (1) the free road in every direction and the
  nearest wall on each side from the real scan; (2) one smooth steering law: towards the longest free road (`K_GAP`
  0.6 per 30°) + to the middle of the corridor (`K_CENTER` 0.8 per m) + away from a wall closer than 0.55 m; the command
  moves at most 3° per call, and there are no modes that switch; (3) speed 0.6 – 1.6 m/s by the free road ahead and the
  wheel angle; (4) wm2's advice ("a step right / as now / a step left", 8 imagined steps + the value head) is added as a
  push of 0.15 that fades in 0.7 s, and a plan starts only when both side walls are 0.55 – 1.10 m away and the free road
  is under 4.5 m; (5) the map only guards against driving backwards. The network code is copied unchanged from
  `claude_controller.py` (checked against torch in E22).
- **In the stand-in** (`python -m ail_cup.garage_sim ail_cup.wm2_controller <track> ail_cup/wm2_c2_value_garage.npz`):

| Track | Direction | Result | Closest wall m | Mean speed m/s | Wheel movement °/s |
|---|---|---|---|---|---|
| oval (79.5 m × 2.57 m) | ccw / cw | lap in 50.1 / 50.4 s | 0.88 / 0.88 | 1.60 / 1.59 | 3.9 / 5.2 |
| wavy | ccw / cw | lap in 42.9 / 42.9 s | 0.73 / 0.72 | 1.54 / 1.54 | 14.2 / 14.2 |
| hairpin (2.2 m radius) | ccw / cw | lap in 30.7 / 31.3 s | 0.56 / 0.56 | 1.50 / 1.47 | 10.5 / 12.2 |

  For comparison on the oval: v19 60.2 m in 60 s (wheel 49.5 °/s), v20 55.6 m in 60 s (74.3 °/s), neither finished.
  Settings tried: `K_CENTER` 0.45 / 0.8 / 1.2 and `K_GAP` 0.9 / 0.6 (all finish; 0.8 with 0.6 is the smoothest and keeps
  the most room); top speed 2.2 m/s also finishes (oval 36 s) but is not the default: the Garage's grip is unknown.
- **What the model does in it: almost nothing.** On these wide tracks no scene counts as "known": 0 – 1 plans per lap,
  advice used in 0 % of the calls. So a Garage result of this controller is a result of the lidar law, not of wm2.
- **Limits of the check:** the stand-in has no tyre slip and its tracks are made up; it catches control-loop faults
  (it reproduced v20's swerve), not grip or the real map shapes.
- **Garage run #1261 (the user, 12 Nov; `wm2_controller.py` v1 + `wm2_c2_value_garage.npz`, the oval): finished in
  50.78 s**, 0.2103 Wh, `step()` p95 10 ms. The stand-in had said 50.1 s. In the pasted part (31 – 50.5 s): steering
  −0.04 … +0.16, speed 1.46 – 1.63 m/s, nearest side wall 0.74 m, the car within 11° of the track's direction. `p0 m0%`:
  the model made no plan and gave no advice in the whole run. Against v18 (run #1258): 68.81 s and 0.4259 Wh.
- **Reading:** the swerve is gone and the lap is 18 s faster on half the energy, but it is the lidar law's lap. The
  stand-in predicted the lap time to 0.7 s, so it is good enough to tune speed and steering before an upload.

### E25 — The corridor test: a judge for widths the model did not train on (13 Nov 2026, night)
- **Decision (the user):** "Step 3 first: the test. Without it we cannot tell whether anything helped."
- **What:** `autoresearch/eval/corridor_test.py` (+ 5 tests of its geometry in `autoresearch/tests/test_corridor_test.py`).
  189 scenes: a corridor with 0.5 / 0.65 / 0.8 / 1.0 / 1.3 / 1.6 / 2.0 m from the middle to each wall; straight, or an
  arc of 4 m radius to the left or to the right; the car in the middle or 30 % of the half width to either side;
  heading along the corridor or 10° off. The scan the car would see is drawn (109 beams, 270°, 10 m), the truth of the
  13 sectors is made as in `wm2/heads.py`, and the wall heads read the encoder's z of that one frame (1.5 m/s, wheel
  straight). No sim, no graphics card, no recorded data. "Known" widths = 0.65 – 1.0 m (the training tracks).
- **Baseline:**

| Model | Side error, known widths | Side error, NEW widths | Front error, known | Front error, new | Mirror gap |
|---|---|---|---|---|---|
| wm2_c2 | 0.13 | **0.59** | 0.13 | 0.45 | 0.38 |
| wm2_c1 | 0.11 | **0.57** | 0.13 | 0.57 | 0.44 |
| wm2_self2 | 0.12 | **0.60** | 0.11 | 0.64 | 0.52 |
| wm2_self1 | 0.10 | **0.55** | 0.11 | 0.64 | 0.42 |

  Side error of wm2_c2 by half width: 0.50 m → 0.44; 0.65 → 0.10; 0.80 → 0.13; 1.00 → 0.17; 1.30 → 0.26; 1.60 → 0.61;
  2.00 → 1.06 (the other three models within 0.07 of these).
- **Findings:** (1) inside the known widths every model is right to ~0.1 m, also on these drawn scans; outside the
  error grows with the distance from the known range, to ~1 m at 2 m; too narrow (0.5 m) is misread as well.
  (2) No model is better than another here: more training on the same two tracks did not help at all (0.55 – 0.60).
  (3) The models are not mirror-symmetric: the same scene flipped gives an answer 0.4 – 0.5 m different.
- **Correction of E23 (Claude's mistake):** E23 says the model reads "a free road of 10 m as 3.3 m". That compared the
  nearest beam within ±7.5° (9.88 m) with the head's front value, which is the nearest wall in the three front sectors
  (±22.5°): in that corridor its truth is 1.29 / sin 22.5° = 3.4 m, so 3.25 m was nearly right. The front column of
  E23's table is wrong in the same way. The side-wall numbers of E23 stand, and this test confirms them. The front
  error is real but smaller: 0.45 m on new widths against 0.13 m on known ones.
- **The number to beat:** side error on new widths 0.59 m, mirror gap 0.38 m (wm2_c2), without losing the 0.13 m on
  known widths. Next: step 1, training with stretched and mirrored scans.

### E26 — `wm2_c3`: training with stretched and mirrored scans (12 – 13 Nov 2026) — done, not driven yet
- **Decision (the user, 13 Nov):** step 1 of the plan after E25: "now you can start training".
- **Idea:** show the same recorded drives as drives in a world 0.6 – 2.2 times bigger, so the model meets other
  corridor widths without new data, and flip them left ↔ right for the heads too.
- **Code:** `wm2/data.py` `windows(..., scale)`: every lidar distance and the speed × k, the throttle × k (steady speed is
  about proportional to it), the wheel angles through `scale_steer` (the same path needs tan(angle) / k), a beam at the
  lidar's range stays there, a beam pushed beyond it is cut. `WM2_SCALE=0.6,2.2` switches it on for `wm2.train` (70 % of
  the windows, log-uniform k; validation unscaled), `wm2.heads` and `wm2.value` (every epoch re-encodes all moments
  with new scales and mirrors; wall answers come from the changed scan, progress and value answers × k). Default: off,
  so every old command trains as before. Tests: `autoresearch/tests/test_scale_augmentation.py` (5).
- **Why it is needed, on one validation batch of wm2_c2:** imagination error / copy error 0.013 unscaled, 0.638 in a
  world 1.6 times bigger, 2.421 at 2.2 times (worse than "nothing changes").
- **Known weak points of the idea:** the shapes stay those of the two tracks; acceleration and the car's size do not
  scale like the world; 3 epochs of fine-tuning may be too little for the encoder to learn other scales.
- **Run** (`night/c3_train.sh`, logs `night/c3_*.log`): `wm2_c2 → wm2_c3`, `data/wm2_c2`, 3 epochs, lr 1e-4, then sector
  and walls5 heads and a balanced value head with the same augmentation, then `corridor_test wm2_c2 wm2_c3`.
- **To beat (E25, wm2_c2):** side error on new widths 0.59 m, mirror gap 0.38 m, keeping ~0.13 m on known widths.
- **Result (the run finished 12 Nov 23:31; logs `night/c3_{train,heads,value,corridor}.log`, files `runs/wm2_c3/`):**
  - **Training** (`night/c3_train.log`, `runs/wm2_c3/loss.csv`): 1,759,225 train / 219,264 val windows, 3 epochs of
    ~26 – 29 min. val pred/copy 0.044 → 0.039 → **0.038**, spread 0.992 → 0.988 (no collapse). wm2_c2 ended at 0.031 on
    the same `data/wm2_c2` val set (E21), so the unscaled imagination is **a little worse** (0.038 against 0.031).
  - **Heads on the test set** (`night/c3_heads.log`; the test answers are made with the same augmentation): walls
    0.06 – 0.13 m by sector (walls5: front 0.12, right 0.06, left 0.07, diagonals 0.08), danger right in 87.1 % (always
    "danger" = 42.6 %), progress error 0.9 cm per step of 6.5 cm. About as wm2_c2 (E21: walls 0.04 – 0.11 m, 88.2 %, 1.0 cm).
  - **Value head** (`night/c3_value.log`, balanced): 1,726,953 train moments with an answer (94.3 %), mean 3.76 m. Test:
    mean error 0.86 m (always the mean: 1.06 m), correlation 0.69. < 1 m driven → says 0.89 m; > 4 m → says 4.82 m.
    Worse than the value head of E19 (0.69 m, correlation 0.74), but its answers are × k in the stretched worlds, so the
    two test sets are not the same.
  - **Corridor test** (`night/c3_corridor.log`, the 189 scenes of E25):

| Model | Side error, known widths | Side error, NEW widths | Front error, known | Front error, NEW | Mirror gap |
|---|---|---|---|---|---|
| wm2_c2 | 0.13 | 0.59 | 0.13 | 0.45 | 0.38 |
| **wm2_c3** | **0.07** | **0.10** | **0.07** | **0.06** | **0.19** |

| Model | 0.50 m | 0.65 m | 0.80 m | 1.00 m | 1.30 m | 1.60 m | 2.00 m |
|---|---|---|---|---|---|---|---|
| wm2_c2 | 0.44 | 0.10 | 0.13 | 0.17 | 0.26 | 0.61 | 1.06 |
| wm2_c3 | 0.05 | 0.07 | 0.07 | 0.07 | 0.10 | 0.11 | 0.13 |

- **Conclusion:** every target of E25 is beaten: side error on new widths 0.59 → **0.10 m** (6 ×), front 0.45 → 0.06 m,
  the mirror gap halved (0.38 → 0.19 m), and the known widths got better too (0.13 → 0.07 m). At the Garage oval's
  1.29 m the side error is now about 0.10 m (wm2_c2: 0.26 m, E23 saw up to 0.5 m on one side).
- **What it does NOT show:** (1) the corridor test reads only the heads on one drawn frame, not the imagination over
  many steps nor a drive; (2) the unscaled val imagination is a little worse (0.038 against 0.031), so the sim drives
  of E21 may not hold; (3) the mirror gap of 0.19 m is still not zero; (4) the drawn tracks have the same simple shapes
  as the test of E25, not the AIL Cup maps.
- **Next:** sim test drives of wm2_c3 with the E20 / E21 setting on both tracks (does it keep wm2_c2's result?), and an
  AIL Cup export: with readings right at 1.29 m, wm2's advice in `ail_cup/wm2_controller.py` may now be allowed in
  wider scenes than 0.55 – 1.10 m (E24).

### E27 — A simple numpy controller for the AIL Cup (13 Nov 2026) — built, `choose()` not written yet, no Garage run
- **Decision (the user, 13 Nov):** "build simple controller with numpy". The user's design: encode → N imagined plans →
  choose the best → drive, every call, no reflexes or modes. Why: the pure-Python controllers needed 1.18 s per plan
  (E22) and spread it over ~15 calls, and their extra rules were patches over wrong wall readings (E23), which wm2_c3
  now reads right on drawn corridors (E26). `make_controller.py` says the Garage allows numpy; not yet confirmed
  (question 12 in `goals/communication.md`).
- **Code:** `ail_cup/simple_controller.py` (the model in numpy: `Wm2`; 9 steering × 3 speed options = 27 plans, each held
  for 20 steps = 1 s; the cost settings of E20 / E21 as constants; `choose()` is the user's part) and
  `ail_cup/check_simple.py` (numpy against torch + time). Model file: `ail_cup/wm2_c3_value_garage.npz` (export of
  13 Nov, 56 arrays, 3.4 MB).
- **Check** (`python -m ail_cup.check_simple wm2_c3`, 4 real frames of `data/self6_compete/drive_0005.npz`): numpy
  (float16 weights) against torch (float32): z within 0.005, imagined z 0.009, speed 0.0007 m/s, walls 0.025 m,
  progress 0.0002 m, value 0.01 m.
- **Time of the model part of one call** (this PC; the Garage was as fast as this PC for the pure-Python controller,
  `claude_controller.py`): 93 ms → 87 ms (the last transformer layer only for the newest moment, the same numbers)
  → **73 ms median, 85 ms slowest** (the tanh form of GELU instead of an erf written in numpy). Cap 250 ms. Under
  50 ms would need fewer plans or a shorter plan.
- **Result:** open (`choose()`, then `garage_sim` on oval / wavy / hairpin, then the Garage).

### E28 — What the training data lacks (13 Nov 2026, no sim, no GPU)
- **Question (the user):** "analyse my dataset and give me variants how to make it more different".
- **Setup:** `autoresearch/data_report.py` over the 12 trip folders (every frame for actions / speed / position,
  every 5th frame for the lidar). Output: `docs/data/data_report_2026-11-13.txt`.
- **Result:** ~44 h of trips. Problems P1 – P9 in `docs/data_problems.md`, the main ones: only 0 – 4 % of frames faster
  than 2 m/s (P1); each track in one direction (practice cw, compete ccw, P2); self-driven trips see a 0.5 m spot from
  1.5 – 2.5 of 8 headings, random data 4 – 6 (P3); practice : compete ≈ 4 : 1 in `wm2_c2`, the newest compete trips
  only 29 min (P4); 17 – 23 % of self-driven compete frames slower than 0.5 m/s (P5); closest wall < 0.2 m in 0 – 2 %
  of frames (P6); `clean_practice` (791 min) unused (P8).
- **Conclusion:** the self-driving rounds made the data bigger, not wider; speed is the biggest gap for week 2.
  Proposed solutions A1 – C1 in `docs/data_solve.md`; none decided yet (question 13 in `goals/communication.md`).

### E29 — Does the synthetic sim match the real one? `practice` rebuilt (14 Nov 2026, no sim, no GPU)
- **Question (the user):** "create a road that is the same as practice to check if the synthetic sim works properly".
- **Setup:** `autoresearch/synth/practice_check.py`: the `practice` road rebuilt in the format of
  `autoresearch/synth/tracks.py` from 120 `clean_practice` trips (one lap = centreline, real lidar hits = the walls);
  then (1) at the poses of 40 OTHER trips the synthetic scan vs the recorded scan, (2) from real poses the real
  commands replayed with the synthetic car for 1 s / 2 s. Picture: `docs/data/synthetic_v1/practice_check.png`.
- **Rebuilt road:** 30.8 m, width 1.23 – 2.34 m (mean 1.56), tightest corner 1.55 m.
- **Lidar:** first try median |error| 0.084 m, bias −0.156 m (synthetic walls too close). Two causes found:
  the wall fit took the near edge of the ~0.3 m thick band of real hits (now: the middle of the band), and the lidar
  sits **0.20 m ahead** of the car's position point (grid 0 – 0.30 m; 0.15 – 0.30 m all ~0.040; time lag between
  pose and scan: none, lag 0 best). Now: **median |error| 0.040 m, 77 % of beams within 10 cm, bias +0.008 m.**
- **Car:** the bicycle geometry with the real speed and wheel angle: 0.017 m after 1 s, 0.032 m after 2 s
  (`clean_practice`); 0.107 m after 2 s on `fast_practice` = tyre slip at speed, which the synthetic car does not have.
  Fitted on real trips: speed lag 0.29 s (guess 0.25), wheel rate limit 3.25 rad/s (guess 3.5; a first-order lag
  fits worse), braking at throttle 0 ≈ −6 m/s². Throttle per m/s: 0.040 is best on `clean_practice` (2 s error
  0.055 m), **0.0428** on `fast_practice` (0.358 m against 0.406 m); 0.0428 is kept (synthetic trips use the fast
  random throttle). From the real commands now: 0.081 m after 1 s, 0.191 m after 2 s (clean_practice).
- **Conclusion:** the synthetic lidar is right to ~4 cm, so the geometry is good enough for the encoder and the wall
  heads. The motion is right over ~1 s but not at speed (no slip), which confirms the rule of
  `docs/track_generator.md` section 6: synthetic trips never train the predictor.
- **Changed:** `autoresearch/synth/collect.py`: `LIDAR_FORWARD = 0.20`, crash = front (lidar) or middle of the car
  closer than 0.17 m, `SPEED_LAG 0.29`, `STEER_RATE 3.25`, `THROTTLE_PER_MS 0.0428`.

### E30 — `wm2_c4` (balanced sampling, D18) in the sim + the speed ladder B1 (night 13 → 14 Nov 2026)
- **Decision (the user, 13 / 14 Nov):** A1 + A2 in the next training; then sim tests without the user and B1 the
  same night. Scripts: `night/c4_train.sh` (training), `night/ladder_night.py` (tests + ladder). Progress lines with
  every number: `night/state_compete.txt` (13 – 14 Nov). Logs: `night/c4_*.log`, `night/compete_test_*.log`,
  `night/compete_ladder_*.log`.
- **Training** (`night/c4_train.log`, `runs/wm2_c4/loss.csv`): `wm2_c3 → wm2_c4`, data `data/wm2_c4` (= the trips of
  c2 + track / source per moment), 3 epochs, lr 1e-4, `WM2_SCALE=0.6,2.2`, `WM2_BALANCE=1`. compete 19.0 % of the
  windows → 50.0 % of the draws. val pred/copy 0.043 → 0.039 → **0.038** (wm2_c3: 0.044 → 0.039 → 0.038), spread
  1.012 → 1.007. Epochs 1763 / 1714 / 1646 s.
- **Heads** (`night/c4_heads.log`): walls5 front 0.13, right 0.06, left 0.07, diagonals 0.09 m (c3: 0.12 / 0.06 / 0.07 /
  0.08); danger right 87.0 % (c3 87.1 %); progress 0.9 cm per step. **Value head** (`night/c4_value.log`, balanced):
  mean error 0.87 m, correlation 0.69 (c3: 0.86, 0.69).
- **Corridor test** (`night/c4_corridor.log`): side error known / new widths c3 0.07 / 0.10, **c4 0.07 / 0.11**; front
  known / new c3 0.07 / 0.06, **c4 0.11 / 0.07**; mirror gap c3 0.19, c4 0.20. Offline: no gain, front a little worse.
- **Sim tests** (5.5 min each, gas 0.12, `WM2_COST=FRONT_BASE=0.25,W_CENTER=1.0,CENTER_CLAMP=0.3,SIDE_MARGIN=0.45,
  W_VALUE=0.3`, each model with its own value head; logs `runs/<model>/drive_<track>_c4night_{a,b}.csv`):

| Model | compete a / b (crashes/min) | compete mean | speed | practice a / b | practice mean | speed |
|---|---|---|---|---|---|---|
| wm2_c2 (E21) | 0.6 / 1.1 | 0.85 | – | 0 / 0 | 0 | – |
| wm2_c3 | 4.06 / 4.33 | **4.2** | 1.56 | 0.00 / 0.18 | 0.09 | 1.69 |
| **wm2_c4** | 1.45 / 0.36 | **0.9** | 1.73 | 0.00 / 0.36 | 0.18 | 1.62 |

- **Speed ladder B1** (`data_collection.collection_random`, the random / gap / safety driver, `COLLECT_THROTTLE_TOP`;
  30 min per step, ~20 min of it driving; folders `data/speed_<track>_t<ceiling>/`, 380 trips, ~700 MB):

| Track | ceiling | trips | crashes/min | mean speed | max | > 2 m/s | > 3 m/s |
|---|---|---|---|---|---|---|---|
| compete | 0.15 | 53 | 1.28 | 1.34 | 3.63 | 17 % | 2 % |
| compete | 0.20 | 51 | 0.84 | 1.46 | 4.54 | 23 % | 4 % |
| compete | 0.25 | 53 | 1.51 | 1.43 | 4.79 | 21 % | 3 % |
| compete | 0.30 | 49 | 1.29 | 1.50 | 4.73 | 24 % | 5 % |
| practice | 0.15 | 44 | 0.09 | 1.20 | 3.67 | 9 % | 1 % |
| practice | 0.20 | 43 | 0.00 | 1.24 | 4.30 | 12 % | 1 % |
| practice | 0.25 | 43 | 0.19 | 1.27 | 4.62 | 13 % | 1 % |
| practice | 0.30 | 44 | 0.05 | 1.27 | 4.51 | 13 % | 1 % |

- **Findings:** (1) **The scale augmentation alone (c3) made compete 5 × worse** (4.2 against 0.85 crashes/min), although
  the corridor test said it was much better (E26): the corridor test is not enough as a judge. (2) **The balanced
  sampling (c4) brought compete back to c2's level** (0.9), a little faster (1.73 m/s), with the corridor gain of c3
  kept. Practice slightly less safe than c2 (one drive 0.36). No model is crash-free on compete yet. (3) The ladder is
  safe at every step (never above 1.51 crashes/min, the stop limit 4 was not reached), but **a higher ceiling barely
  makes faster data**: frames > 2 m/s 17 → 24 % on compete, 9 → 13 % on practice, > 3 m/s at most 5 %. Likely why:
  `data_collection/policy.py` caps random steering at 0.12 (`RANDOM_THROTTLE`) and the safety driver slows down near
  walls, so only gap mode on straights uses the ceiling. P1 (no fast driving) is only partly solved.
- **Night problems:** the first start (02:02) did not connect: the sim opened almost off the screen and the clicks of
  `night/open_sim.ps1` assumed a 39 px title bar (~72 px at 125 % scaling). Fixed (window moved, clicks from the
  client area); Claude's manual re-opens closed the sim and ended the first run at 02:16; restart 02:17 ran clean,
  including the switch to practice.
- **Next:** wm2_c4 is the best model now (compete like c2 + better corridor). For fast data, raise or drop the
  random-steering cap in the ladder, or speed-scaled gap driving. Train on the ladder trips (`data/speed_*`).

### E31 — Where and why the car crashes on `compete`: a wrong turn at a fork (14 Nov 2026, no sim)
- **Question:** the crash analysis of the 13 Nov plan: where and why do wm2_c2 / c3 / c4 still crash on `compete`?
- **Tool:** `autoresearch/eval/crash_report.py` (new): every crash of a drive log with place (metres since the start /
  lap line: the sim puts the car back at the start after a crash), speed, steering, closest wall, and the imagined
  distance to that wall for the last 0.5 s (`<log>_imagined.csv`). Map pictures: `docs/data/compete_crashes.png`,
  `docs/data/compete_corner.png` (walls from 120 `fast_compete` trips, paths of `data/self3-6_compete`).
- **Result 1, one place:** every crash of all three models is 8 – 16 m after the start of the ~47 m lap:
  wm2_c2 9 of 9 (E21 logs), wm2_c3 46 of 46, wm2_c4 10 of 10 (E30 logs). The self-driven trips with positions agree:
  median 13.7 m from the start to the crash, crash points around (−2.6, 4.4) and (−2.8, 6.8).
- **Result 2, not a surprise to the model:** in only 1 of the 10 wm2_c4 crashes was the real wall ≥ 0.3 m closer than
  imagined (wm2_c3: 4 of 46). Speed going into the place does not separate crashes from passes either (wm2_c2: crash
  passes 2.24 m/s, passing 2.59; wm2_c4: 2.81 against 2.65).
- **Result 3, the cause:** at (−2.5, 4.7) the middle channel ends and the road opens into a fork: the real road goes on
  down-left (all 25 passing paths drawn), a branch goes up-left between the outer wall and the divider and narrows into
  a wedge. Most crashing paths turn up-left into that branch and hit the wall at ~(−2.8, 6.8); a few hit the end of the
  divider at ~(−2.6, 4.4). Seen from the fork the wrong branch looks like an open corridor, and the planner has no idea
  of the road's direction: it picks the plan with the most free road ahead.
- **Reading:** a fault of the planner / route, not of the model or of the data. That is why D18 / scale / more laps did
  not change the crash rate (c2 0.85, c4 0.9 crashes/min). E15's left turn (x −0.87, y 9.33, 12 Nov) was an earlier,
  now solved place (the top U-turn; no crash there in the E30 logs).
- **Next:** offline replay at the fork (like E15): the cost of "down-left" against "up-left" plans, term by term, with
  the value head at its weight 0.3 and higher; then a sim test of the best setting (the user's go). Other options: a
  value head trained with more weight on the wedge trips; a heading memory (keep turning the way the road went).

### E32 — Does a heavier value head fix the fork of `compete`? (14 Nov 2026, no sim) — no
- **Decision (the user):** "start option 1" of E31: check a higher `W_VALUE` offline first.
- **Setup:** `autoresearch/eval/fork_replay.py` (new). 407 real moments from `data/self3-6_compete` (trips from the start
  that reached the fork): ±0.5 s around the closest point to the fork (−2.4, 4.8), every 2nd frame; 286 from trips that
  went on down the road ("passed"), 121 from trips that crashed in the wedge. Model wm2_c4 with its walls5 heads and value
  head; the driver's grid (9 steering values held for 1 s × gas 0.04 / 0.08 / 0.12, + `W_SWITCH` 0.3 × steering change)
  and its cost (`FRONT_BASE=0.25,W_CENTER=1.0,CENTER_CLAMP=0.3,SIDE_MARGIN=0.45`); the cheapest plan per moment.
- **What the real cars did** (mean steering of the next 0.5 s): passed +0.31 (10 % right), wedge −0.64 (98 % right):
  at the fork LEFT follows the road, RIGHT goes into the wedge.
- **Cheapest plan, right / left** (right = the wedge side):

| W_VALUE | passed moments | wedge moments | all |
|---|---|---|---|
| 0.0 | 26 % / 53 % | 60 % / 29 % | 36 % / 46 % |
| 0.3 (the driver now) | 23 % / 58 % | **55 % / 33 %** | 32 % / 51 % |
| 1.0 | 22 % / 62 % | 51 % / 38 % | 31 % / 55 % |
| 2.0 | 23 % / 62 % | 46 % / 41 % | 30 % / 56 % |
| 3.0 | 24 % / 63 % | **44 % / 43 %** | 30 % / 57 % |

- **Why so little:** the value head itself is wrong where it matters. Mean value (metres it expects in the 3 s after the
  plan): passed moments right plans 1.44, left 2.11 (right); **wedge moments right 1.56, left 1.21 — it prefers the wedge**.
- **Conclusion:** option 1 does not fix the fork (at 10 × the weight still 44 % of the wedge moments choose the wedge).
  The value head does not know that the wedge is a dead end. Next: option 2 (train the value head so that it does,
  checked with this replay: in wedge moments value(left) must beat value(right)), or option 3 (a heading / route memory).

### E33 — The numpy controller with wm2_c4 / wm2_c3 in the local Garage stand-in (14 Nov 2026, no sim)
- **After D19** (focus on the AIL Cup with numpy). Controller `ail_cup/simple_controller.py` (E27: every call encode →
  27 imagined plans (9 steering × 0.8 / 1.2 / 1.6 m/s, 1 s) → `choose()` with the sim driver's cost of E20 / E21 →
  drive). Model files: `ail_cup/wm2_c4_value_garage.npz` (export 14 Nov, 3.4 MB, 56 arrays) and
  `ail_cup/wm2_c3_value_garage.npz`. numpy vs torch for wm2_c4 (`ail_cup.check_simple`): z 0.004, imagined 0.014,
  walls 0.10 m (readings up to 4.9 m), value 0.012; model part of a call median 73 ms, slowest 84 ms.
- **Test:** `ail_cup/garage_sim.py` (kinematic car, 109-beam lidar, made-up tracks: oval 80 m / 2.57 m wide like map 1,
  wavy 66 m, hairpin 46 m with 2.2 m corners), each direction, 120 s; results `night/garage_sim_c4c3.log`.

| Model | oval ccw / cw | wavy ccw / cw | hairpin ccw / cw | closest wall | slowest call |
|---|---|---|---|---|---|
| wm2_c4 | lap 50.6 / 50.2 s | 41.4 / 41.8 s | 29.7 / 29.8 s | 0.48 m | 189 ms |
| wm2_c3 | lap 51.2 / 50.4 s | 42.5 / 41.8 s | 29.0 / 29.8 s | 0.42 m | **291 ms** |

- **Result:** 12 of 12 laps, no crash; mean speed 1.55 – 1.59 m/s (the fastest option is 1.6 m/s). wm2_c4 keeps a little
  more room on the hairpin (0.48 – 0.62 m against 0.42 – 0.46 m). Call time median 54 – 142 ms on this PC; one wm2_c3
  call took 291 ms (over the Garage cap of 250 ms) while the median of that run was 142 ms: the PC was busier then.
- **Not shown:** the real Garage (other car physics, the real maps, its CPU). The cap is the main risk: fewer plans or a
  shorter plan would give a margin.
- **For the Garage:** controller `ail_cup/simple_controller.py` + model `ail_cup/wm2_c4_value_garage.npz`.

### E34 — The numpy controller in the real Garage: two DNFs on the latency cap, and why (14 Nov 2026, no sim)
- **Runs (pasted by the user):** #1281 (27 plans x 20 steps): DNF "did not answer within the latency cap" on call 4,
  the first plan (calls 1 – 3 only encode, so p95 was 9.5 ms). #1283 (the user's file, 27 x 5 steps): DNF at 11 s,
  "control step took 275 ms > cap 250 ms", p95 175 ms, while step()'s own timer showed <= 80 ms in most half-seconds.
- **Causes found (on this PC):**
  1. Plan size: 27 x 20 steps = ~115 ms here, ~3x more in the Garage. 27 x 5 = ~25 ms here (the user's choice).
  2. OpenBLAS threads: these small matrices run slower on many threads. `garage_sim` oval, 27 x 5: mean call 85 -> 37 ms,
     slowest 157 -> 86 – 108 ms with 1 thread; 1016 calls, none > 150 ms. The `OPENBLAS_NUM_THREADS` variable comes
     too late (numpy is loaded before the controller), so the controller calls OpenBLAS's `set_num_threads(1)` itself.
  3. Likely: `print` inside step() (1 call in 10) waits for the Garage to read the log; this is outside step()'s own
     timer and matches p95 175 ms and the DNF right after a status line. Not proven (no Garage access).
- **Fix (`ail_cup/simple_controller.py` v2):** HORIZON 5; 1 BLAS thread via ctypes; status lines printed by a background
  thread (stopped cleanly at exit); a deadline in `imagine()` (no new step after half the cap, at least 2 steps);
  one warm-up plan at INIT. Stand-in: oval lap 50.5 s, closest wall 0.53 m, mean call 37 ms, clean exit.
- **To check in the next Garage log:** the INIT line "BLAS threads = 1 (...)" and p95 well below 175 ms.
- **Correction (same day): v2 was rejected by the Garage upload check** ("import of 'atexit' is not allowed: only maths
  modules and numpy can be used"). Only `math`, `time`, `numpy` are allowed: no threads, no ctypes, no os.
  **v3** (replaces v2): no background printer and no ctypes. The 1-thread effect comes from the shape of the products:
  OpenBLAS uses threads only for big products, and `x @ w` on (plans, moments, in) runs one small product per plan
  (the 2-D "one product" change of v2 made them big, so it is undone). Prints every 20 calls (1 s) instead of 10,
  and each status line shows how long the previous print took (`printNms`), to test cause 3 in the Garage.
  Stand-in (default threads): oval lap 50.8 s, mean call 34 ms, slowest 118 ms; hairpin lap 29.4 s, slowest 61 ms.
  A cache of the action codes was tried: no difference beyond this PC's noise, not kept.
- **v4 (the user's request, same day):** no prints inside step() at all (status line, beam line, print timer and their
  counters removed); INIT prints only the version and the model. Stand-in oval: lap 50.8 s, mean call 26 ms,
  slowest 44 ms (v3: 34 / 118 ms; same lap, so the driving did not change).
- **Run #1287 (v3): DNF at ~16 s, 265 ms > cap, Garage p95 190 ms.** `print0ms` on every line: prints are NOT the cause.
  step()'s own slowest per second was 78 – 120 ms, so the Garage adds ~70 – 145 ms outside step(). Speed 0.8 was
  never chosen. **v5:** SPEED_OPTIONS (1.2, 1.6) = 18 plans, BUDGET_SHARE 0.3 (rollout deadline 75 ms). Stand-in: oval
  50.8 s, hairpin 29.4 s (same laps), mean call 15 ms, slowest 22 – 23 ms (v4: 26 / 44).

### E35 — The numpy controller completes every AIL Cup road (14 Nov 2026, the user's report in the chat)
- **Result:** `ail_cup/simple_controller.py` (v5 line: 18 plans x 5 steps, 75 ms rollout budget, no prints while
  driving) + `wm2_c4_value_garage.npz` finished all AIL Cup maps in the Garage, with the model as the only driver
  (no safety layer). Times and run numbers: not pasted yet.
- **Why it matters:** first full laps on the real AIL Cup maps with the wm2 world model planning every call (E27 design).
- **Open:** lap times; a faster SPEED_OPTIONS (1.6, 2.0, ...) with SPEED_MAX / THROTTLE_MAX kept at 3.0 / 0.09.

### E36 — The AIL Cup web API works; the baseline on the real FIIT Oval (16 Nov 2026)
- **What:** `autoresearch/ailcup_client.py` (D20 below / docs/ail_cup_api.md) logged in, uploaded `ail_cup/simple_controller.py`
  v5 + `wm2_c4_value_garage.npz` and ran a test drive with no clicking. Output: `night/ailcup_baseline1.log`.
- **Result:** run **#1299**, FIIT Oval (default track), 2 laps **finished in 98.152 s** (~49 s / lap), step() p95 130 ms
  (cap 250), 0.62 Wh, code check ok.
- **Quota:** `per_day = 0` = no daily limit; 69 test drives so far (`docs/data/ailcup_page.json`).
- **Noise estimate (earlier runs of the same line, same page):** FIIT Oval 1-lap #1289 – #1292: 49.9 / 50.3 / 50.3 / 50.7 s
  → about ±0.4 s. #1293 compute_timeout (p95 193 ms), #1296 crash on Hairpin Valley (faster settings of 14 Nov).
- **Real tracks:** `fiit-oval` 79.5 m, `hairpin-valley` 100.8 m, `danube-chicane` 89.6 m; each with centre / left / right
  lines (795 points on the oval) and the start pose, width 2.6 m → `garage_sim` can use the real geometry.
- **Why it matters:** the real AIL Cup can be the loop's judge itself (the user's design, 16 Nov), not only a manual check.

### E37 — Baseline 3× on all 3 real AIL Cup tracks: the noise of the judge (16 Nov 2026)
- **Setup:** `ail_cup/simple_controller.py` v5 + `wm2_c4_value_garage.npz`, 2 laps, `--track` slug, 3 runs per track,
  `autoresearch/ailcup_client.py`. Files: `night/ailcup_base_<track>_<i>.json` (runs #1300 – #1308).
- **Result: 9 / 9 finished, no DNF.** Total time of 2 laps (s), step() p95 130 – 135 ms on every run:

  | Track | Run 1 | Run 2 | Run 3 | Mean | Spread (max − min) |
  |---|---|---|---|---|---|
  | fiit-oval | 98.018 | 97.988 | 98.153 | **98.05** | 0.17 |
  | hairpin-valley | 125.157 | 124.944 | 124.916 | **125.01** | 0.24 |
  | danube-chicane | 114.420 | 115.566 | 114.265 | **114.75** | 1.30 |
  | **Sum (the score)** | 337.595 | 338.498 | 337.334 | **337.81** | **1.16** (sd 0.50) |
- **Conclusion:** the real AIL Cup is an almost deterministic judge (oval sd 0.07 s); the chicane is the noisiest track.
  Proposed keep rule for the loop: a change is kept only if its sum is **below the best by more than 1.5 s** (> the
  baseline spread of 1.16 s, ~0.45 % of the score). The baseline score = **337.8 s**, the first point on the graph.

### E38 — The first real judge run: 3 tracks at once score 2.5 s worse (17 Nov 2026)
- **Setup:** `venv\Scripts\python -m autoresearch.judge` (the baseline: simple_controller v5 + wm2_c4), the 3 test
  drives started **at the same time** (one upload). File: `night/judge_test.json`, runs #1309 – #1311.
- **Result:** all finished, code ok, p95 130 – 135 ms. Oval 98.18, hairpin 125.26, chicane **116.83**; score **340.27 s**
  vs the E37 baseline 337.81 (range 337.33 – 338.50, tracks one after another). The chicane is 1.3 s above its E37 range.
- **Likely cause (not proven):** the controller's rollout has a time budget (`BUDGET_SHARE`, deadline in `imagine()`),
  so with 3 runs at once on the same server it plans fewer steps and drives slower. One run only: could also be noise.
- **Consequence:** the baseline and the experiments must be judged the same way; the 1.5 s keep margin (E37) is only
  valid for runs one after another. **Decision (the user, 17 Nov): the judge runs the tracks one after another**
  (`judge.py`, test `test_tracks_run_one_after_another`).
