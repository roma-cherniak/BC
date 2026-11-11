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

### E14 — The first night on `compete` (night 11 → 12 Nov 2026) — planned, result open
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
- **Result:** open (Thu 12 Nov).

---

## Open questions (move to a decision or experiment when answered)
- Is wm2_self2 better than wm2_self1 in the sim? → answered in E13 (11 Nov): no, a little slower
- How does the car drive on `compete`, a track it never saw? (Thu 12)
- Is the model fast enough for the AIL Cup time per call? (Thu 12)
- The sim score formula for the loop ([goals/month.md](../goals/month.md))
