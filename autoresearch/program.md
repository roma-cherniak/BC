# program.md: the instructions for every agent in the autoresearch loop

You improve a self-driving car controller for the **AIL Cup** (a racing platform). The car is driven by a
**JEPA world model** (`wm2`): every control call (20 per second) the controller encodes the lidar scan, imagines a set
of plans with the model, and drives the cheapest one.

## Your one job in this experiment
Make **one** focused change to `ail_cup/simple_controller.py` that you expect to make the car **faster on all three
tracks without crashing**. Then stop. The loop tests your change on the real AIL Cup and keeps it only if it helps.

## The score (lower is better)
- 3 tracks: `fiit-oval` (79.5 m), `hairpin-valley` (100.8 m), `danube-chicane` (89.6 m), 2 laps each.
- Score = the sum of the 3 total times. A track that does not finish (crash, timeout) counts as **300 s**.
- Code that fails the Garage check counts as 900 s.
- A change is kept only if it beats the best score by more than **1.5 s** (the measured noise).
- The baseline is **337.8 s** (oval 98.05, hairpin 125.01, chicane 114.75).

## Hard rules (a broken rule stops the whole loop)
1. Edit **only** `ail_cup/simple_controller.py`. Do not create, delete or edit any other file.
2. Allowed imports in the controller: `math`, `time`, `numpy` only.
3. The Garage stops a car whose control step takes **more than 250 ms**. The baseline's step p95 is ~135 ms in the
   Garage (the Garage adds 70 – 145 ms of its own); keep extra computation small.
4. The file must stay under 64 KB.
5. Keep the Garage interface unchanged: `class Controller` with `__init__(self, init: dict)` and
   `step(self, obs: dict) -> dict`. The model arrays come in `init["model"]`.
6. The model file is fixed: do not change the model's numbers (`SPEED_MAX`, `THROTTLE_MAX`, `MODEL_MAX_STEER`,
   `BEAMS`, `MAX_RANGE`, the `Wm2` class); the model was trained with them.

## How to work
- Read `autoresearch/runs/<this run>/results.tsv` first: every earlier experiment, its score, per-track times, DNFs,
  and whether it was kept. Do not repeat a change that already failed; build on what was kept.
- Prefer small, explainable changes (a speed option, a margin, a cost weight, the planning horizon) over rewrites.
- Think about the risk: a faster car that crashes on one track loses (300 s).

## Your answer
End your answer with exactly one line:
`CHANGE: <one short sentence: what you changed and why>`
