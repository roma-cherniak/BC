# Stage A: MLP world model on lidar

## Setup
- Data: 1,722 trips (13.2 h, practice track), split by trip 1377 / 172 / 173 (train / val / test, seed 42)
- Input: lidar 271 beams (every 4th, inf -> 10 m, /10), throttle /0.09, steering, speed /3
- Output: next lidar scan (0.05 s later)
- Model: MLP 274 -> 512 -> 512 -> 271, 542,479 parameters
- Training: Adam lr 0.001, batch 256, 40 epochs (~6 s/epoch, RTX 3050), best val at epoch 38
- Baseline "copy": the next scan = the current scan

## One step (0.05 s), MSE
| split | brain     | copy      | brain / copy |
|-------|-----------|-----------|--------------|
| val   | 0.0001658 | 0.0006846 | 0.24         |
| test  | 0.0001702 | 0.0007031 | 0.24         |

## Imagination (rollout), 500 random safe test starts, brain / copy
| 0.05 s | 0.25 s | 0.5 s | 0.75 s | 1.0 s |
|--------|--------|-------|--------|-------|
| 0.25   | 0.22   | 0.35  | 0.48   | 0.67  |
Lines cross at ~1.1 s. After that a few imaginations explode (values < 0 or > 1, never seen in training).

## Conclusions
- The brain learned motion: in imagination it turns the scene like the real car does.
- 4x better than copy for one step, still 1.5x better after 1 s -> imagination is trustworthy for ~1 s.
- Weak points: blurred wall edges (MSE averages uncertain guesses), small jitter, phantom walls,
  compounding error over many steps.
- Data limits: throttle is constant inside a trip, one track, one driver.

## Pictures
loss.png, guess.png, rollout_example.png, rollout.png
