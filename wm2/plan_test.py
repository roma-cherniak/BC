# Offline test of the wm2 CEM planner, the same natural experiment as checks/check_planner.py:
# in "random" test moments a coin steered. If the planner's steering is good, the moments where the coin happened to
# steer like the planner (|coin - plan| < 0.5) must be followed by danger (safety or crash within 0.5 s) less often
# than the same plans shuffled between moments (control). Old lidar models: b109 -12.9, find_gap about -23 points.
# Run from the BC folder:  venv\Scripts\python -m wm2.plan_test wm2_k10

import sys
import time

import torch

from config import DEVICE, RUNS_DIR, THROTTLE_MAX
from wm2 import cem
from wm2.data import load_split, window_starts, windows
from wm2.heads import load_heads
from wm2.model import HISTORY, LATENT, Encoder, Predictor

BRAKE_TIME = 27.5       # s: after this the trip brakes to its end, skip
CHUNK = 128             # moments planned at once (x SAMPLES sequences: 256 nearly filled the 4 GB card)
MOMENTS = 5000          # random subset of the ~20 k random moments (all of them took > 35 min); 0 = all
BINS = [0, 0.5, 1.0, 1.5, 2.01]


def main(run):
    global MOMENTS
    if len(sys.argv) > 2:
        MOMENTS = int(sys.argv[2])
    torch.manual_seed(0)
    encoder, predictor = Encoder().to(DEVICE), Predictor().to(DEVICE)
    encoder.load_state_dict(torch.load(f"{RUNS_DIR}/{run}/encoder.pt", map_location=DEVICE))
    predictor.load_state_dict(torch.load(f"{RUNS_DIR}/{run}/predictor.pt", map_location=DEVICE))
    encoder.eval(), predictor.eval()
    heads = load_heads(run, DEVICE)

    test = load_split("test", DEVICE)
    starts = window_starts(test["trip"], HISTORY)                            # moments with a full history
    now = (starts + HISTORY - 1).numpy()
    keep = (test["mode"][now] == "random") & (test["time"][now] < BRAKE_TIME)
    starts = starts[torch.from_numpy(keep)]
    now = torch.from_numpy(now[keep])
    if MOMENTS and len(starts) > MOMENTS:                                    # a random subset: CEM is slow
        pick = torch.randperm(len(starts))[:MOMENTS].sort().values
        starts, now = starts[pick], now[pick]
    now = now.to(DEVICE)
    print(f"{len(starts)} random test moments", flush=True)

    t0, choices, throttles = time.time(), [], []
    with torch.no_grad():
        for i in range(0, len(starts), CHUNK):
            s = starts[i : i + CHUNK]
            lidar, state, action = windows(test, s, HISTORY)
            z = encoder(lidar.flatten(0, 1), state.flatten(0, 1)).view(len(s), HISTORY, LATENT)
            best, _ = cem.plan(predictor, heads, z, action[:, : HISTORY - 1])
            choices.append(best[:, 0, 1])
            throttles.append(best[:, 0, 0] * THROTTLE_MAX)
            if (i // CHUNK) % 5 == 0:
                print(f"  {i + len(s)} / {len(starts)} moments, {time.time() - t0:.0f} s", flush=True)
    choice, throttle = torch.cat(choices), torch.cat(throttles)
    print(f"planned in {time.time() - t0:.0f} s ({1000 * (time.time() - t0) / len(starts):.1f} ms per moment)")

    coin = test["action"][now, 1]
    danger = test["danger"][now].float()
    control = choice[torch.randperm(len(choice), device=DEVICE)]

    def rates(c):
        diff = (coin - c).abs()
        return [((diff >= lo) & (diff < hi)).sum().item() for lo, hi in zip(BINS[:-1], BINS[1:])], \
               [danger[(diff >= lo) & (diff < hi)].mean().item() for lo, hi in zip(BINS[:-1], BINS[1:])]

    (n, d), (_, c) = rates(choice), rates(control)
    print(f"danger overall {danger.mean().item():.1%}")
    print(f"{'|coin - plan|':<14}{'moments':>9}{'danger':>9}{'control':>9}")
    for (lo, hi), k, a, b in zip(zip(BINS[:-1], BINS[1:]), n, d, c):
        print(f"{lo:.1f} .. {min(hi, 2):.1f}{'':4}{k:>9}{a:>9.1%}{b:>9.1%}")
    print(f"agree vs control: {100 * (d[0] - c[0]):+.1f} points   (old b109 -12.9, find_gap about -23)")
    print(f"planned steering: mean |s| {choice.abs().mean().item():.2f}; "
          f"throttle: mean {throttle.mean().item():.3f}, min {throttle.min().item():.3f}, max {throttle.max().item():.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "wm2_k10")
