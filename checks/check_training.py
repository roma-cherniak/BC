# Training check: the last epoch of runs/<name>/loss.csv (training/train_jepa.py writes one row per epoch).
# Run from the BC folder:  venv\Scripts\python -m checks.check_training k10_50ep

import csv
import sys
from config import RUNS_DIR, CURRENT_RUN

COLLAPSE_SPREAD = 0.5       # spread below this = the encoder gives (almost) the same z for every lidar = collapse


def check_training(folder):
    with open(f"{folder}/loss.csv") as f:
        last = list(csv.DictReader(f))[-1]                  # the last epoch; csv gives every value as text
    spread = float(last["spread"])
    if spread < COLLAPSE_SPREAD:
        print(f"COLLAPSE: spread {spread:.3f} < {COLLAPSE_SPREAD}")
    return {
        "training epochs": int(last["epoch"]) + 1,
        "training val/copy": float(last["val_loss"]) / float(last["copy_loss"]),   # below 1 = better than copying
        "training spread": spread,                                                  # about 1 = healthy
        "training sigreg": float(last["sigreg"]),
    }


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else CURRENT_RUN
    print(check_training(f"{RUNS_DIR}/{run}"))
