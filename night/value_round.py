# One round of "the value head learns from the driver's own crashes" (docs/research_log.md E20):
#   split the new self-driven folders (80 / 10 / 10 by trip), build a data set = everything wm2_c1 learned on + all the
#   compete self-driving folders given, and train a value head of wm2_c1 on it (the encoder stays as it is).
#   venv\Scripts\python night\value_round.py v1 selfc2=data/self2_compete selfc3=data/self3_compete selfc4=data/self4_compete
# -> data/wm2_<round>/{train,val,test}.npz and runs/wm2_c1/value_<round>.pt. The sim must be closed (4 GB card).
# Nothing is overwritten: it stops if the value file exists; lists that exist are not split again.

import os
import subprocess
import sys

BC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BC)
os.chdir(BC)
from wm2 import prepare, value   # noqa: E402

MODEL = "wm2_c1"


def sim_open() -> bool:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq AutoDRIVE Simulator.exe"], capture_output=True, text=True).stdout
    return "autodrive" in out.lower()


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: night\\value_round.py <round> <name>=<folder> [...]")
    name = sys.argv[1]
    extra = dict(item.split("=") for item in sys.argv[2:])
    out_dir, file = f"data/wm2_{name}", f"value_{name}.pt"
    if os.path.exists(f"runs/{MODEL}/{file}"):
        sys.exit(f"runs/{MODEL}/{file} already exists")
    if sim_open():
        sys.exit("the simulator is open: close it first (4 GB card)")
    for short, folder in extra.items():
        if not os.path.exists(f"data/{short}_train.txt"):
            prepare.split_trips(folder, short)
    os.makedirs(out_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        lists = prepare.SETS["c1"][1][split] + [f"data/{short}_{split}.txt" for short in extra]
        prepare.prepare(out_dir, split, lists)
    value.main(MODEL, out_dir, file)


if __name__ == "__main__":
    main()
