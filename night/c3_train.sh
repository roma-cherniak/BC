#!/bin/bash
# 13 Nov 2026 (the user: "now you can start training"): wm2_c2 -> wm2_c3 with the scale + mirror augmentation
# (WM2_SCALE=0.6,2.2: wm2/data.py windows, E26), the same data (data/wm2_c2), 3 epochs, lr 1e-4; then its heads and
# a balanced value head, trained with the same augmentation; then the corridor test against wm2_c2.
#   bash night/c3_train.sh        (from the BC folder; the sim must be closed: 4 GB card). No sim drive is started.
# Logs: night/c3_*.log; progress lines: night/state_compete.txt. Stops at the first step that fails.
cd "$(dirname "$0")/.." || exit 1
PY=venv/Scripts/python
export WM2_SCALE=0.6,2.2
say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  c3: $1" | tee -a night/state_compete.txt; }
step() {   # step <name> <command...>
    local name=$1; shift
    say "$name - start"
    if "$@" >> "night/c3_$name.log" 2>&1; then say "$name - finished"; else say "$name - FAILED, stopped here"; exit 1; fi
}

[ -e runs/wm2_c3 ] && { say "runs/wm2_c3 already exists -> nothing is overwritten"; exit 1; }
tasklist | grep -qi "AutoDRIVE Simulator" && { say "the simulator is open -> not started (4 GB card)"; exit 1; }
step train $PY -u -m wm2.train wm2_c3 3 data/wm2_c2 wm2_c2 1e-4
step heads $PY -u -m wm2.heads wm2_c3 sectors
step heads $PY -u -m wm2.heads wm2_c3 walls5
WM2_VALUE_BALANCE=1 step value $PY -u -m wm2.value wm2_c3
step corridor $PY -m autoresearch.eval.corridor_test wm2_c2 wm2_c3
say "ALL DONE"
