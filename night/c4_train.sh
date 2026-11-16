#!/bin/bash
# 14 Nov 2026 (the user: "Now (training only): A1 + A2 in the next training" -> "start"): wm2_c3 -> wm2_c4 with the
# balanced sampling of docs/data_solve.md (WM2_BALANCE=1: practice / compete 50 / 50, newest compete trips x2, fast
# starts x2, standing car 1 in 5), the scale + mirror augmentation kept (WM2_SCALE=0.6,2.2), the same trips as c2
# (data/wm2_c4 = c2 + track / source per moment), 3 epochs, lr 1e-4; then heads, a balanced value head and the
# corridor test against wm2_c3.
#   bash night/c4_train.sh        (from the BC folder; the sim must be closed: 4 GB card). No sim drive is started.
# Logs: night/c4_*.log; progress lines: night/state_compete.txt. Stops at the first step that fails.
cd "$(dirname "$0")/.." || exit 1
PY=venv/Scripts/python
export WM2_SCALE=0.6,2.2
say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  c4: $1" | tee -a night/state_compete.txt; }
step() {   # step <name> <command...>
    local name=$1; shift
    say "$name - start"
    if "$@" >> "night/c4_$name.log" 2>&1; then say "$name - finished"; else say "$name - FAILED, stopped here"; exit 1; fi
}

[ -e runs/wm2_c4 ] && { say "runs/wm2_c4 already exists -> nothing is overwritten"; exit 1; }
tasklist | grep -qi "AutoDRIVE Simulator" && { say "the simulator is open -> not started (4 GB card)"; exit 1; }
[ -e data/wm2_c4/test.npz ] || step prepare $PY -u -m wm2.prepare c4
WM2_BALANCE=1 step train $PY -u -m wm2.train wm2_c4 3 data/wm2_c4 wm2_c3 1e-4
step heads $PY -u -m wm2.heads wm2_c4 sectors
step heads $PY -u -m wm2.heads wm2_c4 walls5
WM2_VALUE_BALANCE=1 step value $PY -u -m wm2.value wm2_c4
step corridor $PY -m autoresearch.eval.corridor_test wm2_c3 wm2_c4
say "ALL DONE"
