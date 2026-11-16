#!/bin/bash
# 12 Nov 2026 (the user: "you can start training and tests after training"): wm2_c1 -> wm2_c2 on everything, incl. the
# compete laps of E17 - E20, then its heads and a balanced value head, then the test drives of E20's best setting.
#   bash night/c2_train_and_test.sh        (from the BC folder; the sim is closed first, 4 GB card)
# Logs: night/c2_*.log; progress lines: night/state_compete.txt. Stops at the first step that fails.
cd "$(dirname "$0")/.." || exit 1
PY=venv/Scripts/python
say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  c2: $1" | tee -a night/state_compete.txt; }
step() {   # step <name> <command...>
    local name=$1; shift
    say "$name - start"
    if "$@" >> "night/c2_$name.log" 2>&1; then say "$name - finished"; else say "$name - FAILED, stopped here"; exit 1; fi
}

[ -e runs/wm2_c2 ] && { say "runs/wm2_c2 already exists -> nothing is overwritten"; exit 1; }
powershell -NoProfile -ExecutionPolicy Bypass -File night/open_sim.ps1 close
step prepare $PY -u -m wm2.prepare c2
step train $PY -u -m wm2.train wm2_c2 3 data/wm2_c2 wm2_c1 1e-4
step heads $PY -u -m wm2.heads wm2_c2 sectors
step heads $PY -u -m wm2.heads wm2_c2 walls5
WM2_VALUE_BALANCE=1 step value $PY -u -m wm2.value wm2_c2
say "training ALL DONE"

# test drives: the best setting of E20, 2 x 5.5 min on each track
export TEST_SECONDS=330 TEST_COST="FRONT_BASE=0.25,W_CENTER=1.0,CENTER_CLAMP=0.3,SIDE_MARGIN=0.45,W_VALUE=0.3"
powershell -NoProfile -ExecutionPolicy Bypass -File night/open_sim.ps1 compete || { say "the compete sim did not open"; exit 1; }
step test_compete $PY night/test_drives.py wm2_c2 compete c2_test_a c2_test_b
powershell -NoProfile -ExecutionPolicy Bypass -File night/open_sim.ps1 practice || { say "the practice sim did not open"; exit 1; }
step test_practice $PY night/test_drives.py wm2_c2 practice c2_test_a c2_test_b
powershell -NoProfile -ExecutionPolicy Bypass -File night/open_sim.ps1 close
say "tests ALL DONE"
