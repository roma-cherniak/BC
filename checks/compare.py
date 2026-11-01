# All models side by side: reads every runs/*/report.json (written by checks/run_all.py).
# Run from the BC folder:  venv\Scripts\python -m checks.compare

import glob
import json
import os
from config import RUNS_DIR


reports = {}
for path in sorted(glob.glob(f"{RUNS_DIR}/*/report.json")):
    with open(path) as f:
        reports[os.path.basename(os.path.dirname(path))] = json.load(f)    # runs/k10_50ep/report.json -> "k10_50ep"

if not reports:
    exit(f"no report.json in {RUNS_DIR}/*/ yet: run  venv\\Scripts\\python -m checks.run_all <name>  first")

checks = []                     # every number any report has, in the order it first appears
for report in reports.values():
    for name in report:
        if name not in ("run", "date") and name not in checks:
            checks.append(name)

width = max(len(name) for name in checks)
print(f"{'check':<{width}}" + "".join(f"{run:>14}" for run in reports))
for name in checks:
    row = f"{name:<{width}}"
    for report in reports.values():
        value = report.get(name)
        row += f"{value:>14.4g}" if isinstance(value, (int, float)) else f"{'-':>14}"    # "-" = this model has no such number
    print(row)
print(f"{'date':<{width}}" + "".join(f"{report.get('date', '-')[:10]:>14}" for report in reports.values()))
