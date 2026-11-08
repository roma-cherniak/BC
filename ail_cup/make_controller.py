# Build ail_cup/controller_jepa.py for the AIL Cup: controller_template.py + the weights of one lidar model,
# packed into the file itself (the platform allows only numpy and the standard library, no files).
# Run from the BC folder:  venv\Scripts\python -m ail_cup.make_controller k10_50ep
# Then upload ail_cup/controller_jepa.py in the Garage.

import base64
import io
import sys
import zlib
from pathlib import Path

import numpy as np
from config import FACTS, MAX_RANGE, SPEED_MAX, THROTTLE_MAX
from core.models import LATENT, load_run, run_input
from core.planner import STEERING, STEPS, GUARD

DTYPE = np.float32      # np.float16 makes the file 2x smaller, the numbers a bit less exact

HERE = Path(__file__).parent
RUN = sys.argv[1]
if run_input(RUN) != "lidar":
    exit(f"{RUN} is a {run_input(RUN)} model, but the AIL Cup car has only a lidar, no camera")

encoder, predictor, zond = load_run(RUN, "cpu")
weights = {}
for prefix, model in [("encoder", encoder), ("predictor", predictor), ("zond", zond)]:
    for key, value in model.state_dict().items():
        weights[f"{prefix}.{key}"] = value.numpy().astype(DTYPE)        # "encoder.net.0.weight", ...
buffer = io.BytesIO()
np.savez(buffer, **weights)
packed = base64.b64encode(zlib.compress(buffer.getvalue(), 9)).decode()

constants = "\n".join([                 # the same settings as the project (config, core.planner): one truth
    f'MODEL = "{RUN}"',
    f"STEERING = {STEERING}",
    f"STEPS = {STEPS}",
    f"GUARD = {GUARD}",
    f"FACTS = {FACTS}",
    f"MAX_RANGE = {MAX_RANGE}",
    f"SPEED_MAX = {SPEED_MAX}",
    f"THROTTLE_MAX = {THROTTLE_MAX}",
    f"LATENT = {LATENT}",
])
template = (HERE / "controller_template.py").read_text(encoding="utf-8")
out = HERE / "controller_jepa.py"
out.write_text(template.replace("# __CONSTANTS__", constants).replace("__WEIGHTS__", packed), encoding="utf-8")
print(f"saved {out}: {out.stat().st_size / 1024:.0f} KB, {sum(v.size for v in weights.values())} weights of {RUN}")
