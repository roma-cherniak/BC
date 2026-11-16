# Pack a wm2 model (runs/<RUN>/encoder.pt, predictor.pt, heads_sectors.pt) into ONE .npz for the AIL Cup Garage,
# in the shape claude_controller.py v11 reads with pure Python (no numpy there): every array is 1D or 2D,
# names without dots (the Garage rejects dots), float16 (half the size; the planner's choice does not change).
#   conv weights (out, in, k)  -> (out, in * k)        the controller dots one row with a flat window
#   BatchNorm of the projector -> folded into its Linear (eval mode: a fixed scale and shift)
#   attention in_proj (3W, W)  -> q, k, v (W, W) each  so the controller can skip q where it is not needed
#   the 4 blocks               -> one array per part, block number first (49 arrays in all; the Garage allows 64)
# Run from the BC folder:  venv\Scripts\python -m ail_cup.export_wm2 wm2_self1
# Then upload ail_cup/<RUN>_garage.npz as the model of the controller in the Garage.

import datetime
import sys
from pathlib import Path

import numpy as np
import torch

from config import RUNS_DIR
from wm2.model import LAYERS, WIDTH

DTYPE = np.float16


def main(run):
    enc = torch.load(f"{RUNS_DIR}/{run}/encoder.pt", map_location="cpu")
    pred = torch.load(f"{RUNS_DIR}/{run}/predictor.pt", map_location="cpu")
    heads = torch.load(f"{RUNS_DIR}/{run}/heads_sectors.pt", map_location="cpu")
    out = {}

    def put(name, tensor):
        out[name] = tensor.detach().double().numpy()

    for i, layer in enumerate([0, 2, 4, 6]):                         # encoder.conv: Conv1d, GELU, Conv1d, GELU, ...
        w = enc[f"conv.{layer}.weight"]
        put(f"enc_c{i}_w", w.reshape(w.shape[0], -1))
        put(f"enc_c{i}_b", enc[f"conv.{layer}.bias"])
    put("enc_state_w", enc["state.0.weight"])
    put("enc_state_b", enc["state.0.bias"])
    put("enc_mix0_w", enc["mix.0.weight"])
    put("enc_mix0_b", enc["mix.0.bias"])
    put("enc_mix2_w", enc["mix.2.weight"])
    put("enc_mix2_b", enc["mix.2.bias"])
    scale = enc["project.1.weight"] / torch.sqrt(enc["project.1.running_var"] + 1e-5)   # BatchNorm1d, eval
    put("enc_proj_w", enc["project.0.weight"] * scale[:, None])
    put("enc_proj_b", (enc["project.0.bias"] - enc["project.1.running_mean"]) * scale + enc["project.1.bias"])

    put("pred_inp_w", pred["inp.weight"])
    put("pred_inp_b", pred["inp.bias"])
    put("pred_pos", pred["pos"])
    for name in ["act.0", "act.2", "norm", "dz", "speed"]:
        put(f"pred_{name.replace('.', '')}_w", pred[f"{name}.weight"])
        put(f"pred_{name.replace('.', '')}_b", pred[f"{name}.bias"])
    # the 4 transformer blocks have the same shapes: every part is ONE array with the block number first
    # (pred_q_w (4, 128, 128) instead of pred_b0_q_w ... pred_b3_q_w): 14 arrays instead of 56 (the Garage allows 64)
    parts = {}
    for l in range(LAYERS):
        b = f"blocks.{l}."
        w, bias = pred[b + "attn.in_proj_weight"], pred[b + "attn.in_proj_bias"]
        for k, part in enumerate("qkv"):
            parts.setdefault(f"pred_{part}_w", []).append(w[k * WIDTH:(k + 1) * WIDTH])
            parts.setdefault(f"pred_{part}_b", []).append(bias[k * WIDTH:(k + 1) * WIDTH])
        for src, dst in [("attn.out_proj", "o"), ("mlp.0", "m0"), ("mlp.2", "m2"), ("ada", "ada")]:
            parts.setdefault(f"pred_{dst}_w", []).append(pred[f"{b}{src}.weight"])
            parts.setdefault(f"pred_{dst}_b", []).append(pred[f"{b}{src}.bias"])
    for name, tensors in parts.items():
        put(name, torch.stack(tensors))
    for src, dst in [("body.0", "b0"), ("body.2", "b2"), ("out", "out")]:
        put(f"head_{dst}_w", heads[f"{src}.weight"])
        put(f"head_{dst}_b", heads[f"{src}.bias"])

    # the value head (wm2/value.py, 12 Nov 2026): "metres the car still drives in the next 3 s". Optional: without
    # runs/<run>/value.pt the file is as before and the controller plans without it. 6 more arrays (56 of 64).
    value_path = Path(RUNS_DIR) / run / "value.pt"
    if value_path.exists():
        value = torch.load(value_path, map_location="cpu")
        for src, dst in [("body.0", "b0"), ("body.2", "b2"), ("body.4", "out")]:
            put(f"val_{dst}_w", value[f"{src}.weight"])
            put(f"val_{dst}_b", value[f"{src}.bias"])

    # which model this is, inside the file (numbers only: letter codes, the Garage takes plain numeric arrays):
    # the controller prints it, so a Garage run shows what drove
    with_value = "val_out_w" in out
    label = f"{run}{' + value' if with_value else ''} exported {datetime.date.today()}"
    path = Path(__file__).parent / f"{run}{'_value' if with_value else ''}_garage.npz"
    if path.exists():
        sys.exit(f"{path} already exists: nothing is overwritten")
    arrays = {k: v.astype(DTYPE) for k, v in out.items()}
    arrays["meta_name"] = np.array([ord(c) for c in label], dtype=np.int16)
    np.savez(path, **arrays)
    out["meta_name"] = arrays["meta_name"]
    numbers = sum(v.size for v in out.values())
    print(f"saved {path}: {path.stat().st_size / 1024:.0f} KB, {numbers:,} numbers, {len(out)} arrays")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "wm2_self1")
