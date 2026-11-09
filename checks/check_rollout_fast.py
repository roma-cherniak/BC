# Check of the faster wm2 rollout (2026-11-09): wm2/model.py now computes the AdaLN codes once per distinct action and
# can share the first `same` steps of a batch (the planner's prefix). Both must give the SAME numbers as the old code.
# The old code is copied here as the reference. Run from BC:  venv\Scripts\python -m checks.check_rollout_fast [run]
import sys
import time

import torch

from config import THROTTLE_MAX
from wm2 import cem
from wm2.heads import load_heads
from wm2.model import HISTORY, LATENT, GraphRollout, Predictor, rollout

RUN = sys.argv[1] if len(sys.argv) > 1 else "wm2_self1"
TOL = 1e-4


def old_predictor(pred, z, action):                                  # wm2/model.py before 2026-11-09
    T = z.shape[1]
    x = pred.inp(z) + pred.pos[:T]
    cond = pred.act(action)
    for block in pred.blocks:
        shift1, scale1, gate1, shift2, scale2, gate2 = block.ada(cond).chunk(6, dim=-1)
        h = block.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * block.attn(h, h, h, attn_mask=pred.mask[:T, :T], need_weights=False)[0]
        h = block.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * block.mlp(h)
    x = pred.norm(x)
    return z + pred.dz(x), pred.speed(x)[..., 0]


def old_rollout(pred, z_history, actions):
    window_z, window_a = z_history, actions[:, :HISTORY]
    zs, speeds = [], []
    steps = actions.shape[1] - HISTORY + 1
    for s in range(steps):
        z_next, speed_next = old_predictor(pred, window_z, window_a)
        zs.append(z_next[:, -1])
        speeds.append(speed_next[:, -1])
        if s + 1 < steps:
            window_z = torch.cat([window_z[:, 1:], z_next[:, -1:]], dim=1)
            window_a = actions[:, s + 1 : s + 1 + HISTORY]
    return torch.stack(zs, dim=1), torch.stack(speeds, dim=1)


def grid_batch(d, horizon=20):
    # like wm2/drive.py grid_plan: 9 steerings x 8 gas profiles, one z history, every plan starts with the same prefix
    steer = torch.linspace(-1, 1, 9)
    gas = torch.tensor([0.0, 0.04, 0.08, 0.12]) / THROTTLE_MAX
    profiles = [g.repeat(horizon) for g in gas] + [torch.cat([torch.zeros(k), g.repeat(horizon - k)])
                                                   for k in (3, 6) for g in gas[2:]]
    plans = torch.stack([torch.stack([p, torch.full((horizon,), s)], dim=1) for s in steer for p in profiles])
    plans[:, :d] = torch.tensor([0.1, 0.25])
    n = len(plans)
    return torch.randn(1, HISTORY, LATENT).expand(n, -1, -1), torch.rand(1, HISTORY - 1, 2).expand(n, -1, -1), plans


def timed(fn, reps=10):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    return (time.perf_counter() - t) / reps * 1000, out


def main():
    torch.manual_seed(0)
    torch.set_num_threads(6)
    pred = Predictor()
    pred.load_state_dict(torch.load(f"runs/{RUN}/predictor.pt", map_location="cpu"))
    pred.eval()
    heads = load_heads(RUN, "cpu", "sectors")
    ok = True
    with torch.no_grad():
        # 1. training-like windows: every row its own history and actions (same = 0)
        z, a = torch.randn(256, HISTORY, LATENT), torch.rand(256, HISTORY - 1 + 10, 2) * 2 - 1
        diff = max((x - y).abs().max().item() for x, y in zip(old_rollout(pred, z, a), rollout(pred, z, a)))
        ok &= diff < TOL
        print(f"training windows   max difference {diff:.2e}  {'OK' if diff < TOL else 'FAIL'}")
        # 2. planner grid with a shared prefix of d steps
        for d in (1, 4, 8, 16, 20):
            z, past, plans = grid_batch(d)
            ms_old, c_old = timed(lambda: old_cost(pred, heads, z, past, plans))
            ms_new, c_new = timed(lambda: cem.sequence_cost(pred, heads, z, past, plans, same=d))
            # batch 1 vs batch 72 sums in another order (float rounding): compare relative to the cost scale
            diff = ((c_old - c_new).abs().max() / c_old.abs().max()).item()
            same_best = c_old.argmin().item() == c_new.argmin().item()
            ok &= diff < TOL and same_best
            print(f"grid prefix d={d:2d}  max relative cost difference {diff:.2e}  same best plan {same_best}  "
                  f"old {ms_old:5.0f} ms -> new {ms_new:5.0f} ms")
    if torch.cuda.is_available():
        ok &= check_graph(pred.state_dict())
    print("ALL OK" if ok else "SOMETHING DIFFERS")
    sys.exit(0 if ok else 1)


def check_graph(weights):
    # 3. the same grid costs from model.GraphRollout on the graphics card (wm2/drive.py) vs the CPU
    gpu = Predictor().cuda().eval()
    gpu.load_state_dict(weights)
    cpu = Predictor().eval()
    cpu.load_state_dict(weights)
    heads_gpu, heads_cpu = load_heads(RUN, "cuda", "sectors"), load_heads(RUN, "cpu", "sectors")
    roll, ok, agree = GraphRollout(gpu), True, 0
    with torch.no_grad():
        for trial in range(20):
            z, past, plans = grid_batch(1 + trial % 16)
            c_cpu = cem.sequence_cost(cpu, heads_cpu, z, past, plans)
            c_gpu = cem.sequence_cost(gpu, heads_gpu, z.cuda(), past.cuda(), plans.cuda(), roll=roll).cpu()
            diff = ((c_cpu - c_gpu).abs().max() / c_cpu.abs().max()).item()
            agree += c_cpu.argmin().item() == c_gpu.argmin().item()
            ok &= diff < 1e-3
        z, past, plans = [x.cuda() for x in grid_batch(6)]

        def one():
            out = cem.sequence_cost(gpu, heads_gpu, z, past, plans, roll=roll)
            return out.cpu()
        ms, _ = timed(one, 20)
    ok &= agree >= 19                                 # near-ties may flip on float rounding
    print(f"GPU graph          last relative cost difference {diff:.1e}, same best plan {agree}/20, {ms:.0f} ms "
          f"per plan  {'OK' if ok else 'FAIL'}")
    return ok


def old_cost(pred, heads, z, past, plans):
    global rollout
    new, cem.rollout = cem.rollout, lambda p, zh, acts, same=0: old_rollout(p, zh, acts)
    try:
        return cem.sequence_cost(pred, heads, z, past, plans)
    finally:
        cem.rollout = new


if __name__ == "__main__":
    main()
