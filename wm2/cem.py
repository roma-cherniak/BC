# CEM planner for wm2 (Cross-Entropy Method, the planner of LeWM / DINO-WM / PlaNet), over steering AND throttle.
#
# For every moment: try SAMPLES random action sequences of HORIZON steps in the imagination, score each with the cost
# heads, keep the ELITES best, move the sampling distribution towards them, repeat ITERS times; drive the first
# action of the final plan. Next moment starts from the old plan shifted by one step (warm start).
#
# cost of a sequence = - metres driven forward (progress head)
#                      + W_WALL   x sum of (missing safety margin)^2: front needs FRONT_BASE + braking distance v^2/2a,
#                                   the sides and diagonals need SIDE_MARGIN
#                      + W_DANGER x sum of P(danger)            (danger head)
#                      + W_SMOOTH x sum of |steering change|
# Everything is batched: M moments x SAMPLES sequences at once.

import torch

from config import SPEED_MAX, THROTTLE_MAX
from wm2.model import HISTORY, rollout

HORIZON = 20            # 20 x 0.05 s = 1 s
SAMPLES = 128
ELITES = 16
ITERS = 3
THROTTLE_RANGE = (0.0, 0.09)    # raw throttle the model knows: wm2_k10 saw 0.03 .. 0.09 (fast data: up to 0.2)
FRONT_BASE = 0.5        # m kept in front even when slow
DECEL = 4.0             # m/s2, measured braking of throttle 0
SIDE_MARGIN = 0.3       # m to the side walls and diagonals
W_WALL = 20.0
W_DANGER = 0.05
W_SMOOTH = 0.02
STD_START = (0.3, 0.6)  # (throttle in model units, steering)
STD_MIN = 0.05


def sequence_cost(predictor, heads, z_hist, past_actions, plans):
    # z_hist (B, HISTORY, D), past_actions (B, HISTORY - 1, 2), plans (B, HORIZON, 2) in model units -> cost (B,)
    z_im, speed_im = rollout(predictor, z_hist, torch.cat([past_actions, plans], dim=1))
    walls, danger, progress = heads(z_im)
    v = (speed_im * SPEED_MAX).clamp(min=0)
    front_short = (FRONT_BASE + v ** 2 / (2 * DECEL) - walls[..., 0]).clamp(min=0)
    side_short = (SIDE_MARGIN - walls[..., 1:].min(dim=-1).values).clamp(min=0)
    steer = torch.cat([past_actions[:, -1:, 1], plans[..., 1]], dim=1)
    cost = (-progress.sum(1) + W_WALL * (front_short ** 2 + side_short ** 2).sum(1)
            + W_DANGER * torch.sigmoid(danger).sum(1) + W_SMOOTH * steer.diff(dim=1).abs().sum(1))
    return cost


@torch.no_grad()
def plan(predictor, heads, z_hist, past_actions, mean=None, samples=SAMPLES, iters=ITERS):
    # z_hist (M, HISTORY, D) real latest z's, past_actions (M, HISTORY - 1, 2) the actions between them
    # mean (M, HORIZON, 2) warm start or None -> best plan (M, HORIZON, 2) in model units, its cost (M,)
    M, device = len(z_hist), z_hist.device
    low = torch.tensor([THROTTLE_RANGE[0] / THROTTLE_MAX, -1.0], device=device)
    high = torch.tensor([THROTTLE_RANGE[1] / THROTTLE_MAX, 1.0], device=device)
    if mean is None:
        mean = ((low + high) / 2).expand(M, HORIZON, 2).clone()
        mean[..., 1] = 0
    std = torch.tensor(STD_START, device=device).expand(M, HORIZON, 2).clone()
    zh = z_hist.repeat_interleave(samples, dim=0)
    pa = past_actions.repeat_interleave(samples, dim=0)
    for _ in range(iters):
        noise = torch.randn(M, samples, HORIZON, 2, device=device)
        cand = (mean[:, None] + std[:, None] * noise).clamp(low, high)
        cand[:, 0] = mean.clamp(low, high)                                  # always also try the current mean
        cost = sequence_cost(predictor, heads, zh, pa, cand.flatten(0, 1)).view(M, samples)
        best = cost.topk(ELITES, dim=1, largest=False).indices
        elite = torch.gather(cand, 1, best[..., None, None].expand(-1, -1, HORIZON, 2))
        mean, std = elite.mean(dim=1), elite.std(dim=1).clamp(min=STD_MIN)
    final = sequence_cost(predictor, heads, z_hist, past_actions, mean)
    return mean, final


def shift(plan_):
    # warm start for the next moment: drop the action just driven, repeat the last one at the end
    return torch.cat([plan_[:, 1:], plan_[:, -1:]], dim=1)


def check_history():
    assert HISTORY >= 2, "the planner needs at least one past action"
