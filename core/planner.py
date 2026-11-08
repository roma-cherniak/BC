# Planner: imagine every steering option and score how close the walls get. The driver and check_planner use it.

import torch
from config import FACTS
from core.imagine import imagine_step


STEERING = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
STEPS = 10              # 10 x 0.05 s = 0.5 s of imagination
GUARD = False           # no guard B: the driver's settings


def score(facts):
    # the closest wall; a diagonal counts /1.4 (in a corridor it is ~1.4x farther), the front /2. bigger = safer
    f = dict(zip(FACTS, facts.T))                    # f["front"] = facts[:, 0], ...
    walls = torch.stack([f["right"], f["left"], f["front_right"] / 1.4, f["front_left"] / 1.4, f["front"] / 2], dim=1)
    return walls.min(dim=1).values


def score_options(encoder, predictor, zond, obs, throttle, speed):
    # obs = what the encoder eats for M moments (lidar (M, 271) or photos (M, 1, 36, 64)),
    # throttle (M,), speed (M, 1), already scaled for the model -> scores (M, 9)
    M, K = len(obs), len(STEERING)
    options = torch.tensor(STEERING, device=obs.device)
    with torch.no_grad():
        z = encoder(obs.float()).repeat_interleave(K, dim=0)                   # (M*9, 32): m0 x9, m1 x9, ...
        act = torch.stack([throttle.repeat_interleave(K), options.repeat(M)], dim=1)
        spd = speed.repeat_interleave(K, dim=0)
        worst = torch.full((M * K,), float("inf"), device=obs.device)         # the closest wall on the WHOLE path
        for _ in range(STEPS):
            z = imagine_step(predictor, z, act, spd, GUARD)
            worst = torch.minimum(worst, score(zond(z).clamp(0, 1)))
    return worst.view(M, K)
