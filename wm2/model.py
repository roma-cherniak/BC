# wm2: a LeWorldModel-style JEPA for the lidar (LeWM: arXiv 2603.19312, in literarute/), sized for the laptop.
#
#   Encoder   one moment (109 lidar beams + the car's state) -> z (LATENT numbers)
#             1D convolutions along the beams: the same filters work at every angle, so a corner looks the same
#             wherever it is; then a projector with BatchNorm, like LeWM (SIGReg needs a projected, non-normalised z)
#   Predictor a small causal transformer over the last HISTORY z's: it sees how things MOVE, not one photo.
#             The action of every moment steers every layer through AdaLN (zero-initialised, like LeWM) instead of
#             being glued to the input. It predicts the CHANGE of z (z_next = z + dz) and the next speed.
#
# Sizes for 109 beams: encoder ~0.45 M, predictor ~1.3 M numbers. Everything works for any batch B.

import math

import torch
import torch.nn as nn

BEAMS = 109
STATE = 2           # [speed / SPEED_MAX, wheel angle / MAX_STEER_RAD]
ACTION = 2          # [throttle / THROTTLE_MAX, steering]
LATENT = 64
HISTORY = 4         # frames the predictor looks back: 4 x 0.05 s = 0.2 s
WIDTH = 128         # transformer width
LAYERS = 4
HEADS = 4


class Encoder(nn.Module):
    def __init__(self, beams=BEAMS):
        super().__init__()
        self.conv = nn.Sequential(                                  # (B, 1, 109) -> lengths 55, 28, 14, 7
            nn.Conv1d(1, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Flatten(),
        )
        flat = self.conv(torch.zeros(1, 1, beams)).shape[1]        # 128 x 7 = 896 for 109 beams
        self.state = nn.Sequential(nn.Linear(STATE, 32), nn.GELU())
        self.mix = nn.Sequential(nn.Linear(flat + 32, 256), nn.GELU(), nn.Linear(256, LATENT))
        self.project = nn.Sequential(nn.Linear(LATENT, LATENT), nn.BatchNorm1d(LATENT))

    def forward(self, lidar, state):
        # lidar (B, beams) 0..1, state (B, 2) -> z (B, LATENT)
        x = torch.cat([self.conv(lidar[:, None]), self.state(state)], dim=1)
        return self.project(self.mix(x))


class Block(nn.Module):
    # one transformer layer; the action of each moment shifts and scales its inputs (AdaLN) and gates its outputs
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(WIDTH, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.norm2 = nn.LayerNorm(WIDTH, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH))
        self.ada = nn.Linear(WIDTH, 6 * WIDTH)
        nn.init.zeros_(self.ada.weight)                             # AdaLN-zero: every block starts as "do nothing"
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, cond, mask):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = self.norm2(x) * (1 + scale2) + shift2
        return x + gate2 * self.mlp(h)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(LATENT, WIDTH)
        self.pos = nn.Parameter(torch.zeros(HISTORY, WIDTH))
        self.act = nn.Sequential(nn.Linear(ACTION, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH))
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(WIDTH)
        self.dz = nn.Linear(WIDTH, LATENT)
        self.speed = nn.Linear(WIDTH, 1)
        nn.init.zeros_(self.dz.weight)                              # an untrained predictor = "nothing changes" (copy)
        nn.init.zeros_(self.dz.bias)
        mask = torch.triu(torch.full((HISTORY, HISTORY), -math.inf), diagonal=1)
        self.register_buffer("mask", mask)                          # moment i sees only moments <= i

    def forward(self, z, action):
        # z (B, T, LATENT), action (B, T, 2), T <= HISTORY  ->  next z after every moment (B, T, LATENT)
        # and the next speed / SPEED_MAX (B, T)
        T = z.shape[1]
        x = self.inp(z) + self.pos[:T]
        cond = self.act(action)
        for block in self.blocks:
            x = block(x, cond, self.mask[:T, :T])
        x = self.norm(x)
        return z + self.dz(x), self.speed(x)[..., 0]


def rollout(predictor, z_history, actions):
    # imagine len(actions) steps ahead: z_history (B, HISTORY, LATENT) real past, actions (B, HISTORY - 1 + S, 2) =
    # the actions of the past moments and then the S actions to imagine. -> imagined z (B, S, LATENT), speed (B, S)
    window_z = z_history
    window_a = actions[:, :HISTORY]
    zs, speeds = [], []
    steps = actions.shape[1] - HISTORY + 1
    for s in range(steps):
        z_next, speed_next = predictor(window_z, window_a)
        zs.append(z_next[:, -1])
        speeds.append(speed_next[:, -1])
        if s + 1 < steps:                                           # slide the window: drop the oldest moment
            window_z = torch.cat([window_z[:, 1:], z_next[:, -1:]], dim=1)
            window_a = actions[:, s + 1 : s + 1 + HISTORY]
    return torch.stack(zs, dim=1), torch.stack(speeds, dim=1)


def count(module):
    return sum(p.numel() for p in module.parameters())
