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

    def forward(self, x, code, mask):
        # code = self.ada(action features) (B, T, 6 x WIDTH): the caller computes it (once per action, see codes())
        shift1, scale1, gate1, shift2, scale2, gate2 = code.chunk(6, dim=-1)
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

    def codes(self, action, distinct=True):
        # action (B, T, 2) -> (B, T, LAYERS, 6 x WIDTH): what every action tells every layer (AdaLN). It depends on
        # the action only, not on its place in the window, so a rollout computes it once per distinct action
        # (the grid planner's 72 plans use only a few dozen) instead of once per step and window slot.
        # distinct=False skips torch.unique (it waits for the GPU, which a CUDA graph does not allow).
        flat = action.reshape(-1, ACTION)
        if distinct:
            flat, index = torch.unique(flat, dim=0, return_inverse=True)
        cond = self.act(flat)
        per_layer = torch.stack([block.ada(cond) for block in self.blocks], dim=1)
        if distinct:
            per_layer = per_layer[index]
        return per_layer.view(*action.shape[:-1], LAYERS, -1)

    def forward(self, z, action, codes=None):
        # z (B, T, LATENT), action (B, T, 2), T <= HISTORY  ->  next z after every moment (B, T, LATENT)
        # and the next speed / SPEED_MAX (B, T). codes: self.codes(action) if already computed (rollout)
        T = z.shape[1]
        x = self.inp(z) + self.pos[:T]
        if codes is None:
            codes = self.codes(action)
        for i, block in enumerate(self.blocks):
            x = block(x, codes[:, :, i], self.mask[:T, :T])
        x = self.norm(x)
        return z + self.dz(x), self.speed(x)[..., 0]


def rollout(predictor, z_history, actions, same=0, distinct=True):
    # imagine len(actions) steps ahead: z_history (B, HISTORY, LATENT) real past, actions (B, HISTORY - 1 + S, 2) =
    # the actions of the past moments and then the S actions to imagine. -> imagined z (B, S, LATENT), speed (B, S)
    # same: the first `same` imagined steps are identical for the whole batch (the planner: one z history and a
    # prefix every plan shares) -> they are computed for row 0 only and copied. 0 = every row on its own.
    codes = predictor.codes(actions, distinct)                      # once for the whole rollout
    B, steps = len(actions), actions.shape[1] - HISTORY + 1
    same = min(same, steps)
    rows = 1 if same > 0 else B
    window_z = z_history[:rows]
    zs, speeds = [], []
    for s in range(steps):
        if s == same and rows < B:                                  # the plans part here: one row -> every row
            rows = B
            window_z = window_z.expand(B, -1, -1)
            zs, speeds = [z.expand(B, -1) for z in zs], [v.expand(B) for v in speeds]
        z_next, speed_next = predictor(window_z, actions[:rows, s : s + HISTORY], codes[:rows, s : s + HISTORY])
        zs.append(z_next[:, -1])
        speeds.append(speed_next[:, -1])
        if s + 1 < steps:                                           # slide the window: drop the oldest moment
            window_z = torch.cat([window_z[:, 1:], z_next[:, -1:]], dim=1)
    if rows < B:                                                    # every step was shared
        zs, speeds = [z.expand(B, -1) for z in zs], [v.expand(B) for v in speeds]
    return torch.stack(zs, dim=1), torch.stack(speeds, dim=1)


class GraphRollout:
    # rollout() on the graphics card as ONE CUDA graph per batch shape (the sim driver, wm2/drive.py). On Windows
    # starting one GPU operation costs ~50 us, and a 20-step rollout of the 4-layer predictor has ~2000 tiny ones:
    # 119 ms as normal GPU code, 253 ms on the CPU, ~54 ms as a graph (2026-11-09, with the sim running).
    # A graph replays fixed operations on fixed tensors: new inputs are copied into them, so `same` is not used.
    def __init__(self, predictor):
        self.predictor, self.graphs = predictor, {}

    def __call__(self, z_history, actions, same=0):
        key = (tuple(z_history.shape), tuple(actions.shape))
        if key not in self.graphs:
            self.graphs[key] = self.capture(z_history, actions)
        graph, z_in, a_in, z_out, v_out = self.graphs[key]
        z_in.copy_(z_history)
        a_in.copy_(actions)
        graph.replay()
        return z_out.clone(), v_out.clone()

    @torch.no_grad()
    def capture(self, z_history, actions):
        z_in, a_in = z_history.clone(), actions.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                               # warm-up runs, as torch.cuda.graph asks
            for _ in range(3):
                rollout(self.predictor, z_in, a_in, 0, distinct=False)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            z_out, v_out = rollout(self.predictor, z_in, a_in, 0, distinct=False)
        return graph, z_in, a_in, z_out, v_out


def count(module):
    return sum(p.numel() for p in module.parameters())
