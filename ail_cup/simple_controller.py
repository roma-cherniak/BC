"""A simple wm2 driver for the AIL Cup, with numpy (13 Nov 2026; the user's design, docs/research_log.md E27).

Every call (20 per second), nothing spread over several calls:
  1. ENCODE   the newest scan -> z; the last HISTORY z's and the actions sent between them are kept
  2. IMAGINE  N plans (every steering option x every speed option, each held for HORIZON steps of 0.05 s) in ONE
              batched rollout of the predictor; the heads read walls and progress, the value head the road after it
  3. CHOOSE   the plan with the lowest cost: choose() below
  4. DRIVE    the steering and speed of that plan; the next call plans again from the new scan
No reflexes, no modes, no map guard: if the model reads the walls right (wm2_c3, E26), the plan is the driver.

Why numpy: the pure-Python controllers (claude_controller.py, wm2_controller.py) need 1.18 s for ONE plan of 3 options
(E22) and had to spread it over ~15 calls. make_controller.py says the platform allows numpy (question 12 in
goals/communication.md: not yet confirmed in the Garage).

Model file: "venv\\Scripts\\python -m ail_cup.export_wm2 <RUN>" -> ail_cup/<RUN>_value_garage.npz (the same file the
older controllers read). Check against torch + time:   venv\\Scripts\\python -m ail_cup.check_simple wm2_c3
Try on this PC:  venv\\Scripts\\python -m ail_cup.garage_sim ail_cup.simple_controller oval ail_cup/wm2_c3_value_garage.npz
"""
import math
import time

import numpy as np

VERSION = "simple_controller v5 (numpy, 18 plans x 5 steps, 75 ms budget, no prints while driving)"
# ----- what the model was trained with (the same numbers as wm2_controller.py)
BEAMS = 109                     # -135 .. +135 degrees, 2.5 degrees apart, right -> left
MAX_RANGE = 10.0                # m: lidar / MAX_RANGE -> 0 .. 1
SPEED_MAX = 3.0                 # m/s: the model gets speed / SPEED_MAX
THROTTLE_MAX = 0.09             # the model gets AutoDRIVE throttle / THROTTLE_MAX
MODEL_MAX_STEER = 0.5236        # rad: model steering +-1 = +-30 degrees of the AutoDRIVE car
GAS_PER_MS = 1 / 24.5           # AutoDRIVE throttle per m/s of steady speed
HISTORY, WIDTH, HEADS, LAYERS, LATENT = 4, 128, 4, 4, 64
# ----- the plans
# Garage (run #1283): 27 x 20 steps hit the 250 ms cap; 27 x 5 = ~75 ms a call there (the user's choice, 14 Nov)
HORIZON = 5                     # imagined steps of 0.05 s = 0.25 s (the value head sees the 3 s after it)
STEER_OPTIONS = np.linspace(-0.8, 0.8, 9)   # model units, + = left; 0.8 x 30 deg = 24 deg = the AIL Cup maximum
SPEED_OPTIONS = (1.2, 1.6)                  # m/s (run #1287: 0.8 was never chosen; 18 plans instead of 27)
VALUE_STEPS = 5                 # the value head is read on the last steps of the plan (as wm2/cem.py)
START_SPEED = 0.8               # m/s for the first HISTORY - 1 calls, while there is no history to plan from
# ----- the cost settings of the best sim driver (E20 / E21), for choose()
FRONT_BASE, DECEL = 0.25, 4.0   # m kept in front + the braking distance v^2 / (2 DECEL)
SIDE_MARGIN = 0.45              # m to the side and diagonal walls
W_WALL, W_CENTER, CENTER_CLAMP = 20.0, 1.0, 0.3
W_VALUE, VALUE_CAP = 0.3, 6.0
# ----- time: the Garage ends the race (DNF) at the FIRST call slower than init["latency_cap_ms"] (250 ms so far);
# its CPU is ~3x slower than this PC (run #1097), and run #1281 died on call 4, the first plan (~115 ms here)
BUDGET_SHARE = 0.3              # the rollout stops adding steps after this share of the cap. Low because the Garage
                                # adds ~70-145 ms OUTSIDE step() (run #1287: step() own <= 120 ms, Garage p95 190 / 265 ms)
MIN_STEPS = 2                   # imagined steps always done, even past the budget (0.1 s)

GELU_C = math.sqrt(2 / math.pi)

def choose(walls: np.ndarray, progress: np.ndarray, speed: np.ndarray, value: np.ndarray | None,
           options: np.ndarray) -> int:
    """Pick the plan to drive. N plans, each HORIZON imagined steps of 0.05 s:
      walls     (N, H, 13)  imagined nearest wall in 13 sectors of 15 degrees, metres:
                            [..., 0] = right (-90), [..., 5:8] = ahead (-15, 0, +15), [..., 12] = left (+90)
      progress  (N, H)      imagined metres driven forward in each step
      speed     (N, H)      imagined speed, m/s (already >= 0)
      value     (N,)        value head: metres the car still drives in the 3 s after the plan; None without a head
      options   (N, 2)      the plan's [steering in model units (+ = left), speed m/s]
    -> the index of the plan with the lowest cost (0 .. N-1). The constants above are the sim driver's settings.
    """
    # the cost of the sim driver (wm2/cem.py sequence_cost, the settings of E20 / E21), summed over the steps
    front = walls[..., 5:8].min(-1)                                      # -15, 0, +15 degrees
    side = np.concatenate([walls[..., :5], walls[..., 8:]], -1).min(-1)  # the sides and diagonals
    front_short = np.maximum(0.0, FRONT_BASE + speed ** 2 / (2 * DECEL) - front)
    side_short = np.maximum(0.0, SIDE_MARGIN - side)
    center = np.clip(walls[..., 12] - walls[..., 0], -CENTER_CLAMP, CENTER_CLAMP) ** 2
    cost = (-progress + W_WALL * (front_short ** 2 + side_short ** 2) + W_CENTER * center).sum(1)
    if value is not None:                                                # the road after the plan (value head)
        cost = cost - W_VALUE * np.clip(value, 0.0, VALUE_CAP)
    return int(np.argmin(cost))


# ----- numpy pieces ----------------------------------------------------------------------------------------------
def gelu(x: np.ndarray) -> np.ndarray:
    # the tanh form of GELU: numpy has no erf, and an erf written in numpy took 25 % of a call. Checked against torch's
    # exact GELU (ail_cup/check_simple.py, wm2_c3): imagined walls within 0.025 m, value within 0.01 m
    return 0.5 * x * (1 + np.tanh(GELU_C * (x + 0.044715 * x * x * x)))


def layer_norm(x: np.ndarray) -> np.ndarray:
    d = x - x.mean(-1, keepdims=True)
    return d / np.sqrt((d * d).mean(-1, keepdims=True) + 1e-5)


class Wm2:
    # the world model of an export_wm2 file in numpy: encoder, predictor, sector heads, value head (optional)
    CONVS = [(5, 2), (5, 2), (5, 2), (3, 1)]          # kernel, padding of the 4 Conv1d (stride 2), wm2/model.py

    def __init__(self, arrays: dict):
        def get(name):
            return np.asarray(arrays[name], dtype=np.float32)

        self.convs = [(get(f"enc_c{i}_w").T.copy(), get(f"enc_c{i}_b"), k, pad) for i, (k, pad) in enumerate(self.CONVS)]
        names = ["enc_state", "enc_mix0", "enc_mix2", "enc_proj", "pred_inp", "pred_act0", "pred_act2", "pred_dz",
                 "pred_speed", "head_b0", "head_b2", "head_out"]
        self.has_value = all(f"val_{p}_w" in arrays for p in ["b0", "b2", "out"])
        if self.has_value:
            names += ["val_b0", "val_b2", "val_out"]
        self.dense_w = {n: (get(n + "_w").T.copy(), get(n + "_b")) for n in names}      # x @ w + b
        self.norm_w, self.norm_b = get("pred_norm_w"), get("pred_norm_b")
        self.pos = get("pred_pos")
        self.block_w = {p: (get(f"pred_{p}_w").transpose(0, 2, 1).copy(), get(f"pred_{p}_b"))   # (layer, in, out)
                        for p in ["q", "k", "v", "o", "m0", "m2", "ada"]}
        self.mask = np.triu(np.full((HISTORY, HISTORY), -np.inf, dtype=np.float32), 1)   # moment i sees <= i
        try:
            self.name = "".join(chr(int(c)) for c in np.asarray(arrays["meta_name"]))
        except Exception:
            self.name = "unknown (no meta_name in the file)"

    # x @ w on (plans, moments, in) runs as one small product per plan. Keep it so: OpenBLAS uses several threads only
    # for big products, and for these sizes its threads made calls slower and spikier (E34: one 2-D product per call
    # gave mean 85 / slowest 157 ms in garage_sim). The Garage allows no way to set the thread count (only maths + numpy).
    def dense(self, x: np.ndarray, name: str) -> np.ndarray:
        w, b = self.dense_w[name]
        return x @ w + b

    def layer(self, x: np.ndarray, l: int, part: str) -> np.ndarray:
        w, b = self.block_w[part]
        return x @ w[l] + b[l]

    def encode(self, lidar: np.ndarray, state: np.ndarray) -> np.ndarray:
        # lidar (109,) 0..1, state (2,) -> z (64,)
        x = lidar[None].astype(np.float32)                               # channels x length
        for w, b, k, pad in self.convs:
            padded = np.pad(x, ((0, 0), (pad, pad)))
            n = (padded.shape[1] - k) // 2 + 1
            cols = padded[:, 2 * np.arange(n)[:, None] + np.arange(k)]   # (channels, n, k)
            cols = cols.transpose(1, 0, 2).reshape(n, -1)                # one window per row, channel by channel
            x = gelu(cols @ w + b).T                                     # (out channels, n)
        flat = np.concatenate([x.reshape(-1), gelu(self.dense(state.astype(np.float32), "enc_state"))])
        return self.dense(self.dense(gelu(self.dense(flat, "enc_mix0")), "enc_mix2"), "enc_proj")

    def codes(self, actions: np.ndarray) -> np.ndarray:
        # actions (A, 2) -> (A, LAYERS, 6, WIDTH): what every action tells every layer (AdaLN)
        c = self.dense(gelu(self.dense(actions, "pred_act0")), "pred_act2")
        w, b = self.block_w["ada"]
        per_layer = c[None] @ w + b[:, None]                             # (LAYERS, A, 6 x WIDTH)
        return per_layer.transpose(1, 0, 2).reshape(len(actions), LAYERS, 6, WIDTH)

    def attention(self, h: np.ndarray, l: int, queries: int) -> np.ndarray:
        # causal self-attention; only the last `queries` moments ask (the others still give their keys and values)
        B, T, _ = h.shape
        hd = WIDTH // HEADS
        k, v = (self.layer(h, l, p).reshape(B, T, HEADS, hd).transpose(0, 2, 1, 3) for p in "kv")
        q = self.layer(h[:, -queries:], l, "q").reshape(B, queries, HEADS, hd).transpose(0, 2, 1, 3)
        s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd) + self.mask[T - queries : T, :T]
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        return self.layer((s @ v).transpose(0, 2, 1, 3).reshape(B, queries, WIDTH), l, "o")

    def predict(self, z: np.ndarray, code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # z (B, T, 64), code (B, T, LAYERS, 6, WIDTH) -> next z after the newest moment (B, 64), its speed / SPEED_MAX (B,)
        x = self.dense(z, "pred_inp") + self.pos[: z.shape[1]]
        for l in range(LAYERS):
            c = code[:, :, l]                                            # shift1, scale1, gate1, shift2, scale2, gate2
            h = layer_norm(x) * (1 + c[:, :, 1]) + c[:, :, 0]
            if l == LAYERS - 1:                                          # only the newest moment's output is used:
                x, c = x[:, -1:], c[:, -1:]                              # the last layer skips the other moments
            x = x + c[:, :, 2] * self.attention(h, l, x.shape[1])
            h = layer_norm(x) * (1 + c[:, :, 4]) + c[:, :, 3]
            x = x + c[:, :, 5] * self.layer(gelu(self.layer(h, l, "m0")), l, "m2")
        y = (layer_norm(x) * self.norm_w + self.norm_b)[:, -1]
        return z[:, -1] + self.dense(y, "pred_dz"), self.dense(y, "pred_speed")[:, 0]

    def imagine(self, z_hist: np.ndarray, past: np.ndarray, plans: np.ndarray,
                deadline: float = math.inf) -> tuple[np.ndarray, np.ndarray]:
        # z_hist (HISTORY, 64) real, past (HISTORY - 1, 2) the actions between them, plans (N, H, 2) model units
        # -> imagined z (N, h, 64), speed / SPEED_MAX (N, h). Like wm2.model.rollout, all plans at once.
        # deadline (time.perf_counter() seconds): no new step is started if it would end after it, so h can be < H
        # (at least MIN_STEPS). The Garage ends the race at the first call over its latency cap.
        n, steps = plans.shape[:2]
        actions = np.concatenate([np.broadcast_to(past, (n, HISTORY - 1, 2)), plans], axis=1)
        distinct, index = np.unique(actions.reshape(-1, 2), axis=0, return_inverse=True)
        codes, index = self.codes(distinct.astype(np.float32)), index.reshape(n, -1)
        window = np.broadcast_to(z_hist, (n, HISTORY, LATENT)).astype(np.float32)
        zs, speeds = [], []
        for s in range(steps):
            step_began = time.perf_counter()
            z_next, speed_next = self.predict(window, codes[index[:, s : s + HISTORY]])
            zs.append(z_next)
            speeds.append(speed_next)
            window = np.concatenate([window[:, 1:], z_next[:, None]], axis=1)
            now = time.perf_counter()
            if s + 1 >= MIN_STEPS and now + (now - step_began) > deadline:   # the next step would end too late
                break
        return np.stack(zs, 1), np.stack(speeds, 1)

    def heads(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # z (..., 64) -> 13 sector walls (..., 13) m, progress (...) m per step
        out = self.dense(gelu(self.dense(gelu(self.dense(z, "head_b0")), "head_b2")), "head_out")
        return out[..., :13], out[..., 14]

    def value(self, z: np.ndarray) -> np.ndarray:
        # z (..., 64) -> metres the car still drives in the next 3 s (wm2/value.py)
        return self.dense(gelu(self.dense(gelu(self.dense(z, "val_b0")), "val_b2")), "val_out")[..., 0]


# ----- the controller the Garage calls ---------------------------------------------------------------------------
class Controller:
    def __init__(self, init: dict):
        print(VERSION)
        init = init if isinstance(init, dict) else {}
        self.max_steer = float((init.get("car") or {}).get("max_steer", 0.4189))
        self.lidar_info = init.get("lidar") or {}
        self.model = None
        if init.get("model") is not None:
            try:                                                         # a bad model file must not stop the run
                self.model = Wm2(init["model"])
                print(f"INIT: MODEL = {self.model.name}, value head = {self.model.has_value}")
            except Exception as error:
                print("INIT: model NOT loaded, driving towards the longest free road:", repr(error))
        else:
            print("INIT: no model, driving towards the longest free road")
        steer, speed = np.meshgrid(STEER_OPTIONS, SPEED_OPTIONS, indexing="ij")
        self.options = np.column_stack([steer.ravel(), speed.ravel()])  # (N, 2): steering, m/s
        self.plans = np.repeat(np.column_stack([self.options[:, 1] * GAS_PER_MS / THROTTLE_MAX, self.options[:, 0]]
                                               )[:, None], HORIZON, axis=1).astype(np.float32)   # (N, H, 2) model units
        self.beams, self.angles = None, None
        self.z_hist, self.past = [], []
        self.budget_s = BUDGET_SHARE * float(init.get("latency_cap_ms") or 250.0) / 1000
        if self.model is not None:                                       # INIT has no cap: the first plan of the
            self.model.imagine(np.zeros((HISTORY, LATENT), np.float32),  # race must not pay numpy's first-use cost
                               np.zeros((HISTORY - 1, 2), np.float32), self.plans)

    def setup_beams(self, n: int) -> None:
        # which platform beam is each of the model's 109 beams (by angle)
        start = self.lidar_info.get("angle_min", math.radians(-135))
        inc = self.lidar_info.get("angle_increment", math.radians(270) / (n - 1))
        self.angles = np.degrees(start + inc * np.arange(n))
        want = np.radians(-135 + 2.5 * np.arange(BEAMS))
        self.beams = np.clip(np.round((want - start) / inc).astype(int), 0, n - 1)

    def step(self, obs: dict) -> dict:
        began = time.perf_counter()
        scan = np.nan_to_num(np.asarray(obs["lidar"], dtype=np.float32), nan=MAX_RANGE, posinf=MAX_RANGE)
        if self.beams is None:
            self.setup_beams(len(scan))
        speed_now, wheel_now = float(obs.get("speed", 0.0)), float(obs.get("steer", 0.0))
        if self.model is None:                                           # no model: the longest free road, slowly
            ahead = np.abs(self.angles) <= 60
            steer, speed = float(np.radians(self.angles[ahead][np.argmax(scan[ahead])])) * 0.8, START_SPEED
        else:
            lidar = np.minimum(scan[self.beams], MAX_RANGE) / MAX_RANGE
            z = self.model.encode(lidar, np.array([speed_now / SPEED_MAX, wheel_now / MODEL_MAX_STEER]))
            self.z_hist = (self.z_hist + [z])[-HISTORY:]
            if len(self.z_hist) < HISTORY:                               # the first calls: straight, slowly
                steer, speed = 0.0, START_SPEED
            else:
                z_im, v_im = self.model.imagine(np.stack(self.z_hist), np.array(self.past, np.float32), self.plans,
                                                deadline=began + self.budget_s)
                walls, progress = self.model.heads(z_im)
                value = self.model.value(z_im[:, -VALUE_STEPS:]).mean(1) if self.model.has_value else None
                best = choose(walls, progress, np.maximum(v_im * SPEED_MAX, 0.0), value, self.options)
                steer, speed = self.options[best, 0] * MODEL_MAX_STEER, float(self.options[best, 1])
        steer = float(max(-self.max_steer, min(self.max_steer, steer)))
        self.past = (self.past + [[speed * GAS_PER_MS / THROTTLE_MAX, steer / MODEL_MAX_STEER]])[-(HISTORY - 1):]
        return {"steer": steer, "speed": speed}
