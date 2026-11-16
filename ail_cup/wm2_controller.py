"""wm2 driver for the AIL Cup, written anew on 12 Nov 2026 (replaces claude_controller.py v11 - v20).

WHO STEERS: the real lidar, every call (20 per second). The world model only ADVISES, and only in scenes it knows.
Why (docs/research_log.md E22, E23): until v20 the world model steered. In the Garage it decides once per ~0.75 s
(pure Python), and it reads the wide Garage tracks wrongly: a corridor with 1.29 m to each side as 1.39 m right /
0.79 m left. It learned on the AutoDRIVE tracks (0.65 - 1.0 m to each side). v19 and v20 steered away from walls
that were not there. (An earlier line here said it also reads 10 m of free road as 3.3 m: that compared two
different things and was wrong, see E25.)

Every call:
  1. LOOK   the real scan -> the free road in every direction, the nearest wall on each side, the wall ahead
  2. STEER  a smooth law: turn towards the direction with the longest free road (within +-60 degrees), move to the
            middle of the corridor, and push away from a wall that is close. One formula, no modes that switch.
            The wheel moves at most RATE per call, so a decision can never become a swerve.
  3. ADVISE the world model (encoder + predictor + heads + value head, the file <RUN>_value_garage.npz of
            "python -m ail_cup.export_wm2 <RUN>") compares "a bit more right / as now / a bit more left" in its
            imagination, like the sim driver (wm2/cem.py). Its answer is added to the law's steering as a small push
            that fades within ADVICE_CALLS calls. A plan starts only when the scene is like the model's training
            (KNOWN_SIDE, KNOWN_FRONT); the status line shows how often that was (m = share of calls with advice).
  4. SPEED  slower when the road ahead is short or the wheel is turned
  5. GUARD  the platform's map only says which way is forward: if the car points more than TURN_DEG off, it turns back.

Pure Python, no numpy. The network part is the code of claude_controller.py, checked against the torch model by
ail_cup/check_controller.py (option costs equal to 0.001). Try it on this PC first:
    venv/Scripts/python -m ail_cup.garage_sim ail_cup.wm2_controller oval ail_cup/wm2_c2_value_garage.npz
"""
import math
import operator
import time

VERSION = "wm2_controller v1 (lidar law steers every call, wm2 advises in known scenes)"
# status line: t time, s steering sent (+ = left), g the law's direction (deg), c centre push, a the model's push,
#              [model costs: a step right, as now, a step left], v speed, F R L real walls (m), e degrees to the
#              track's forward, p plans / m share of calls with advice (%), slowest call, prog
BEAMS = 109                     # the beams wm2 knows: -135..+135 degrees, 2.5 degrees apart, right -> left
MAX_RANGE = 10                  # m: lidar / MAX_RANGE -> 0..1, like core/sensors.clean_lidar
SPEED_MAX = 3                   # m/s: the model gets speed / SPEED_MAX
THROTTLE_MAX = 0.09             # the model gets AutoDRIVE throttle / THROTTLE_MAX
MODEL_MAX_STEER = 0.5236        # rad: model steering +-1 = +-30 degrees of the AutoDRIVE car
GAS_PER_MS = 1 / 24.5           # AutoDRIVE throttle per m/s of steady speed
HISTORY, WIDTH, HEADS, LAYERS = 4, 128, 4, 4

# ----- the lidar law
CONE_DEG = 60                   # degrees: directions looked at for the longest free road
GAP_HALF = 12                   # degrees: half width of the window whose NEAREST beam is "the free road" that way
FREE_CAP = 6.0                  # m: a free road longer than this is not better (stops chasing far corners)
STRAIGHT_BONUS = 0.012          # m per degree: of two equal directions the one nearer to straight ahead wins
K_GAP = 0.6                     # steering (model units) per 30 degrees of direction to the longest free road
K_CENTER = 0.8                  # steering per metre the car is off the middle between the side walls
CENTER_CAP = 0.8                # m: the off-middle distance counts up to this
SIDE_SAFE = 0.55                # m: a side or diagonal wall closer than this pushes the wheel away ...
K_SAFE = 1.6                    # ... by this much steering per missing metre
RATE = 0.10                     # steering (model units) the command may change per call: 3 degrees, 60 degrees / s
STEER_MAX = 0.8                 # model units; 0.8 x 30 deg = 24 deg = the AIL Cup maximum
# ----- the speed
SPEED_TOP = 1.6                 # m/s on a long free road with the wheel straight (v18 - v20: 1.2 everywhere)
SPEED_LOW = 0.6                 # m/s at least
SLOW_FRONT = 3.5                # m: with less free road ahead the speed falls linearly, to SPEED_LOW at STOP_FRONT
STOP_FRONT = 0.5
SLOW_TURN = 0.45                # share of the speed taken away at full lock
# ----- the model's advice
STEPS = 8                       # imagined steps of 0.05 s per option (0.4 s); the value head looks after them
STEER_STEP = 0.25               # the options: the wheel as it is -+ this (7.5 degrees)
ADVICE_MIN = 0.10               # cost by which the best option must beat "as now", or there is no advice
ADVICE_PUSH = 0.15              # steering added by an advice ...
ADVICE_CALLS = 14               # ... fading to 0 within this many calls (0.7 s)
KNOWN_SIDE = (0.55, 1.10)       # m: the model reads a scene right when BOTH side walls are in this range ...
KNOWN_FRONT = 4.5               # ... and the free road ahead is shorter than this (E23)
CALL_BUDGET = 0.09              # s for the whole call (the Garage cap: 250 ms; v18 - v20 ran with this)
HARD_LIMIT = 0.12               # s: the first piece of a call may start while a call stays below this
FRONT_BASE, DECEL = 0.4, 4.0    # m in front + the braking distance v^2 / (2 DECEL)
SIDE_MARGIN = 0.45              # m to the side walls in the imagination
W_WALL, W_CENTER, CENTER_CLAMP = 20.0, 1.0, 0.3
W_VALUE = 0.3 * STEPS / 20      # per metre of the value head (the sim driver's 0.3 belongs to 20-step plans)
VALUE_STEPS, VALUE_CAP = 3, 6.0
# ----- the map guard
LOOKAHEAD = 1.5                 # m along the centreline: "forward" = the direction to that point
TRACK_SEARCH = 30               # centreline points around the last one searched for the nearest
TURN_DEG = 100                  # degrees off forward: the car faces back -> turn round slowly
TURN_SPEED = 0.5                # m/s
PRINT_EVERY = 10                # calls between two status lines (the Garage cuts a long output)

if hasattr(math, "sumprod"):    # Python 3.12+: a fast dot product written in C
    dot = math.sumprod
else:
    def dot(a, b):
        return sum(map(operator.mul, a, b))

ERF_SCALE = 1 / math.sqrt(2)
ATT_SCALE = 1 / math.sqrt(WIDTH // HEADS)


def as_list(array):
    return array.tolist() if hasattr(array, "tolist") else array


def lin(x, w, b):
    return [dot(row, x) + c for row, c in zip(w, b)]


def gelu(v):
    erf = math.erf
    return [0.5 * a * (1 + erf(a * ERF_SCALE)) for a in v]


def norm(v):
    n = len(v)
    m = sum(v) / n
    d = [a - m for a in v]
    s = 1 / math.sqrt(dot(d, d) / n + 1e-5)
    return [a * s for a in d]


def short(value):
    # a short description for printing: numbers and text as they are, big lists / dicts only by their size and keys
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {short(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return f"list of {len(value)}" + (f", first {short(value[0])}" if value else "")
    if hasattr(value, "shape"):
        return f"array {value.shape}"
    return repr(value)


def pick(box, key, default):
    # box[key], but if box is not a dict or the key is missing / None -> default (and say so)
    if isinstance(box, dict) and box.get(key) is not None:
        return box[key]
    print(f"INIT: no {key!r}, using {default!r}")
    return default


class Controller:
    def __init__(self, init):
        print(VERSION)
        print("INIT keys:", list(init) if isinstance(init, dict) else type(init))
        car = pick(init, "car", {})
        lidar = pick(init, "lidar", None)
        if isinstance(car, dict):
            print("INIT car:", car)
        if isinstance(lidar, dict):
            print("INIT lidar:", lidar)
        self.max_steer = pick(car, "max_steer", 0.4189)
        self.lidar_info = lidar if isinstance(lidar, dict) else None
        self.angles = None                                                   # degrees of every beam, built on the first call
        self.track, self.track_i, self.track_step, self.track_ahead = None, None, 0.0, 1
        try:
            line = [(float(p[0]), float(p[1])) for p in as_list(init["track"]["centerline"])]
            length = sum(math.dist(a, b) for a, b in zip(line, line[1:] + line[:1]))
            self.track, self.track_step = line, length / len(line)
            self.track_ahead = max(1, round(LOOKAHEAD / self.track_step))
            print(f"INIT track: length {length:.1f} m, {len(line)} centreline points")
        except Exception as error:
            print("INIT: no track map, no wrong-way guard:", repr(error))
        self.w, self.has_value, self.model_name = None, False, "none"
        model = init.get("model") if isinstance(init, dict) else None
        if model is not None:
            try:                                                             # a bad model must not stop the run
                self.unpack(model)
                print(f"INIT: wm2 loaded, MODEL = {self.model_name}, value head = {self.has_value}")
            except Exception as error:
                self.w = None
                print("INIT: wm2 NOT loaded, the lidar law drives alone:", repr(error))
        else:
            print("INIT: no model, the lidar law drives alone")
        self.history = []                                                    # [(lidar, state), action sent] of the last calls
        self.codes_cache = {}
        self.thinking = None                                                 # the advice being computed (a generator)
        self.moment_s = 0.0
        self.slowest_ms = 0.0
        self.last_costs = []
        self.wheel = 0.0                                                     # the steering sent last (model units)
        self.advice, self.advice_at = 0.0, -10 ** 6
        self.calls = self.plans = self.advised_calls = 0

    # ----- the network ------------------------------------------------------------------------------------------
    def unpack(self, model):
        def get(name):
            return as_list(model[name])
        w = {}
        for i in range(4):
            w[f"enc_c{i}"] = (get(f"enc_c{i}_w"), get(f"enc_c{i}_b"))
        for name in ["enc_state", "enc_mix0", "enc_mix2", "enc_proj", "pred_inp", "pred_act0", "pred_act2",
                     "pred_norm", "pred_dz", "pred_speed", "head_b0", "head_b2", "head_out"]:
            w[name] = (get(name + "_w"), get(name + "_b"))
        w["pred_pos"] = get("pred_pos")
        self.has_value = all(f"val_{part}_w" in model for part in ["b0", "b2", "out"])   # v18: optional
        if self.has_value:
            for part in ["b0", "b2", "out"]:
                w[f"val_{part}"] = (get(f"val_{part}_w"), get(f"val_{part}_b"))
        stacked = {part: (get(f"pred_{part}_w"), get(f"pred_{part}_b"))     # (4 blocks, ...) each, see export_wm2
                   for part in ["q", "k", "v", "o", "m0", "m2", "ada"]}
        w["layers"] = [{part: (weight[l], bias[l]) for part, (weight, bias) in stacked.items()} for l in range(LAYERS)]
        try:                                                                 # the model's name, written by export_wm2
            self.model_name = "".join(chr(int(c)) for c in get("meta_name"))
        except Exception:
            self.model_name = "unknown (no meta_name in the file)"
        self.w = w

    def conv(self, x, name, k, pad):
        # x: channels x length -> GELU(Conv1d stride 2): out channels x new length
        weight, bias = self.w[name]
        padded = [[0.0] * pad + ch + [0.0] * pad for ch in x]
        n_out = (len(x[0]) + 2 * pad - k) // 2 + 1
        columns = []
        for j in range(n_out):
            window = []
            for ch in padded:
                window.extend(ch[2 * j:2 * j + k])
            columns.append(gelu(lin(window, weight, bias)))
        return [list(row) for row in zip(*columns)]                         # positions x channels -> channels x positions



    def encode_pieces(self, lidar, state):
        # the encoder as a generator: a pause after every layer (each ~3-8 ms on the laptop); returns z (64)
        x = self.conv([lidar], "enc_c0", 5, 2)
        yield
        x = self.conv(x, "enc_c1", 5, 2)
        yield
        x = self.conv(x, "enc_c2", 5, 2)
        yield
        x = self.conv(x, "enc_c3", 3, 1)
        yield
        flat = [v for ch in x for v in ch]                                   # like nn.Flatten: channel by channel
        flat += gelu(lin(state, *self.w["enc_state"]))
        h = lin(gelu(lin(flat, *self.w["enc_mix0"])), *self.w["enc_mix2"])
        return lin(h, *self.w["enc_proj"])

    def encode(self, lidar, state):
        # the same in one go (tests)
        pieces = self.encode_pieces(lidar, state)
        while True:
            try:
                next(pieces)
            except StopIteration as done:
                return done.value

    def codes_pieces(self, action):
        # what this action tells every layer (AdaLN): shift1, scale1, gate1, shift2, scale2, gate2 per layer, cached;
        # a generator with a pause after every layer (v12: one piece of all 4 was the biggest one)
        key = (round(action[0], 4), round(action[1], 4))
        if key not in self.codes_cache:
            c = lin(gelu(lin(list(key), *self.w["pred_act0"])), *self.w["pred_act2"])
            per_layer = []
            for layer in self.w["layers"]:
                yield
                full = lin(c, *layer["ada"])
                per_layer.append([full[i * WIDTH:(i + 1) * WIDTH] for i in range(6)])
            self.codes_cache[key] = per_layer
        return self.codes_cache[key]

    def token_pieces(self, z, action, pos, kv, last):
        # one moment of the predictor window, causal: kv holds the keys/values of the earlier moments in every layer
        # and gets this moment's. Returns (next z, next speed) for the newest moment (last=True), else None.
        # A generator with a pause after every layer (~3-8 ms each on the laptop).
        code = yield from self.codes_pieces(action)
        x = [a + p for a, p in zip(lin(z, *self.w["pred_inp"]), self.w["pred_pos"][pos])]
        for l, layer in enumerate(self.w["layers"]):
            if l:
                yield
            sh1, sc1, g1, sh2, sc2, g2 = code[l]
            h = [n * (1 + s) + t for n, s, t in zip(norm(x), sc1, sh1)]
            keys, values = kv[l]
            keys.append(lin(h, *layer["k"]))
            values.append(lin(h, *layer["v"]))
            if l == LAYERS - 1 and not last:                                 # its output is never used
                return None
            q = lin(h, *layer["q"])
            out = []
            hd = WIDTH // HEADS
            for head in range(HEADS):
                a, b = head * hd, (head + 1) * hd
                qh = q[a:b]
                scores = [dot(qh, key[a:b]) * ATT_SCALE for key in keys]
                top = max(scores)
                e = [math.exp(s - top) for s in scores]
                total = sum(e)
                mix = [0.0] * hd
                for weight, value in zip(e, values):
                    wgt = weight / total
                    mix = [m + wgt * v for m, v in zip(mix, value[a:b])]
                out += mix
            o = lin(out, *layer["o"])
            x = [xa + g * oa for xa, g, oa in zip(x, g1, o)]
            h = [n * (1 + s) + t for n, s, t in zip(norm(x), sc2, sh2)]
            m = lin(gelu(lin(h, *layer["m0"])), *layer["m2"])
            x = [xa + g * ma for xa, g, ma in zip(x, g2, m)]
        nw, nb = self.w["pred_norm"]
        y = [n * a + b for n, a, b in zip(norm(x), nw, nb)]
        dz = lin(y, *self.w["pred_dz"])
        sw, sb = self.w["pred_speed"]
        return [a + b for a, b in zip(z, dz)], dot(y, sw[0]) + sb[0]

    def heads(self, z):
        # z -> 13 sector walls (m, -90 right .. +90 left), danger logit, progress (m per step)
        h = gelu(lin(gelu(lin(z, *self.w["head_b0"])), *self.w["head_b2"]))
        out = lin(h, *self.w["head_out"])
        return out[:13], out[14]

    def value(self, z):
        # z -> metres the car is expected to still drive in the next 3 s (wm2/value.py)
        h = gelu(lin(gelu(lin(z, *self.w["val_b0"])), *self.w["val_b2"]))
        return lin(h, *self.w["val_out"])[0]


    def beam_at(self, angle, n):
        # the index of the platform beam closest to this angle (degrees)
        return min(max(round((math.radians(angle) - self.angle_start) / self.angle_step), 0), n - 1)

    # ----- where is "forward" on the track (the map the platform gives) ---------------------------------------------
    def track_error(self, pose):
        # degrees from the car's heading to the track's forward direction (a centerline point LOOKAHEAD m ahead):
        # + = forward is to the left, |e| > 90 = the car drives the wrong way. None without a map or pose.
        if not self.track or not isinstance(pose, dict) or "x" not in pose:
            return None
        x, y, yaw = pose["x"], pose["y"], pose.get("yaw", 0.0)
        pts, n = self.track, len(self.track)
        if self.track_i is None:                                             # first time: the whole map, and which
            self.track_i = min(range(n), key=lambda i: (pts[i][0] - x) ** 2 + (pts[i][1] - y) ** 2)   # way is forward
            i = self.track_i
            dx, dy = pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1]
            if dx * math.cos(yaw) + dy * math.sin(yaw) < 0:                  # the car starts facing forward
                self.track.reverse()
                self.track_i = n - 1 - i
            print(f"INIT track: {n} points, step {self.track_step:.2f} m, start at point {self.track_i}")
        else:                                                                # then: near the last point only ...
            near = [(self.track_i + k) % n for k in range(-TRACK_SEARCH, TRACK_SEARCH + 1)]
            self.track_i = min(near, key=lambda i: (pts[i][0] - x) ** 2 + (pts[i][1] - y) ** 2)
            if (pts[self.track_i][0] - x) ** 2 + (pts[self.track_i][1] - y) ** 2 > 1.5 ** 2:   # ... unless lost
                self.track_i = min(range(n), key=lambda i: (pts[i][0] - x) ** 2 + (pts[i][1] - y) ** 2)
        tx, ty = pts[(self.track_i + self.track_ahead) % n]
        e = math.degrees(math.atan2(ty - y, tx - x) - yaw)
        return (e + 180) % 360 - 180


    # ----- the model's advice: a generator, paused after every piece (like claude_controller.think) ----------------
    def advise(self, frames, past, wheel, speed):
        zh = []
        for lidar, state in frames:
            z = yield from self.encode_pieces(lidar, state)
            zh.append(z)
        gas = speed * GAS_PER_MS / THROTTLE_MAX
        now = round(wheel * 20) / 20
        options = [max(-STEER_MAX, min(STEER_MAX, round(now + d, 3))) for d in (-STEER_STEP, 0.0, STEER_STEP)]
        costs = []
        for steer in options:
            action, imagined, cost = [gas, steer], [], 0.0
            for s in range(STEPS):
                kv = [([], []) for _ in range(LAYERS)]
                result = None
                for i in range(HISTORY):
                    n = s + i                                                # moment number from the plan's start
                    z = zh[n] if n < HISTORY else imagined[n - HISTORY]
                    result = yield from self.token_pieces(z, past[n] if n < HISTORY - 1 else action, i, kv,
                                                          i == HISTORY - 1)
                    yield
                z_next, v_next = result
                imagined.append(z_next)
                walls, progress = self.heads(z_next)
                v = max(0.0, v_next * SPEED_MAX)
                front, side = min(walls[5:8]), min(walls[:5] + walls[8:])
                front_short = max(0.0, FRONT_BASE + v * v / (2 * DECEL) - front)
                side_short = max(0.0, SIDE_MARGIN - side)
                center = max(-CENTER_CLAMP, min(CENTER_CLAMP, walls[12] - walls[0])) ** 2
                cost += -progress + W_WALL * (front_short ** 2 + side_short ** 2) + W_CENTER * center
                yield
            if self.has_value:
                ahead = sum(self.value(z) for z in imagined[-VALUE_STEPS:]) / VALUE_STEPS
                cost -= W_VALUE * max(0.0, min(VALUE_CAP, ahead))
                yield
            costs.append(cost)
        best = costs.index(min(costs))
        self.last_costs = costs
        self.plans += 1
        if best != 1 and costs[1] - costs[best] >= ADVICE_MIN:               # clearly better than "as now"
            self.advice, self.advice_at = (ADVICE_PUSH if best == 2 else -ADVICE_PUSH), self.calls
        else:
            self.advice = 0.0

    # ----- the lidar law ------------------------------------------------------------------------------------------
    def nearest(self, scan, low, high):
        # the nearest beam between two angles (degrees)
        return min((r for a, r in zip(self.angles, scan) if low <= a <= high), default=MAX_RANGE)

    def look(self, scan):
        # -> direction of the longest free road (deg, + = left), free road straight ahead, nearest wall right / left
        best_angle, best_score = 0.0, -1.0
        for angle in range(-CONE_DEG, CONE_DEG + 1, 4):
            free = min(self.nearest(scan, angle - GAP_HALF, angle + GAP_HALF), FREE_CAP)
            score = free - STRAIGHT_BONUS * abs(angle)
            if score > best_score:
                best_angle, best_score = angle, score
        ahead = self.nearest(scan, -GAP_HALF, GAP_HALF)
        return best_angle, ahead, self.nearest(scan, -105, -25), self.nearest(scan, 25, 105)

    def law(self, gap, right, left):
        # the steering the lidar alone asks for (model units, + = left), and its centre part for the status line
        centre = K_CENTER * max(-CENTER_CAP, min(CENTER_CAP, (left - right) / 2))
        steer = K_GAP * gap / 30 + centre
        if left < SIDE_SAFE:
            steer -= K_SAFE * (SIDE_SAFE - left)
        if right < SIDE_SAFE:
            steer += K_SAFE * (SIDE_SAFE - right)
        return max(-STEER_MAX, min(STEER_MAX, steer)), centre

    # ----- one call -------------------------------------------------------------------------------------------
    def step(self, obs):
        began = time.perf_counter()
        self.calls += 1
        scan = [r if math.isfinite(r) else MAX_RANGE for r in obs["lidar"]]
        n = len(scan)
        if self.angles is None:
            info = self.lidar_info
            if info and "angle_min" in info and "angle_increment" in info:
                start, inc = info["angle_min"], info["angle_increment"]
            else:
                start, inc = math.radians(-135), math.radians(270) / (n - 1)
                print(f"INIT: no lidar angles, assuming 270 degrees over {n} beams")
            self.angle_start, self.angle_step = start, inc
            self.angles = [math.degrees(start + i * inc) for i in range(n)]
            self.beams = [self.beam_at(-135 + i * 270 / (BEAMS - 1), n) for i in range(BEAMS)]
            print(f"INIT: {n} beams from {self.angles[0]:.0f} to {self.angles[-1]:.0f} degrees; OBS keys:",
                  [k for k in obs] if isinstance(obs, dict) else type(obs))
        speed_now = obs.get("speed", 0.0)
        gap, ahead, right, left = self.look(scan)
        target, centre = self.law(gap, right, left)

        known = (KNOWN_SIDE[0] <= right <= KNOWN_SIDE[1] and KNOWN_SIDE[0] <= left <= KNOWN_SIDE[1]
                 and ahead <= KNOWN_FRONT)
        if self.w is not None:
            lidar = [min(scan[b], MAX_RANGE) / MAX_RANGE for b in self.beams]
            frame = (lidar, [speed_now / SPEED_MAX, obs.get("steer", 0.0) / MODEL_MAX_STEER])
            self.history = (self.history + [[frame, None]])[-HISTORY:]
            if self.thinking is None and known and len(self.history) == HISTORY:
                self.thinking = self.advise([f[0] for f in self.history], [f[1] for f in self.history[:-1]],
                                            self.wheel, max(speed_now, SPEED_LOW))
            done = 0
            while self.thinking is not None:                                 # piece by piece within the budget
                elapsed = time.perf_counter() - began
                if elapsed + self.moment_s > (CALL_BUDGET if done else HARD_LIMIT) and (done or elapsed > 0.02):
                    break
                done += 1
                t0 = time.perf_counter()
                try:
                    next(self.thinking)
                except StopIteration:
                    self.thinking = None
                self.moment_s = max(time.perf_counter() - t0, 0.9 * self.moment_s)
        push = 0.0
        if known and self.advice:                                            # the advice fades, and only counts in
            age = self.calls - self.advice_at                                # a scene the model knows
            push = self.advice * max(0.0, 1 - age / ADVICE_CALLS)
            self.advised_calls += push != 0.0
        target = max(-STEER_MAX, min(STEER_MAX, target + push))

        e = self.track_error(obs.get("pose"))
        speed = SPEED_TOP
        if ahead < SLOW_FRONT:
            speed = SPEED_LOW + (SPEED_TOP - SPEED_LOW) * max(0.0, (ahead - STOP_FRONT) / (SLOW_FRONT - STOP_FRONT))
        if e is not None and abs(e) > TURN_DEG:                              # facing back: turn round, slowly
            target, speed = (STEER_MAX if e > 0 else -STEER_MAX), TURN_SPEED
        self.wheel += max(-RATE, min(RATE, target - self.wheel))
        speed = max(SPEED_LOW if speed > TURN_SPEED else TURN_SPEED, speed * (1 - SLOW_TURN * abs(self.wheel) / STEER_MAX))
        steer = max(-self.max_steer, min(self.max_steer, self.wheel * MODEL_MAX_STEER))
        if self.w is not None:                                               # what the model sees as this action
            self.history[-1][1] = [speed * GAS_PER_MS / THROTTLE_MAX, steer / MODEL_MAX_STEER]

        self.slowest_ms = max(self.slowest_ms, 1000 * (time.perf_counter() - began))
        if self.calls % PRINT_EVERY == 0:
            costs = " ".join(f"{c:+.2f}" for c in self.last_costs) if self.last_costs else "-"
            print(f"t{self.calls / 20:5.1f} s{self.wheel:+.2f} g{gap:+.0f} c{centre:+.2f} a{push:+.2f} [{costs}]" 
                  f" v{speed_now:.2f} F{ahead:.2f} R{right:.2f} L{left:.2f} e{(e if e is not None else 0):+.0f}"
                  f" p{self.plans} m{100 * self.advised_calls / self.calls:.0f}% {self.slowest_ms:.0f}ms"
                  f" prog{obs.get('progress', 0):.1f}")
            self.slowest_ms = 0.0
        return {"steer": steer, "speed": speed}
#       t 31.0 s+0.01 g+0 c+0.01 a+0.00 [-] v1.56 F6.01 R1.29 L1.30 e-3 p0 m0% 1ms prog48.0
