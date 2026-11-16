"""wm2 lidar driver for the AIL Cup (v14): the world model wm2_self1 in pure Python (no numpy), no weights in the file.
Claude's working file: Claude edits it, the user uploads it to the Garage and reports what happened.

The platform gives the network as init["model"]: ail_cup/<RUN>_garage.npz from `python -m ail_cup.export_wm2 wm2_self1`
(encoder + predictor + 13 sector heads, ~1.67 M numbers, float16, 49 arrays).

Every call (20 per second):
  1. the newest lidar (109 beams = the AIL Cup lidar itself) + speed + wheel angle are kept for the last 4 calls
     (the predictor's history, 0.2 s); the encoder turns them into z's only when a new plan starts
  2. the planner (a generator, think()) goes on piece by piece until the call has used CALL_BUDGET s: it encodes the
     4 frames, then for every steering option imagines STEPS steps with the predictor, reads the 13 sector walls +
     progress from every imagined z with the heads and adds up a cost like the sim driver (wm2/cem.py
     sequence_cost: walls closer than a speed-dependent margin, progress, centre; W_SWITCH scaled to STEPS)
  3. when all options are done, the cheapest one becomes the steering; if even it comes closer than RESCUE_M to a
     wall, the simple rule of v10 steers (deepest opening + push away from close side walls)
  4. reflex every call: a REAL side / diagonal wall closer than REFLEX_M -> the simple rule steers at once
Pure Python is slow (a full plan ~1.4 s of work on the laptop; run #1223: the Garage is as fast), so a plan is
spread over ~15 calls; until it is finished the car keeps the last decision. Pieces of ~1-10 ms: one encoder layer,
one action-code layer or one predictor layer. Faster than a plain port:
  - the first imagined steps share their first window moments (real z's + past actions) for every option:
    those moments are computed once per plan (token by token with a cache of keys/values, like a causal LM)
  - in the last layer, moments before the newest one need only their keys/values (their output is not used)
  - the action codes (AdaLN) are cached per distinct action; the encoder runs only for the frames of a new plan

Versions: v11 first wm2 port (DNF 270 ms: encoder every call), v12 generator, v13 layer-sized pieces (run #1223: no
DNF, but steering always 0 -> wall), v14 W_SWITCH scaled to the plan length + real-lidar reflex.

Safe start: anything missing in init (car, lidar, model) gets a default and is printed. Without the network the
car drives by the simple rule, so a run still shows the prints.
"""
import math
import operator
import time

VERSION = "claude_controller v17 (wm2, 3x10 plan, reflexes, track map: forward guard)"
# status line: t time, mode (brain / RESCUE / REFLEX / TRACK / TURN), s steering (+ = left),
#              [option costs: right, straight, left], v speed, F R L real walls (m),
#              e degrees to the track's forward direction (+ = left),
#              p plans done / calls since the last decision, w closest imagined wall (m), slowest call, prog
BEAMS = 109                     # the beams wm2 knows: -135..+135 degrees, 2.5 degrees apart, right -> left
MAX_RANGE = 10                  # m: lidar / MAX_RANGE -> 0..1, like core/sensors.clean_lidar
SPEED_MAX = 3                   # m/s: the model gets speed / SPEED_MAX
THROTTLE_MAX = 0.09             # the model gets AutoDRIVE throttle / THROTTLE_MAX
MODEL_MAX_STEER = 0.5236        # rad: model steering +-1 = +-30 degrees of the AutoDRIVE car
GAS_PER_MS = 1 / 24.5           # AutoDRIVE throttle per m/s of steady speed (0.05 -> 1.22 m/s, 0.12 -> ~2.9 m/s)
HISTORY, WIDTH, HEADS, LAYERS = 4, 128, 4, 4

SPEED = 1.2                     # m/s, the command speed (the plan imagines driving at it)
OPTIONS = [-0.8, 0.0, 0.8]      # steering options in model units; 0.8 x 30 deg = 24 deg = the AIL Cup maximum
STEPS = 10                      # imagined steps of 0.05 s per option (0.5 s, ~0.6 m). v14 had 5 options x 4 steps: their
                                # costs differed by ~0.01 (run #1224: -0.17 -0.22 -0.28 -0.27 -0.24), 0.2 s of
                                # imagination cannot tell left from straight -> fewer options, longer look
CALL_BUDGET = 0.09              # s for the WHOLE call; the Garage cap is 250 ms (run #1223 with 0.08: calls 76-80 ms,
                                # Garage step() p95 165 ms, pieces 4-10 ms like the laptop). Thinking goes one piece (an encoder
                                # layer, an action code layer or one predictor layer, ~1-8 ms laptop) at a time and
                                # starts the next one only if it still fits, so a call ends ~one piece after CALL_BUDGET
                                # (v5's BUDGET 0.2 broke the cap in run #1097, v11's encoder every call: 270 ms,
                                # v12 with 0.12 s and 30-ms pieces was still too slow in the Garage)
HARD_LIMIT = 0.12               # s: the first piece of a call may start while a call stays below this
FRONT_BASE = 0.4                # m kept in front even when slow ...
DECEL = 4.0                     # ... plus the braking distance v^2 / (2 DECEL)
SIDE_MARGIN = 0.35              # m to the side walls and diagonals
W_WALL, W_CENTER = 20.0, 0.3   # cost weights per imagined step, as the sim driver wm2/drive.py
W_SWITCH = 0.3 * STEPS / 20     # per unit of steering change, once per plan: drive.py's 0.3 belongs to 20-step plans.
                                # Unscaled (v13, run #1223) it outweighed everything a 4-step plan could gain: the
                                # brain kept steering 0 while the road turned left and drove into the right wall
REFLEX_M = 0.7                  # m: a real side / diagonal wall closer than this -> the simple rule steers at once
RESCUE_M = 0.25                 # m: if the best imagined path comes closer than this to a wall, the simple rule steers
SAFE_FRONT = 1.6                # m: closer in front -> slow down linearly ...
STOP_FRONT = 0.35               # m: ... to MIN_SPEED at this distance
MIN_SPEED = 0.3                 # m/s
FRONT_REFLEX_M = 1.0            # m: a real wall ahead (+-10 deg) closer than this -> the simple rule steers at once
LOOKAHEAD = 1.5                 # m along the track's centerline: "forward" = the direction to that point
TRACK_SEARCH = 30               # centerline points around the last one searched for the nearest (~6 m each way)
CONE_DEG = 60                   # degrees: the simple rule looks for openings only this far from "forward"
TRACK_DEG = 45                  # degrees: the brain's steering is overruled when the car points this far off forward
TURN_DEG = 100                  # degrees: beyond this the car faces back -> turn round slowly (mode TURN)
TURN_SPEED = 0.5                # m/s
PRINT_EVERY = 10                # calls between two status lines (0.5 s): the Garage cuts a long output (run #1224)
GAP_HALF = 10                   # degrees: half width of an opening the simple rule steers into
SIDE_MIN = 0.5                  # m: a side wall closer than this pushes the simple rule's steering away from it

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
        print(VERSION)                                                       # which file really runs in the Garage
        print("INIT type:", type(init), "keys:", list(init) if isinstance(init, dict) else "-")   # no dunders: the Garage forbids them
        car = pick(init, "car", {})
        lidar = pick(init, "lidar", None)
        if isinstance(car, dict):
            print("INIT car:", car)
        if isinstance(lidar, dict):
            print("INIT lidar:", lidar)
        if isinstance(init, dict):
            for key in init:
                if key not in ("car", "lidar", "model", "models", "track"):  # big ones: one short line each
                    print(f"INIT {key}: {short(init[key])}")
            model_in = init.get("model")
            if isinstance(model_in, dict):                                   # not every array: the Garage cuts long
                print(f"INIT model: {len(model_in)} arrays")                  # output (run #1224 lost its crash)
            else:
                print("INIT model:", type(model_in))
        self.max_steer = pick(car, "max_steer", 0.4189)
        self.lidar_info = lidar if isinstance(lidar, dict) else None
        self.beams = None                                                    # built on the first step()
        self.track, self.track_i, self.track_step, self.track_ahead = None, None, 0.0, 1
        try:                                                                 # the map: centerline points [x, y, ...]
            line = [(float(p[0]), float(p[1])) for p in as_list(init["track"]["centerline"])]
            length = sum(math.dist(a, b) for a, b in zip(line, line[1:] + line[:1]))
            self.track, self.track_step = line, length / len(line)
            self.track_ahead = max(1, round(LOOKAHEAD / self.track_step))
            print(f"INIT track: length {length:.1f} m, {len(line)} centerline points")
        except Exception as error:
            print("INIT: no track map, no wrong-way guard:", repr(error))
        self.w = None                                                        # None = no network: simple rule
        model = pick(init, "model", None)
        if model is not None:
            try:                                                             # a bad model must not stop the run
                self.unpack(model)                                           # not "load": the Garage forbids that name
                print(f"INIT: wm2 loaded, MODEL = {self.model_name}, controller = {VERSION}")
            except Exception as error:
                self.w = None
                print("INIT: wm2 NOT loaded, driving by the simple rule:", repr(error))
        self.history = []                                                    # [z, action sent then] of the last frames
        self.codes_cache = {}
        self.thinking = None                                                 # the plan being computed (a generator)
        self.moment_s = 0.0                                                  # s, how long one paused piece took
        self.slowest_ms = 0.0                                                # the slowest call since the last print
        self.last_costs = []                                                 # the option costs of the last plan
        self.mode = "brain "
        self.previous = 0.0                                                  # the steering decided last (model units)
        self.rescue = False
        self.calls = self.plans = self.rescue_calls = 0
        self.decided_at = 0
        self.best_cost, self.best_wall = 0.0, 0.0

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

    # ----- the planner: a generator, paused after every predictor moment ------------------------------------------
    def think(self, frames, past, steer_now):
        # One whole plan. "yield" = pause here; the next call goes on from exactly this point (step() calls next()
        # as long as the call budget allows), so a call never waits for more than one piece (~3-15 ms laptop).
        # frames: (lidar, state) of the 4 newest calls, past: the actions sent at the first 3 of them,
        # steer_now: wheel angle (model units). The encoder runs here, only for these 4 frames: v11 encoded every
        # call (30-60 ms laptop, ~3x more in the Garage) and broke the 250 ms cap (DNF, 270 ms).
        zh = []
        for lidar, state in frames:
            z = yield from self.encode_pieces(lidar, state)
            zh.append(z)
        gas = SPEED * GAS_PER_MS / THROTTLE_MAX
        shared = {}
        for s in range(HISTORY - 1):
            # window of step s = moments s..s+3; those with index <= 2 (a real z and a past action) are the same for
            # every option -> their keys/values once
            kv = [([], []) for _ in range(LAYERS)]
            for i in range(HISTORY - 1 - s):
                yield from self.token_pieces(zh[s + i], past[s + i], i, kv, False)
                yield
            shared[s] = kv
        costs, walls_min = [], []
        for steer in OPTIONS:
            action, imagined, cost, closest = [gas, steer], [], 0.0, math.inf
            for s in range(STEPS):
                base = shared.get(s)
                kv = [(keys[:], values[:]) for keys, values in base] if base else [([], []) for _ in range(LAYERS)]
                result = None
                for i in range(len(kv[0][0]), HISTORY):
                    n = s + i                                                # moment number from the plan's start
                    z = zh[n] if n < HISTORY else imagined[n - HISTORY]
                    result = yield from self.token_pieces(z, past[n] if n < HISTORY - 1 else action, i, kv,
                                                          i == HISTORY - 1)
                    yield
                z_next, speed = result
                imagined.append(z_next)
                walls, progress = self.heads(z_next)
                v = max(0.0, speed * SPEED_MAX)
                front = min(walls[5:8])                                      # -15, 0, +15 degrees
                side = min(walls[:5] + walls[8:])
                front_short = max(0.0, FRONT_BASE + v * v / (2 * DECEL) - front)
                side_short = max(0.0, SIDE_MARGIN - side)
                center = max(-2.0, min(2.0, walls[12] - walls[0])) ** 2
                cost += -progress + W_WALL * (front_short ** 2 + side_short ** 2) + W_CENTER * center
                closest = min(closest, front, side)
                yield
            costs.append(cost + W_SWITCH * abs(steer - steer_now))
            walls_min.append(closest)
        best = costs.index(min(costs))                                       # the plan is done: decide
        self.previous = OPTIONS[best]
        self.best_cost, self.best_wall = costs[best], walls_min[best]
        self.last_costs = costs
        self.rescue = self.best_wall < RESCUE_M
        self.plans += 1
        self.decided_at = self.calls

    # ----- the simple rule (v10) --------------------------------------------------------------------------------
    def make_beams(self, n):
        # the 109 beams wm2 knows, spread over -135..+135 degrees -> indexes in this platform's scan of n beams
        info = self.lidar_info
        if info and "angle_min" in info and "angle_increment" in info:
            self.angle_start, self.angle_step = info["angle_min"], info["angle_increment"]
        else:                                                                # no angles: assume 270 deg, right -> left
            self.angle_start, self.angle_step = math.radians(-135), math.radians(270) / (n - 1)
            print(f"INIT: no lidar angles, assuming 270 degrees over {n} beams")
        beams = [self.beam_at(-135 + i * 270 / (BEAMS - 1), n) for i in range(BEAMS)]
        print(f"INIT: model eats {BEAMS} beams, platform gives {n}, distinct beams used {len(set(beams))}")
        return beams

    def beam_at(self, angle, n):
        # the index of the platform beam closest to this angle (degrees); outside the fov: the nearest edge beam
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

    def simple_rule(self, scan, center=0.0):
        # steer into the deepest opening within +-CONE_DEG of `center` (the track's forward direction, degrees; also
        # never beyond -90..+90), pushed away from side walls closer than SIDE_MIN. Returns steering in model units
        # (1 = 30 degrees). Without the cone (v16, run #1226) the deepest opening once was BACK: the car turned round.
        n = len(scan)
        center = max(-90.0, min(90.0, center))
        low, high = int(max(-90, center - CONE_DEG)), int(min(90, center + CONE_DEG))
        best_angle, best_depth = center, -1.0
        for angle in range(low - low % 5, high + 1, 5):
            depth = min(scan[self.beam_at(a, n)] for a in range(angle - GAP_HALF, angle + GAP_HALF + 1, 2))
            if depth > best_depth + 0.05 or (abs(depth - best_depth) <= 0.05
                                             and abs(angle - center) < abs(best_angle - center)):
                best_angle, best_depth = angle, depth
        steer = best_angle / 30
        left = min(scan[self.beam_at(a, n)] for a in range(60, 121, 5))
        right = min(scan[self.beam_at(a, n)] for a in range(-120, -59, 5))
        if left < SIDE_MIN:
            steer -= (SIDE_MIN - left) / SIDE_MIN * 1.5
        if right < SIDE_MIN:
            steer += (SIDE_MIN - right) / SIDE_MIN * 1.5
        return max(-1.0, min(1.0, steer))

    # ----- one call -------------------------------------------------------------------------------------------
    def step(self, obs):
        began = time.perf_counter()
        self.calls += 1
        scan = [r if math.isfinite(r) else MAX_RANGE for r in obs["lidar"]]
        if self.beams is None:
            self.beams = self.make_beams(len(scan))
            print("OBS keys:", list(obs) if isinstance(obs, dict) else type(obs), "beams:", len(scan))
            if isinstance(obs, dict):
                print("OBS first:", short({k: v for k, v in obs.items() if k != "lidar"}))
        speed_now = obs.get("speed", 0.0)
        steer_now = obs.get("steer", 0.0) / MODEL_MAX_STEER                  # wheel angle, model units

        if self.w is None:
            self.previous, self.rescue = self.simple_rule(scan), True
        else:
            lidar = [min(scan[b], MAX_RANGE) / MAX_RANGE for b in self.beams]
            frame = (lidar, [speed_now / SPEED_MAX, steer_now])              # kept raw: encoded only when a plan starts
            self.history = (self.history + [[frame, None]])[-HISTORY:]      # the action is filled in below
            if self.thinking is None and len(self.history) == HISTORY:
                self.thinking = self.think([f[0] for f in self.history], [f[1] for f in self.history[:-1]], steer_now)
            # think piece by piece; start the next piece only if it still fits into CALL_BUDGET (the first piece of a
            # call may use up to HARD_LIMIT, so a slow CPU still makes progress)
            done = 0
            while self.thinking is not None:
                elapsed = time.perf_counter() - began
                limit = CALL_BUDGET if done else HARD_LIMIT
                if elapsed + self.moment_s > limit and (done or elapsed > 0.02):
                    break                                                    # a fresh call always does one piece:
                done += 1                                                    # thinking never stops for good
                t0 = time.perf_counter()
                try:
                    next(self.thinking)
                except StopIteration:                                        # the plan is done (think decided)
                    self.thinking = None
                took = time.perf_counter() - t0
                self.moment_s = max(took, 0.9 * self.moment_s)               # a careful guess of the next piece
        # reflex on the REAL lidar every call: the brain decides only every ~15 calls (run #1223: the right wall came
        # from 1.28 to 0.38 m between its decisions)
        n = len(scan)
        near = min(scan[self.beam_at(a, n)] for a in list(range(-90, -29, 5)) + list(range(30, 91, 5)))
        front = min(scan[self.beam_at(a, n)] for a in range(-10, 11, 2))     # +-10 degrees ahead
        reflex = near < REFLEX_M or front < FRONT_REFLEX_M                   # run #1225: hit the wall AHEAD at 1.2 m/s
        e = self.track_error(obs.get("pose"))                                # run #1226: drove 40 s the WRONG way
        center = e if e is not None else 0.0
        speed = SPEED
        if e is not None and abs(e) > TURN_DEG:                              # facing back: turn towards forward, slowly
            steering, self.mode, speed = (1.0 if e > 0 else -1.0), "TURN  ", TURN_SPEED
        elif self.rescue or reflex:
            steering = self.simple_rule(scan, center)
            self.mode = "REFLEX" if reflex else "RESCUE"
        elif e is not None and abs(e) > TRACK_DEG:                           # the brain leads away from the track
            steering, self.mode = self.simple_rule(scan, center), "TRACK "
        else:
            steering, self.mode = self.previous, "brain "

        if front < SAFE_FRONT:                                               # brake earlier and harder: v15 kept
            speed = min(speed, max(MIN_SPEED, (front - STOP_FRONT) / (SAFE_FRONT - STOP_FRONT) * SPEED))
        steer = max(-self.max_steer, min(self.max_steer, steering * MODEL_MAX_STEER))
        if self.w is not None:                                               # what the model will see as this action
            self.history[-1][1] = [speed * GAS_PER_MS / THROTTLE_MAX, steer / MODEL_MAX_STEER]

        self.rescue_calls += self.rescue
        call_ms = 1000 * (time.perf_counter() - began)
        self.slowest_ms = max(self.slowest_ms, call_ms)
        if self.calls % PRINT_EVERY == 0:
            wall = {name: min(scan[self.beam_at(a, len(scan))] for a in angles)
                    for name, angles in [("R", range(-95, -84)), ("L", range(85, 96))]}
            costs = " ".join(f"{c:+.2f}" for c in self.last_costs) if self.last_costs else "-"
            print(f"t{self.calls / 20:5.1f} {self.mode} s{steering:+.2f} [{costs}] v{speed_now:.2f}"
                  f" F{front:.2f} R{wall['R']:.2f} L{wall['L']:.2f} p{self.plans}/{self.calls - self.decided_at}"
                  f" w{self.best_wall:.2f} e{center:+.0f} {self.slowest_ms:.0f}ms prog{obs.get('progress', 0):.1f}")
            self.slowest_ms = 0.0
        return {"steer": steer, "speed": speed}
