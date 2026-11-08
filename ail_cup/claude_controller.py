"""JEPA lidar driver for the AIL Cup: pure Python (no numpy), no weights inside the file.
Claude's working file: Claude edits it, the user uploads it to the Garage and reports what happened.

The platform gives the network as init["model"] (an .npz: encoder.net.*, predictor.net.*, zond.*).
Every step: the lidar is re-sampled to the 271 beams the model knows (-135..+135 degrees, right -> left,
cut at MAX_RANGE, / MAX_RANGE), the encoder describes it, the predictor imagines STEPS steps for every
steering option, the probe reads the walls on the whole path, and the option whose closest wall stays
farthest away wins. Only the standard library (math, operator, time).

Pure Python is slow: one full plan (9 options x 10 steps) takes ~0.08 s, but the platform asks 20x per second.
So the plan is spread over several calls: every call thinks at most BUDGET seconds (a few options), then
answers with the last finished decision. A new decision comes every ~3 calls (~0.15 s); the speed and the
SAFE_FRONT brake use the newest lidar every call.

Safe start: anything missing in init (car, lidar, model) gets a default and is printed. Without the network the
car steers to the side with more space, so a run still shows the prints.
"""
import math
import operator
import time

STEERING = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
STEPS = 10
GUARD = False
FACTS = ['front', 'right', 'left', 'steering', 'speed', 'front_right', 'front_left']
MAX_RANGE = 10
SPEED_MAX = 3
THROTTLE_MAX = 0.09
LATENT = 32

SPEED = 1.22                    # m/s, constant command speed = what throttle 0.05 gave in AutoDRIVE (1.8 crashed in 4 s)
THROTTLE = 0.05                 # the AutoDRIVE throttle the model is told it drives with
AUTODRIVE_MAX_STEER = 0.5236    # rad: steering +-1 of the model = +-30 degrees of the AutoDRIVE car
SMOOTH = 0.01                   # penalty per unit of steering change -> less zig-zag
SAFE_FRONT = 1.0                # m, slow down if something is closer in front (other cars)
BUDGET = 0.1                    # s of thinking per call: 0.2 broke the 250 ms cap on the slower Garage CPU (run #1097)
PRINT_EVERY = 5                 # calls between two status lines (0.25 s)
GAP_HALF = 10                   # degrees: half width of an opening the rescue rule steers into (~ the car's width at 1 m)
SIDE_MIN = 0.5                  # m: a side wall closer than this pushes the rescue steering away from it
RESCUE_M = 0.25               # m: if the best imagined path comes closer than this to a wall, use the simple rule

if hasattr(math, "sumprod"):    # Python 3.12+: a fast dot product written in C
    dot = math.sumprod
else:
    def dot(a, b):
        return sum(map(operator.mul, a, b))


def as_list(array):
    return array.tolist() if hasattr(array, "tolist") else array


def relu(x):
    return [v if v > 0 else 0.0 for v in x]


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
        print("claude_controller v10")                                        # which file really runs in the Garage
        print("INIT type:", type(init), "keys:", list(init) if isinstance(init, dict) else "-")   # no dunders: the Garage forbids them
        car = pick(init, "car", {})
        lidar = pick(init, "lidar", None)
        if isinstance(car, dict):
            print("INIT car:", car)
        if isinstance(lidar, dict):
            print("INIT lidar:", lidar)
        if isinstance(init, dict):
            for key in init:
                if key not in ("car", "lidar", "model"):
                    print(f"INIT {key}: {short(init[key])}")
            print("INIT model:", short(init.get("model")))
        self.max_steer = pick(car, "max_steer", AUTODRIVE_MAX_STEER)
        self.lidar_info = lidar if isinstance(lidar, dict) else None
        self.beams = None                                                    # built on the first step()
        model = pick(init, "model", None)
        self.w = None                                                        # None = no network: simple rule
        if model is not None:
            try:                                                             # a bad model must not stop the run
                w = {}
                for name in ["encoder.net.0", "encoder.net.2", "predictor.net.0", "predictor.net.2",
                             "predictor.net.4", "zond.0", "zond.2"]:
                    key = name.replace(".", "_")                             # the Garage allows no dots in .npz names
                    w[name] = (as_list(model[key + "_weight"]), as_list(model[key + "_bias"]))
                self.w = w
                print("INIT: network loaded")
            except Exception as error:
                print("INIT: network NOT loaded, driving by the simple rule:", repr(error))
        self.fact = {name: i for i, name in enumerate(FACTS)}
        self.previous = 0.0                                                  # the last steering option (-1 .. 1)
        self.todo = []                                                       # options of the plan not imagined yet
        self.values = {}                                                     # option index -> its value
        self.option_time = 0.0                                               # s, how long one option took last time
        self.calls = 0
        self.rescue = False                                                  # True = the simple rule steers now
        self.rescue_calls = 0

    def make_beams(self, n):
        # the beams the model knows (271 at 1 degree, or 109 at 2.5 degrees = the AIL Cup lidar itself), spread over
        # -135..+135 degrees -> indexes in this platform's scan of n beams
        info = self.lidar_info
        if info and "angle_min" in info and "angle_increment" in info:
            self.angle_start, self.angle_step = info["angle_min"], info["angle_increment"]
        else:                                                                # no angles: assume 270 deg, right -> left
            self.angle_start, self.angle_step = math.radians(-135), math.radians(270) / (n - 1)
            print(f"INIT: no lidar angles, assuming 270 degrees over {n} beams")
        inputs = len(self.w["encoder.net.0"][0][0]) if self.w else 271        # how many beams the encoder eats
        angles = [-135 + i * 270 / (inputs - 1) for i in range(inputs)]
        beams = [self.beam_at(a, n) for a in angles]
        print(f"INIT: model eats {inputs} beams, platform gives {n}, distinct beams used {len(set(beams))}")
        return beams

    def beam_at(self, angle, n):
        # the index of the platform beam closest to this angle (degrees); outside the fov: the nearest edge beam
        return min(max(round((math.radians(angle) - self.angle_start) / self.angle_step), 0), n - 1)

    def layer(self, x, name):
        weight, bias = self.w[name]
        return [dot(row, x) + b for row, b in zip(weight, bias)]

    def sparse_layer(self, h, name):
        # like layer(), but after relu: most of h is 0 (55-85 %), so only the columns where h is not 0 are multiplied
        keep = [i for i, v in enumerate(h) if v > 0]
        if len(keep) < 2:                                                    # itemgetter of 1 index gives a number
            return self.layer(h, name)
        pick_h = operator.itemgetter(*keep)
        weight, bias = self.w[name]
        kept = pick_h(h)
        return [dot(pick_h(row), kept) + b for row, b in zip(weight, bias)]

    def encode(self, lidar):
        return self.sparse_layer(relu(self.layer(lidar, "encoder.net.0")), "encoder.net.2")

    def predict(self, z, act, spd):
        h = relu(self.layer(z + act + [spd], "predictor.net.0"))
        h = relu(self.sparse_layer(h, "predictor.net.2"))
        return self.sparse_layer(h, "predictor.net.4")

    def read(self, z):                                                       # the probe (zond): z -> FACTS
        return [min(max(v, 0.0), 1.0) for v in self.sparse_layer(relu(self.layer(z, "zond.0")), "zond.2")]

    def score(self, facts):
        # the closest wall; a diagonal counts /1.4 (in a corridor it is ~1.4x farther), the front /2. bigger = safer
        f = self.fact
        return min(facts[f["right"]], facts[f["left"]], facts[f["front_right"]] / 1.4,
                   facts[f["front_left"]] / 1.4, facts[f["front"]] / 2)

    def imagine(self, option):
        # drive STEPS steps with this steering in the imagination: the closest wall on the WHOLE path
        z, act = self.start, [THROTTLE / THROTTLE_MAX, option]
        worst = math.inf
        for _ in range(STEPS):
            z = self.predict(z, act, self.spd)
            if GUARD:
                norm = math.sqrt(dot(z, z))
                z = [v / norm * math.sqrt(LATENT) for v in z]
            worst = min(worst, self.score(self.read(z)))
        return worst

    def simple_rule(self, scan):
        # no network / rescue: steer into the deepest opening ahead (-90..+90 degrees). An opening counts only as deep
        # as its closest beam within +-GAP_HALF degrees, so a single far beam between two walls does not attract.
        # (The old rule "average of the right vs left third" looked behind the car too and steered left into a
        # right turn, run #1100.)
        n = len(scan)
        best_angle, best_depth = 0.0, -1.0
        for angle in range(-90, 91, 5):
            depth = min(scan[self.beam_at(a, n)] for a in range(angle - GAP_HALF, angle + GAP_HALF + 1, 2))
            if depth > best_depth + 0.05 or (abs(depth - best_depth) <= 0.05 and abs(angle) < abs(best_angle)):
                best_angle, best_depth = angle, depth                       # ties: the straighter opening
        steer = best_angle / 30                                              # steering 1 = 30 degrees
        # ...and push away from a side wall that is too close: the deepest opening alone drove along the left wall
        # at 0.2 m and even steered into it (run #1102)
        left = min(scan[self.beam_at(a, n)] for a in range(60, 121, 5))
        right = min(scan[self.beam_at(a, n)] for a in range(-120, -59, 5))
        if left < SIDE_MIN:
            steer -= (SIDE_MIN - left) / SIDE_MIN * 1.5
        if right < SIDE_MIN:
            steer += (SIDE_MIN - right) / SIDE_MIN * 1.5
        return max(-1.0, min(1.0, steer))

    def step(self, obs):
        began = time.perf_counter()
        self.calls += 1
        scan = [r if math.isfinite(r) else MAX_RANGE for r in obs["lidar"]]
        if self.beams is None:
            self.beams = self.make_beams(len(scan))
            print("OBS keys:", list(obs) if isinstance(obs, dict) else type(obs), "beams:", len(scan))
            if isinstance(obs, dict):
                print("OBS first:", short({k: v for k, v in obs.items() if k != "lidar"}))

        if self.w is None:
            self.previous = self.simple_rule(scan)
        else:
            if not self.todo:                                                # no plan in progress: start one
                lidar = [min(scan[b], MAX_RANGE) / MAX_RANGE for b in self.beams]    # like clean_lidar
                self.start = self.encode(lidar)                              # one start for every option
                self.spd = obs["speed"] / SPEED_MAX
                self.todo = list(range(len(STEERING)))
                self.values = {}
            done = 0                                                         # options imagined in THIS call
            while self.todo:                                                 # think while the next option fits
                used = time.perf_counter() - began
                if done and used + self.option_time > BUDGET:                # at least one per call, or no progress
                    break
                k = self.todo.pop(0)
                self.values[k] = self.imagine(STEERING[k]) - SMOOTH * abs(STEERING[k] - self.previous)
                self.option_time = time.perf_counter() - began - used
                done += 1
            if not self.todo:                                                # the plan is finished: decide
                best = max(self.values, key=self.values.get)                # the first best, like np.argmax
                self.previous = STEERING[best]
                self.rescue = self.values[best] * MAX_RANGE < RESCUE_M       # the brain sees no way out
            if self.rescue:
                # every imagined path hits a wall: the brain gives no answer (all scores 0, SMOOTH then keeps the
                # old steering, run #1099). Steer to the side with more space until the brain sees a way again.
                self.previous = self.simple_rule(scan)

        speed = SPEED
        mid = len(scan) // 2
        front = min(scan[mid - 4 : mid + 5])
        if front < SAFE_FRONT:
            speed = min(speed, max(0.5, front * 2.0))
        steer = max(-self.max_steer, min(self.max_steer, self.previous * AUTODRIVE_MAX_STEER))

        self.rescue_calls += self.rescue
        if self.calls % PRINT_EVERY == 0:
            wall = {name: min(scan[self.beam_at(a, len(scan))] for a in angles)          # real walls, m
                    for name, angles in [("R", range(-95, -84)), ("L", range(85, 96))]}
            best = max(self.values.values()) * MAX_RANGE if self.values else 0.0
            print(f"t {self.calls / 20:5.2f}  {'RESCUE' if self.rescue else 'brain '}  steer {self.previous:+.2f}"
                  f"  v {obs.get('speed', 0):.2f}"
                  f"  F {front:.2f} R {wall['R']:.2f} L {wall['L']:.2f}  imagined {best:.2f}"
                  f"  ms {1000 * (time.perf_counter() - began):.0f}  prog {obs.get('progress', 0):.1f}"
                  f"  bat {obs.get('battery_left_wh', 0):.3f}")
        return {"steer": steer, "speed": speed}
