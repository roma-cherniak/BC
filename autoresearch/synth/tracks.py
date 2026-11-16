# Random closed roads for synthetic training data (C1 in docs/data_solve.md, settings in docs/track_generator.md).
# A road = a centreline (N, 2) with points every 0.2 m, driven counter-clockwise, and a width per point.
# Families: blob (random smooth loop), oval (stadium), chicane (stadium with side steps), hairpin (long thin stadium),
# wavy (loop with 3 - 5 bends each way). Every road is checked: tightest corner >= MIN_RADIUS, no part of the road
# closer than its width to another part (the walls never cross).
#   venv\Scripts\python -m autoresearch.synth.tracks [n]      -> pictures of n random roads in docs/data/synthetic_v1/

import math
import os
import sys

import numpy as np

STEP = 0.2                              # m between centreline points
MIN_RADIUS = 2.0                        # m, tightest corner of the centreline
WIDTH_RANGE = (1.2, 3.0)                # m, log-uniform
LENGTH_RANGE = (40.0, 120.0)            # m
FAMILIES = {"blob": 0.40, "oval": 0.15, "chicane": 0.15, "hairpin": 0.15, "wavy": 0.15}
WIDTH_CHANGE_SHARE, WIDTH_CHANGE = 0.30, 0.20   # 30 % of roads: width changes slowly by up to +-20 %
CLEARANCE = 0.6                         # m, extra gap between two parts of the road besides its width


def resample(pts: np.ndarray, step: float = STEP) -> np.ndarray:
    closed = np.vstack([pts, pts[:1]])
    along = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    want = np.arange(0, along[-1], step)
    return np.column_stack([np.interp(want, along, closed[:, 0]), np.interp(want, along, closed[:, 1])])


def length_of(line: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.vstack([line, line[:1]]), axis=0), axis=1).sum())


def radius_of(line: np.ndarray) -> np.ndarray:
    # turning radius at every point (m), from the circle through the neighbours 3 points away on each side
    a, b, c = np.roll(line, 3, axis=0), line, np.roll(line, -3, axis=0)
    ab, bc, ca = (np.linalg.norm(p - q, axis=1) for p, q in ((a, b), (b, c), (c, a)))
    cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    return np.where(cross < 1e-9, np.inf, ab * bc * ca / (2 * np.maximum(cross, 1e-9)))


def closest_other_part(line: np.ndarray, skip_m: float) -> float:
    # smallest distance between two centreline points that are more than skip_m apart ALONG the road
    sub = line[::2]
    d = np.linalg.norm(sub[:, None] - sub[None], axis=2)
    i = np.arange(len(sub))
    gap = np.abs(i[:, None] - i[None]) * 2 * STEP
    along = np.minimum(gap, len(line) * STEP - gap)
    return float(d[along > skip_m].min())


def _stadium(straight: float, r: float) -> np.ndarray:
    pts = [(x, -r) for x in np.arange(-straight / 2, straight / 2, STEP)]
    pts += [(straight / 2 + r * math.sin(a), -r * math.cos(a)) for a in np.arange(0, math.pi, STEP / r)]
    pts += [(x, r) for x in np.arange(straight / 2, -straight / 2, -STEP)]
    pts += [(-straight / 2 - r * math.sin(a), r * math.cos(a)) for a in np.arange(0, math.pi, STEP / r)]
    return np.array(pts)


def _fourier(rng: np.random.Generator, orders: list[int], amp: float) -> np.ndarray:
    a = np.linspace(0, 2 * math.pi, 3000, endpoint=False)
    r = np.ones_like(a)
    for k in orders:
        r += rng.uniform(0.3, 1.0) * amp / len(orders) * np.cos(k * a + rng.uniform(0, 2 * math.pi))
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def _chicanes(line: np.ndarray, rng: np.random.Generator, count: int) -> np.ndarray:
    # side steps of 0.5 - 1.5 m over 3 - 6 m on the bottom straight (y < 0, |x| small)
    out = line.copy()
    xs = rng.uniform(-0.35, 0.35, count) * (line[:, 0].max() - line[:, 0].min())
    for x0 in xs:
        offset, length = rng.uniform(0.5, 1.5) * rng.choice([-1, 1]), rng.uniform(3, 6)
        bottom = line[:, 1] < 0
        bump = offset * np.exp(-(((line[:, 0] - x0) / (length / 2)) ** 2))
        out[:, 1] += np.where(bottom, bump, 0)
    return out


def shape(family: str, rng: np.random.Generator) -> np.ndarray:
    # an unscaled closed shape of the family, counter-clockwise
    if family == "oval":
        return _stadium(rng.uniform(10, 30), rng.uniform(4, 9))
    if family == "hairpin":
        return _stadium(rng.uniform(15, 35), rng.uniform(MIN_RADIUS, 3.0))
    if family == "chicane":
        return _chicanes(_stadium(rng.uniform(20, 35), rng.uniform(4, 8)), rng, int(rng.integers(1, 3)))
    if family == "wavy":
        return _fourier(rng, [int(rng.integers(3, 6))], rng.uniform(0.2, 0.4))
    return _fourier(rng, list(range(2, int(rng.integers(4, 8)))), rng.uniform(0.3, 0.7))       # blob


def make_road(seed: int) -> dict | None:
    # -> {"line", "width" (per point), "family", "seed"} or None if this seed gives a bad road
    rng = np.random.default_rng(seed)
    family = rng.choice(list(FAMILIES), p=list(FAMILIES.values()))
    raw = resample(shape(family, rng), 0.05)
    if family in ("blob", "wavy"):                                  # free shapes: scale to a random length
        raw *= rng.uniform(*LENGTH_RANGE) / length_of(raw)
    line = resample(raw)
    width = math.exp(rng.uniform(math.log(WIDTH_RANGE[0]), math.log(WIDTH_RANGE[1]))) * np.ones(len(line))
    if rng.random() < WIDTH_CHANGE_SHARE:
        a = np.linspace(0, 2 * math.pi, len(line), endpoint=False)
        width *= 1 + WIDTH_CHANGE * np.sin(int(rng.integers(1, 4)) * a + rng.uniform(0, 2 * math.pi))
    if rng.random() < 0.5:                                          # half of the roads clockwise
        line, width = line[::-1].copy(), width[::-1].copy()
    road = {"line": line, "width": width, "family": str(family), "seed": seed}
    return road if is_valid(road) else None


def is_valid(road: dict) -> bool:
    line, width = road["line"], road["width"]
    if not LENGTH_RANGE[0] * 0.8 <= length_of(line) <= LENGTH_RANGE[1] * 1.3:
        return False
    if radius_of(line).min() < max(MIN_RADIUS, width.max() / 2 + 0.3):
        return False
    return closest_other_part(line, skip_m=3 * width.max() + 2) > width.max() + CLEARANCE


def walls_of(road: dict) -> np.ndarray:
    # -> wall segments (M, 4): x1, y1, x2, y2 of both walls
    line, half = road["line"], road["width"] / 2
    ahead = np.roll(line, -1, axis=0) - np.roll(line, 1, axis=0)
    normal = np.column_stack([-ahead[:, 1], ahead[:, 0]]) / np.linalg.norm(ahead, axis=1, keepdims=True)
    walls = [line + side * half[:, None] * normal for side in (+1, -1)]
    return np.vstack([np.column_stack([w, np.roll(w, -1, axis=0)]) for w in walls])


def roads(first_seed: int, count: int) -> list[dict]:
    # the first `count` valid roads from seed first_seed on (bad seeds are skipped, so seeds are not contiguous)
    out, seed = [], first_seed
    while len(out) < count:
        road = make_road(seed)
        if road is not None:
            out.append(road)
        seed += 1
    return out


def draw(road_list: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 5
    rows = math.ceil(len(road_list) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    for ax, road in zip(np.ravel(axes), road_list):
        w = walls_of(road)
        ax.plot(w[:, [0, 2]].T, w[:, [1, 3]].T, "k-", lw=0.8)
        ax.plot(*road["line"][0], "ro", ms=4)
        ax.set_title(f"{road['family']} #{road['seed']}  w {road['width'].mean():.1f} m  "
                     f"{length_of(road['line']):.0f} m", fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in np.ravel(axes)[len(road_list):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    os.makedirs("docs/data/synthetic_v1", exist_ok=True)
    draw(roads(0, n), "docs/data/synthetic_v1/roads.png")
    print("docs/data/synthetic_v1/roads.png")
