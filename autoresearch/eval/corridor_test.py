# Does a wm2 model read walls right in corridors it did NOT train on? (the second part of the judge, 13 Nov 2026)
#
# Why (docs/research_log.md E23): wm2_c2 reads the walls of `practice` and `compete` to 0.04 - 0.11 m, but a straight
# corridor with 1.29 m to each side (the AIL Cup oval) as 1.39 m right / 0.79 m left.
# Every number we had came from the two training tracks, so nothing could show this. This test can: it draws simple
# corridors of many widths, makes the lidar scan the car would see, and compares the model's wall heads with the truth.
#
# A scene = a corridor (straight, or an arc that bends left or right) + where the car stands in it (in the middle or
# off to one side, heading along it or 10 degrees off). No simulator, no graphics card, no recorded data.
#   KNOWN widths: 0.65 - 1.0 m to each side, like the training tracks.  NEW widths: everything else.
#   side error   mean |head - truth| over the 10 sectors from 30 to 90 degrees on both sides (m)
#   front error  |head - truth| of the nearest wall in the 3 front sectors (m; truth at most 10 m, the lidar's range)
#   mirror gap   the same scene flipped left <-> right must give the flipped answer; mean difference (m)
# The one number to watch: the side error on NEW widths ("new_side"). Lower is better.
#
# Run from the BC folder:  venv\Scripts\python -m autoresearch.eval.corridor_test wm2_c2 [more runs]

import math
import sys
from dataclasses import dataclass

import numpy as np
import torch

from config import MAX_RANGE, RUNS_DIR, SPEED_MAX
from wm2.heads import KINDS, load_heads
from wm2.model import BEAMS, Encoder

FOV = math.radians(270)
HALF_WIDTHS = [0.5, 0.65, 0.8, 1.0, 1.3, 1.6, 2.0]      # m from the middle of the corridor to each wall
KNOWN = (0.65, 1.0)                                     # the widths of the training tracks (E23)
OFFSETS = [-0.3, 0.0, 0.3]                              # the car's place, as a share of the half width (+ = to the left)
HEADINGS = [-10.0, 0.0, 10.0]                           # degrees off the corridor's direction (+ = to the left)
RADII = [None, 4.0, -4.0]                               # None = straight; + = the corridor bends left, - = right (m)
SPEED = 1.5                                             # m/s the model is told
SECTORS = KINDS["sectors"][0]                           # the beams of the 13 sectors, -90 (right) .. +90 (left)
SIDE = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
FRONT = [5, 6, 7]


@dataclass(frozen=True)
class CorridorScore:
    run: str
    known_side: float
    known_front: float
    new_side: float
    new_front: float
    mirror_gap: float
    by_width: tuple[tuple[float, float, float], ...]    # (half width, side error, front error)


def walls(half: float, radius: float | None) -> np.ndarray:
    # wall segments (M, 4) of a corridor that starts along +x at the origin; the car's "middle" is the origin
    if radius is None:
        return np.array([[-60.0, half, 60.0, half], [-60.0, -half, 60.0, -half]])
    centre = np.array([0.0, radius])                    # the bend's centre: to the left for a left bend
    angles = np.linspace(-math.pi, math.pi, 721)
    segments = []
    for r in (abs(radius) - half, abs(radius) + half):
        ring = centre + r * np.column_stack([np.cos(angles), np.sin(angles)])
        segments.append(np.column_stack([ring[:-1], ring[1:]]))
    return np.vstack(segments)


def scan_of(x: float, y: float, yaw: float, segments: np.ndarray) -> np.ndarray:
    # the lidar at (x, y) looking along yaw: BEAMS distances, right -> left over 270 degrees, at most MAX_RANGE m
    angles = yaw + np.linspace(-FOV / 2, FOV / 2, BEAMS)
    dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]
    x1, y1 = segments[None, :, 0] - x, segments[None, :, 1] - y
    ex, ey = segments[None, :, 2] - segments[None, :, 0], segments[None, :, 3] - segments[None, :, 1]
    det = dx * ey - dy * ex
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    along = (x1 * ey - y1 * ex) / det
    place = (x1 * dy - y1 * dx) / det
    hit = np.where((along > 0) & (place >= 0) & (place <= 1), along, np.inf)
    return np.minimum(hit.min(axis=1), MAX_RANGE)


def scenes() -> list[tuple[float, np.ndarray]]:
    # every scene: (half width, scan)
    out = []
    for half in HALF_WIDTHS:
        for radius in RADII:
            if radius is not None and abs(radius) - half < 0.5:
                continue                                # the inner wall would be a point: not a corridor
            segments = walls(half, radius)
            for offset in OFFSETS:
                for heading in HEADINGS:
                    out.append((half, scan_of(0.0, offset * half, math.radians(heading), segments)))
    return out


def truth(scan: np.ndarray) -> np.ndarray:
    # the nearest wall of every sector (m), exactly as wm2/heads.py makes its training answers
    return np.array([scan[beams].min() for beams in SECTORS])


def read(encoder, heads, scans: np.ndarray) -> np.ndarray:
    # what the model's wall heads say for these scans (N, 13), wheel straight, SPEED m/s
    state = torch.tensor([[SPEED / SPEED_MAX, 0.0]]).repeat(len(scans), 1)
    with torch.no_grad():
        return heads(encoder(torch.tensor(scans / MAX_RANGE, dtype=torch.float32), state))[0].numpy()


def corridor_score(run: str) -> CorridorScore:
    encoder = Encoder()
    encoder.load_state_dict(torch.load(f"{RUNS_DIR}/{run}/encoder.pt", map_location="cpu"))
    encoder.eval()
    heads = load_heads(run, "cpu", "sectors")
    halves = np.array([half for half, _ in scenes()])
    scans = np.array([scan for _, scan in scenes()])
    true = np.array([truth(scan) for scan in scans])
    said = read(encoder, heads, scans)
    flipped = read(encoder, heads, scans[:, ::-1].copy())[:, ::-1]     # the mirrored scene, answer flipped back
    side = np.abs(said - true)[:, SIDE].mean(axis=1)
    front = np.abs(said[:, FRONT].min(axis=1) - true[:, FRONT].min(axis=1))
    known = (halves >= KNOWN[0]) & (halves <= KNOWN[1])
    return CorridorScore(
        run=run, known_side=float(side[known].mean()), known_front=float(front[known].mean()),
        new_side=float(side[~known].mean()), new_front=float(front[~known].mean()),
        mirror_gap=float(np.abs(said - flipped).mean()),
        by_width=tuple((half, float(side[halves == half].mean()), float(front[halves == half].mean()))
                       for half in HALF_WIDTHS))


def table(scores: list[CorridorScore]) -> str:
    lines = ["| Model | Side error, known widths | Side error, NEW widths | Front error, known | Front error, NEW | Mirror gap |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {s.run} | {s.known_side:.2f} | **{s.new_side:.2f}** | {s.known_front:.2f} | {s.new_front:.2f} | "
              f"{s.mirror_gap:.2f} |" for s in scores]
    lines += ["", "Side error (m) by the distance from the middle to each wall:", "",
              "| Model | " + " | ".join(f"{half:.2f} m" for half in HALF_WIDTHS) + " |",
              "|---|" + "---|" * len(HALF_WIDTHS)]
    lines += [f"| {s.run} | " + " | ".join(f"{side:.2f}" for _, side, _ in s.by_width) + " |" for s in scores]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: venv\\Scripts\\python -m autoresearch.eval.corridor_test <run> [more runs]")
    print(f"{len(scenes())} scenes: half widths {HALF_WIDTHS}, known {KNOWN}\n")
    print(table([corridor_score(run) for run in sys.argv[1:]]))
