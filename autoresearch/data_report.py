# Dataset diversity report: per data folder -> time, track, who drove, how trips end, throttle / steering / speed spread,
# driving direction, corridor width, closest wall, free road ahead, and track coverage (0.5 m cells x 8 headings).
# Run from the BC folder:  venv\Scripts\python -m autoresearch.data_report [folder ...]   (default: every trip folder)
# Output of 13 Nov 2026: docs/data/data_report_2026-11-13.txt (E28).
import glob, os, sys, collections
import numpy as np

FOLDERS = ["clean_practice", "random_practice", "fast_practice", "fast_compete", "self_practice", "self_compete",
           "self2_practice", "self2_compete", "self3_compete", "self4_compete", "self5_compete", "self6_compete"]
FRONT, LEFT, RIGHT = 540, 900, 180      # 1081 beams over 270 deg, 0.25 deg per beam, index 540 = straight ahead
LIDAR_EVERY = 5                          # read the lidar of every 5th frame only (speed)
CELL = 0.5                               # m, grid for track coverage

def pct(a, edges):
    h, _ = np.histogram(a, edges)
    return " ".join(f"{100 * x / max(len(a), 1):4.0f}" for x in h)

def folder_report(name):
    files = sorted(glob.glob(f"data/{name}/*.npz"))
    acc = collections.defaultdict(list)
    modes, reasons, tracks = collections.Counter(), collections.Counter(), collections.Counter()
    cells = collections.defaultdict(set)   # cell -> set of heading bins (8)
    seconds, turn_sign = 0.0, []
    for f in files:
        with np.load(f) as z:
            n = len(z["throttle"])
            if n < 2:
                continue
            t = z["time"]; seconds += float(t[-1] - t[0])
            for k in ("throttle", "steering", "speed"):
                acc[k].append(z[k])
            if "mode" in z.files:
                modes.update(map(str, z["mode"]))
            reasons[str(z["reason"])] += 1; tracks[str(z["track"])] += 1
            pos, yaw = z["position"], z["orientation"]
            # direction around the track: sign of angular motion around the trip's centre
            c = pos - pos.mean(0)
            cross = c[:-1, 0] * np.diff(pos[:, 1]) - c[:-1, 1] * np.diff(pos[:, 0])
            turn_sign.append(np.sign(cross.sum()) * (len(pos) > 100))
            hb = ((yaw % (2 * np.pi)) / (2 * np.pi) * 8).astype(int)
            for (x, y), h in zip(np.floor(pos / CELL).astype(int), hb):
                cells[(x, y)].add(h)
            lid = z["lidar"][::LIDAR_EVERY].astype(np.float32)
            acc["width"].append(np.minimum(lid[:, LEFT], 10) + np.minimum(lid[:, RIGHT], 10))
            acc["front"].append(np.minimum(lid[:, FRONT], 10))
            acc["closest"].append(lid.min(1))
    if not acc:
        return
    a = {k: np.concatenate(v) for k, v in acc.items()}
    ts = np.array(turn_sign)
    print(f"\n=== {name}: {len(files)} trips, {seconds / 60:.0f} min, {len(a['speed'])} frames, tracks {dict(tracks)}")
    print(f"  modes   {dict(modes.most_common(6))}")
    print(f"  end     {dict(reasons)}   crashes/min {reasons['crash'] / max(seconds / 60, 1e-9):.2f}")
    print(f"  throttle mean {a['throttle'].mean():.3f}  p5/p95 {np.percentile(a['throttle'], 5):.3f}/{np.percentile(a['throttle'], 95):.3f}  max {a['throttle'].max():.3f}")
    print(f"  speed m/s  %  [0,.5,1,1.5,2,3,5]: {pct(a['speed'], [0, .5, 1, 1.5, 2, 3, 5])}   mean {a['speed'].mean():.2f}")
    print(f"  steering %  [-1,-.6,-.2,.2,.6,1]: {pct(a['steering'], [-1.01, -.6, -.2, .2, .6, 1.01])}   mean {a['steering'].mean():+.3f}")
    print(f"  laps direction (long trips): ccw {int((ts > 0).sum())}  cw {int((ts < 0).sum())}")
    print(f"  corridor width m  %  [0,1.2,1.6,2,2.6,3.5,20]: {pct(a['width'], [0, 1.2, 1.6, 2, 2.6, 3.5, 20])}")
    print(f"  closest wall m    %  [0,.2,.35,.5,.8,20]: {pct(a['closest'], [0, .2, .35, .5, .8, 20])}")
    print(f"  front free m      %  [0,1,2,4,8,11]: {pct(a['front'], [0, 1, 2, 4, 8, 11])}")
    nh = np.array([len(h) for h in cells.values()])
    print(f"  coverage: {len(cells)} cells of {CELL} m, headings per cell mean {nh.mean():.1f} of 8, cells with 1 heading {100 * (nh == 1).mean():.0f}%")
    return cells

if __name__ == "__main__":
    for name in sys.argv[1:] or FOLDERS:
        folder_report(name)
