# Track generator: settings (plan of 13 Nov, to build on Sat 14)

Why: the model knows only the roads it drove (E25, E28, `docs/data_problems.md` P9), and the sim has only 2 tracks.
The generator makes many new roads so the encoder and the wall heads learn **roads in general**.
Solution C1 in [data_solve.md](data_solve.md). Code: `autoresearch/track_gen.py` (planned), built on `ail_cup/garage_sim.py`.

Every value below is a **proposal**. Values marked **(you)** need the user's answer.

## 1. Track shape
| Setting | Proposed value | Why |
|---|---|---|
| Kind of track | closed loop (a lap) | like the sim and AIL Cup tracks; no dead ends |
| Track length | 40 – 120 m | AIL Cup oval is 79.5 m; shorter = more corners per minute |
| Shape families (share of tracks) | random blob 40 %, oval / stadium 15 %, oval + chicanes 15 %, hairpins 15 %, S-bends / wavy 15 % | covers the 3 AIL Cup maps and shapes no real track has |
| Random blob | 6 – 14 control points around a circle, radius noise ±40 %, smoothed spline | the most variety per track |
| Tightest corner radius (centreline) | **2.0 m (you)** | the car turns at ≥ 0.74 m (wheelbase 0.33 m, 24°); AIL Cup hairpin ~2.2 m |
| Widest corner | straight (infinite radius) | |
| Chicane | offset 0.5 – 1.5 m over 3 – 6 m | like AIL Cup map 3 |
| Self-crossing tracks | rejected | a road must not cross itself |
| Direction | clockwise and counter-clockwise, 50 / 50 | fixes P2 (each real track only one way) |

## 2. Width
| Setting | Proposed value | Why |
|---|---|---|
| Track width | **1.2 – 3.0 m (you)**, log-uniform | real data: practice ~1.2 – 1.6 m, AIL Cup oval 2.57 m; scale test of E26 up to 2 × 2.0 m |
| Width changes along the track | 30 % of tracks: ±20 % slowly | real roads are not one width |
| Walls | both sides, closed lines; points every 0.2 m | as `garage_sim.walls_of` |
| Obstacles / gaps in walls | none (first version) | can be added later |

## 3. The car (as `garage_sim.py`)
| Setting | Value | Note |
|---|---|---|
| Model | kinematic bicycle, no tyre slip | **guessed physics**: this data does NOT train the predictor (see 6) |
| Wheelbase | 0.3302 m | |
| Max steering | ±24° (AIL Cup) | the AutoDRIVE car has ±30° |
| Time step | 0.05 s (20 Hz) | as the sim and the Garage |
| Speed lag / steering rate | 0.29 s / 3.25 rad/s | fitted on real trips (E29) |
| Lidar position | 0.20 m ahead of the car point | fitted (E29): lidar error 4 cm |
| Crash | closer than 0.17 m to a wall | `CAR_RADIUS` |

## 4. The lidar
| Setting | Value |
|---|---|
| Beams / field / range | 109 beams, 270°, 10 m (= AIL Cup and the training input after `clean_lidar`) |
| Noise | ±1 – 2 cm per beam, 0 – 3 % beams dropped to max range (A3) |
| Saved as | the 109 beams directly (not 1081 raw) + a flag, so `wm2/prepare.py` does not take every 10th beam again |

## 5. Who drives (the actions)
| Setting | Proposed value | Why |
|---|---|---|
| Driver mix | random steering segments 40 %, gap driver with noise 40 %, recovery pushes 20 % | like the real random data (best coverage) + P3 / P6 |
| Random segment | steer −1 … 1 held 0.3 – 1.5 s | as the real random collector |
| Recovery push | steer to one side 0.3 – 0.8 s, then the gap driver takes back | states off the line (P3) |
| Safety driver | takes over when the closest wall < 0.35 m | to keep trips long |
| Speed | 0.5 – 4 m/s, throttle ceiling picked per trip | fixes P1 for the geometry part only |
| Start | random place on the track, heading ±30° from the road, random offset ±40 % of half width | more headings per place (P3) |
| Trip end | crash, or 60 s | |

## 6. How it is used in training
| Part of wm2 | Synthetic data? | Why |
|---|---|---|
| Encoder (lidar → z) | **yes**, 30 – 50 % of windows | geometry does not depend on the physics |
| Wall heads (sectors, walls5) | **yes** | the same |
| Predictor (imagination) | **no** | the physics of the generator are guesses |
| Progress / value heads | **no** (first version) | they depend on the real car's motion |

## 7. Size and output
| Setting | Proposed value |
|---|---|
| Tracks | 200 first (test), then ~2,000 |
| Trips per track | 3 – 5 (different starts, directions, drivers) |
| Frames | ~1 M (about the size of `random_practice`) |
| Folder | `data/synthetic_v1/` (trip `.npz` files, mode `synthetic`, track = `gen_<seed>`) |
| Split | by **track**, not by trip: test tracks are never seen in training |
| Seed | fixed per track, so every track can be drawn again |
| Pictures | `docs/data/synthetic_v1/*.png`, 20 random tracks, for the user to check |

## 8. Checks that it is right
- Tests in `autoresearch/tests/`: no self-crossing, width inside range, tightest radius ≥ the limit, lidar of a straight
  corridor = the drawn walls.
- `autoresearch.data_report` on the set: widths and headings per place must cover more than the real data (E28).
- The shape test (Sat 14) before and after `wm2_c4`.
- Risk: synthetic scans look different from the sim's → the model learns two worlds. Check: wall error of `wm2_c4` on
  the **real** test set must not get worse.

## Questions for the user
1. Narrowest width and widest width the model must handle? (proposed 1.2 – 3.0 m)
2. Tightest corner? (proposed 2.0 m radius)
3. Should obstacles (boxes, gaps) be part of "every road", or only walls?
