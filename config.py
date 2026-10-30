# One place for the numbers that several files must agree on.
# A setting that only one file uses (BATCH, EPOCHS, STEPS of one check, ...) stays in that file.

# --- where things run and live
DEVICE = "cuda"                 # training and checks; the sim driver uses its own "cpu"
DATA_DIR = "data/random_practice"

# --- recording: collection_random.py writes it, checks/check_data.py checks it
RECORD_STEP = 0.05              # s between two rows
EPISODE_TIME = 30               # s, one trip
BRAKE_TIME = 28                 # s, after this the car brakes until the end of the trip
WAIT_MIN = 60                   # messages to wait after a reset (random between WAIT_MIN and WAIT_MAX)
WAIT_MAX = 1200
THROTTLE_MIN = 0.03             # random throttle while driving
THROTTLE_MAX = 0.09             # ... and the scale for the model: throttle / THROTTLE_MAX

# --- what the model sees: prepare_data.py and the driver must clean the lidar the same way
LIDAR_BEAMS = 1081              # beams in one raw scan from the game
BEAM_STEP = 4                   # keep every 4th lidar beam: 1081 -> 271
MAX_RANGE = 10                  # m, farther walls count as 10; lidar / MAX_RANGE -> 0..1
SPEED_MAX = 3                   # m/s, the model gets speed / SPEED_MAX

# --- what the probe reads, in this order (zond.pt has len(FACTS) outputs)
FACTS = ["front", "right", "left", "steering", "speed", "front_right", "front_left"]
FRONT = slice(130, 140)         # lidar beams of every wall (271 beams, index = angle + 135)
RIGHT = slice(40, 50)           # -90°
LEFT = slice(220, 230)          # +90°
FRONT_RIGHT = slice(85, 95)     # -45°
FRONT_LEFT = slice(175, 185)    # +45°
WALLS = [FACTS.index(name) for name in ["front", "right", "left", "front_right", "front_left"]]   # [0, 1, 2, 5, 6]
