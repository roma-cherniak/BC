# One place for the numbers that several files must agree on.
# A setting that only one file uses (BATCH, EPOCHS, STEPS of one check, ...) stays in that file.

# --- where things run and live
DEVICE = "cuda"                 # training and checks; the sim driver uses its own "cpu"
DATA_DIR = "data/random_practice"
RUNS_DIR = "runs"               # one trained model = one folder runs/<name>/ (encoder, predictor, zond, loss, reports)
CURRENT_RUN = "k10_50ep"        # the model a check or an old script loads when you give it no name

# --- recording: collection_random.py writes it, checks/check_data.py checks it
RECORD_STEP = 0.05              # s between two rows
EPISODE_TIME = 30               # s, one trip
BRAKE_TIME = 28                 # s, after this the car brakes until the end of the trip
WAIT_MIN = 60                   # messages to wait after a reset (random between WAIT_MIN and WAIT_MAX)
WAIT_MAX = 1200
THROTTLE_MIN = 0.03             # random throttle while driving
THROTTLE_MAX = 0.09             # ... and the scale for the model: throttle / THROTTLE_MAX
THROTTLE_TOP = 0.20             # fast collection (data/fast_<track>): random throttle up to this (~4.9 m/s);
                                # the model scale stays THROTTLE_MAX, so fast throttle gives inputs up to 2.2

# --- what the model sees: prepare_data.py and the driver must clean the lidar the same way
LIDAR_BEAMS = 1081              # beams in one raw scan from the game
BEAM_STEP = 10                  # keep every 10th lidar beam: 1081 -> 109 at 2.5 degrees = the AIL Cup lidar
                                # (it was 4: 1081 -> 271, the models before k10_50ep_b109; their pairs: data/beams271/)
MAX_RANGE = 10                  # m, farther walls count as 10; lidar / MAX_RANGE -> 0..1
SPEED_MAX = 3                   # m/s, the model gets speed / SPEED_MAX


#prepare photo
PHOTO_WIDTH = 64
PHOTO_HEIGHT = 36

# --- what the probe reads, in this order (zond.pt has len(FACTS) outputs)
FACTS = ["front", "right", "left", "steering", "speed", "front_right", "front_left"]


def beams_between(first, last):
    # the cleaned lidar beams from angle first to last (degrees, -135 = right ... +135 = left), for any BEAM_STEP:
    # raw beam = (angle + 135) * 4, cleaned beam = raw / BEAM_STEP  (BEAM_STEP 4: 130:140, BEAM_STEP 10: 52:56)
    return slice(round((first + 135) * 4 / BEAM_STEP), round((last + 135) * 4 / BEAM_STEP))


FRONT = beams_between(-5, 5)            # lidar beams of every wall
RIGHT = beams_between(-95, -85)         # -90°
LEFT = beams_between(85, 95)            # +90°
FRONT_RIGHT = beams_between(-50, -40)   # -45°
FRONT_LEFT = beams_between(40, 50)      # +45°
WALLS = [FACTS.index(name) for name in ["front", "right", "left", "front_right", "front_left"]]   # [0, 1, 2, 5, 6]
