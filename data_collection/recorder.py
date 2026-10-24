import numpy as np



KEYS = ["lidar", "throttle", "steering", "speed", "time", "collisions", "image",
        "position", "orientation", "yaw_rate", "real_steering"]          # the same names as in the old npz files

rows = []



def new_episod():
    rows.clear()

def add_row(obs, steering, throttle, time):

    obs["steering"] = steering
    obs["throttle"] = throttle
    obs["time"] = time
    rows.append(obs)

    
def save(path, track, reason, warmup):
    columns = {key : [row[key] for row in rows] for key in KEYS}
    columns["lidar"] = np.array(columns["lidar"], dtype=np.float16)
    np.savez_compressed(path, **columns, track=track, reason=reason, warmup=warmup)























