import numpy as np
from config import THROTTLE_MAX, SPEED_MAX
from core.sensors import clean_lidar



def clean_action(f):
    throttle = f["throttle"] / THROTTLE_MAX
    steering = f["steering"]

    return np.stack([throttle, steering], axis=1)


def make_pairs(lidar, action, speed):
    lidar_now = lidar[:-1]
    lidar_next = lidar[1:]

    return lidar_now, action[:-1], speed[:-1], lidar_next


def prepare_split(name):

    with open(f"data/{name}.txt") as t:
        files = t.read().split("\n")


    all_now, all_act, all_spd, all_next = [], [], [], []

    for file in files:
        with np.load(file) as f:
            lidar = clean_lidar(f["lidar"])
            action = clean_action(f)
            speed = f["speed"] / SPEED_MAX
        lidar_now, act, spd, lidar_next = make_pairs(lidar, action, speed)
        assert len(lidar_now) == len(act) == len(lidar_next)
        all_now.append(lidar_now)
        all_act.append(act)
        all_spd.append(spd)
        all_next.append(lidar_next)


    np.savez(f"data/{name}_pairs.npz",
             lidar_now = np.concatenate(all_now),
             act = np.concatenate(all_act),
             spd = np.concatenate(all_spd),
             lidar_next = np.concatenate(all_next)
             
             )

    print(name, "pairs:", sum(len(a) for a in all_now))


if __name__ == "__main__":         # only when this file is run, not when another file imports it
    for name in ["train", "val", "test"]:
        prepare_split(name)
