import base64
import gzip




KEYS = ["Ranges", "Speed", "Time", "Steering", "Throttle", "Colisions", "Position", "Orientation", "Yaw_rate", "Real_steering"]
rows = {key : [] for key in KEYS}



def read_laser(data):
    packed = data['V1 LIDAR Range Array']                              # the zipped text
    try:
        text = gzip.decompress(base64.b64decode(packed)).decode('utf-8')   # unzip → normal text
    except Exception:
        print("LIDAR is not zipped? first symbols:", packed[:80])          # TEMP: to find the padding bug
        raise
    return [float(v) for v in text.split()]



def read_packet(data):

    obs = {}
    obs["lidar"] = read_laser(data)
    obs["speed"] = float(data['V1 Speed'])
    obs["real_steering"] = float(data['V1 Steering'])                  # what the wheels really do (rad)
    obs["collisions"] = int(data['V1 Collisions'])
    obs["image"] = base64.b64decode(data['V1 Front Camera Image'])     # JPEG bytes

    parts = data['V1 Position'].split()
    obs["position"] = [float(p) for p in parts[0:2]]                  # x, y

    parts = data['V1 Orientation Euler Angles'].split()
    obs["orientation"] = float(parts[2])                               # yaw

    parts = data['V1 Angular Velocity'].split()
    obs["yaw_rate"] = float(parts[2])

    obs["lap_time"] = float(data['V1 Lap Time'])                       # for the clock, not saved
    return obs
