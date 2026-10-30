
import base64
import gzip
import numpy as np
from config import BEAM_STEP, MAX_RANGE


def read_laser(data):
    packed = data['V1 LIDAR Range Array']
    try:
        text = gzip.decompress(base64.b64decode(packed)).decode('utf-8')   # practice / compete builds: zipped text
    except Exception:
        text = packed                                                      # some builds send plain text
    return [float(v) for v in text.split()]                                # broken text still fails here (ValueError)


def clean_lidar(lidar):
    
    lidar = np.asarray(lidar)[..., ::BEAM_STEP]
    lidar = np.minimum(lidar, MAX_RANGE)
    return lidar / MAX_RANGE
