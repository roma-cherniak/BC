import eventlet
import eventlet.wsgi
import socketio
import base64
import gzip
from jepa.jepa_model import Encoder, Predictor, LATENT
import torch
import torch.nn as nn
import numpy as np
from driver_lidar.decision import make_decesion



DEVICE = 'cuda'


sio = socketio.Server() 



@sio.on('connect')
def connect(sid, environ):
    print('Connected! Caller ID:', sid)



def read_laser(data):
    packed = data['V1 LIDAR Range Array']                              # the zipped text
    try:
        text = gzip.decompress(base64.b64decode(packed)).decode('utf-8')   # unzip → normal text
    except Exception:
        print("LIDAR is not zipped? first symbols:", packed[:80])          # TEMP: to find the padding bug
        raise
    return [float(v) for v in text.split()]




encoder = Encoder().to(DEVICE)
encoder.load_state_dict(torch.load("jepa_encoder.pt", map_location=DEVICE))

predictor = Predictor().to(DEVICE)
predictor.load_state_dict(torch.load("jepa_predoctor.pt", map_location=DEVICE))

@sio.on("Bridge")                                  
def bridge(sid, data):

    lidar = np.array(read_laser(data))
    lidar = np.minimum(lidar[::4], 10) / 10 


    speed = float(data['V1 Speed'])
    throttle = float(data['V1 Throttle'])


    x = torch.tensor(lidar, dtype=torch.float32, device=DEVICE)[None]



    with torch.no_grad():
        z = encoder(x)

        steering = make_decesion(z, speed , throttle)

    
    throttle = 0.05

    sio.emit('Bridge', data={'V1 Throttle': str(throttle), 'V1 Steering': str(steering), 'Reset': 'False'})
    


print('Door is open, waiting for the game...')
eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), socketio.WSGIApp(sio))    
