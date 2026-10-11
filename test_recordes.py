import eventlet
import eventlet.wsgi
import socketio
import base64
import gzip
import csv 
import time
from find_gap import find_gap
import time
import random 


sio = socketio.Server()

log_file = open('my_drive.csv', 'w', newline='') 
writer = csv.writer(log_file)                         # a helper that writes CSV rows for us
writer.writerow(['episode', 'time', 'speed', 'throttle', 'steering',
                 'right', 'front', 'left', 'collisions', 'global_timer'])      # the header row: column names
packets = 0                                   # counter: how many messages so far
episod = 0
waiting = 0
global_timer = 0
lap_timer = 0
next_write = 0.0
throttle = random.uniform(0.03 , 0.1)
r_steering = 0
random_steering = 0

@sio.on('connect')
def connect(sid, environ):
    print('Connected! Caller ID:', sid)

def read_laser(data):
    packed = data['V1 LIDAR Range Array']                        # the zipped text
    text = gzip.decompress(base64.b64decode(packed)).decode('utf-8')   # unzip → normal text
    return [float(v) for v in text.split()]                      


def send_every_60_packets(data, ranges):

    print('speed:', data['V1 Speed'])
    
    print('beams:', len(ranges),
                  ' right:', ranges[180],      
                  ' front:', ranges[540],      
                  ' left:', ranges[900])       
    print(data['V1 Collisions'])





@sio.on("Bridge")                                  
def bridge(sid, data):
    global packets , episod, waiting, global_timer, lap_timer, next_write, throttle, r_steering, random_steering
    packets = packets + 1
    reset = "False"
    steering = 0
    episod
    
    ranges = read_laser(data)

    if packets % 60 == 0:                            
        send_every_60_packets(data, ranges)


    if int(data['V1 Collisions'])  > 0: 
        global_timer = 0
        next_write = 0.0
        packets = 0
        reset = "True"
        waiting =  60
        episod += 1
        throttle = random.uniform(0.03 , 0.1)
        r_steering = 0


    

    episod_time = float(data['V1 Lap Time'])
    if lap_timer < episod_time :
        global_timer += episod_time - lap_timer
        lap_timer = episod_time
    elif lap_timer > episod_time :
        lap_timer = episod_time
        global_timer += episod_time




    if global_timer > 28:   # time that car gonna  move forward 
        throttle = 0

    if global_timer  >= 30:
        global_timer = 0
        next_write = 0.0
        packets = 0
        reset = "True"
        waiting = 120
        episod += 1
        throttle = random.uniform(0.03 , 0.1)
        r_steering = 0


    
    steering  = find_gap(ranges)

    if global_timer > r_steering :
        random_steering = random.uniform(-0.3 , 0.3)
        r_steering += random.uniform(0.3 , 2)


    steering = steering + random_steering
    steering = max(-1, min(1, steering))

    if waiting > 0:
        waiting -= 1

    if global_timer > next_write and waiting  == 0:

        writer.writerow([episod, round(time.time(), 3), data['V1 Speed'],throttle ,steering ,
                  ranges[180], ranges[540],  ranges[900],data['V1 Collisions'], global_timer ])
        log_file.flush()
        next_write += 0.05



    sio.emit('Bridge', data={'V1 Throttle': str(throttle) , 'V1 Steering': str(steering) , 'Reset': str(reset)})


print('Door is open, waiting for the game...')
eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), socketio.WSGIApp(sio))
