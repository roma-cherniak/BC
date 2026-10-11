import eventlet
import eventlet.wsgi
import socketio
import base64
import gzip
import csv 
import time
from find_gap import find_gap
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


WAIT_AFTER_TIMEOUT = 120
WAIT_AFTER_CRASH = 60
EPISODE_TIME = 30   
BRAKE_TIME = 28  
RECORD_STEP = 0.05  
THROTTLE_MIN = 0.03
THROTTLE_MAX = 0.1
STEERING_NOISE = 0.3

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


def clock(episod_time):
    global lap_timer , global_timer
    if lap_timer < episod_time :
        global_timer += episod_time - lap_timer
        lap_timer = episod_time
    elif lap_timer > episod_time :
        lap_timer = episod_time
        global_timer += episod_time    


def start_new_episod(wait):
        
        global global_timer, next_write, waiting, episod, throttle, r_steering
        global_timer = 0
        next_write = 0.0
        waiting = wait
        episod += 1
        throttle = random.uniform(THROTTLE_MIN, THROTTLE_MAX)
        r_steering = 0
        return 'True'


def add_steering_noise(steering):
    global global_timer, r_steering, random_steering
    if global_timer > r_steering :
        random_steering = random.uniform(-STEERING_NOISE , STEERING_NOISE)
        r_steering += random.uniform(0.3 , 2)


    steering = steering + random_steering
    return max(-1, min(1, steering))


def write_row(data, ranges, noise_steering):
    writer.writerow([episod, round(time.time(), 3), data['V1 Speed'],throttle ,noise_steering ,
            ranges[180], ranges[540],  ranges[900],data['V1 Collisions'], global_timer ])
    log_file.flush()


@sio.on("Bridge")                                  
def bridge(sid, data):

    global packets , waiting,  next_write, throttle
    packets = packets + 1
    reset = "False"

    ranges = read_laser(data)

    if packets == 1:   # Clear all data that was before
        reset = start_new_episod(WAIT_AFTER_TIMEOUT)
    if packets % 60 == 0:                  # Send info message every 60 packets           
        send_every_60_packets(data, ranges)

    clock(float(data['V1 Lap Time']))

    # time that car gonna  move forward
    if global_timer > BRAKE_TIME:    
        throttle = 0
    if global_timer  >= EPISODE_TIME:
        reset = start_new_episod(WAIT_AFTER_TIMEOUT)

    # steering angle with noise 
    steering  = add_steering_noise(find_gap(ranges))   

    #  When collision happen write that in csv file and restart game
    if waiting == 0 and int(data['V1 Collisions']) > 0:  
        write_row(data, ranges, steering)
        reset = start_new_episod(WAIT_AFTER_CRASH)

    # Wait calculate
    if waiting > 0:   
        waiting -= 1
        next_write = global_timer

    # write new row to csv file every RECORD_STEP seconds 
    if global_timer > next_write and waiting  == 0:
        write_row(data, ranges, steering)
        next_write += RECORD_STEP

    # Send my throttle and steering to the simulator
    sio.emit('Bridge', data={'V1 Throttle': str(throttle) , 'V1 Steering': str(steering) , 'Reset': str(reset)})


print('Door is open, waiting for the game...')
eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), socketio.WSGIApp(sio))
