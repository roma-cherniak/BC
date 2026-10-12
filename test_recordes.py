import eventlet
import eventlet.wsgi
import socketio
import base64
import gzip
import csv 
import time
from find_gap import find_gap
from lidar_collection import save_npz
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

ranges_list = []
speed_list = []
throttle_list = []
steering_list = []
time_list = []
colisions_list = []


WAIT_MIN = 60
WAIT_MAX = 1200
EPISODE_TIME = 30   
BRAKE_TIME = 28  
RECORD_STEP = 0.05  
THROTTLE_MIN = 0.03
THROTTLE_MAX = 0.093
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
    print(f"time - {global_timer}")


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
        global ranges_list, speed_list, throttle_list, steering_list, time_list, colisions_list

        if len(ranges_list) > 0:
            save_npz(ranges_list, 
                     speed_list, 
                     time_list, 
                     steering_list, 
                     throttle_list, 
                     episod,
                     colisions_list)

        ranges_list.clear()
        speed_list.clear()
        time_list.clear()
        steering_list.clear()
        throttle_list.clear()
        colisions_list.clear()


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

    global ranges_list, speed_list, throttle_list, steering_list, time_list, colisions_list 

    ranges_list.append(ranges)
    speed_list.append(float(data['V1 Speed']))
    throttle_list.append(throttle)
    steering_list.append(noise_steering)
    time_list.append(global_timer)
    colisions_list.append(int(data['V1 Collisions']))


    writer.writerow([episod, round(time.time(), 3), data['V1 Speed'],throttle ,noise_steering ,
            ranges[180], ranges[540],  ranges[900],data['V1 Collisions'], global_timer ])
    log_file.flush()


@sio.on("Bridge")                                  
def bridge(sid, data):

    global packets , waiting,  next_write, throttle, lap_timer, global_timer
    packets = packets + 1
    reset = "False"

    ranges = read_laser(data)
    if packets % 60 == 0:
        send_every_60_packets(data, ranges)

    if packets == 1:   # Clear all data that was before
        lap_timer = float(data['V1 Lap Time'])
        reset = start_new_episod(random.randint(WAIT_MIN,WAIT_MAX))

    clock(float(data['V1 Lap Time']))

    if global_timer > BRAKE_TIME and waiting == 0:    
        throttle = 0
    if global_timer  >= EPISODE_TIME and waiting == 0:
        reset = start_new_episod(random.randint(WAIT_MIN,WAIT_MAX))

    steering  = add_steering_noise(find_gap(ranges))   



    if waiting == 0 and int(data['V1 Collisions']) > 0:  
        write_row(data, ranges, steering)
        reset = start_new_episod(random.randint(WAIT_MIN,WAIT_MAX) )

    elif waiting > 0 and int(data['V1 Collisions']) > 0:
        reset = "True"
        waiting = random.randint(WAIT_MIN, WAIT_MAX)


    if waiting > 1:   
        waiting -= 1

    if waiting == 1:
        global_timer = 0
        next_write = 0
        waiting -= 1


    if global_timer > next_write and waiting  == 0:
        write_row(data, ranges, steering)
        next_write += RECORD_STEP

    # Send my throttle and steering to the simulator
    sio.emit('Bridge', data={'V1 Throttle': str(throttle) , 'V1 Steering': str(steering) , 'Reset': str(reset)})


print('Door is open, waiting for the game...')
eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), socketio.WSGIApp(sio))
