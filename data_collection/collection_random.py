import eventlet
import eventlet.wsgi
import socketio
from data_collection.packet import read_packet
import random 
from data_collection.policy import decide, reset_policy
from data_collection.recorder import rows, add_row, save, new_episod

sio = socketio.Server()


packets = 0                                   
episod = 0
waiting = 0
global_timer = 0
lap_timer = 0
next_write = 0.0


TRACK = "practice"
warmup = 0   
WAIT_MIN = 60
WAIT_MAX = 1200
EPISODE_TIME = 30   
BRAKE_TIME = 28  
RECORD_STEP = 0.05



@sio.on('connect')
def connect(sid, environ):
    print('Connected! Caller ID:', sid)
               

def clock(episod_time):
    global lap_timer , global_timer
    if lap_timer < episod_time :
        global_timer += episod_time - lap_timer
        lap_timer = episod_time
    elif lap_timer > episod_time :
        lap_timer = episod_time
        global_timer += episod_time    


def start_new_episod(wait, reason):
        
        global global_timer, next_write, waiting, episod, warmup

        if len(rows) > 0:
            save(f"data/random_practice/drive_{episod:04d}.npz", TRACK, reason, warmup)

        new_episod()
        reset_policy()


        global_timer = 0
        next_write = 0.0
        waiting = wait
        warmup = wait
        episod += 1
        return 'True'


@sio.on("Bridge")                                  
def bridge(sid, data):

    global packets , waiting,  next_write, lap_timer, global_timer, warmup
    packets = packets + 1
    reset = "False"

    obs = read_packet(data)

    if packets == 1:   # Clear all data that was before
        lap_timer = obs["lap_time"]
        reset = start_new_episod(random.randint(WAIT_MIN,WAIT_MAX), "start")

    clock(obs["lap_time"])

    if global_timer  >= EPISODE_TIME and waiting == 0:
        reset = start_new_episod(random.randint(WAIT_MIN,WAIT_MAX), "timeout")


    throttle, steering, mode = decide(obs["lidar"], global_timer, waiting > 0)
    if global_timer > BRAKE_TIME and waiting == 0:
        throttle = 0


    if waiting == 0 and obs["collisions"] > 0:
        add_row(obs, steering, throttle, global_timer, mode)
        reset = start_new_episod(random.randint(WAIT_MIN,WAIT_MAX), "crash" )

    elif waiting > 0 and obs["collisions"] > 0:
        reset = "True"
        waiting = random.randint(WAIT_MIN, WAIT_MAX)
        warmup = waiting


    if waiting > 1:   
        waiting -= 1

    if waiting == 1:
        global_timer = 0
        reset_policy()
        next_write = 0
        waiting -= 1


    if global_timer > next_write and waiting  == 0:
        add_row(obs, steering, throttle, global_timer, mode)
        next_write += RECORD_STEP

    # Send my throttle and steering to the simulator
    sio.emit('Bridge', data={'V1 Throttle': str(throttle) , 'V1 Steering': str(steering) , 'Reset': str(reset)})


print('Door is open, waiting for the game...')
eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), socketio.WSGIApp(sio))
