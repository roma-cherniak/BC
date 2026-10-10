import eventlet
import eventlet.wsgi
import socketio
import base64
import gzip
import csv 
import time



sio = socketio.Server()

log_file = open('my_drive.csv', 'w', newline='') 
writer = csv.writer(log_file)                         # a helper that writes CSV rows for us
writer.writerow(['episode', 'time', 'speed', 'throttle', 'steering',
                 'right', 'front', 'left', 'collisions'])      # the header row: column names
packets = 0                                   # counter: how many messages so far
episod = 0




@sio.on('connect')
def connect(sid, environ):
    print('Connected! Caller ID:', sid)

def read_laser(data):
    packed = data['V1 LIDAR Range Array']                        # the zipped text
    text = gzip.decompress(base64.b64decode(packed)).decode('utf-8')   # unzip → normal text
    return [float(v) for v in text.split()]                      





@sio.on("Bridge")                                  # BLANK 1
def bridge(sid, data):
    global packets , episod
    packets = packets + 1
    reset = "False"
    steering = 0
    episod
    ranges = read_laser(data)
    if packets == 1:
        print('The game sends these drawers:', list(data.keys()))   # BLANK 2

    if packets % 60 == 0:                            # BLANK 3
        print('speed:', data['V1 Speed'])

        print('beams:', len(ranges),
              ' right:', ranges[180],      # BLANK: beam number for right
              ' front:', ranges[540],      # BLANK: beam number for front
              ' left:', ranges[900])       # BLANK: beam number for left
        print(data['V1 Collisions'])



    if int(data['V1 Collisions']) > 3: 
        reset = "True"
        episod += 1


    if ranges[180] < 0.5:
        steering = 0.5
    elif ranges[900] < 0.5:
        steering = -0.5
    elif ranges[540] < 0.5:
        if ranges[180] < ranges[900]:
            steering = 0.8
        else:
            steering = -0.8


    if packets < 11800:
        throttle = 0.05
    else :  
        throttle = 0

    if packets == 12000:
        packets = 0
        reset = "True"
        episod += 1



    if packets % 6 == 0 :

        writer.writerow([episod, round(time.time(), 3), data['V1 Speed'],throttle ,steering ,
                  ranges[180], ranges[540],  ranges[900],data['V1 Collisions'] ])
        log_file.flush()

        print(episod)



    # politely answer "do nothing": no gas, no steering, no reset
    sio.emit('Bridge', data={'V1 Throttle': str(throttle) , 'V1 Steering': str(steering), 'Reset': str(reset)})


print('Door is open, waiting for the game...')
eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), socketio.WSGIApp(sio))
