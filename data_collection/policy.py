import math 
import numpy as np
import random

# working range from 180 to 900
# 900 - left , 180 - rinht

STEERING_NOISE = 0.3
r_steering = 0  
random_steering = 0

def dangarous_angle(distance, half_width=0.25):
    distance = max(distance, 0.1)
    angle = math.degrees(math.atan(half_width / distance))
    return int(angle * 4) 


def reset_noise():          
    global r_steering
    r_steering = 0

def add_steering_noise(steering, time):
    global r_steering, random_steering
    if time > r_steering :
        random_steering = random.uniform(-STEERING_NOISE , STEERING_NOISE)
        r_steering += random.uniform(0.3 , 2)


    steering = steering + random_steering
    return max(-1, min(1, steering))



def find_gap(ranges):   # ranges has type list
    front = ranges[180:901]  # range that i will work with from right to left 
    max_count = 0
    index = 0
    front = [min(r,3) for r in front]
    original = front.copy()
    jumps = np.diff(front)
    corners = np.where(np.abs(jumps) > 0.5)[0]

    for corner in corners:

        near = min(original[corner],original[corner + 1])
        dan_angle = dangarous_angle(near)
        for k in range((corner - dan_angle), (corner + dan_angle)):      # Safty buble 
            if 0 <= k < len(front):         
                front[k] = 0                



    winners = []    
    for i in range(0, len(front) - 80 + 1, 20):     # Choose the best 20 degres angle to ride  
        count = sum(front[i: i + 80])
        if count > max_count:          
            max_count = count
            winners = [i]             
        elif count == max_count:       
            winners.append(i)

    groups = [[winners[0]]]  
    for w in winners[1:]:
        if w - groups[-1][-1] == 20:             
            groups[-1].append(w)
        else:                                     
            groups.append([w])
    best = groups[0]
    for g in groups:
        if len(g) > len(best):
            best = g


    index = (best[0] + best[-1]) / 2
    index = index + 180
    center = index + 40
    angle = (center / 4) - 135
    steering_angle = max(-1, min(1, angle / 30))





    SIDE_LIMIT = 0.35                            # if wall is near from side just move forward
    right_side = min(ranges[90:300])           
    left_side = min(ranges[780:990])          
    if steering_angle < 0 and right_side < SIDE_LIMIT:
        steering_angle = 0                    
    if steering_angle > 0 and left_side < SIDE_LIMIT:
        steering_angle = 0          


    return steering_angle   


def steering_angle(ranges, time):
    return add_steering_noise(find_gap(ranges), time)
