# lidar , steering , throtle , speed , time 


import numpy as np

# DATA_DIR - directory in what save datset
DATA_DIR = "data/random_practice"
def save_npz(ranges,speed_list,time_list, noise_steering , my_throttle, episod, collisions, image_list, position_list, orientation_list, yaw_rate_list,real_steering_list,track, reason, warmup):
    np.savez_compressed(f"{DATA_DIR}/drive_{episod:04d}.npz", 
                        lidar = np.array(ranges, dtype = np.float16), 
                        throttle = my_throttle,
                        steering = noise_steering,
                        speed = speed_list,
                        time = time_list,
                        collisions = collisions,
                        image = image_list,
                        position = position_list,
                        orientation = orientation_list,
                        yaw_rate = yaw_rate_list,
                        real_steering = real_steering_list,
                        track = track,
                        reason = reason,
                        warmup = warmup
                        )

def show_npz(episod):

    with np.load(f"{DATA_DIR}/drive_{episod:04d}.npz") as file:
        lidar = file["lidar"]
        throttle = file["throttle"]
        steering = file["steering"]
        speed = file["speed"]
        time = file["time"]
        image = file["image"]

        print(file.files, lidar.shape , throttle.shape, steering.shape, speed.shape, time.shape, image.shape)
        print(lidar[:5])                    
        print(throttle[:5]) 
        print(steering[:5])  
        print(speed[:5])       
        print(time[:5]) 
        print(image[:5])  

    


