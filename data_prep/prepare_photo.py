import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from core.data import load_images
from data_prep.prepare_data import clean_action
from config import SPEED_MAX , PHOTO_WIDTH, PHOTO_HEIGHT
from core.sensors import clean_lidar


def prepare_photo(name):

    with open(f"data/{name}.txt") as t:
        files = t.read().split("\n")
        image_list = []
        action_list = []
        speed_list = []
        trip_list = []
        lidar_list = []
        for file in files:
            with np.load(file) as f:
                images = f["image"]
                actions = clean_action(f)
                speed = f["speed"] / SPEED_MAX
                lidar = clean_lidar(f["lidar"])  

            number = int(file[-8:-4])
            trip = np.full(len(images), number)
            

            for image in images:

                picture = Image.open(io.BytesIO(image))
                picture = picture.resize((PHOTO_WIDTH,PHOTO_HEIGHT))
                picture = picture.convert("L")
                pixels = np.array(picture)
                image_list.append(pixels)


            action_list.append(actions)
            speed_list.append(speed)
            trip_list.append(trip)
            lidar_list.append(lidar)


        np.savez(f"data/{name}_photos.npz",
                 image = np.stack(image_list),
                 act = np.concatenate(action_list),
                 spd = np.concatenate(speed_list),
                 trip = np.concatenate(trip_list),
                 lidar = np.concatenate(lidar_list))


            

if __name__ == "__main__":
    prepare_photo("test")
    prepare_photo("train")
    prepare_photo("val")










