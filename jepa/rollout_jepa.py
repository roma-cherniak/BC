import torch
from jepa.jepa_model import LATENT
import numpy as np
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from utils import load_pairs, DEVICE, load_jepa



BATCH = 256
LR = 0.001
PROBE_EPOCHS = 20
TRAIN_MOMENTS = 100000
STEPS = 20 # 1 second


torch.manual_seed(0)                                  # the same random moments and random encoder every run




test = load_pairs("test")
lidar_now, action, speed, lidar_next = test

same = (lidar_next[:-1] == lidar_now[1:]).all(dim=1)  



print(same[0])
safe = []
for i in range(len(same) - STEPS):
    if same[i : i + STEPS].all():
        safe.append(i)



random.seed(0)
starts = random.sample(safe, 500)              
starts = torch.tensor(starts)                   




encoder, predictor = load_jepa()

def rollout_error():

    guess_loss, copy_loss = [], []

    with torch.no_grad():
        lidar_now, action, speed, lidar_next = test
        latent_prediction = encoder(lidar_now[starts].float())
        start_prediction = latent_prediction

        for i in range(STEPS):

            latent_prediction = predictor(latent_prediction, action[starts + i], speed[starts + i])
            latent_prediction = latent_prediction / latent_prediction.norm(dim=1, keepdim=True) * 32 ** 0.5
            real_latent = encoder(lidar_now[starts + i + 1].float())

            loss = ((real_latent - latent_prediction) ** 2).mean().item()
            start_loss = ((real_latent - start_prediction) ** 2).mean().item()

            guess_loss.append(loss)
            copy_loss.append(start_loss)

    return guess_loss, copy_loss


def draw_graph(brain_err, copy_err):
    seconds = [(k + 1) * 0.05 for k in range(STEPS)]

    plt.plot(seconds, brain_err, label="brain error", color="gray")
    plt.plot(seconds, copy_err, label="copy error", color="green")

    plt.xlabel("seconds")
    plt.ylabel("error")
    plt.legend()
    plt.savefig("png/rollout_jepa.png")
    plt.show()

brain_err, copy_err = rollout_error()
draw_graph(brain_err, copy_err)


