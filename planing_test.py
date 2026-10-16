import numpy as np 
from simple_model.model import NeuralModel
import matplotlib.pyplot as plt
import torch 

guess = 0
seconds = 2 * 20  # one second is 20 


d = np.load("data/test_pairs.npz")
lidar_now = torch.from_numpy(d["lidar_now"]).float()
action = torch.from_numpy(d["act"]).float()
speed = torch.from_numpy(d["spd"]).float()[:, None]
lidar_next = torch.from_numpy(d["lidar_next"]).float()



model = NeuralModel()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))

start = 4340


def show_graph(r_start, r_end , guess):




    angles = np.linspace(-135, 135, 271)

    plt.plot(angles, r_start * 10 , label="start", color="gray")
    plt.plot(angles, r_end * 10 , label="end", color="green")
    plt.plot(angles, guess * 10, "--", label=f"brain's guess for {seconds}", color="red")

    plt.xlabel("angle (degrees, + = left)")
    plt.ylabel("distance (m)")
    plt.legend()
    plt.savefig("guess.png")
    plt.show()






with torch.no_grad():
    guess = model(lidar_now[start : start + 1], action[start : start + 1], speed[start : start + 1]) 
    for i in range(start+1 , start + seconds):     
        guess = model(guess, action[i : i+1], speed[i : i+1])



show_graph(lidar_now[start],lidar_now[start + seconds], guess[0])


