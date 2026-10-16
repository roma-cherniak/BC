import numpy as np 
from simple_model.model import NeuralModel
import matplotlib.pyplot as plt
import torch 



d = np.load("data/test_pairs.npz")
lidar_now = torch.from_numpy(d["lidar_now"]).float()
action = torch.from_numpy(d["act"]).float()
speed = torch.from_numpy(d["spd"]).float()[:, None]
lidar_next = torch.from_numpy(d["lidar_next"]).float()

    
model = NeuralModel()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
def show_graph(moment):
    i = moment
    with torch.no_grad():
        guess = model(lidar_now[i : i+1], action[i : i+1], speed[i : i+1])[0]


    angles = np.linspace(-135, 135, 271)

    plt.plot(angles, lidar_now[i] * 10 , label="now", color="gray")
    plt.plot(angles, lidar_next[i] * 10, label="real next", color="green")
    plt.plot(angles, guess * 10, "--", label="brain's guess", color="red")

    plt.xlabel("angle (degrees, + = left)")
    plt.ylabel("distance (m)")
    plt.legend()
    plt.savefig("guess.png")
    plt.show()


def show_loss():
    with torch.no_grad():
        guess = model(lidar_now, action, speed)

    loss = ((guess - lidar_next) ** 2).mean().item()
    copy_loss = ((lidar_next - lidar_now) ** 2).mean().item()

    print(f"Test loss = {loss:.4} , copy_loss = {copy_loss:.4}")

show_loss()
