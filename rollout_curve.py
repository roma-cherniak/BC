import numpy as np 
from simple_model.model import NeuralModel
import matplotlib.pyplot as plt
import torch 
import random 


guess = 0

STEPS = 20


d = np.load("data/test_pairs.npz")
lidar_now = torch.from_numpy(d["lidar_now"]).float()
action = torch.from_numpy(d["act"]).float()
speed = torch.from_numpy(d["spd"]).float()[:, None]
lidar_next = torch.from_numpy(d["lidar_next"]).float()





same = (lidar_next[:-1] == lidar_now[1:]).all(dim=1)  



print(same[0])
safe = []
for i in range(len(same) - STEPS):
    if same[i : i + STEPS].all():
        safe.append(i)



random.seed(0)
starts = random.sample(safe, 500)              
starts = torch.tensor(starts)                   
print(starts[:10], len(starts))



model = NeuralModel()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))




def draw_graph(brain_err, copy_err):
    seconds = [(k + 1) * 0.05 for k in range(STEPS)]

    plt.plot(seconds, brain_err, label="brain error", color="gray")
    plt.plot(seconds, copy_err, label="copy error", color="green")

    plt.xlabel("seconds")
    plt.ylabel("error")
    plt.legend()
    plt.savefig("rollout.png")
    plt.show()





def rollout_error(starts):

    guess_loss, copy_loss = [], []
    with torch.no_grad():
        guess = lidar_now[starts]
        for i in range(STEPS):     
            guess = model(guess, action[starts + i], speed[starts + i])
            real = lidar_now[starts + i + 1]    
            guess_loss.append(((guess - real) ** 2).mean().item())
            copy_loss.append(((lidar_now[starts] - real) ** 2).mean().item())

    return guess_loss, copy_loss


brain_err, copy_err = rollout_error(starts)
draw_graph(brain_err, copy_err)
