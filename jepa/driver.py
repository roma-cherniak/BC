import torch
from utils import load_pairs, DEVICE, load_jepa, load_zond

STEERING = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
MOMENT = 1000
STEPS = 5



test = load_pairs("test")

lidar_now, action, speed, lidar_next = test


encoder, predictor = load_jepa()

zond = load_zond("zond.pt")


with torch.no_grad():
    prediction = encoder(lidar_now[MOMENT : MOMENT + 1].float())
    zond_g = zond(prediction)
    front_t, right_t, left_t, steering_t, speed_t_t = zond_g[0][:5]
    
    print(f"Start ----- front = {front_t} , right = {right_t}, left = {left_t}")
    best_steering = 0    
    sum = -1000
    for steering in STEERING:
        prediction = encoder(lidar_now[MOMENT : MOMENT + 1].float())
        

        for i in range(STEPS):
            act = torch.tensor([[action[MOMENT + i , 0].item(), steering]], device=DEVICE)   
            prediction = predictor(prediction , act , speed[MOMENT + i : MOMENT + i + 1])
            zond_guess = zond(prediction)
            front, right, left, steering_g, speed_g = zond_guess[0][:5]

            if i == STEPS - 1:
                print(f"Finish for steering = {steering} -------- front =  {front} , right = {right}, left = {left}")
                if front + right + left > sum:
                    sum = front + right + left
                    best_steering = steering

        print(" ################################################  \n")

print(best_steering, sum)
print(action[MOMENT, 1].item())























