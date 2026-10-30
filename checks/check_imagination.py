import torch
from jepa.jepa_model import LATENT
import random
from utils import load_pairs, load_jepa



STEPS = 20 # 1 second

torch.manual_seed(0)   







def check_imagination(encoder, predictor, test):

    guess_loss, copy_loss = [], []

    with torch.no_grad():
        lidar_now, action, speed, lidar_next = test

        same = (lidar_next[:-1] == lidar_now[1:]).all(dim=1)  
        safe = []
        for i in range(len(same) - STEPS):
            if same[i : i + STEPS].all():
                safe.append(i)



        random.seed(0)
        starts = random.sample(safe, 500)              
        starts = torch.tensor(starts) 


        latent_prediction = encoder(lidar_now[starts].float())
        start_prediction = latent_prediction

        for i in range(STEPS):

            latent_prediction = predictor(latent_prediction, action[starts + i], speed[starts + i])
            latent_prediction = latent_prediction / latent_prediction.norm(dim=1, keepdim=True) * LATENT ** 0.5
            real_latent = encoder(lidar_now[starts + i + 1].float())

            loss = ((real_latent - latent_prediction) ** 2).mean().item()
            start_loss = ((real_latent - start_prediction) ** 2).mean().item()

            guess_loss.append(loss)
            copy_loss.append(start_loss)

           

    return {"1 s": guess_loss[-1] / copy_loss[-1]}


if __name__ == "__main__":
    test = load_pairs("test")
    encoder, predictor = load_jepa()
    print(check_imagination(encoder, predictor, test))



















