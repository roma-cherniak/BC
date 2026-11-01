import sys
import torch
import random
from core.data import load_pairs, safe_starts
from core.imagine import imagine_step
from core.models import load_run
from config import DEVICE, CURRENT_RUN



STEPS = 20 # 1 second

torch.manual_seed(0)   


def check_imagination(encoder, predictor, test):

    guess_loss, copy_loss = [], []

    with torch.no_grad():
        lidar_now, action, speed, lidar_next = test

        safe = safe_starts(lidar_now, lidar_next, STEPS)

        random.seed(0)
        starts = random.sample(safe.tolist(), 500)      # random.sample needs a plain list, not a tensor
        starts = torch.tensor(starts)


        latent_prediction = encoder(lidar_now[starts].float())
        start_prediction = latent_prediction

        for i in range(STEPS):

            latent_prediction = imagine_step(predictor, latent_prediction, action[starts + i], speed[starts + i],
                                             guard=True)
            real_latent = encoder(lidar_now[starts + i + 1].float())

            loss = ((real_latent - latent_prediction) ** 2).mean().item()
            start_loss = ((real_latent - start_prediction) ** 2).mean().item()

            guess_loss.append(loss)
            copy_loss.append(start_loss)

           

    return {"1 s, quess loss / copy loss": guess_loss[-1] / copy_loss[-1]}


if __name__ == "__main__":
    test = load_pairs("test")
    encoder, predictor, _ = load_run(sys.argv[1] if len(sys.argv) > 1 else CURRENT_RUN, DEVICE, with_zond=False)
    print(check_imagination(encoder, predictor, test))



















