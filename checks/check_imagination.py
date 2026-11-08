import sys
import torch
import random
from core.data import load_pairs, safe_starts, pair_obs
from core.imagine import imagine_step
from core.models import load_run, run_input
from config import DEVICE, CURRENT_RUN




STEPS = 20 # 1 second

torch.manual_seed(0)


def check_imagination(encoder, predictor, obs, action, speed , safe):


    guess_loss, copy_loss = [], []

    with torch.no_grad():


        random.seed(0)
        starts = random.sample(safe.tolist(), 500)      # random.sample needs a plain list, not a tensor
        starts = torch.tensor(starts)


        latent_prediction = encoder(obs[starts])
        start_prediction = latent_prediction

        for i in range(STEPS):

            latent_prediction = imagine_step(predictor, latent_prediction, action[starts + i], speed[starts + i],
                                             guard=True)
            real_latent = encoder(obs[starts + i + 1])

            loss = ((real_latent - latent_prediction) ** 2).mean().item()
            start_loss = ((real_latent - start_prediction) ** 2).mean().item()

            guess_loss.append(loss)
            copy_loss.append(start_loss)

           

    return {"1 s, quess loss / copy loss": guess_loss[-1] / copy_loss[-1]}


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else CURRENT_RUN
    encoder, predictor, _ = load_run(run, DEVICE, with_zond=False)
    lidar_now, action, speed, lidar_next = load_pairs("test")
    obs = pair_obs("test", run_input(run))                  # lidar or photos, one row per test pair
    safe = safe_starts(lidar_now, lidar_next, STEPS)        # the same trips for both: the lidar pairs know them
    print(check_imagination(encoder, predictor, obs, action, speed, safe))


















