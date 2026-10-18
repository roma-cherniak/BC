import numpy as np
import torch
from jepa_model import Encoder, Predictor
import time
import csv



DEVICE = "cuda" 
BATCH = 256
LR = 0.001
EPOCHS = 10


log_file = open('jepa_loss.csv', 'w', newline='') 
writer = csv.writer(log_file)                         # a helper that writes CSV rows for us
writer.writerow(['epoch', 'train_loss', 'val_loss', 'spread', 'copy_loss'])      # the header row: column names



def load_pairs(name):
    d = np.load(f"data/{name}_pairs.npz")
    lidar_now = torch.from_numpy(d["lidar_now"]).to(DEVICE)
    action = torch.from_numpy(d["act"]).float().to(DEVICE)
    speed = torch.from_numpy(d["spd"]).float().to(DEVICE)[:, None]
    lidar_next = torch.from_numpy(d["lidar_next"]).to(DEVICE)
    return lidar_now, action, speed, lidar_next



train = load_pairs("train")
val = load_pairs("val")


encoder = Encoder().to(DEVICE)
predictor = Predictor().to(DEVICE)

optimizer = torch.optim.Adam(list(encoder.parameters()) +  list(predictor.parameters()), lr=LR)



def train_one_epoch(data):
    lidar_now, action, speed, lidar_next = data
    order = torch.randperm(len(lidar_now), device=DEVICE)
    total = 0
    for start in range(0, len(order), BATCH):
        idx = order[start : start + BATCH]
        latent_now = encoder(lidar_now[idx].float())   
        latent_next = encoder(lidar_next[idx].float())
        guess = predictor(latent_now, action[idx].float(), speed[idx].float())
        loss = ((guess - latent_next) ** 2).mean() 
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()

        total += loss.item() * len(idx)                     
    return total / len(order) 




def val_loss(data):
    lidar_now, action, speed, lidar_next = data
    total = 0
    copy_total = 0
    all_z = []
    with torch.no_grad():
        for start in range(0, len(lidar_now), BATCH):
            end = start + BATCH
            latent_now = encoder(lidar_now[start : end].float())   
            latent_next = encoder(lidar_next[start : end].float())
            guess = predictor(latent_now, action[start : end].float(), speed[start : end].float())
            loss = ((guess - latent_next) ** 2).mean() 
            copy_loss = ((latent_next - latent_now) ** 2).mean() 
            total += loss.item() * len(guess)
            copy_total += copy_loss.item() * len(guess)

            all_z.append(latent_now)
    spread = torch.cat(all_z)
    spread = spread.std(dim = 0).mean().item()

    return total / len(lidar_now), copy_total / len(lidar_now) , spread   

best = 1
for epoch in range(EPOCHS):
    t0 = time.time()
    loss = train_one_epoch(train)
    v_loss, copy_loss, spread = val_loss(val)

    torch.save(predictor.state_dict(), "jepa_predoctor.pt")
    torch.save(encoder.state_dict(), "jepa_encoder.pt")
    print("model saved")
    print(f"epoch {epoch}  train {loss:.7f}  val {v_loss:.7f} spread {spread:.7} copy_loss {copy_loss:.7}")
    print(f"time - {time.time() - t0:.0f}")
    writer.writerow([epoch, loss, v_loss,spread, copy_loss])
    log_file.flush()

                 
        