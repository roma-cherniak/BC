import numpy as np
import torch
from jepa_model import Encoder, Predictor
import time
import csv
from sigreg import sigreg


DEVICE = "cuda" 
BATCH = 256
LR = 0.001
EPOCHS = 50
SIGREG_WEIGHT = 0.09
K = 10

log_file = open('jepa_loss.csv', 'w', newline='') 
writer = csv.writer(log_file)                         # a helper that writes CSV rows for us
writer.writerow(['epoch', 'train_loss', 'val_loss', 'spread', 'copy_loss', 'sigreg'])      # the header row: column names





def load_pairs(name):
    d = np.load(f"data/{name}_pairs.npz")
    lidar_now = torch.from_numpy(d["lidar_now"]).to(DEVICE)
    action = torch.from_numpy(d["act"]).float().to(DEVICE)
    speed = torch.from_numpy(d["spd"]).float().to(DEVICE)[:, None]
    lidar_next = torch.from_numpy(d["lidar_next"]).to(DEVICE)
    return lidar_now, action, speed, lidar_next



train = load_pairs("train")
val = load_pairs("val")

lidar_now, action, speed, lidar_next = train

same_train = (lidar_next[:-1] == lidar_now[1:]).all(dim=1).cpu()



print(same_train[0])
safe_train = []
for i in range(len(same_train) - K):
    if same_train[i : i + K].all():
        safe_train.append(i)

safe_train = torch.tensor(safe_train, device=DEVICE)

encoder = Encoder().to(DEVICE)
predictor = Predictor().to(DEVICE)

optimizer = torch.optim.Adam(list(encoder.parameters()) +  list(predictor.parameters()), lr=LR)



def train_one_epoch(data):
    lidar_now, action, speed, lidar_next = data
    order = safe_train[torch.randperm(len(safe_train), device=DEVICE)]
    total = 0
    total_reg = 0
    number_of_batches = 0
    for start in range(0, len(order), BATCH):
        idx = order[start : start + BATCH]
        latent_now = encoder(lidar_now[idx].float())   
        latent_next = encoder(lidar_next[idx].float())
        reg = (sigreg(latent_now) + sigreg(latent_next)) / 2
        z = latent_now
        loss = 0
        for k in range(K):
            z = predictor(z, action[idx + k].float(), speed[idx + k].float())
            loss = loss + ((z - encoder(lidar_next[idx + k].float())) ** 2).mean()
        loss = loss / K
        sig_loss = loss + SIGREG_WEIGHT * reg 
        optimizer.zero_grad()   
        sig_loss.backward()
        optimizer.step()

        total += loss.item() * len(idx)   
        total_reg += reg.item() 
        number_of_batches += 1
    return total / len(order), total_reg / number_of_batches




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
    loss, sigloss = train_one_epoch(train)
    v_loss, copy_loss, spread = val_loss(val)

    torch.save(predictor.state_dict(), "jepa_predoctor.pt")
    torch.save(encoder.state_dict(), "jepa_encoder.pt")
    print("model saved")
    print(f"epoch {epoch}  train {loss:.7f}  val {v_loss:.7f} spread {spread:.7} copy_loss {copy_loss:.7} sigloss {sigloss}")
    print(f"time - {time.time() - t0:.0f}")
    writer.writerow([epoch, loss, v_loss,spread, copy_loss,sigloss])
    log_file.flush()

                 
        