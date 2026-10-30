import torch
from simple_model.model import NeuralModel
import time
import csv
from utils import load_pairs, DEVICE



BATCH = 256
LR = 0.001
EPOCHS = 40


log_file = open('loss.csv', 'w', newline='') 
writer = csv.writer(log_file)                         # a helper that writes CSV rows for us
writer.writerow(['epoch', 'train_loss', 'val_loss'])      # the header row: column names


train = load_pairs("train")
val = load_pairs("val")


model = NeuralModel().to(DEVICE)                             
optimizer = torch.optim.Adam(model.parameters(), lr=LR)




def train_one_epoch(data):
    lidar_now, action, speed, lidar_next = data
    order = torch.randperm(len(lidar_now), device=DEVICE)
    total = 0
    for start in range(0, len(order), BATCH):
        idx = order[start : start + BATCH]
        guess = model(lidar_now[idx].float(), action[idx], speed[idx])
        loss = ((guess - lidar_next[idx]) ** 2).mean()                                          
        optimizer.zero_grad()                               
        loss.backward()                                      
        optimizer.step()                                     
        total += loss.item() * len(idx)                     
    return total / len(order)                                



def val_loss(data):
    lidar_now, action, speed, lidar_next = data
    total = 0
    with torch.no_grad():
        for start in range(0, len(lidar_now), BATCH):
            end = start + BATCH
            guess = model(lidar_now[start:end].float(), action[start:end], speed[start:end])
            loss = ((guess - lidar_next[start:end]) ** 2).mean()
            total += loss.item() * len(guess)
    return total / len(lidar_now)    

best = 1
for epoch in range(EPOCHS):
    t0 = time.time()
    loss = train_one_epoch(train)
    v = val_loss(val)
    if v < best :
            best = v
            torch.save(model.state_dict(), "model.pt")
            print("model saved")
    print(f"epoch {epoch}  train {loss:.7f}  val {v:.7f}  val/copy {v / 0.0006847:.2f}")
    print(f"time - {time.time() - t0:.0f}")
    writer.writerow([epoch, loss, v])
    log_file.flush()



