import numpy as np

d = np.load("data/val_pairs.npz")
now = d["lidar_now"]
nxt = d["lidar_next"]

loss = ((now - nxt) ** 2).mean()
print("copy baseline MSE:", loss)
