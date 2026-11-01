import matplotlib.pyplot as plt
import pandas as pd


file = pd.read_csv("loss.csv")

print(file.train_loss)

fig, ax = plt.subplots()
ax.plot(file.epoch, file.val_loss, label = "val_loss") 
ax.plot(file.epoch, file.train_loss, label = "train_loss")

ax.legend()
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Training results")
ax.grid()
plt.savefig("loss.png")
plt.show()

  
