import torch
import torch.nn as nn


LIDAR = 271
HIDDEN = 512
SPEED = 1
ACTION = 2

class NeuralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LIDAR + ACTION + SPEED, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, LIDAR),
        )

    def forward(self, lidar_now, action, speed):           
        x = torch.cat([lidar_now, action, speed], dim=1)  
        return self.net(x)


if __name__ == "__main__":
    model = NeuralModel()
# guess = model(torch.zeros(8, LIDAR), torch.zeros(8, 2), torch.zeros(8, 1))
# print(print(sum(p.numel() for p in model.parameters())))



    
