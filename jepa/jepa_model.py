import torch
import torch.nn as nn


LIDAR = 271
ACTION = 2
SPEED = 1
LATENT = 32
HIDDEN = 256


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LIDAR, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, LATENT)
        )

    def forward(self, lidar):
        return self.net(lidar)


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT + ACTION + SPEED, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, LATENT)
        )

    def forward(self, latent, action, speed):
        x = torch.cat([latent , action , speed], dim = 1)
        return self.net(x)


if __name__ == "__main__":

    encoder = Encoder()
    predictor = Predictor()
    latent_quess = encoder(torch.zeros(8,LIDAR))
    print(latent_quess.shape)
    quess = predictor(latent_quess, torch.zeros(8,ACTION), torch.zeros(8, SPEED))
    print(quess.shape)


