import torch.nn as nn
import config 

kernel_size = config.config.kernel_size
stride = config.config.stride
padding = config.config.padding

class CNN64_Model(nn.Module):
    def __init__(self):
        super(CNN64_Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 2, kernel_size=kernel_size, stride=stride, padding=padding),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
