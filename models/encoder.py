import torch.nn as nn
from models.rhc_layer import RHCLayer

class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()

        self.layer1 = RHCLayer(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.layer2 = RHCLayer(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool2d(2)

        self.layer3 = RHCLayer(base_channels*2, base_channels*4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)

        x = self.layer2(x)
        x = self.pool2(x)

        x = self.layer3(x)

        return x
