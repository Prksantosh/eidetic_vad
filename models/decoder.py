
import torch.nn as nn
from models.ae_rhc_layer import AERHCLayer

class Decoder(nn.Module):
    def __init__(self, base_channels, out_channels):
        super().__init__()

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer1 = AERHCLayer(base_channels*4, base_channels*2)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer2 = AERHCLayer(base_channels*2, base_channels)

        self.final = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.layer1(x)

        x = self.up2(x)
        x = self.layer2(x)


        return self.final(x)
