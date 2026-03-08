import torch
import torch.nn as nn

class EMU3D(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv_r = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv_i = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv_g = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv_o = nn.Conv3d(channels, channels, 3, padding=1)

        self.norm = nn.BatchNorm3d(channels)

    def forward(self, x, prev_c):
        r = torch.sigmoid(self.conv_r(x))
        i = torch.sigmoid(self.conv_i(x))
        g = torch.tanh(self.conv_g(x))
        o = torch.sigmoid(self.conv_o(x))

        recall = torch.softmax(r, dim=2) * prev_c
        c = i * g + self.norm(prev_c + recall)
        h = o * torch.tanh(c)


        return h, c
