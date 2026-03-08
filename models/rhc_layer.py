# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:07:03 2026

@author: Santosh Prakash
"""

import torch
import torch.nn as nn

class RHCLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        half = in_channels // 2

        # Regular branch
        self.reg_conv = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, 3, padding=1),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True)
        )

        # Offset branch (simulated with dilated conv)
        self.off_conv = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=2, dilation=2),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, 3, padding=2, dilation=2),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True)
        )

        self.shuffle = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        c = x.shape[1] // 2
        x1, x2 = x[:, :c], x[:, c:]

        r = self.reg_conv(x1)
        o = self.off_conv(x2)

        out = torch.cat([r, o], dim=1)
        out = self.shuffle(out)

        return self.norm(out + x)