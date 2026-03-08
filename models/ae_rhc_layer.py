# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:07:52 2026

@author: Santosh Prakash
"""

import torch
import torch.nn as nn
from models.rhc_layer import RHCLayer

class SEResidual(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class AERHCLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.rhc = RHCLayer(in_channels, out_channels)
        self.se = SEResidual(out_channels)

    def forward(self, x):
        x = self.rhc(x)
        return self.se(x)