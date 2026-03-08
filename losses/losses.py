# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:11:24 2026

@author: Santosh Prakash
"""

import torch.nn as nn

class PredictionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, gt):
        return self.mse(pred, gt)