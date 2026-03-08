# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:08:35 2026

@author: Santosh Prakash
"""

def timestamp_transform(x):
    """
    Input:  B x T x C x H x W
    Output: B x C x T x H x W
    """
    return x.permute(0, 2, 1, 3, 4).contiguous()