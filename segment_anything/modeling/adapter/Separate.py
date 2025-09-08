
from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from typing import Type

from numpy import random


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        output = self.sigmoid(max_result + avg_result)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class Separate(nn.Module):
    def __init__(self, kernel_size=7, channel=768):
        super().__init__()
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.ca = ChannelAttention(channel)


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        w = self.sa(x)*self.ca(x)#
        xc = x * w
        xs = x - xc
        return xc, xs

