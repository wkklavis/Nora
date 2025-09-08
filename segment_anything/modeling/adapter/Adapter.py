from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from typing import Type

from numpy import random

from segment_anything.modeling.adapter.DistributionUncertainty import DistributionUncertainty, DistributionNoise
from segment_anything.modeling.adapter.Separate import Separate


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

        self.noise = DistributionNoise()

    def forward(self, x):
        # x is (BT, HW+1, D)

        x = self.noise(x)

        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)


        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

