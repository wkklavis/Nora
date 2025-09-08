import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import torchvision.transforms
from torch.nn import functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"


class DistributionNoise(nn.Module):
    def __init__(self, p=0.5):
        super(DistributionNoise, self).__init__()
        self.p = p
        self.alpha = nn.Parameter(torch.ones(768), requires_grad=True)

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x
        x = x.permute(0, 3, 1, 2)
        N, C, H, W = x.shape

        mean = x.mean(dim=[2, 3], keepdim=False).reshape(x.shape[0], x.shape[1], 1, 1)
        std = (x.var(dim=[2, 3], keepdim=False) + 1e-6).sqrt().reshape(x.shape[0], x.shape[1], 1, 1)

        # Gaussian noise ∼ N(mean, std)
        gaussian_noise = torch.randn(N, C, H, W).to(device)
        gaussian_noise = gaussian_noise * std + mean

        gaussian_noise = gaussian_noise * self.alpha.unsqueeze(-1).unsqueeze(-1)

        # using torch.sigmoid() function to generate sigmoid weight
        weight = torch.sigmoid(x)
        # Take the average in the channel dimension  [8, 1, 16, 16]
        weight = weight.mean(dim=1, keepdim=True)

        gaussian_noise = gaussian_noise * weight

        gaussian_feature = torch.add(x, gaussian_noise)
        return gaussian_feature.permute(0, 2, 3, 1)