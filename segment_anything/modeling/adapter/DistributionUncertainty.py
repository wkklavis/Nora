import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import torchvision.transforms
from torch.nn import functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x
        x = x.permute(0, 3, 1, 2)

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        x = x.permute(0, 2, 3, 1)
        return x


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

        # 使用 torch.sigmoid() 函数生成 sigmoid weight
        weight = torch.sigmoid(x)
        # 在通道维度上取平均，以获得形状为 [8, 1, 16, 16] 的 weight
        weight = weight.mean(dim=1, keepdim=True)

        gaussian_noise = gaussian_noise * weight

        gaussian_feature = torch.add(x, gaussian_noise)
        return gaussian_feature.permute(0, 2, 3, 1)