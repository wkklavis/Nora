import numpy as np
import torch
from torch import nn
from typing import Any, Optional, Tuple, Type
import torch.nn.functional as F
from segment_anything.modeling.prompt.gumbel_sigmoid import GumbelSigmoid

device = "cuda" if torch.cuda.is_available() else "cpu"
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class Separation(torch.nn.Module):
    def __init__(self, size, num_channel=32, tau=0.1):
        super(Separation, self).__init__()
        C, H, W = size
        self.C, self.H, self.W = C, H, W
        self.tau = tau

        self.sep_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, feat):
        rob_map = self.sep_net(feat)

        mask = rob_map.reshape(rob_map.shape[0], 1, -1)
        mask = torch.nn.Sigmoid()(mask)
        mask = GumbelSigmoid(tau=self.tau)(mask)
        mask = mask[:, 0].reshape(mask.shape[0], self.C, self.H, self.W)

        pos_feat = feat * mask
        neg_feat = feat * (1 - mask)

        return pos_feat, neg_feat, mask


class ContrastPromptGenerate(nn.Module):
    def __init__(self, embed_dim=256, image_embedding_size=16, tau=0.1, num_classes=1):
        super(ContrastPromptGenerate, self).__init__()
        self.tau = tau
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.image_embedding_size = (image_embedding_size, image_embedding_size)
        self.separation = Separation(size=(embed_dim, image_embedding_size, image_embedding_size), tau=self.tau)
        self.aux1 = nn.ConvTranspose2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux2 = nn.ConvTranspose2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.pos_prompt = nn.Parameter(torch.Tensor(1, embed_dim, image_embedding_size, image_embedding_size))
        self.neg_prompt = nn.Parameter(torch.Tensor(1, embed_dim, image_embedding_size, image_embedding_size))

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, feature):
        N, C, H, W = feature.shape
        pos_feat, neg_feat, mask = self.separation(feature)

        pos_prompt = self.pos_prompt.repeat(N, 1, 1, 1)+pos_feat
        neg_prompt = self.pos_prompt.repeat(N, 1, 1, 1)+neg_feat

        pos_out = self.aux1(pos_prompt)
        pos_out = F.interpolate(pos_out, scale_factor=16, mode='bilinear', align_corners=False)

        neg_out = self.aux2(neg_prompt)
        neg_out = F.interpolate(neg_out, scale_factor=16, mode='bilinear', align_corners=False)


        # pos_prompt = self.pos_prompt.repeat(N, 1, 1, 1)
        # neg_prompt = self.pos_prompt.repeat(N, 1, 1, 1)
        # pos_out = self.aux(pos_prompt)
        # pos_out = F.interpolate(pos_out, scale_factor=16, mode='bilinear', align_corners=False)
        #
        # neg_out = self.aux(neg_prompt)
        # neg_out = F.interpolate(neg_out, scale_factor=16, mode='bilinear', align_corners=False)
        sparse_embeddings = torch.empty((1, 0, C), device=device)

        return sparse_embeddings, pos_prompt, neg_prompt, pos_out, neg_out

