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





class SpatialPromptGenerate(nn.Module):
    def __init__(self, embed_dim=256, image_embedding_size=16, tau=0.1, num_classes=1):
        super(SpatialPromptGenerate, self).__init__()
        self.tau = tau
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.image_embedding_size = (image_embedding_size, image_embedding_size)


        self.conv1 = nn.Sequential(
                nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Sigmoid())
        self.conv2 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))


    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, feature):
        N, C, H, W = feature.shape

        spatial_attention_map = self.conv1(feature)

        prompt = self.conv2(feature * (1 - spatial_attention_map) * self.alpha + feature * spatial_attention_map * self.beta)

        sparse_embeddings = torch.empty((1, 0, C), device=device)

        return sparse_embeddings, prompt
