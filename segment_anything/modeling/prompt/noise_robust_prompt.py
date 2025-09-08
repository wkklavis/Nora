import numpy as np
import torch
from torch import nn
from typing import Any, Optional, Tuple
import torch.nn.functional as F
from functools import reduce
from operator import mul
import math
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

class NoiseRobustPrompt(nn.Module):
    def __init__(self, dim=256, image_embedding_size=16):
        super(NoiseRobustPrompt, self).__init__()
        self.pe_layer = PositionEmbeddingRandom(dim // 2)
        self.image_embedding_size = (image_embedding_size, image_embedding_size)
        self.dim = dim
        self.num = 50

        self.tokens = nn.Parameter(torch.empty([self.num, self.dim]))
        self.proj = nn.Linear(self.dim, self.dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

        val = math.sqrt(6.0 / float(3 * reduce(mul, (image_embedding_size, image_embedding_size), 1) + self.dim))
        nn.init.uniform_(self.tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def robust_feat(self, feats, tokens):
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        attn = attn * (self.dim**-0.5)
        attn = F.softmax(attn, dim=-1)
        robust_fea = torch.einsum(
            "nbm,mc->nbc",
            attn,
            tokens
        )
        robust_fea = self.proj(robust_fea + feats)
        return robust_fea

    def forward(self, x):
        B, C, H, W = x.shape

        tokens = self.tokens
        # normalization tokens
        tokens = (tokens - tokens.mean(dim=0, keepdim=False)) / (tokens.var(dim=0, keepdim=False) + 1e-6).sqrt()

        # Reshape to (H*W, B, C) for processing
        x = x.permute(2, 3, 0, 1).reshape(H * W, B, C)

        if self.training:
            noise = torch.randn(self.num, self.dim).to(device)
            w = ((tokens @ tokens.T - torch.eye(self.num).to(device)) / float(self.dim)).mean(1)
            noise = noise * w.unsqueeze(1)
            tokens = tokens + noise * self.scale

        robust_feat = self.robust_feat(x, tokens)

        # Reshape back to (B, C, H, W)
        robust_prompt = robust_feat.permute(1, 2, 0).reshape(B, C, H, W)

        #tokens for spare prompt
        sparse_embeddings = torch.empty((1, 0, C), device=device)
        sparse_embeddings = torch.cat([sparse_embeddings, tokens.unsqueeze(0)], dim=1)

        return sparse_embeddings, robust_prompt